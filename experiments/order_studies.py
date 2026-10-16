#!/usr/bin/env python3
"""Inference-time error-recovery and slot-order studies (App. I, "Early errors and slot order").

Trains (or reloads from a checkpoint) ``evidence_linear`` and ``vanilla`` on
task A (sinusoids) with the synthetic harness, then runs three eval-only
studies on the held-out test set:

(a) ``forced_error`` -- replace the first processed slot's attention with a
    corrupted allocation (uniform over tokens, or one-hot on the peak-energy
    token of the weakest true source); depletion proceeds normally for the
    remaining slots. Measures how many true sources the *remaining* slots still
    recover, against the unperturbed model and against vanilla SA under the
    same perturbation (do early errors cascade?).
(b) ``order_sensitivity`` -- canonical order vs. ``--n-orders`` random slot
    orders at inference, plus an ensemble averaging predictions over orders.
(c) ``evidence_ordered`` -- greedy evidence-guided order
    (:meth:`EvidenceDepletionSlotAttention.forward_evidence_greedy`) vs. canonical.

Records (``sweep="order_studies"`` with a ``study`` field) go to
``<output-dir>/order_studies_results.json``; trained weights are checkpointed
under ``<output-dir>/checkpoints`` so a resumed job never retrains.

Usage::

    python experiments/order_studies.py --seeds 3 --epochs 200
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from evidence_depletion.data.synthetic import TASKS
from evidence_depletion.io import load_results
from evidence_depletion.metrics import (
    cosine_overlap,
    param_recovery_error,
    peak_overlap_rate,
    source_recovery,
)
from evidence_depletion.models import build_model
from evidence_depletion.training import default_device, train_synthetic

TASK_NAME = "sinusoids"
METHODS = ["evidence_linear", "vanilla"]
PERTURBATIONS = ["none", "uniform", "wrong_peak"]
RESULTS_FILE = "order_studies_results.json"

Data = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def make_config(args: argparse.Namespace) -> dict[str, Any]:
    """The synthetic harness configuration for task A."""
    return {
        "n_train": args.n_train,
        "n_test": args.n_test,
        "N": 256,
        "K_range": (1, 4),
        "d_model": 64,
        "n_tokens": 16,
        "batch_size": 128,
        "lr": 3e-4,
        "epochs": args.epochs,
        "n_iter": 3,
        "gamma": 3.0,
        "tau": 0.3,
    }


def get_model(
    method: str,
    seed: int,
    config: dict[str, Any],
    ckpt_dir: Path,
    device: torch.device,
    use_checkpoint: bool = True,
) -> tuple[nn.Module, dict[str, Any] | None]:
    """Return ``(model, train_record)``: train with the synthetic harness and
    checkpoint, or reload an existing checkpoint if ``use_checkpoint``
    (``train_record`` is then ``None``)."""
    n_slots = config.get("n_slots") or (config["K_range"][1] + 1)
    ckpt = ckpt_dir / f"{method}_seed{seed}_ep{config['epochs']}.pt"

    if use_checkpoint and ckpt.exists():
        model = build_model(
            method=method,
            n_params=TASKS[TASK_NAME]["n_params"],
            N=config["N"],
            d_model=config["d_model"],
            n_tokens=config["n_tokens"],
            n_slots=n_slots,
            tokenizer_type="fft",
            gamma=config["gamma"],
            tau=config["tau"],
            n_iter=config["n_iter"],
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device)
        print(f"    loaded checkpoint {ckpt.name}")
        return model, None

    train_record, model = train_synthetic(TASK_NAME, method, seed, {**config, "return_model": True})
    torch.save(model.state_dict(), ckpt)
    return model, train_record


def batched_forward(
    model: nn.Module,
    signals: torch.Tensor,
    device: torch.device,
    eval_seed: int,
    batch_size: int = 128,
    sa_kwargs_fn: Callable[[int, int], dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eval-mode forward over the test set: ``(attn, exist_logits, params_pred)``.

    The slot-initialization noise is identical across conditions: each batch
    runs under a forked RNG seeded with ``eval_seed + batch_start``.
    ``sa_kwargs_fn(start, stop)`` supplies slot-attention keyword arguments.
    """
    model.eval()
    fork_devices = [device] if device.type == "cuda" else []
    attns, exists, params = [], [], []
    with torch.no_grad():
        for i in range(0, signals.shape[0], batch_size):
            x = signals[i : i + batch_size].to(device)
            kwargs = sa_kwargs_fn(i, i + x.shape[0]) if sa_kwargs_fn else {}
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(eval_seed + i)
                _, attn, exist_logits, params_pred, _ = model(x, **kwargs)
            attns.append(attn.cpu())
            exists.append(exist_logits.cpu())
            params.append(params_pred.cpu())
    return torch.cat(attns), torch.cat(exists), torch.cat(params)


def full_metrics(
    attn: torch.Tensor,
    exist: torch.Tensor,
    params_pred: torch.Tensor,
    params_true: torch.Tensor,
    K_true: torch.Tensor,
    n_params: int,
) -> dict[str, float]:
    """Peak overlap, cosine overlap, mean and frequency parameter MAE."""
    per_param_mae, n_matched = param_recovery_error(
        params_pred, params_true, exist, K_true, n_params
    )
    return {
        "collapse_rate": round(peak_overlap_rate(attn, K_true), 4),
        "mean_attn_overlap": round(cosine_overlap(attn, K_true), 4),
        "param_mae": round(float(per_param_mae.mean().item()), 6) if n_matched else float("nan"),
        "mae_frequency": round(float(per_param_mae[0].item()), 6) if n_matched else float("nan"),
    }


def wrong_peak_attention(
    components: torch.Tensor, K_true: torch.Tensor, params_true: torch.Tensor, n_tokens: int
) -> torch.Tensor:
    """Corrupted first-slot allocation: one-hot on the peak-energy token of the
    weakest true source (the source a trained first slot would not claim).

    Token energy is the sum of squares within each tokenizer chunk. Sinusoids
    are near-stationary, so this is only weakly peaked; the uniform corruption
    is the complementary, token-agnostic case.
    """
    B, _, N = components.shape
    chunk = N // n_tokens
    forced = torch.zeros(B, n_tokens)
    for b in range(B):
        K = int(K_true[b].item())
        k_weak = int(params_true[b, :K, 1].argmin().item()) if K > 0 else 0
        energy = components[b, k_weak].reshape(n_tokens, chunk).pow(2).sum(-1)
        forced[b, energy.argmax()] = 1.0
    return forced


def study_forced_error(
    model: nn.Module,
    method: str,
    seed: int,
    data: Data,
    device: torch.device,
    eval_seed: int,
    freq_tol: float,
    n_params: int,
) -> list[dict[str, Any]]:
    """(a) Forced-error recovery. Slot 0 is excluded from the candidate pool
    for the "remaining" metrics: in eval mode the first processed slot is
    slot 0 for the sequential models and, by convention, for vanilla SA."""
    signals, params_true, K_true, components = data
    forced_wrong = wrong_peak_attention(components, K_true, params_true, n_tokens=16)
    uniform = torch.full((signals.shape[0], 16), 1.0 / 16)
    records = []
    for pert in PERTURBATIONS:
        if pert == "none":
            fn = None
        else:
            forced = uniform if pert == "uniform" else forced_wrong

            def fn(a: int, b: int, t: torch.Tensor = forced) -> dict[str, Any]:
                return {"forced_attn_first": t[a:b].to(device)}

        attn, exist, params_pred = batched_forward(
            model, signals, device, eval_seed, sa_kwargs_fn=fn
        )
        rec_rem, mae_rem = source_recovery(
            params_pred, params_true, exist, K_true, freq_tol=freq_tol, exclude_slot=0
        )
        rec_all, mae_all = source_recovery(
            params_pred, params_true, exist, K_true, freq_tol=freq_tol
        )
        records.append(
            {
                "task": TASK_NAME,
                "method": method,
                "seed": seed,
                "sweep": "order_studies",
                "study": "forced_error",
                "perturbation": pert,
                "freq_tol": freq_tol,
                "recovery_remaining": round(rec_rem, 4),
                "mae_remaining": round(mae_rem, 6),
                "recovery_all_slots": round(rec_all, 4),
                "mae_all_slots": round(mae_all, 6),
                **full_metrics(attn, exist, params_pred, params_true, K_true, n_params),
            }
        )
    return records


def study_order_sensitivity(
    model: nn.Module,
    method: str,
    seed: int,
    data: Data,
    device: torch.device,
    eval_seed: int,
    n_orders: int,
    n_params: int,
) -> list[dict[str, Any]]:
    """(b) Canonical order vs. random slot orders (plus a prediction ensemble)."""
    signals, params_true, K_true, _ = data
    n_slots = model.slot_attn.n_slots

    def eval_order(order: list[int] | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fn = (lambda a, b: {"slot_order": order}) if order is not None else None
        return batched_forward(model, signals, device, eval_seed, sa_kwargs_fn=fn)

    attn_c, exist_c, params_c = eval_order(None)
    canon = full_metrics(attn_c, exist_c, params_c, params_true, K_true, n_params)

    rng = np.random.RandomState(1234 + seed)
    per_order, sum_attn, sum_exist, sum_params = [], 0, 0, 0
    for _ in range(n_orders):
        order = rng.permutation(n_slots).tolist()
        attn_o, exist_o, params_o = eval_order(order)
        per_order.append(full_metrics(attn_o, exist_o, params_o, params_true, K_true, n_params))
        sum_attn = sum_attn + attn_o
        sum_exist = sum_exist + exist_o
        sum_params = sum_params + params_o

    # Outputs are indexed by slot identity, so per-slot averaging is well defined.
    ens = full_metrics(
        sum_attn / n_orders,
        sum_exist / n_orders,
        sum_params / n_orders,
        params_true,
        K_true,
        n_params,
    )

    record: dict[str, Any] = {
        "task": TASK_NAME,
        "method": method,
        "seed": seed,
        "sweep": "order_studies",
        "study": "order_sensitivity",
        "perturbation": None,
        "n_orders": n_orders,
    }
    for k, v in canon.items():
        record[f"canonical_{k}"] = v
    for key in ["collapse_rate", "mean_attn_overlap", "param_mae", "mae_frequency"]:
        vals = [m[key] for m in per_order]
        record[f"random_{key}_mean"] = round(float(np.mean(vals)), 6)
        record[f"random_{key}_std"] = round(float(np.std(vals)), 6)
    for k, v in ens.items():
        record[f"ensemble_{k}"] = v
    return [record]


def study_evidence_ordered(
    model: nn.Module,
    method: str,
    seed: int,
    data: Data,
    device: torch.device,
    eval_seed: int,
    n_params: int,
) -> list[dict[str, Any]]:
    """(c) Greedy evidence-guided order vs. canonical order."""
    signals, params_true, K_true, _ = data
    attn_c, exist_c, params_c = batched_forward(model, signals, device, eval_seed)
    canon = full_metrics(attn_c, exist_c, params_c, params_true, K_true, n_params)

    attn_g, exist_g, params_g = batched_forward(
        model,
        signals,
        device,
        eval_seed,
        sa_kwargs_fn=lambda a, b: {"order_policy": "evidence_greedy"},
    )
    greedy = full_metrics(attn_g, exist_g, params_g, params_true, K_true, n_params)

    record: dict[str, Any] = {
        "task": TASK_NAME,
        "method": method,
        "seed": seed,
        "sweep": "order_studies",
        "study": "evidence_ordered",
        "perturbation": None,
    }
    for k, v in canon.items():
        record[f"canonical_{k}"] = v
    for k, v in greedy.items():
        record[f"greedy_{k}"] = v
    return [record]


def expected_keys(method: str) -> list[tuple[str, str | None]]:
    """(study, perturbation) pairs produced for ``method``."""
    keys: list[tuple[str, str | None]] = [("forced_error", p) for p in PERTURBATIONS]
    if method.startswith("evidence"):
        keys += [("order_sensitivity", None), ("evidence_ordered", None)]
    return keys


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    out_path = out_dir / RESULTS_FILE

    all_results = load_results(out_path, resume=not args.no_resume)

    def done(method: str, seed: int, study: str, perturbation: str | None = None) -> bool:
        return any(
            r["method"] == method
            and r["seed"] == seed
            and r["study"] == study
            and r.get("perturbation") == perturbation
            for r in all_results
            if r.get("sweep") == "order_studies"
        )

    def save() -> None:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    config = make_config(args)
    n_params = TASKS[TASK_NAME]["n_params"]
    device = default_device()
    task = TASKS[TASK_NAME]

    for method in METHODS:
        for seed in range(args.seeds):
            pending = [k for k in expected_keys(method) if not done(method, seed, *k)]
            if not pending:
                print(f"method={method} seed={seed} ... SKIPPED (all studies done)")
                continue

            print(
                f"method={method} seed={seed} — training/loading ({config['epochs']} epochs) ...",
                flush=True,
            )
            model, train_record = get_model(
                method, seed, config, ckpt_dir, device, use_checkpoint=not args.no_resume
            )
            if train_record is not None:
                train_record["sweep"] = "order_studies"
                train_record["study"] = "train"
                train_record["perturbation"] = None
                all_results.append(train_record)
                save()

            # The generator re-seeds with seed + 10000: the harness's test set,
            # here with the per-source components for the wrong-peak corruption.
            data = task["generate"](
                args.n_test,
                K_range=config["K_range"],
                N=config["N"],
                dominant=True,
                seed=seed + 10000,
            )
            eval_seed = 5000 + seed

            studies = set(pending)
            if any(s == "forced_error" for s, _ in studies):
                # Recompute all three perturbations together so they stay paired.
                all_results[:] = [
                    r
                    for r in all_results
                    if not (
                        r.get("sweep") == "order_studies"
                        and r["method"] == method
                        and r["seed"] == seed
                        and r["study"] == "forced_error"
                    )
                ]
                recs = study_forced_error(
                    model, method, seed, data, device, eval_seed, args.freq_tol, n_params
                )
                all_results.extend(recs)
                save()
                for r in recs:
                    print(
                        f"  forced_error[{r['perturbation']:>10}]: "
                        f"recovery_remaining={r['recovery_remaining']:.3f} "
                        f"mae_remaining={r['mae_remaining']:.4f} "
                        f"collapse={r['collapse_rate']:.3f}"
                    )

            if ("order_sensitivity", None) in studies:
                recs = study_order_sensitivity(
                    model, method, seed, data, device, eval_seed, args.n_orders, n_params
                )
                all_results.extend(recs)
                save()
                r = recs[0]
                print(
                    f"  order_sensitivity: canonical collapse="
                    f"{r['canonical_collapse_rate']:.3f} random="
                    f"{r['random_collapse_rate_mean']:.3f}"
                    f"±{r['random_collapse_rate_std']:.3f} ensemble="
                    f"{r['ensemble_collapse_rate']:.3f}"
                )

            if ("evidence_ordered", None) in studies:
                recs = study_evidence_ordered(
                    model, method, seed, data, device, eval_seed, n_params
                )
                all_results.extend(recs)
                save()
                r = recs[0]
                print(
                    f"  evidence_ordered: canonical collapse="
                    f"{r['canonical_collapse_rate']:.3f} "
                    f"mae={r['canonical_param_mae']:.4f} | greedy collapse="
                    f"{r['greedy_collapse_rate']:.3f} "
                    f"mae={r['greedy_param_mae']:.4f}"
                )

    save()
    print(f"\nSaved {len(all_results)} results to {out_path}")
    return all_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-time error-recovery / slot-order studies"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument(
        "--n-orders", type=int, default=10, help="Random slot orders for the sensitivity study"
    )
    parser.add_argument(
        "--freq-tol",
        type=float,
        default=0.5,
        help="Frequency tolerance (Hz; range 2-10) for a true source to count as recovered",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore (and overwrite) existing results and retrain instead of reloading checkpoints",
    )
    parser.add_argument("--output-dir", type=str, default="results/runs/order_studies")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
