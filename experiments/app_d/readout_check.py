#!/usr/bin/env python3
"""Readout geometry of trained models (App. D, "Empirical check on trained models").

Loads vanilla and evidence-depletion (linear) checkpoints written by
``experiments/order_studies.py`` (task A, 200 epochs by default), extracts the
trained attention exactly (an instrumented forward that is verified against
``model(x)``), and measures on 1024 test mixtures with ``K in {2, 3, 4}`` the
minimum pairwise cosine over the five slots of

* the attention rows themselves (and their mean pairwise cosine);
* readouts in the magnitude token space ``u_s = alpha_s @ |rfft(chunk)|``
  (the primary surrogate space of App. D); for evidence depletion also with the
  depletion factors, ``u_s = alpha_s @ (e_s * m)``;
* diagnostics without a claim in the paper: readouts in the model's value
  space, in the signed linear chunk-FFT space and the complex-modulus cosine.

Paper (3 seeds each): vanilla readouts median ``0.997-0.999`` with attention
min-pair cosine ``0.24-0.55``; evidence depletion attention mean-pair cosine
``0.05-0.19`` and readouts ``0.81-0.92``; with vs. without depletion factors
differs by < 0.01.

Usage::

    python experiments/order_studies.py --seeds 3            # trains + checkpoints
    python experiments/app_d/readout_check.py --ckpt-dir results/runs/order_studies/checkpoints
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evidence_depletion.data.synthetic import generate_sinusoids
from evidence_depletion.models import build_model

N, N_TOKENS, D_MODEL, N_SLOTS = 256, 16, 64, 5
CHUNK = N // N_TOKENS


def load_model(ckpt_dir: Path, method: str, seed: int, epochs: int) -> nn.Module:
    """Rebuild the order-studies model and load its checkpoint."""
    model = build_model(
        method=method,
        n_params=3,
        N=N,
        d_model=D_MODEL,
        n_tokens=N_TOKENS,
        n_slots=N_SLOTS,
        tokenizer_type="fft",
        gamma=3.0,
        tau=0.3,
        n_iter=3,
    )
    ckpt = ckpt_dir / f"{method}_seed{seed}_ep{epochs}.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


@torch.no_grad()
def run_vanilla(model: nn.Module, x: torch.Tensor, seed: int) -> tuple[torch.Tensor, ...]:
    """Instrumented vanilla forward: ``(attn_wm, v, attn)`` of the last iteration.

    ``attn_wm`` are the token-normalized weights the readout uses, ``attn`` the
    softmax over slots returned by the model.
    """
    sa = model.slot_attn
    tokens = model.tokenizer(x)
    B, L, D = tokens.shape
    torch.manual_seed(seed)
    slots = sa.sample_slots(B, tokens.device)
    inp = sa.norm_in(tokens)
    k = sa.proj_k(inp)
    v = sa.proj_v(inp)
    for _ in range(sa.n_iter):
        q = sa.proj_q(sa.norm_s(slots))
        attn = F.softmax(torch.einsum("bsd,bld->bsl", q, k) / (D**0.5), dim=1)
        attn_wm = attn / attn.sum(-1, keepdim=True).clamp(min=1e-8)
        slots = sa.update_slot(slots, torch.einsum("bsl,bld->bsd", attn_wm, v))
    return attn_wm, v, attn


@torch.no_grad()
def run_evidence(model: nn.Module, x: torch.Tensor, seed: int) -> tuple[torch.Tensor, ...]:
    """Instrumented evidence-depletion forward (eval order ``0..S-1``).

    Returns ``attns [B,S,L]``, per-slot depleted values ``[B,S,L,D]`` and the
    evidence each slot saw ``[B,S,L]``.
    """
    sa = model.slot_attn
    tokens = model.tokenizer(x)
    B, L, D = tokens.shape
    torch.manual_seed(seed)
    slots_init = sa.sample_slots(B, tokens.device)
    inp = sa.norm_in(tokens)
    evidence = torch.ones(B, L)
    attns, values, evids = [], [], []
    for s in range(sa.n_slots):
        slot = slots_init[:, s : s + 1, :]
        masked_inp = inp * evidence.unsqueeze(-1)
        k = sa.proj_k(masked_inp)
        v = sa.proj_v(masked_inp)
        for _ in range(sa.n_iter):
            attn = sa.attend(slot, k, sa.evidence_bias(evidence))
            slot = sa.update_slot(slot, torch.einsum("bsl,bld->bsd", attn, v))
        attns.append(attn.squeeze(1))
        values.append(v)
        evids.append(evidence.clone())
        evidence = sa.deplete(evidence, attn.squeeze(1))
    return torch.stack(attns, 1), torch.stack(values, 1), torch.stack(evids, 1)


@torch.no_grad()
def verify_replication(model: nn.Module, method: str, x: torch.Tensor, seed: int) -> float:
    """Max |difference| between the instrumented and the model's attention (< 1e-6)."""
    torch.manual_seed(seed)
    _, attn_ref, _, _, _ = model(x)
    attn = (
        run_vanilla(model, x, seed)[2] if method == "vanilla" else run_evidence(model, x, seed)[0]
    )
    err = (attn_ref - attn).abs().max().item()
    if err >= 1e-6:
        raise RuntimeError(f"replication mismatch: {err}")
    return err


def pairwise_cos_stats(U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``U [B, S, F]`` -> (min, mean) over off-diagonal slot pairs, each ``[B]``."""
    Un = U / U.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    vals = torch.einsum("bsd,btd->bst", Un, Un)[:, ~torch.eye(U.shape[1], dtype=torch.bool)]
    return vals.min(-1).values, vals.mean(-1)


def min_pairwise_abs_cos_complex(U: torch.Tensor) -> torch.Tensor:
    """``U [B, S, F]`` complex -> min over pairs of ``|<u, u'>| / (|u| |u'|)``, ``[B]``."""
    Un = U / U.abs().pow(2).sum(-1).sqrt().clamp(min=1e-12).unsqueeze(-1)
    cosm = torch.einsum("bsf,btf->bst", Un, Un.conj()).abs()
    return cosm[:, ~torch.eye(U.shape[1], dtype=torch.bool)].min(dim=-1).values


def chunk_fft(x: torch.Tensor) -> torch.Tensor:
    """Complex rFFT of each tokenizer chunk: ``[B, L, CHUNK // 2 + 1]``."""
    return torch.fft.rfft(x.reshape(x.shape[0], N_TOKENS, CHUNK), dim=-1)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Readout geometry of trained checkpoints")
    ap.add_argument(
        "--ckpt-dir",
        type=str,
        default="results/runs/order_studies/checkpoints",
        help="Directory with <method>_seed<s>_ep<epochs>.pt from experiments/order_studies.py",
    )
    ap.add_argument("--epochs", type=int, default=200, help="Epoch tag of the checkpoints")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=1024)
    ap.add_argument("--data-seed", type=int, default=777)
    ap.add_argument("--eval-seed", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--output", type=str, default="results/runs/app_d/readout_check.json")
    args = ap.parse_args(argv)

    signals, _, _, _ = generate_sinusoids(
        args.n_test, K_range=(2, 4), N=N, noise_std=0.1, dominant=True, seed=args.data_seed
    )
    z = chunk_fft(signals)
    lin = torch.cat([z.real, z.imag], dim=-1)
    mag = z.abs()

    def q(a: np.ndarray, p: float) -> float:
        return float(np.quantile(a, p))

    results: dict[str, Any] = {"config": vars(args), "models": {}}
    header = (
        f"{'model':>16} {'seed':>4} | {'attn min':>8} {'attn mean':>9} | {'mag':>6} "
        f"{'mag-depl':>8} | {'model':>6} {'signed':>6} {'|cplx|':>6}"
    )
    print("Median over samples of the min pairwise cosine across slots\n" + header)
    print("-" * len(header))
    for method in ("vanilla", "evidence_linear"):
        for seed in range(args.seeds):
            model = load_model(Path(args.ckpt_dir), method, seed, args.epochs)
            err = verify_replication(model, method, signals[:64], args.eval_seed)
            acc: dict[str, list[torch.Tensor]] = {
                k: [] for k in ("attn_min", "attn_mean", "mag", "mag_dep", "model", "lin", "cplx")
            }
            for i in range(0, args.n_test, args.batch_size):
                xb, sl = signals[i : i + args.batch_size], slice(i, i + args.batch_size)
                bseed = args.eval_seed + i
                if method == "vanilla":
                    alpha, v, _ = run_vanilla(model, xb, bseed)
                    u_model = torch.einsum("bsl,bld->bsd", alpha, v)
                else:
                    alpha, vals, evid = run_evidence(model, xb, bseed)
                    u_model = torch.einsum("bsl,bsld->bsd", alpha, vals)
                    u_dep = torch.einsum(
                        "bsl,bslf->bsf", alpha, evid.unsqueeze(-1) * mag[sl].unsqueeze(1)
                    )
                    acc["mag_dep"].append(pairwise_cos_stats(u_dep)[0])
                a_min, a_mean = pairwise_cos_stats(alpha)
                acc["attn_min"].append(a_min)
                acc["attn_mean"].append(a_mean)
                acc["mag"].append(
                    pairwise_cos_stats(torch.einsum("bsl,blf->bsf", alpha, mag[sl]))[0]
                )
                acc["model"].append(pairwise_cos_stats(u_model)[0])
                acc["lin"].append(
                    pairwise_cos_stats(torch.einsum("bsl,blf->bsf", alpha, lin[sl]))[0]
                )
                u_c = torch.einsum("bsl,blf->bsf", alpha.to(z.dtype), z[sl])
                acc["cplx"].append(min_pairwise_abs_cos_complex(u_c))

            med = {k: q(torch.cat(v).numpy(), 0.5) for k, v in acc.items() if v}
            results["models"][f"{method}_seed{seed}"] = {"replication_max_err": err, "median": med}
            dep = f"{med['mag_dep']:>8.3f}" if "mag_dep" in med else f"{'-':>8}"
            print(
                f"{method:>16} {seed:>4} | {med['attn_min']:>8.3f} {med['attn_mean']:>9.3f} | "
                f"{med['mag']:>6.3f} {dep} | {med['model']:>6.3f} {med['lin']:>6.3f} "
                f"{med['cplx']:>6.3f}"
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")
    return results


if __name__ == "__main__":
    main()
