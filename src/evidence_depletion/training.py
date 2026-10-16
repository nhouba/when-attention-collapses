"""Shared training and evaluation loops for the synthetic and FUSS benchmarks.

All benchmarks optimize the same objective (Sec. 3.1; the synthetic harness has
no posterior head and no auxiliary losses beyond the default diversity term)::

    loss = BCE(existence) + 5 * MSE(matched parameters) + w_div * L_div
           [+ w_ortho * L_ortho]          (loss-based baselines, App. H)
           [+ w_coverage * L_coverage]    (coverage-loss baseline, App. H)

with Adam (lr 3e-4), cosine annealing over the epochs and gradient clipping at
norm 1. Parameters are matched greedily between the top-``K`` slots by
existence logit and the ``K`` true sources.

Entry points:

* :func:`train_synthetic` -- one run of the synthetic harness
  (``experiments/synthetic.py``).
* :func:`train_loss_baseline` -- vanilla / sequential SA with stronger
  auxiliary losses (``experiments/loss_baselines.py``).
* :func:`train_fuss` -- one run on FUSS (``experiments/fuss.py``).

Each returns the JSON result record written by the corresponding script.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evidence_depletion.data.synthetic import TASKS
from evidence_depletion.metrics import (
    cosine_overlap,
    evidence_depletion_split,
    greedy_match,
    k_detection_accuracy,
    mean_slot_embedding_cosine,
    param_mae_exact,
    param_recovery_error,
    peak_overlap_rate,
)
from evidence_depletion.models import build_model
from evidence_depletion.separation import evaluate_sisdr

LOSS_BASELINES: dict[str, tuple[str, dict[str, float]]] = {
    "vanilla_divloss": ("vanilla", {"w_div": 5.0, "w_ortho": 0.0}),
    "vanilla_ortho": ("vanilla", {"w_div": 0.5, "w_ortho": 3.0}),
    "vanilla_divloss_ortho": ("vanilla", {"w_div": 5.0, "w_ortho": 3.0}),
    "seq_divloss": ("sequential", {"w_div": 5.0, "w_ortho": 0.0}),
    "seq_ortho": ("sequential", {"w_div": 0.5, "w_ortho": 3.0}),
    "seq_divloss_ortho": ("sequential", {"w_div": 5.0, "w_ortho": 3.0}),
}
"""Loss-based baselines (App. H, Table 10): name -> (slot attention, loss weights)."""


def default_device() -> torch.device:
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Losses
# ============================================================


def existence_loss(
    exist_logits: torch.Tensor, K_vals: torch.Tensor, n_slots: int, device: torch.device
) -> torch.Tensor:
    """BCE between existence logits and the target "first ``K`` slots exist"."""
    exist_target = torch.zeros(exist_logits.shape[0], n_slots, device=device)
    for b in range(exist_logits.shape[0]):
        exist_target[b, : int(K_vals[b].item())] = 1.0
    return F.binary_cross_entropy_with_logits(exist_logits, exist_target, reduction="mean")


def matched_param_loss(
    params_pred: torch.Tensor,
    params_true: torch.Tensor,
    exist_logits: torch.Tensor,
    K_vals: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Mean over matched pairs of the parameter MSE.

    The top-``K`` slots by existence logit are matched greedily to the ``K``
    true sources (:func:`~evidence_depletion.metrics.greedy_match`).
    """
    n_slots = params_pred.shape[1]
    losses = []
    for b in range(params_pred.shape[0]):
        K = int(K_vals[b].item())
        if K == 0:
            continue
        top_k = min(K, n_slots)
        _, top_idx = exist_logits[b].topk(top_k)
        pred = params_pred[b, top_idx]
        true = params_true[b, :K]
        cost = torch.cdist(pred.unsqueeze(0), true.unsqueeze(0)).squeeze(0)
        for i, j in greedy_match(cost, min(top_k, K)):
            losses.append(F.mse_loss(pred[i], true[j]))

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _offdiag_cosine_sq(x: torch.Tensor, n_slots: int, device: torch.device) -> torch.Tensor:
    x_norm = F.normalize(x, dim=-1)
    sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
    mask = ~torch.eye(n_slots, dtype=torch.bool, device=device).unsqueeze(0)
    return (sim * mask.float()).pow(2).mean()


def slot_diversity_loss(slots: torch.Tensor, n_slots: int, device: torch.device) -> torch.Tensor:
    """Mean squared off-diagonal cosine similarity of the slot embeddings."""
    return _offdiag_cosine_sq(slots, n_slots, device)


def attention_orthogonality_loss(
    attn: torch.Tensor, n_slots: int, device: torch.device
) -> torch.Tensor:
    """Mean squared off-diagonal cosine similarity of the slot attention vectors."""
    return _offdiag_cosine_sq(attn, n_slots, device)


def coverage_penalty(attn: torch.Tensor) -> torch.Tensor:
    """Coverage loss (adaptation of Tu et al. 2016): ``sum_l (sum_s a_sl - 1)^2``.

    Computed on the token-normalized attention (each slot's distribution over
    tokens): vanilla SA's raw softmax over slots sums to one over slots, which
    would make the penalty identically zero. Mean over the batch.
    """
    attn_wm = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    coverage = attn_wm.sum(dim=1)
    return ((coverage - 1.0) ** 2).sum(dim=-1).mean()


def training_loss(
    outputs: tuple,
    params: torch.Tensor,
    K_vals: torch.Tensor,
    n_slots: int,
    device: torch.device,
    w_div: float = 0.5,
    w_ortho: float | None = None,
    w_coverage: float | None = None,
) -> torch.Tensor:
    """Full training objective (see the module docstring).

    Args:
        outputs: the 5-tuple returned by
            :class:`~evidence_depletion.models.DecompositionModel`.
        params: ``[B, max_K, P]`` true parameters.
        K_vals: ``[B]`` true counts.
        n_slots: ``S``.
        device: device of the loss terms.
        w_div: weight of :func:`slot_diversity_loss`.
        w_ortho: weight of :func:`attention_orthogonality_loss`; ``None`` omits the term.
        w_coverage: weight of :func:`coverage_penalty`; ``None`` omits the term.
    """
    slots, attn, exist_logits, params_pred, _ = outputs
    exist_loss = existence_loss(exist_logits, K_vals, n_slots, device)
    param_loss = matched_param_loss(params_pred, params, exist_logits, K_vals, device)
    if n_slots > 1 and w_div > 0:
        diversity_loss = slot_diversity_loss(slots, n_slots, device)
    else:
        diversity_loss = torch.tensor(0.0, device=device)

    loss = exist_loss + 5.0 * param_loss + w_div * diversity_loss
    if w_ortho is not None:
        if n_slots > 1 and w_ortho > 0:
            ortho_loss = attention_orthogonality_loss(attn, n_slots, device)
        else:
            ortho_loss = torch.tensor(0.0, device=device)
        loss = loss + w_ortho * ortho_loss
    if w_coverage is not None:
        loss = loss + w_coverage * coverage_penalty(attn)
    return loss


def fit(
    model: nn.Module,
    train_loader: Iterable,
    n_slots: int,
    device: torch.device,
    epochs: int,
    lr: float = 3e-4,
    epoch_callback: Callable[[int], None] | None = None,
    **loss_weights: float | None,
) -> float:
    """Train ``model`` with Adam + cosine annealing + gradient clipping at 1.

    Args:
        model: a :class:`~evidence_depletion.models.DecompositionModel`.
        train_loader: yields ``(signals, params, K)``.
        n_slots: ``S``.
        device: training device.
        epochs: number of epochs (also the cosine period).
        lr: Adam learning rate.
        epoch_callback: called with the epoch index after each epoch's
            scheduler step.
        **loss_weights: ``w_div`` / ``w_ortho`` / ``w_coverage`` for :func:`training_loss`.

    Returns:
        Wall-clock training time in seconds (callbacks included).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for signals, params, K_vals in train_loader:
            signals = signals.to(device)
            params = params.to(device)
            K_vals = K_vals.to(device)

            outputs = model(signals)
            loss = training_loss(outputs, params, K_vals, n_slots, device, **loss_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        if epoch_callback is not None:
            epoch_callback(epoch)
    return time.time() - t0


# ============================================================
# Evaluation
# ============================================================


@dataclass
class EvalAccumulator:
    """Per-batch metric values collected by :func:`evaluate`."""

    collapse: list[float] = field(default_factory=list)
    overlap: list[float] = field(default_factory=list)
    k_acc: list[float] = field(default_factory=list)
    param_errors: list[torch.Tensor] = field(default_factory=list)
    exact_mae: list[np.ndarray] = field(default_factory=list)
    exact_weights: list[int] = field(default_factory=list)
    per_example: list[tuple[float, float]] = field(default_factory=list)
    depleted_matched: list[float] = field(default_factory=list)
    depleted_unmatched: list[float] = field(default_factory=list)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    n_params: int,
    cosine: bool = True,
    matching: str = "greedy",
    log_depletion: bool = False,
) -> EvalAccumulator:
    """One eval-mode pass over ``loader`` (yielding ``(signals, params, K)``).

    Args:
        model: model to evaluate (switched to eval mode).
        loader: evaluation batches.
        device: evaluation device.
        n_params: parameters per source.
        cosine: also collect :func:`~evidence_depletion.metrics.cosine_overlap`.
        matching: ``"greedy"`` (:func:`~evidence_depletion.metrics.param_recovery_error`,
            synthetic) or ``"exact"`` (:func:`~evidence_depletion.metrics.param_mae_exact`,
            FUSS; match-count weighted).
        log_depletion: collect the matched / unmatched depleted evidence mass
            (only for models that return evidence).
    """
    model.eval()
    acc = EvalAccumulator()
    for signals, params, K_vals in loader:
        signals = signals.to(device)
        params = params.to(device)
        K_vals = K_vals.to(device)

        _, attn, exist_logits, params_pred, evidence = model(signals)

        cr = peak_overlap_rate(attn, K_vals)
        acc.collapse.append(cr)
        if cosine:
            acc.overlap.append(cosine_overlap(attn, K_vals))
        acc.k_acc.append(k_detection_accuracy(exist_logits, K_vals))

        if matching == "greedy":
            per_param_mae, n_matched = param_recovery_error(
                params_pred, params, exist_logits, K_vals, n_params
            )
            if n_matched > 0:
                acc.param_errors.append(per_param_mae.cpu())
        else:
            per_param_mae, n_matched = param_mae_exact(
                params_pred, params, exist_logits, K_vals, n_params
            )
            if n_matched > 0:
                pm = per_param_mae.detach().cpu().numpy()
                acc.exact_mae.append(pm * n_matched)
                acc.exact_weights.append(n_matched)
                acc.per_example.append((cr, float(pm.mean())))

        if log_depletion and isinstance(evidence, list) and len(evidence) > 1:
            m_vals, u_vals = evidence_depletion_split(evidence, exist_logits, K_vals)
            acc.depleted_matched.extend(m_vals)
            acc.depleted_unmatched.extend(u_vals)
    return acc


def _mean_or_none(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 6) if values else None


def _mechanism_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in ("gamma", "tau", "n_iter") if key in config}


def _make_loaders(
    task_name: str, seed: int, config: dict[str, Any], n_train: int, n_test: int, **gen_kwargs: Any
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Train / test loaders of a synthetic task (test data uses ``seed + 10000``)."""
    generate_fn = TASKS[task_name]["generate"]
    N = config.get("N", 256)
    K_range = config.get("K_range", (1, 4))
    train_signals, train_params, train_K, _ = generate_fn(
        n_train, K_range=K_range, N=N, dominant=True, seed=seed, **gen_kwargs
    )
    test_signals, test_params, test_K, _ = generate_fn(
        n_test, K_range=K_range, N=N, dominant=True, seed=seed + 10000, **gen_kwargs
    )
    batch_size = config.get("batch_size", 128)
    train_loader = DataLoader(
        TensorDataset(train_signals, train_params, train_K),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_signals, test_params, test_K), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, test_signals, test_K


# ============================================================
# Synthetic harness
# ============================================================


def train_synthetic(
    task_name: str, method: str, seed: int, config: dict[str, Any]
) -> dict[str, Any] | tuple[dict[str, Any], nn.Module]:
    """Train and evaluate one model on a synthetic task (Sec. 3.1).

    Args:
        task_name: key of :data:`~evidence_depletion.data.synthetic.TASKS`.
        method: a :data:`~evidence_depletion.models.METHODS` entry.
        seed: seeds ``torch`` / ``numpy``, the training data (``seed``) and
            the test data (``seed + 10000``).
        config: hyper-parameters, with defaults ``n_train=10000``, ``n_test=2000``,
            ``N=256``, ``K_range=(1, 4)``, ``n_slots=K_max + 1``, ``batch_size=128``,
            ``d_model=64``, ``n_tokens=16``, ``lr=3e-4``, ``epochs=300``; optional
            ``gamma`` / ``tau`` / ``n_iter`` (mechanism), ``amp_ratio``
            (amplitude-ratio sweep), ``w_coverage`` (``coverage_loss`` only,
            default 1.0), ``track_dynamics`` (per-epoch slot-embedding cosine,
            peak and cosine overlap on one fixed held-out batch),
            ``log_depletion`` (matched / unmatched depleted evidence) and
            ``return_model``.

    Returns:
        The result record (and the trained model if ``config["return_model"]``).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    task = TASKS[task_name]
    n_params = task["n_params"]
    K_range = config.get("K_range", (1, 4))
    n_slots = config.get("n_slots") or (K_range[1] + 1)

    gen_kwargs = {}
    if config.get("amp_ratio") is not None:
        gen_kwargs["amp_ratio"] = config["amp_ratio"]
    train_loader, test_loader, test_signals, test_K = _make_loaders(
        task_name,
        seed,
        config,
        config.get("n_train", 10000),
        config.get("n_test", 2000),
        **gen_kwargs,
    )
    batch_size = config.get("batch_size", 128)

    model = build_model(
        method=method,
        n_params=n_params,
        N=config.get("N", 256),
        d_model=config.get("d_model", 64),
        n_tokens=config.get("n_tokens", 16),
        n_slots=n_slots,
        tokenizer_type="fft" if task_name != "gaussian_bumps" else "direct",
        **_mechanism_kwargs(config),
    )
    device = default_device()
    model = model.to(device)

    track_dynamics = bool(config.get("track_dynamics", False))
    dynamics: list[dict[str, float]] = []
    callback = None
    if track_dynamics:
        dyn_signals = test_signals[:batch_size].to(device)
        dyn_K = test_K[:batch_size].to(device)
        fork_devices = [device] if device.type == "cuda" else []

        def callback(epoch: int) -> None:
            # Forked RNG + fixed seed: identical slot-init noise in every epoch,
            # and the training randomness is unchanged by the logging.
            model.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(seed + 999)
                slots_d, attn_d, _, _, _ = model(dyn_signals)
                dynamics.append(
                    {
                        "epoch": epoch,
                        "emb_cos": round(mean_slot_embedding_cosine(slots_d, dyn_K), 6),
                        "peak_overlap": round(peak_overlap_rate(attn_d, dyn_K), 6),
                        "attn_overlap": round(cosine_overlap(attn_d, dyn_K), 6),
                    }
                )

    w_coverage = config.get("w_coverage", 1.0) if method == "coverage_loss" else None
    train_time = fit(
        model,
        train_loader,
        n_slots,
        device,
        epochs=config.get("epochs", 300),
        lr=config.get("lr", 3e-4),
        epoch_callback=callback,
        w_div=0.5,
        w_coverage=w_coverage,
    )

    log_depletion = bool(config.get("log_depletion", False))
    acc = evaluate(model, test_loader, device, n_params, log_depletion=log_depletion)

    results: dict[str, Any] = {
        "task": task_name,
        "method": method,
        "seed": seed,
        "train_time_s": round(train_time, 1),
        "collapse_rate": round(np.mean(acc.collapse), 4),
        "mean_attn_overlap": round(np.mean(acc.overlap), 4),
        "k_accuracy": round(np.mean(acc.k_acc), 4),
    }
    if acc.param_errors:
        mean_errors = torch.stack(acc.param_errors).mean(dim=0)
        for i, name in enumerate(task["param_names"]):
            results[f"mae_{name}"] = round(mean_errors[i].item(), 6)
    else:
        for name in task["param_names"]:
            results[f"mae_{name}"] = float("nan")

    if track_dynamics:
        results["dynamics"] = dynamics
    if log_depletion:
        results["depleted_matched"] = _mean_or_none(acc.depleted_matched)
        results["depleted_unmatched"] = _mean_or_none(acc.depleted_unmatched)

    if config.get("return_model", False):
        return results, model
    return results


def train_loss_baseline(
    task_name: str, method: str, seed: int, config: dict[str, Any]
) -> dict[str, Any]:
    """Train and evaluate one loss-based baseline (App. H, Table 10).

    Same data, model and evaluation as :func:`train_synthetic`, but the slot
    attention and loss weights come from :data:`LOSS_BASELINES`, ``n_slots`` is
    always ``K_max + 1``, and the defaults are ``n_train=5000``, ``n_test=1000``,
    ``epochs=200``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    task = TASKS[task_name]
    n_params = task["n_params"]
    K_range = config.get("K_range", (1, 4))
    n_slots = K_range[1] + 1

    train_loader, test_loader, _, _ = _make_loaders(
        task_name, seed, config, config.get("n_train", 5000), config.get("n_test", 1000)
    )
    if method not in LOSS_BASELINES:
        raise ValueError(f"Unknown method: {method}. Valid: {list(LOSS_BASELINES)}")
    sa_method, loss_config = LOSS_BASELINES[method]

    model = build_model(
        method=sa_method,
        n_params=n_params,
        N=config.get("N", 256),
        d_model=config.get("d_model", 64),
        n_tokens=config.get("n_tokens", 16),
        n_slots=n_slots,
        tokenizer_type="fft" if task_name != "gaussian_bumps" else "direct",
        n_iter=config.get("n_iter", 3),
        gamma=config.get("gamma", 3.0),
        tau=config.get("tau", 0.3),
    )
    device = default_device()
    model = model.to(device)

    w_div = loss_config.get("w_div", 0.5)
    w_ortho = loss_config.get("w_ortho", 0.0)
    train_time = fit(
        model,
        train_loader,
        n_slots,
        device,
        epochs=config.get("epochs", 200),
        lr=config.get("lr", 3e-4),
        w_div=w_div,
        w_ortho=w_ortho,
    )

    acc = evaluate(model, test_loader, device, n_params, cosine=False)
    results: dict[str, Any] = {
        "task": task_name,
        "method": method,
        "seed": seed,
        "train_time_s": round(train_time, 1),
        "collapse_rate": round(np.mean(acc.collapse), 4),
        "k_accuracy": round(np.mean(acc.k_acc), 4),
        "w_div": w_div,
        "w_ortho": w_ortho,
    }
    if acc.param_errors:
        mean_errors = torch.stack(acc.param_errors).mean(dim=0)
        for i, name in enumerate(task["param_names"]):
            results[f"mae_{name}"] = round(mean_errors[i].item(), 6)
    return results


# ============================================================
# FUSS
# ============================================================

FUSS_PARAM_NAMES = ("freq", "amp", "centroid")


def train_fuss(
    method: str,
    seed: int,
    config: dict[str, Any],
    train_loader: Iterable,
    test_loader: Iterable,
    n_slots: int,
    n_params: int = 3,
    sisdr_loader: Iterable | None = None,
    mask_mode: str = "rank1",
) -> dict[str, Any]:
    """Train and evaluate one model on FUSS (Sec. 3.2, App. J).

    Args:
        method: a :data:`~evidence_depletion.models.METHODS` entry.
        seed: seeds ``torch`` / ``numpy`` before the model is built.
        config: ``signal_length`` (default 16000), ``d_model`` (64), ``n_tokens``
            (16), ``lr`` (3e-4), ``epochs`` (200), optional ``gamma`` / ``tau`` /
            ``n_iter`` and ``w_coverage`` (``coverage_loss`` only, default 1.0).
        train_loader, test_loader: yield ``(mixture, params, K)``.
        n_slots: ``S``.
        n_params: parameters per source.
        sisdr_loader: optional loader over a ``return_sources=True`` dataset; if
            given, PIT SI-SDR of the mask readout is evaluated
            (:func:`~evidence_depletion.separation.evaluate_sisdr`); otherwise the
            ``si_sdr*`` keys are ``None``.
        mask_mode: readout mode for the SI-SDR evaluation.

    Returns:
        The result record, including the known-K exact-Hungarian parameter MAE
        and per-batch ``(collapse, param MAE)`` pairs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = default_device()

    model = build_model(
        method=method,
        n_params=n_params,
        N=config.get("signal_length", 16000),
        d_model=config.get("d_model", 64),
        n_tokens=config.get("n_tokens", 16),
        n_slots=n_slots,
        tokenizer_type="fft",
        **_mechanism_kwargs(config),
    )
    model = model.to(device)

    w_coverage = config.get("w_coverage", 1.0) if method == "coverage_loss" else None
    train_time = fit(
        model,
        train_loader,
        n_slots,
        device,
        epochs=config.get("epochs", 200),
        lr=config.get("lr", 3e-4),
        w_div=0.5,
        w_coverage=w_coverage,
    )

    acc = evaluate(model, test_loader, device, n_params, matching="exact")

    sisdr: dict[str, Any] = {"si_sdr": None, "si_sdri": None, "si_sdr_mixture": None}
    if sisdr_loader is not None:
        s = evaluate_sisdr(model, sisdr_loader, device, mask_mode=mask_mode)
        print(
            f"[SI-SDR: {s['si_sdr']} dB, SI-SDRi: {s['si_sdri']} dB, "
            f"mix: {s['si_sdr_mixture']} dB, n={s['n_mixtures']}] ",
            end="",
            flush=True,
        )
        sisdr = {k: s[k] for k in ("si_sdr", "si_sdri", "si_sdr_mixture")}

    if acc.exact_weights:
        tot = float(np.sum(acc.exact_weights))
        mae = np.sum(acc.exact_mae, axis=0) / tot
        pnames = FUSS_PARAM_NAMES[:n_params]
        param_mae: dict[str, Any] = {
            f"param_mae_{pnames[i]}": round(float(mae[i]), 5) for i in range(len(pnames))
        }
        param_mae["param_mae_mean"] = round(float(mae.mean()), 5)
    else:
        param_mae = {"param_mae_mean": None}

    return {
        "task": "fuss",
        "method": method,
        "seed": seed,
        "train_time_s": round(train_time, 1),
        "collapse_rate": round(np.mean(acc.collapse), 4),
        "mean_attn_overlap": round(np.mean(acc.overlap), 4),
        "k_accuracy": round(np.mean(acc.k_acc), 4),
        **param_mae,
        "per_example_collapse_mae": acc.per_example,
        **sisdr,
    }
