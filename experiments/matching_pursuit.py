#!/usr/bin/env python3
"""Classical, training-free matching pursuit on the sinusoid task (App. H).

Iterative FFT peak picking with least-squares subtraction on an oversampled
Fourier dictionary; ``K`` is estimated by an amplitude threshold. The
evaluation mirrors ``experiments/synthetic.py``: the test set of seed ``s`` is
``generate_sinusoids(n_test, K_range=(1, 4), N=256, dominant=True, seed=s + 10000)``,
and K accuracy and greedy-matched parameter MAE are computed by packing the
decomposition into the ``(B, S, 3)`` parameter / ``(B, S)`` existence-logit
format. Attention-based collapse metrics do not apply and are reported as 0.0.

Records (method ``"matching_pursuit"``, sweep ``"method"``) are appended to
``<output-dir>/ablation_results.json``, the file written by
``experiments/synthetic.py``.

``--n-test`` defaults to 1000 (the paper's test-set size). ``--no-resume``
recomputes existing matching-pursuit records only.

Usage::

    python experiments/matching_pursuit.py --seeds 5 --n-test 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evidence_depletion.data.synthetic import TASKS, generate_sinusoids
from evidence_depletion.io import load_results
from evidence_depletion.metrics import k_detection_accuracy, param_recovery_error


def _ls_sinusoid_fit(
    resid: np.ndarray, t: np.ndarray, freq: float
) -> tuple[float, float, np.ndarray, float]:
    """Least-squares fit of ``A sin(2 pi f t + phi)`` at fixed ``f``.

    Returns:
        ``(amp, phase, fitted, sse)``.
    """
    s = np.sin(2 * np.pi * freq * t)
    c = np.cos(2 * np.pi * freq * t)
    # A sin(x + phi) = a sin(x) + b cos(x)
    M = np.stack([s, c], axis=1)
    coef, _, _, _ = np.linalg.lstsq(M, resid, rcond=None)
    a, b = coef
    fitted = a * s + b * c
    amp = float(np.hypot(a, b))
    phase = float(np.arctan2(b, a)) % (2 * np.pi)
    sse = float(np.sum((resid - fitted) ** 2))
    return amp, phase, fitted, sse


def matching_pursuit_sinusoids(
    signal: np.ndarray,
    N: int = 256,
    max_K: int = 4,
    amp_threshold: float = 0.15,
    oversample: int = 41,
) -> list[tuple[float, float, float]]:
    """Decompose a signal into sinusoids by matching pursuit.

    1. Pick the largest rFFT magnitude bin of the residual (DC excluded).
    2. Refine the frequency on ``oversample`` points within +/- 1 bin by
       minimizing the least-squares residual.
    3. Fit amplitude / phase at the refined frequency and subtract.
    4. Stop when the amplitude drops below ``amp_threshold`` or after ``max_K``
       components. The default 0.15 lies between the noise floor (std 0.1) and
       the minimum source amplitude (0.2) of the generator.

    Returns:
        Up to ``max_K`` tuples ``(freq, amp, phase)``.
    """
    t = np.linspace(0, 1, N)  # the generator's grid (endpoint included)
    dt = 1.0 / (N - 1)
    resid = np.asarray(signal, dtype=np.float64).copy()

    freqs_fft = np.fft.rfftfreq(N, d=dt)
    bin_width = freqs_fft[1] - freqs_fft[0]

    comps = []
    for _ in range(max_K):
        mag = np.abs(np.fft.rfft(resid))
        peak = int(np.argmax(mag[1:]) + 1)
        f0 = freqs_fft[peak]

        grid = np.linspace(f0 - bin_width, f0 + bin_width, oversample)
        grid = grid[grid > 0]
        best = None
        for f in grid:
            amp, phase, fitted, sse = _ls_sinusoid_fit(resid, t, f)
            if best is None or sse < best[3]:
                best = (f, amp, phase, sse, fitted)

        f_hat, amp_hat, phase_hat, _, fitted = best
        if amp_hat < amp_threshold:
            break
        comps.append((f_hat, amp_hat, phase_hat))
        resid = resid - fitted
    return comps


def evaluate_matching_pursuit(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate matching pursuit on the test split of task A for ``seed``."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_test = config.get("n_test", 2000)
    N = config.get("N", 256)
    K_range = config.get("K_range", (1, 4))
    max_K = K_range[1]
    n_slots = max_K + 1
    n_params = TASKS["sinusoids"]["n_params"]
    amp_threshold = config.get("amp_threshold", 0.15)

    test_signals, test_params, test_K, _ = generate_sinusoids(
        n_test, K_range=K_range, N=N, dominant=True, seed=seed + 10000
    )

    t0 = time.time()
    B = test_signals.shape[0]
    params_pred = torch.zeros(B, n_slots, n_params)
    exist_logits = torch.full((B, n_slots), -10.0)
    for b in range(B):
        comps = matching_pursuit_sinusoids(
            test_signals[b].numpy(), N=N, max_K=max_K, amp_threshold=amp_threshold
        )
        for k, (freq, amp, phase) in enumerate(comps):
            params_pred[b, k] = torch.tensor([freq, amp, phase])
            exist_logits[b, k] = 10.0
    eval_time = time.time() - t0

    k_acc = k_detection_accuracy(exist_logits, test_K)
    per_param_mae, n_matched = param_recovery_error(
        params_pred, test_params, exist_logits, test_K, n_params
    )

    results: dict[str, Any] = {
        "task": "sinusoids",
        "method": "matching_pursuit",
        "seed": seed,
        "train_time_s": round(eval_time, 1),
        "collapse_rate": 0.0,
        "mean_attn_overlap": 0.0,
        "k_accuracy": round(k_acc, 4),
    }
    for i, name in enumerate(TASKS["sinusoids"]["param_names"]):
        results[f"mae_{name}"] = (
            round(per_param_mae[i].item(), 6) if n_matched > 0 else float("nan")
        )
    return results


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(
        description="Classical matching pursuit baseline (sinusoid task only)"
    )
    parser.add_argument(
        "--seeds", type=int, default=5, help="Number of random seeds (default: 5, i.e. 0-4)"
    )
    parser.add_argument(
        "--n-test", type=int, default=1000, help="Test samples (default: 1000, as in the paper)"
    )
    parser.add_argument(
        "--amp-threshold", type=float, default=0.15, help="Amplitude threshold for K estimation"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute seeds that already have a matching_pursuit record (other records "
        "in ablation_results.json are kept)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/runs/synthetic",
        help="Output directory (appends to ablation_results.json)",
    )
    args = parser.parse_args(argv)

    seeds = list(range(args.seeds))
    config = {
        "n_test": args.n_test,
        "N": 256,
        "K_range": (1, 4),
        "amp_threshold": args.amp_threshold,
    }

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "ablation_results.json"

    # The file is shared with experiments/synthetic.py: --no-resume only drops
    # previous matching-pursuit records and keeps everything else.
    all_results = load_results(out_path)
    if args.no_resume:
        all_results = [r for r in all_results if r.get("method") != "matching_pursuit"]

    def already_done(seed: int) -> bool:
        return any(
            r["task"] == "sinusoids"
            and r["method"] == "matching_pursuit"
            and r["seed"] == seed
            and r.get("sweep") == "method"
            for r in all_results
        )

    print(f"=== Matching pursuit baseline: sinusoids x {len(seeds)} seeds ===\n")
    for seed in seeds:
        if already_done(seed):
            print(f"  seed={seed} ... SKIPPED (exists)")
            continue
        print(f"  seed={seed} ... ", end="", flush=True)
        result = evaluate_matching_pursuit(seed, config)
        result["sweep"] = "method"
        all_results.append(result)
        print(
            f"K_acc={result['k_accuracy']:.3f} "
            f"mae_freq={result['mae_frequency']:.4f} "
            f"({result['train_time_s']:.0f}s)"
        )
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} results to {out_path}")
    return all_results


if __name__ == "__main__":
    main()
