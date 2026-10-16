#!/usr/bin/env python3
"""Latency microbenchmark of the slot-attention module alone (App. I, Table 12).

Forward-pass wall clock (no tokenizer, no heads) for vanilla SA, sequential SA
and evidence depletion (linear) at ``S in {5, 10, 20, 40}`` slots and
``L in {16, 128}`` tokens (``d_model=64``, batch 32): warm-up, then timed
iterations, on CPU and, when available, CUDA. Also fits the ``O(S * I)``
scaling (least squares of ms vs. ``S``) per (device, mode, ``L``).

Records (``sweep="latency"``, ``record_type`` ``"measurement"`` or
``"scaling_fit"``) go to ``<output-dir>/latency_results.json``. Resume-safe:
measurements already present for the available devices are not repeated;
scaling fits are recomputed on every run.

Usage::

    python experiments/latency.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from evidence_depletion.attention import EvidenceDepletionSlotAttention
from evidence_depletion.baselines import SequentialSA, VanillaSA
from evidence_depletion.io import load_results

MODES = ["vanilla", "sequential", "evidence_linear"]
S_VALUES = [5, 10, 20, 40]
L_VALUES = [16, 128]
RESULTS_FILE = "latency_results.json"


def build_sa(mode: str, n_slots: int, d_model: int, n_iter: int) -> nn.Module:
    if mode == "vanilla":
        return VanillaSA(n_slots=n_slots, d_model=d_model, n_iter=n_iter)
    if mode == "sequential":
        return SequentialSA(n_slots=n_slots, d_model=d_model, n_iter=n_iter)
    if mode == "evidence_linear":
        return EvidenceDepletionSlotAttention(
            n_slots=n_slots, d_model=d_model, n_iter=n_iter, depletion="linear"
        )
    raise ValueError(f"Unknown mode: {mode}")


def bench_forward(
    module: nn.Module, tokens: torch.Tensor, warmup: int, iters: int, device: torch.device
) -> tuple[float, float]:
    """Mean and std wall clock (ms) of ``module(tokens)`` under ``torch.no_grad``."""
    module.eval()
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            module(tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(iters):
            t0 = time.perf_counter()
            module(tokens)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times)), float(np.std(times))


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares ``y = a x + b``: ``(slope, intercept, r2)``."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / RESULTS_FILE

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    all_results = load_results(out_path, resume=not args.no_resume)

    measurements = [
        r
        for r in all_results
        if r.get("sweep") == "latency" and r.get("record_type") == "measurement"
    ]

    def find(device: str, mode: str, L: int, S: int) -> dict[str, Any] | None:
        for r in measurements:
            if (
                r["device"] == device
                and r["mode"] == mode
                and r["seq_len"] == L
                and r["n_slots"] == S
                and r["iters"] == args.iters
                and r["batch_size"] == args.batch_size
            ):
                return r
        return None

    combos = [
        (dev, mode, L, S) for dev in devices for mode in MODES for L in L_VALUES for S in S_VALUES
    ]
    missing = [c for c in combos if find(c[0].type, c[1], c[2], c[3]) is None]

    if not missing:
        print(
            "All measurements already present for available devices "
            f"({[d.type for d in devices]}) — skipping benchmark, recomputing scaling fits only."
        )
    else:
        print(
            f"Benchmarking {len(missing)} of {len(combos)} combos "
            f"(devices: {[d.type for d in devices]}, batch={args.batch_size}, "
            f"warmup={args.warmup}, iters={args.iters})"
        )

    torch.manual_seed(0)
    for dev, mode, L, S in missing:
        module = build_sa(mode, S, args.d_model, args.n_iter).to(dev)
        tokens = torch.randn(args.batch_size, L, args.d_model, device=dev)
        mean_ms, std_ms = bench_forward(module, tokens, args.warmup, args.iters, dev)
        if not (math.isfinite(mean_ms) and math.isfinite(std_ms)):
            raise RuntimeError(f"non-finite timing for {dev.type}/{mode}/L={L}/S={S}")
        rec = {
            "sweep": "latency",
            "record_type": "measurement",
            "mode": mode,
            "n_slots": S,
            "seq_len": L,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "n_iter": args.n_iter,
            "device": dev.type,
            "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "warmup": args.warmup,
            "iters": args.iters,
            "mean_ms": round(mean_ms, 4),
            "std_ms": round(std_ms, 4),
            "torch_version": torch.__version__,
        }
        all_results.append(rec)
        measurements.append(rec)
        print(f"  {dev.type:>4} {mode:<16} L={L:<4} S={S:<3} {mean_ms:8.3f} ± {std_ms:.3f} ms")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # O(S * I) scaling fits (ms per slot) per (device, mode, L), recomputed each run.
    all_results = [
        r
        for r in all_results
        if not (r.get("sweep") == "latency" and r.get("record_type") == "scaling_fit")
    ]
    print("\nScaling fits (ms vs S):")
    for dev_type in sorted({r["device"] for r in measurements}):
        for mode in MODES:
            for L in L_VALUES:
                sel = sorted(
                    [
                        r
                        for r in measurements
                        if r["device"] == dev_type and r["mode"] == mode and r["seq_len"] == L
                    ],
                    key=lambda r: r["n_slots"],
                )
                if len(sel) < 2:
                    continue
                xs = [r["n_slots"] for r in sel]
                ys = [r["mean_ms"] for r in sel]
                slope, intercept, r2 = linear_fit(xs, ys)
                all_results.append(
                    {
                        "sweep": "latency",
                        "record_type": "scaling_fit",
                        "mode": mode,
                        "seq_len": L,
                        "device": dev_type,
                        "n_iter": args.n_iter,
                        "S_values": xs,
                        "ms_per_slot": round(slope, 5),
                        "intercept_ms": round(intercept, 5),
                        "r2": round(r2, 5),
                    }
                )
                print(
                    f"  {dev_type:>4} {mode:<16} L={L:<4} "
                    f"{slope:8.4f} ms/slot (intercept {intercept:.3f} ms, R²={r2:.4f})"
                )

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} records to {out_path}")
    return all_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slot-attention forward-pass latency microbenchmark"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-iter", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore (and overwrite) an existing results file instead of skipping runs "
        "that already have a record",
    )
    parser.add_argument("--output-dir", type=str, default="results/runs/latency")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
