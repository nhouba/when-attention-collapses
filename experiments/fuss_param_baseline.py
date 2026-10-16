#!/usr/bin/env python3
"""Trivial-predictor calibration of the FUSS parameter MAE (App. J).

A constant predictor (per-parameter mean or median of all *training* sources)
is evaluated unchanged on the validation split, the model's test set, with the
same parameter normalization and per-source weighting as the model's MAE. A
learned model's MAE is only meaningful if it is clearly below this baseline.

Usage::

    python experiments/fuss_param_baseline.py --fuss-dir data/fuss/ssdata
"""

from __future__ import annotations

import argparse

import numpy as np

from evidence_depletion.data.fuss import FUSSDataset

PARAM_NAMES = ["freq", "amp", "centroid"]


def collect_params(ds: FUSSDataset, limit: int) -> np.ndarray:
    """Parameters of every source of the first ``limit`` mixtures: ``[n_sources, 3]``."""
    P = []
    for i in range(min(len(ds), limit)):
        _, params, K = ds[i]
        P.append(params[: int(K)].numpy())
    return np.concatenate(P, axis=0)


def main(argv: list[str] | None = None) -> dict[str, dict[str, float]]:
    ap = argparse.ArgumentParser(description="Train-fit constant predictor for FUSS parameters")
    ap.add_argument("--fuss-dir", required=True)
    ap.add_argument("--signal-length", type=int, default=64000)
    ap.add_argument("--max-train", type=int, default=5000)
    ap.add_argument("--max-test", type=int, default=1000)
    args = ap.parse_args(argv)

    train = FUSSDataset(args.fuss_dir, split="train", max_K=4, signal_length=args.signal_length)
    val = FUSSDataset(args.fuss_dir, split="validation", max_K=4, signal_length=args.signal_length)
    Ptr = collect_params(train, args.max_train)
    Pval = collect_params(val, args.max_test)

    mean_pred = Ptr.mean(axis=0)
    median_pred = np.median(Ptr, axis=0)

    print(f"train_sources={len(Ptr)}  val_sources={len(Pval)}")
    print(f"{'param':10}{'MAE(train-mean)':>16}{'MAE(train-median)':>18}")
    out: dict[str, dict[str, float]] = {"mean": {}, "median": {}}
    for j, name in enumerate(PARAM_NAMES):
        out["mean"][name] = float(np.abs(Pval[:, j] - mean_pred[j]).mean())
        out["median"][name] = float(np.abs(Pval[:, j] - median_pred[j]).mean())
        print(f"{name:10}{out['mean'][name]:>16.4f}{out['median'][name]:>18.4f}")
    print("\nConstant fit on TRAIN, evaluated on VALIDATION.")
    print("A model's MAE must be clearly below MAE(train-median) per parameter.")
    return out


if __name__ == "__main__":
    main()
