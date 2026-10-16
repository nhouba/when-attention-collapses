#!/usr/bin/env python3
"""Numerical check of Proposition 1 (App. D): ``cos(u, u') >= 1 - 2 kappa^2``.

1. Random instances: tokens ``v_l = a_l h + r_l`` with ``||r_l|| <= kappa a_l`` and
   readouts under one-hot and random nonnegative weights; the minimum slack
   ``min cos(u, u') - (1 - 2 kappa^2)`` must be ``>= 0``.
2. Tightness: the two-token construction of the proof attains the bound.
3. The inequality chain ``1 - 2 d^2 >= (1 - 2d) / (1 - 2d + 2d^2)`` for ``d < 1/2``.

Usage::

    python experiments/app_d/check_bound.py
"""

from __future__ import annotations

import argparse

import numpy as np


def main(argv: list[str] | None = None) -> float:
    ap = argparse.ArgumentParser(description="Numerical check of Proposition 1")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    worst = 1e9
    for _ in range(args.trials):
        d = rng.integers(2, 8)
        L = rng.integers(2, 6)
        kap = rng.uniform(0, 0.999)
        h = np.zeros(d)
        h[0] = 1
        a = rng.uniform(0.1, 5, L)
        r = rng.normal(size=(L, d))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        r *= (a * kap * rng.uniform(0.9, 1, L))[:, None]
        V = a[:, None] * h + r
        k = (np.linalg.norm(r, axis=1) / a).max()
        W = np.vstack([np.eye(L), rng.exponential(size=(50, L))])
        U = W @ V
        Un = U / np.linalg.norm(U, axis=1, keepdims=True)
        worst = min(worst, (Un @ Un.T).min() - (1 - 2 * k**2))
    print("min slack (should be >=0):", worst)

    k = 0.6
    th = np.arcsin(k)
    h = np.array([1, 0.0])
    n = np.array([0, 1.0])
    vp = h + k * (-np.sin(th) * h + np.cos(th) * n)
    vm = h + k * (-np.sin(th) * h - np.cos(th) * n)
    print("tight:", vp @ vm / np.linalg.norm(vp) / np.linalg.norm(vm), 1 - 2 * k * k)

    ds = np.linspace(1e-4, 0.4999, 10000)
    gap = ((1 - 2 * ds**2) - (1 - 2 * ds) / (1 - 2 * ds + 2 * ds**2)).min()
    print("min of (1-2d^2)-(1-2d)/(1-2d+2d^2):", gap)
    return float(worst)


if __name__ == "__main__":
    main()
