#!/usr/bin/env python3
"""Premise of Proposition 1 in the magnitude token space (App. D, "Empirical check").

For sinusoid mixtures (task A, ``K in {2, 3, 4}``) with tokens ``m_l = |rfft(chunk_l)|``
(16 chunks of 16 samples), computes per sample

* ``kappa*``: the sine of the half-angle of the smallest spherical cap that
  contains all token directions (the smallest achievable ``kappa``), and the
  bound ``1 - 2 kappa*^2``;
* the geometric floor: the minimum pairwise cosine between token directions;
* ``rho``: the cross-token amplitude ratio ``max_l a_res,l / min_l a_dom,l``
  (``a_dom`` from the dominant source's chunk energy, ``a_res`` from the rest).

Statistics are reported over all tokens and over energetic tokens (chunk energy
above 1% of the mean). Paper: median ``kappa* = 0.56``, median bound ``0.37``,
median ``rho = 0.21`` (which does not bound ``kappa`` in this space).

Usage::

    python experiments/app_d/kappa_mag.py
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from evidence_depletion.data.synthetic import generate_sinusoids


def min_enclosing_cap_angle(X: np.ndarray, iters: int = 3000) -> float:
    """Half-angle of an (approximately) minimal spherical cap containing the rows of ``X``."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    h = Xn.mean(0)
    h /= np.linalg.norm(h)
    for t in range(1, iters):
        far = Xn[np.argmin(Xn @ h)]
        h = h + (far - h) / (t + 1)
        h /= np.linalg.norm(h)
    return float(np.arccos(np.clip((Xn @ h).min(), -1, 1)))


def main(argv: list[str] | None = None) -> dict[str, np.ndarray]:
    ap = argparse.ArgumentParser(description="kappa* and rho in the magnitude token space")
    ap.add_argument("--n", type=int, default=1024, help="number of test mixtures")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args(argv)

    sig, _, K, comps = generate_sinusoids(
        args.n, K_range=(2, 4), N=256, noise_std=0.1, dominant=True, seed=args.seed
    )
    L, C = 16, 16
    m = torch.fft.rfft(sig.reshape(-1, L, C), dim=-1).abs().numpy()

    kap_all, kap_keep, floor_all, floor_keep, rho = [], [], [], [], []
    for b in range(args.n):
        k = int(K[b])
        c = comps[b, :k].numpy()
        e = (c.reshape(k, L, C) ** 2).sum(-1)
        dom = e.sum(1).argmax()
        s = sig[b].numpy().reshape(L, C)
        tot = (s**2).sum(-1)
        keep = tot > 0.01 * tot.mean()
        a_dom = np.sqrt(e[dom])
        a_res = np.sqrt(((s - c[dom].reshape(L, C)) ** 2).sum(-1))
        rho.append(a_res[keep].max() / a_dom[keep].min())
        for X, kap, floor in ((m[b], kap_all, floor_all), (m[b][keep], kap_keep, floor_keep)):
            kap.append(np.sin(min_enclosing_cap_angle(X)))
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            floor.append((Xn @ Xn.T).min())

    def q(a):
        return np.round(np.quantile(a, [0.05, 0.5, 0.95]), 3)

    rho, ka, kk = np.array(rho), np.array(kap_all), np.array(kap_keep)
    print("rho p05/med/p95", q(rho))
    print("kappa* (min-enclosing-cap, all tokens)", q(ka), " keep:", q(kk))
    print("bound 1-2kappa*^2 all", q(1 - 2 * ka**2), " keep", q(1 - 2 * kk**2))
    print("geometric floor = min pairwise token cos (all)", q(floor_all), " keep", q(floor_keep))
    print(
        "frac kappa* <= rho:",
        (kk <= rho).mean(),
        " bound with rho 1-2rho^2 median",
        np.median(1 - 2 * rho**2),
    )
    print(
        "frac floor_keep < 1-2rho^2 (i.e. kappa<=rho impossible):",
        (np.array(floor_keep) < 1 - 2 * rho**2).mean(),
    )
    return {"rho": rho, "kappa_all": ka, "kappa_keep": kk}


if __name__ == "__main__":
    main()
