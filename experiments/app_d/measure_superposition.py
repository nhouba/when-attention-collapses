#!/usr/bin/env python3
"""Token-space superposition strength of each synthetic task (App. D).

Tokens are time chunks (16 chunks of 16 samples, as in the tokenizers). For
every noise-free mixture with ``K >= 2``, the dominant source is the
max-energy component and its per-chunk energy share is
``s_l = E_dom,l / sum_k E_k,l``, restricted to chunks with more than 1% of the
mean chunk energy. Reports the mean over samples of ``min_l s_l``, of the mean
share and of the fraction of chunks the dominant source owns (> 50%), next to
vanilla SA's peak overlap on that task (Table 1).

Usage::

    python experiments/app_d/measure_superposition.py
"""

from __future__ import annotations

import argparse

import torch

from evidence_depletion.data.synthetic import TASKS

VANILLA_PEAK_OVERLAP = {"sinusoids": 0.319, "gaussian_bumps": 0.263, "multiscale": 0.219}
"""Vanilla SA peak overlap per task (Table 1, mean over 5 seeds)."""


def main(argv: list[str] | None = None) -> dict[str, dict[str, float]]:
    ap = argparse.ArgumentParser(description="Dominant-source energy share per token")
    ap.add_argument("--n", type=int, default=2048, help="mixtures per task")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    N, NTOK = 256, 16
    torch.manual_seed(args.seed)
    print(
        f"{'task':>16} | {'mean min_l s_l':>14} {'mean s_l':>9} {'frac chunks>50% dom':>20} "
        f"| {'vanilla collapse':>16}"
    )
    out = {}
    for name, task in TASKS.items():
        _, _, K_vals, comps = task["generate"](args.n, K_range=(2, 4), N=N, noise_std=0.0)
        mins, means, fracs = [], [], []
        for b in range(args.n):
            K = int(K_vals[b])
            e = (comps[b, :K].reshape(K, NTOK, N // NTOK) ** 2).sum(-1)
            dom = e.sum(1).argmax()
            tot = e.sum(0)
            keep = tot > 0.01 * tot.mean()
            share = (e[dom] / tot.clamp(min=1e-12))[keep]
            if len(share) == 0:
                continue
            mins.append(share.min().item())
            means.append(share.mean().item())
            fracs.append((share > 0.5).float().mean().item())
        out[name] = {
            "min_share": sum(mins) / len(mins),
            "mean_share": sum(means) / len(means),
            "frac_dominated": sum(fracs) / len(fracs),
        }
        r = out[name]
        print(
            f"{name:>16} | {r['min_share']:>14.3f} {r['mean_share']:>9.3f} "
            f"{r['frac_dominated']:>20.3f} | {VANILLA_PEAK_OVERLAP[name]:>16.3f}"
        )
    return out


if __name__ == "__main__":
    main()
