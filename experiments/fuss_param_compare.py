#!/usr/bin/env python3
"""Compare FUSS parameter MAE across seeds with the trivial-predictor baseline (App. J).

Reads ``fuss_results.json`` files written by ``experiments/fuss.py`` and
prints, per method, mean ± std over seeds of peak overlap, cosine overlap and
the known-K exact-Hungarian parameter MAE; then classifies ``evidence_linear``
per parameter against the train-median constant predictor:

* model clearly below the baseline -> parameter recovery is a valid task metric;
* within tolerance                 -> tie, avoid downstream claims;
* baseline clearly below the model -> only collapse transfers.

"Clearly" means mean ± std does not reach the baseline +/- ``--tol``.

Usage::

    python experiments/fuss_param_compare.py   # one sub-directory per FUSS run/seed slice

The default ``--results-glob`` is ``results/runs/fuss/*/fuss_results.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

# Train-fit constant-predictor MAE (output of experiments/fuss_param_baseline.py
# with the paper configuration: 4 s crops, 5000 train / 1000 validation mixtures).
BASELINE_MEDIAN = {"freq": 0.0388, "amp": 0.0041, "centroid": 0.0621}
BASELINE_MEAN = {"freq": 0.0549, "amp": 0.0048, "centroid": 0.0806}

PARAMS = ("freq", "amp", "centroid")
METHODS = ["vanilla", "sequential", "evidence_linear", "evidence_quadratic", "mesh", "sparsemax"]


def main(argv: list[str] | None = None) -> dict[str, dict[str, list[float]]]:
    ap = argparse.ArgumentParser(description="FUSS parameter MAE vs. trivial baseline")
    ap.add_argument("--results-glob", default="results/runs/fuss/*/fuss_results.json")
    ap.add_argument(
        "--tol", type=float, default=0.002, help="abs-MAE tolerance for 'clearly' vs 'ties'"
    )
    args = ap.parse_args(argv)

    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fn in glob.glob(args.results_glob):
        with open(fn) as f:
            records = json.load(f)
        for x in records:
            m = x["method"]
            for p in PARAMS:
                v = x.get(f"param_mae_{p}")
                if v is not None:
                    by[m][p].append(v)
            if x.get("collapse_rate") is not None:
                by[m]["collapse"].append(x["collapse_rate"])
            if x.get("mean_attn_overlap") is not None:
                by[m]["cos"].append(x["mean_attn_overlap"])

    def ms(values: list[float]) -> str:
        return f"{np.mean(values):.4f}±{np.std(values):.4f}" if values else "--"

    print(
        f"{'method':20}{'n':>3}  {'collapse':>9}{'cos-ovl':>9}   "
        f"{'freq(mean±std)':>18}{'amp':>16}{'centroid':>18}"
    )
    for m in METHODS:
        if m not in by:
            continue
        d = by[m]
        cr = f"{np.mean(d['collapse']):.3f}" if d.get("collapse") else "--"
        co = f"{np.mean(d['cos']):.3f}" if d.get("cos") else "--"
        print(
            f"{m:20}{len(d.get('freq', [])):>3}  {cr:>9}{co:>9}   "
            f"{ms(d.get('freq', [])):>18}{ms(d.get('amp', [])):>16}"
            f"{ms(d.get('centroid', [])):>18}"
        )

    print(
        f"\nTRAIN-fit baseline  median: freq {BASELINE_MEDIAN['freq']:.4f} "
        f"amp {BASELINE_MEDIAN['amp']:.4f} centroid {BASELINE_MEDIAN['centroid']:.4f}"
    )
    print("\n=== VERDICT (evidence_linear vs train-median baseline, per param) ===")
    if "evidence_linear" in by:
        d = by["evidence_linear"]
        for p in PARAMS:
            v = d.get(p, [])
            if not v:
                print(f"  {p}: no data yet")
                continue
            mu, sd = np.mean(v), np.std(v)
            b = BASELINE_MEDIAN[p]
            if mu + sd < b - args.tol:
                verdict = "MODEL CLEARLY BEATS baseline -> valid task metric"
            elif mu - sd > b + args.tol:
                verdict = "BASELINE CLEARLY BEATS model -> only collapse transfers"
            else:
                verdict = "TIES -> weak; avoid downstream claims"
            print(f"  {p:9} model {mu:.4f}±{sd:.4f}  vs baseline {b:.4f}  -> {verdict}")
    return {m: dict(v) for m, v in by.items()}


if __name__ == "__main__":
    main()
