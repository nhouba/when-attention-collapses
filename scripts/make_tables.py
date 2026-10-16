#!/usr/bin/env python3
"""Aggregate run JSONs into the paper's synthetic / FUSS tables (markdown on stdout).

Reads every *.json below the given directories (default: results/runs) and
recognises the record schemas written by the experiment scripts:

  experiments/synthetic.py, experiments/matching_pursuit.py -> ablation_results.json
      records with "task", "method", "seed", "sweep" in {method, sensitivity,
      amp_ratio, stress, arch}, "collapse_rate" (= peak overlap),
      "mean_attn_overlap" (= cosine overlap), "k_accuracy", ...
  experiments/loss_baselines.py   -> baseline_results.json (records with "w_div")
  experiments/fuss.py             -> fuss_results.json (records with task == "fuss")
  experiments/order_studies.py    -> order_studies_results.json (records with "study")
  experiments/latency.py          -> latency_results.json (records with "record_type")

Aggregation follows the paper: mean and POPULATION std (numpy default, ddof=0)
over seeds, printed with two decimals. Duplicate records (same key) are
collapsed; conflicting duplicates are reported on stderr.

Usage:
    python scripts/make_tables.py                     # results/runs
    python scripts/make_tables.py results/runs/synthetic_main results/runs/fuss
    python scripts/make_tables.py results/paper/raw   # the original per-seed result files
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = ["sinusoids", "gaussian_bumps", "multiscale"]
TASK_HDR = ["Sinusoids", "Gaussians", "Multi-scale"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def classify(rec):
    if not isinstance(rec, dict):
        return None
    if "kind" in rec and "source" in rec:
        # log-derived reference records (results/paper/raw/log_records.json): only the
        # FUSS run behind the paper's FUSS numbers is used; the other log records are
        # summary rows without a matching per-seed result file
        return "fuss" if str(rec["source"]).startswith("fuss_rerun/") else None
    if "record_type" in rec:
        return "latency"
    if "study" in rec:
        return "order"
    if rec.get("task") == "fuss":
        return "fuss"
    if "w_div" in rec:
        return "loss"
    if "task" in rec and "method" in rec and "collapse_rate" in rec:
        return "ablation"
    return None


EXTRA_KEYS = (
    "sweep",
    "gamma",
    "tau",
    "amp_ratio",
    "stress_config",
    "n_slots",
    "l_tokens",
    "d_model",
    "n_iterations",
    "study",
    "perturbation",
    "record_type",
    "mode",
    "seq_len",
    "device",
)


def load_records(dirs):
    store = defaultdict(dict)  # kind -> key -> record
    for d in dirs:
        d = Path(d)
        files = [d] if d.is_file() else sorted(d.rglob("*.json"))
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(data, list):
                continue
            for rec in data:
                kind = classify(rec)
                if kind is None:
                    continue
                key = (rec.get("task"), rec.get("method"), rec.get("seed")) + tuple(
                    rec.get(k) for k in EXTRA_KEYS
                )
                old = store[kind].get(key)
                if (
                    old is not None
                    and "collapse_rate" in rec
                    and old.get("collapse_rate") != rec.get("collapse_rate")
                ):
                    print(
                        f"[warn] conflicting duplicate {kind} {key}: "
                        f"{old.get('collapse_rate')} vs {rec.get('collapse_rate')} "
                        f"(keeping {f})",
                        file=sys.stderr,
                    )
                store[kind][key] = rec
    return {k: list(v.values()) for k, v in store.items()}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def sel(recs, **flt):
    out = []
    for r in recs:
        ok = True
        for k, v in flt.items():
            rv = r.get(k, "method" if k == "sweep" else None)
            if rv != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def fmt(vals, std=True, nd=2):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "---"
    a = np.asarray(vals, float)
    s = f"{a.mean():.{nd}f}"
    if std:
        s += f" ± {a.std():.{nd}f}"
    return s + (f" (n={a.size})" if a.size != 5 else "")


def table(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def per_task_rows(recs, methods, metric="collapse_rate", extra_col=None, labels=None):
    rows = []
    for m in methods:
        cells = [fmt([r.get(metric) for r in sel(recs, task=t, method=m)]) for t in TASKS]
        if all(c == "---" for c in cells):
            continue
        if extra_col is not None:
            cells.append(extra_col(m))
        rows.append([(labels or {}).get(m, m)] + cells)
    return rows


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

LABELS = {
    "vanilla": "Vanilla SA",
    "sequential": "Sequential SA",
    "evidence_binary": "Evidence binary",
    "evidence_linear": "Evidence linear (1-a)",
    "evidence_quadratic": "Evidence quadratic (1-a^2)",
    "evidence_cubic": "Evidence cubic (1-a^3)",
    "dataspace_quadratic": "Data-space quad. (no bias)",
    "dataspace_quadratic_bias": "Data-space quad. (with bias)",
    "sparsemax": "Sparsemax",
    "entmax15": "Entmax-1.5",
    "sa_sh": "SA-SH",
    "mesh": "MESH (full)",
    "coverage": "Coverage (mech.)",
    "coverage_loss": "Coverage (loss)",
    "stickbreaking": "Stick-breaking",
    "dspn": "DSPN-lite set predictor (text only)",
}


def synthetic_tables(abl, fuss):
    meth = sel(abl, sweep="method")
    if not meth and not fuss:
        return

    def fuss_col(m):
        return fmt([r["collapse_rate"] for r in fuss if r["method"] == m])

    t1 = [
        "vanilla",
        "sequential",
        "evidence_linear",
        "evidence_quadratic",
        "dataspace_quadratic",
        "dataspace_quadratic_bias",
    ]
    table(
        "Table 1 - peak overlap, mean ± pop. std",
        ["Method"] + TASK_HDR + ["FUSS"],
        per_task_rows(meth, t1, extra_col=fuss_col, labels=LABELS),
    )
    table(
        "Table 9 (App. G) - peak overlap",
        ["Method"] + TASK_HDR,
        per_task_rows(
            meth,
            ["evidence_binary", "evidence_linear", "evidence_quadratic", "evidence_cubic"],
            labels=LABELS,
        ),
    )
    rows = []
    for m in ["vanilla", "sequential", "evidence_linear"]:
        cells = []
        for t in TASKS:
            rr = sel(meth, task=t, method=m)
            cells.append(
                f"{fmt([r['collapse_rate'] for r in rr], std=False)} / "
                f"{fmt([r.get('mean_attn_overlap') for r in rr], std=False)}"
            )
        ff = [r for r in fuss if r["method"] == m]
        cells.append(
            f"{fmt([r['collapse_rate'] for r in ff], std=False)} / "
            f"{fmt([r.get('mean_attn_overlap') for r in ff], std=False)}"
        )
        rows.append([LABELS[m]] + cells)
    table(
        "Table 8 (App. F) - peak / cosine, means",
        ["Method"] + TASK_HDR + ["FUSS"],
        rows,
    )
    alt = [
        "vanilla",
        "sparsemax",
        "entmax15",
        "sa_sh",
        "mesh",
        "coverage",
        "coverage_loss",
        "stickbreaking",
        "dspn",
        "evidence_linear",
    ]
    table(
        "Table 2 - peak overlap",
        ["Method"] + TASK_HDR,
        per_task_rows(meth, alt, labels=LABELS),
    )
    table(
        "App. H text - cosine overlap of the alternatives",
        ["Method"] + TASK_HDR,
        per_task_rows(meth, alt, metric="mean_attn_overlap", labels=LABELS),
    )
    mp = sel(meth, task="sinusoids", method="matching_pursuit")
    if mp:
        print(
            f"\nMatching pursuit (sinusoids, eval-only; collapse is hard-coded 0): "
            f"K-acc {fmt([r['k_accuracy'] for r in mp], nd=3)}, "
            f"MAE(freq) {fmt([r.get('mae_frequency') for r in mp], nd=3)}"
        )


def loss_table(loss):
    if not loss:
        return
    order = [
        "vanilla_divloss",
        "vanilla_ortho",
        "vanilla_divloss_ortho",
        "seq_divloss",
        "seq_ortho",
        "seq_divloss_ortho",
    ]
    table(
        "Table 10 (App. H) - peak overlap",
        ["Variant"] + TASK_HDR,
        per_task_rows(loss, order),
    )


def sensitivity_table(abl):
    sens = sel(abl, sweep="sensitivity")
    if not sens:
        return
    gammas = sorted({r["gamma"] for r in sens})
    taus = sorted({r["tau"] for r in sens})
    rows = [
        [f"gamma={g}"]
        + [
            fmt([r["collapse_rate"] for r in sens if r["gamma"] == g and r["tau"] == t])
            for t in taus
        ]
        for g in gammas
    ]
    table(
        "Table 7 (App. E) - sinusoids, evidence_quadratic",
        [""] + [f"tau={t}" for t in taus],
        rows,
    )


def stress_tables(abl):
    amp = sel(abl, sweep="amp_ratio")
    if amp:
        ratios = sorted({r["amp_ratio"] for r in amp})
        rows = []
        for m in ["vanilla", "sequential", "evidence_linear"]:
            rr = [
                fmt([r["collapse_rate"] for r in amp if r["method"] == m and r["amp_ratio"] == a])
                for a in ratios
            ]
            if any(c != "---" for c in rr):
                rows.append([LABELS[m]] + rr)
        table(
            "Table 11 (App. I) - peak overlap, mean ± pop. std",
            ["Ratio"] + [f"{a:g}" for a in ratios],
            rows,
        )
    st = sel(abl, sweep="stress")
    if st:
        rows = []
        for cfg in ["s_sweep", "bigk", "weak"]:
            for S in sorted({r["n_slots"] for r in st if r["stress_config"] == cfg}):
                for m in ["vanilla", "sequential", "evidence_linear"]:
                    rr = [
                        r
                        for r in st
                        if r["stress_config"] == cfg and r["n_slots"] == S and r["method"] == m
                    ]
                    if not rr:
                        continue
                    rows.append(
                        [
                            cfg,
                            str(S),
                            m,
                            fmt([r["collapse_rate"] for r in rr], nd=3),
                            fmt([r["k_accuracy"] for r in rr], std=False, nd=3),
                            fmt([r.get("depleted_matched") for r in rr], std=False),
                            fmt([r.get("depleted_unmatched") for r in rr], std=False),
                        ]
                    )
        table(
            "App. I - slot over-provisioning (s_sweep) and cardinality (bigk, weak)",
            [
                "config",
                "S",
                "method",
                "peak overlap",
                "exact-K acc",
                "depl. matched",
                "depl. unmatched",
            ],
            rows,
        )
    arch = sel(abl, sweep="arch")
    if arch:
        rows = []
        for key in sorted({(r["l_tokens"], r["d_model"], r["n_iterations"]) for r in arch}):
            for m in ["vanilla", "evidence_linear"]:
                rr = [
                    r
                    for r in arch
                    if (r["l_tokens"], r["d_model"], r["n_iterations"]) == key and r["method"] == m
                ]
                if rr:
                    rows.append(
                        [
                            str(key[0]),
                            str(key[1]),
                            str(key[2]),
                            m,
                            fmt([r["collapse_rate"] for r in rr], nd=3),
                            fmt([r["mean_attn_overlap"] for r in rr], nd=3),
                        ]
                    )
        table(
            "App. I - token count / architecture sweep (paper reports the L axis only)",
            ["L", "d", "I", "method", "peak overlap", "cosine"],
            rows,
        )


def fuss_tables(fuss):
    if not fuss:
        return
    methods = [
        m
        for m in [
            "vanilla",
            "sequential",
            "evidence_linear",
            "evidence_quadratic",
            "sa_sh",
            "mesh",
            "sparsemax",
        ]
        if any(r["method"] == m for r in fuss)
    ]
    seeds = sorted({r["seed"] for r in fuss})
    rows = []
    for m in methods:
        rr = {r["seed"]: r for r in fuss if r["method"] == m}
        rows.append(
            [LABELS.get(m, m)]
            + [f"{rr[s]['collapse_rate']:.2f}" if s in rr else "---" for s in seeds]
            + [
                fmt([r["collapse_rate"] for r in rr.values()]),
                fmt([r.get("mean_attn_overlap") for r in rr.values()], std=False),
                fmt([r.get("si_sdri") for r in rr.values()]),
            ]
        )
    table(
        "Table 13 (App. J) - FUSS per seed",
        ["Method"] + [f"Seed {s}" for s in seeds] + ["Peak mean ± std", "Cosine", "SI-SDRi [dB]"],
        rows,
    )


def order_latency(order, lat):
    if order:
        ev = [r for r in order if r["method"] == "evidence_linear"]
        rows = []
        for pert in ["none", "uniform", "wrong_peak"]:
            rr = [r for r in ev if r["study"] == "forced_error" and r["perturbation"] == pert]
            if rr:
                rows.append(
                    [
                        pert,
                        fmt([r["collapse_rate"] for r in rr], std=False, nd=3),
                        fmt([r["recovery_remaining"] for r in rr], std=False, nd=3),
                    ]
                )
        table(
            "App. I - early errors (evidence_linear, forced first-slot allocation)",
            ["perturbation", "collapse", "recovery_remaining"],
            rows,
        )
        eo = [r for r in ev if r["study"] == "evidence_ordered"]
        if eo:
            keys = [k for k in eo[0] if isinstance(eo[0][k], (int, float)) and k != "seed"]
            print(
                "\nEvidence-ordered inference (per-seed means): "
                + ", ".join(f"{k}={np.mean([r[k] for r in eo]):.4f}" for k in keys)
            )
        os_ = [r for r in ev if r["study"] == "order_sensitivity"]
        if os_:
            keys = [k for k in os_[0] if isinstance(os_[0][k], (int, float)) and k != "seed"]
            print(
                "Order sensitivity (per-seed means): "
                + ", ".join(f"{k}={np.mean([r[k] for r in os_]):.4f}" for k in keys)
            )
    if lat:
        meas = [r for r in lat if r["record_type"] == "measurement"]
        for dev in sorted({r["device"] for r in meas}):
            for L in sorted({r["seq_len"] for r in meas}):
                Ss = sorted({r["n_slots"] for r in meas})
                rows = []
                for mode in ["vanilla", "sequential", "evidence_linear"]:
                    rr = {
                        r["n_slots"]: r["mean_ms"]
                        for r in meas
                        if r["device"] == dev and r["seq_len"] == L and r["mode"] == mode
                    }
                    if rr:
                        rows.append([mode] + [f"{rr[s]:.1f}" if s in rr else "---" for s in Ss])
                if rows:
                    table(
                        f"Table 12 - ms per forward pass, device={dev}, L={L}",
                        ["S"] + [str(s) for s in Ss],
                        rows,
                    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="*", default=["results/runs"])
    args = ap.parse_args()
    recs = load_records(args.dirs)
    abl, fuss = recs.get("ablation", []), recs.get("fuss", [])
    print("Loaded: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(recs.items())))
    synthetic_tables(abl, fuss)
    loss_table(recs.get("loss", []))
    sensitivity_table(abl)
    stress_tables(abl)
    fuss_tables(fuss)
    order_latency(recs.get("order", []), recs.get("latency", []))


if __name__ == "__main__":
    main()
