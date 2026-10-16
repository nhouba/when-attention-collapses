#!/usr/bin/env python3
"""Synthetic benchmark harness: method comparison and sweeps (Sec. 3.1, App. E-I).

Trains every method x task x seed, evaluates collapse / K accuracy / parameter
MAE, and appends one record per run to ``<output-dir>/ablation_results.json``
(resume-safe: finished runs are skipped).

Usage::

    python experiments/synthetic.py                                   # Table 1 / Table 9
    python experiments/synthetic.py --methods vanilla evidence_linear --tasks sinusoids
    python experiments/synthetic.py --methods mesh sa_sh sparsemax entmax15 \\
        coverage coverage_loss stickbreaking dspn                     # Table 2
    python experiments/synthetic.py --gamma-sweep --tau-sweep         # App. E
    python experiments/synthetic.py --amp-ratio-sweep --track-dynamics  # App. I
    python experiments/synthetic.py --stress-sweep                    # App. I
    python experiments/synthetic.py --arch-sweep                      # App. I

Defaults are the paper's budget (Sec. 3.1: 200 epochs, 5000 training and 1000
test mixtures). Resuming skips runs that already have a record without checking their budget
(a warning is printed); ``--no-resume`` starts from an empty results file.

Record ``sweep`` values: ``method``, ``sensitivity``, ``amp_ratio``, ``stress``, ``arch``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evidence_depletion.data.synthetic import TASKS
from evidence_depletion.io import load_results
from evidence_depletion.training import train_synthetic

ALL_METHODS = [
    "vanilla",
    "sequential",
    "evidence_binary",
    "evidence_linear",
    "evidence_quadratic",
    "evidence_cubic",
    "dataspace_quadratic",
    "dataspace_quadratic_bias",
]
"""Default method comparison (Table 1, Table 9)."""

ALTERNATIVE_METHODS = [
    "mesh",
    "sa_sh",
    "sparsemax",
    "entmax15",
    "coverage",
    "coverage_loss",
    "stickbreaking",
    "dspn",
]
"""Alternative anti-collapse mechanisms (Table 2, App. H); run via ``--methods``."""

# Joint gamma x tau sensitivity grid for evidence_quadratic (App. E, Table 7).
GAMMA_VALUES = [1, 3, 10]
TAU_VALUES = [0.1, 0.3, 1.0]

# Amplitude-ratio sweep on task A (App. I, Table 11).
AMP_RATIO_VALUES = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
AMP_RATIO_METHODS = ["vanilla", "sequential", "evidence_linear"]

# Stress tests on task A (App. I, "Slot over-provisioning and cardinality"):
#   s_sweep: many empty slots (S >> K); bigk: K up to 8 with S = 9;
#   weak: K = 6 sources of comparable amplitude (amp_ratio = 1).
STRESS_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "s_sweep",
        "n_slots_list": [5, 8, 12, 16],
        "K_range": (1, 4),
        "methods": ["vanilla", "evidence_linear"],
        "amp_ratio": None,
    },
    {
        "name": "bigk",
        "n_slots_list": [9],
        "K_range": (1, 8),
        "methods": ["vanilla", "sequential", "evidence_linear"],
        "amp_ratio": None,
    },
    {
        "name": "weak",
        "n_slots_list": [7],
        "K_range": (6, 6),
        "methods": ["vanilla", "sequential", "evidence_linear"],
        "amp_ratio": 1.0,
    },
]

# Architecture sensitivity on task A (App. I, "Token count"): one axis at a time
# from the default (n_tokens=16, d_model=64, n_iter=3); 8 unique configurations.
ARCH_DEFAULT = {"n_tokens": 16, "d_model": 64, "n_iter": 3}
ARCH_CONFIGS = (
    [{"n_tokens": L} for L in [8, 16, 32, 64]]
    + [{"d_model": d} for d in [32, 128]]
    + [{"n_iter": i} for i in [1, 5]]
)
ARCH_METHODS = ["vanilla", "evidence_linear"]

ALL_TASKS = list(TASKS.keys())
RESULTS_FILE = "ablation_results.json"


def run_ablation(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the requested comparison and sweeps; returns all records."""
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / RESULTS_FILE

    methods = args.methods or ALL_METHODS
    tasks = args.tasks or ALL_TASKS
    seeds = list(range(args.seeds))

    base_config = {
        "n_train": args.n_train,
        "n_test": args.n_test,
        "N": 256,
        "K_range": (1, 4),
        "d_model": 64,
        "n_tokens": 16,
        "batch_size": 128,
        "lr": 3e-4,
        "epochs": args.epochs,
        "n_iter": 3,
        "gamma": 3.0,
        "tau": 0.3,
        "track_dynamics": args.track_dynamics,
    }

    all_results = load_results(out_path, resume=not args.no_resume)

    def save() -> None:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    def already_done(task, method, seed, sweep="method", extra=None) -> bool:
        for r in all_results:
            if (
                r["task"] == task
                and r["method"] == method
                and r["seed"] == seed
                and r["sweep"] == sweep
            ):
                if extra and any(r.get(k) != v for k, v in extra.items()):
                    continue
                return True
        return False

    any_special_sweep = (
        args.gamma_sweep
        or args.tau_sweep
        or args.amp_ratio_sweep
        or args.stress_sweep
        or args.arch_sweep
    )

    # Method comparison (skipped when only sweeps are requested without --methods).
    if not any_special_sweep or args.methods:
        total = len(methods) * len(tasks) * len(seeds)
        print(
            f"=== Method comparison: {len(methods)} methods × "
            f"{len(tasks)} tasks × {len(seeds)} seeds = {total} runs ===\n"
        )
    else:
        methods = []
        total = 0

    run_idx = 0
    for task in tasks:
        for method in methods:
            for seed in seeds:
                run_idx += 1
                if already_done(task, method, seed):
                    print(
                        f"[{run_idx}/{total}] task={task} method={method} seed={seed} "
                        "... SKIPPED (exists)"
                    )
                    continue
                print(
                    f"[{run_idx}/{total}] task={task} method={method} seed={seed} ... ",
                    end="",
                    flush=True,
                )
                result = train_synthetic(task, method, seed, {**base_config})
                result["sweep"] = "method"
                all_results.append(result)
                print(
                    f"collapse={result['collapse_rate']:.3f} "
                    f"K_acc={result['k_accuracy']:.3f} "
                    f"({result['train_time_s']:.0f}s)"
                )
                save()

    # Joint gamma x tau sensitivity grid (App. E).
    if args.gamma_sweep or args.tau_sweep:
        n_combos = len(GAMMA_VALUES) * len(TAU_VALUES)
        print(
            f"\n=== Sensitivity grid: {len(GAMMA_VALUES)} γ × {len(TAU_VALUES)} τ "
            f"= {n_combos} combos × {len(tasks)} tasks × {len(seeds)} seeds "
            f"= {n_combos * len(tasks) * len(seeds)} runs ===\n"
        )
        for task in tasks:
            for gamma in GAMMA_VALUES:
                for tau in TAU_VALUES:
                    for seed in seeds:
                        method_key = f"evidence_g{gamma}_t{tau}"
                        if already_done(task, method_key, seed, sweep="sensitivity"):
                            print(
                                f"  γ={gamma} τ={tau} task={task} seed={seed} ... SKIPPED (exists)"
                            )
                            continue
                        print(
                            f"  γ={gamma} τ={tau} task={task} seed={seed} ... ",
                            end="",
                            flush=True,
                        )
                        config = {**base_config, "gamma": gamma, "tau": tau}
                        result = train_synthetic(task, "evidence_quadratic", seed, config)
                        result["sweep"] = "sensitivity"
                        result["gamma"] = gamma
                        result["tau"] = tau
                        result["method"] = method_key
                        all_results.append(result)
                        print(
                            f"collapse={result['collapse_rate']:.3f} "
                            f"({result['train_time_s']:.0f}s)"
                        )
                        save()

    # Amplitude-ratio sweep (App. I, Table 11; task A only).
    if args.amp_ratio_sweep:
        amp_methods = [
            m for m in AMP_RATIO_METHODS if args.sweep_methods is None or m in args.sweep_methods
        ]
        n_runs = len(AMP_RATIO_VALUES) * len(amp_methods) * len(seeds)
        print(
            f"\n=== Amplitude-ratio sweep: {len(AMP_RATIO_VALUES)} ratios × "
            f"{len(amp_methods)} methods × {len(seeds)} seeds "
            f"= {n_runs} runs (task: sinusoids) ===\n"
        )
        for ratio in AMP_RATIO_VALUES:
            for method in amp_methods:
                for seed in seeds:
                    extra = {"amp_ratio": ratio}
                    if already_done("sinusoids", method, seed, sweep="amp_ratio", extra=extra):
                        print(f"  ratio={ratio} method={method} seed={seed} ... SKIPPED (exists)")
                        continue
                    print(f"  ratio={ratio} method={method} seed={seed} ... ", end="", flush=True)
                    config = {**base_config, "amp_ratio": ratio}
                    result = train_synthetic("sinusoids", method, seed, config)
                    result["sweep"] = "amp_ratio"
                    result["amp_ratio"] = ratio
                    all_results.append(result)
                    print(
                        f"collapse={result['collapse_rate']:.3f} "
                        f"overlap={result['mean_attn_overlap']:.3f} "
                        f"({result['train_time_s']:.0f}s)"
                    )
                    save()

    # Stress tests (App. I; task A only).
    if args.stress_sweep:
        n_runs = sum(len(c["n_slots_list"]) * len(c["methods"]) for c in STRESS_CONFIGS) * len(
            seeds
        )
        print(
            f"\n=== Stress sweep: {[c['name'] for c in STRESS_CONFIGS]} × {len(seeds)} seeds "
            f"= {n_runs} runs (task: sinusoids) ===\n"
        )
        for cfg in STRESS_CONFIGS:
            for n_slots in cfg["n_slots_list"]:
                for method in cfg["methods"]:
                    for seed in seeds:
                        label = f"  {cfg['name']} S={n_slots} method={method} seed={seed} ... "
                        extra = {"stress_config": cfg["name"], "n_slots": n_slots}
                        if already_done("sinusoids", method, seed, sweep="stress", extra=extra):
                            print(label + "SKIPPED (exists)")
                            continue
                        print(label, end="", flush=True)
                        config = {
                            **base_config,
                            "K_range": cfg["K_range"],
                            "n_slots": n_slots,
                            "log_depletion": True,
                        }
                        if cfg["amp_ratio"] is not None:
                            config["amp_ratio"] = cfg["amp_ratio"]

                        result = train_synthetic("sinusoids", method, seed, config)
                        result["sweep"] = "stress"
                        result["stress_config"] = cfg["name"]
                        result["n_slots"] = n_slots
                        result["max_K"] = cfg["K_range"][1]
                        if cfg["amp_ratio"] is not None:
                            result["amp_ratio"] = cfg["amp_ratio"]
                        all_results.append(result)

                        dep = ""
                        if result.get("depleted_matched") is not None:
                            dep = (
                                f" depl_m={result['depleted_matched']:.3f}"
                                f" depl_u={result['depleted_unmatched']:.3f}"
                            )
                        print(
                            f"collapse={result['collapse_rate']:.3f} "
                            f"K_acc={result['k_accuracy']:.3f}{dep} "
                            f"({result['train_time_s']:.0f}s)"
                        )
                        save()

    # Architecture sensitivity (App. I; task A only).
    if args.arch_sweep:
        n_runs = len(ARCH_CONFIGS) * len(ARCH_METHODS) * len(seeds)
        print(
            f"\n=== Architecture sweep: {len(ARCH_CONFIGS)} configs × "
            f"{len(ARCH_METHODS)} methods × {len(seeds)} seeds "
            f"= {n_runs} runs (task: sinusoids) ===\n"
        )
        for arch in ARCH_CONFIGS:
            resolved = {**ARCH_DEFAULT, **arch}
            for method in ARCH_METHODS:
                for seed in seeds:
                    label = (
                        f"  L={resolved['n_tokens']} d={resolved['d_model']} "
                        f"I={resolved['n_iter']} method={method} seed={seed} ... "
                    )
                    extra = {
                        "l_tokens": resolved["n_tokens"],
                        "d_model": resolved["d_model"],
                        "n_iterations": resolved["n_iter"],
                    }
                    if already_done("sinusoids", method, seed, sweep="arch", extra=extra):
                        print(label + "SKIPPED (exists)")
                        continue
                    print(label, end="", flush=True)
                    result = train_synthetic("sinusoids", method, seed, {**base_config, **resolved})
                    result["sweep"] = "arch"
                    result["l_tokens"] = resolved["n_tokens"]
                    result["d_model"] = resolved["d_model"]
                    result["n_iterations"] = resolved["n_iter"]
                    all_results.append(result)
                    print(
                        f"collapse={result['collapse_rate']:.3f} "
                        f"K_acc={result['k_accuracy']:.3f} "
                        f"({result['train_time_s']:.0f}s)"
                    )
                    save()

    save()
    print(f"\nSaved {len(all_results)} results to {out_path}")
    print_summary(all_results)
    return all_results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Mean ± std tables of every sweep present in ``results``."""
    print("\n" + "=" * 80)
    print("SUMMARY: Method Comparison (mean ± std across seeds)")
    print("=" * 80)

    method_results = [r for r in results if r["sweep"] == "method"]
    tasks = sorted({r["task"] for r in method_results})
    methods = sorted({r["method"] for r in method_results})
    for task in tasks:
        print(f"\n{'─' * 60}")
        print(f"Task: {task}")
        print(f"{'Method':<25} {'Collapse↓':>10} {'K Acc↑':>10} {'Param MAE↓':>12}")
        print(f"{'─' * 60}")
        for method in methods:
            runs = [r for r in method_results if r["task"] == task and r["method"] == method]
            if not runs:
                continue
            collapses = [r["collapse_rate"] for r in runs]
            k_accs = [r["k_accuracy"] for r in runs]
            mae_keys = [k for k in runs[0] if k.startswith("mae_")]
            if mae_keys:
                avg_maes = [np.mean([r[k] for k in mae_keys]) for r in runs]
                mae_str = f"{np.mean(avg_maes):.4f}±{np.std(avg_maes):.4f}"
            else:
                mae_str = "N/A"
            print(
                f"{method:<25} "
                f"{np.mean(collapses):.3f}±{np.std(collapses):.3f} "
                f"{np.mean(k_accs):.3f}±{np.std(k_accs):.3f} "
                f"{mae_str:>12}"
            )

    sens_results = [r for r in results if r["sweep"] == "sensitivity"]
    for task in sorted({r["task"] for r in sens_results}):
        print(f"\n{'─' * 60}")
        print(f"Sensitivity grid: {task}")
        gammas = sorted({r["gamma"] for r in sens_results})
        taus = sorted({r["tau"] for r in sens_results})
        gt_label = "g \\ t"
        print(f"{gt_label:<10}" + "".join(f"{'t=' + str(t):>16}" for t in taus))
        print(f"{'─' * 60}")
        for g in gammas:
            row = f"{'g=' + str(g):<10}"
            for t in taus:
                runs = [
                    r
                    for r in sens_results
                    if r["task"] == task and r["gamma"] == g and r["tau"] == t
                ]
                if runs:
                    vals = [r["collapse_rate"] for r in runs]
                    row += f"{np.mean(vals):.3f}±{np.std(vals):.3f}".rjust(16)
                else:
                    row += "---".rjust(16)
            print(row)

    _print_grouped_summary(results, "amp_ratio", ["amp_ratio"])
    _print_grouped_summary(results, "stress", ["stress_config", "n_slots"])
    _print_grouped_summary(results, "arch", ["l_tokens", "d_model", "n_iterations"])


def _print_grouped_summary(
    results: list[dict[str, Any]], sweep: str, group_keys: list[str]
) -> None:
    """Mean ± std of collapse and cosine overlap per configuration and method."""
    runs = [r for r in results if r["sweep"] == sweep]
    if not runs:
        return
    print(f"\n{'─' * 60}")
    print(f"Sweep: {sweep}")
    print(f"{'Config':<30} {'Method':<18} {'Collapse↓':>12} {'Overlap↓':>12}")
    print(f"{'─' * 60}")
    groups = sorted(
        {tuple(r.get(k) for k in group_keys) for r in runs},
        key=lambda g: tuple(str(x) for x in g),
    )
    methods = sorted({r["method"] for r in runs})
    for group in groups:
        label = " ".join(f"{k}={v}" for k, v in zip(group_keys, group))
        for method in methods:
            sel = [
                r
                for r in runs
                if r["method"] == method and tuple(r.get(k) for k in group_keys) == group
            ]
            if not sel:
                continue
            col = [r["collapse_rate"] for r in sel]
            ovl = [r["mean_attn_overlap"] for r in sel]
            print(
                f"{label:<30} {method:<18} "
                f"{np.mean(col):.3f}±{np.std(col):.3f} "
                f"{np.mean(ovl):.3f}±{np.std(ovl):.3f}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evidence depletion ablations")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to test (default: all)")
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to test (default: all)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds (default: 5)")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Training epochs per run (default: 200)"
    )
    parser.add_argument(
        "--n-train", type=int, default=5000, help="Training samples (default: 5000)"
    )
    parser.add_argument("--n-test", type=int, default=1000, help="Test samples (default: 1000)")
    parser.add_argument("--gamma-sweep", action="store_true", help="Run gamma sensitivity sweep")
    parser.add_argument("--tau-sweep", action="store_true", help="Run tau sensitivity sweep")
    parser.add_argument(
        "--amp-ratio-sweep",
        action="store_true",
        help="Amplitude-ratio sweep on task A (sinusoids), App. I",
    )
    parser.add_argument(
        "--sweep-methods",
        nargs="+",
        default=None,
        help="Restrict the method list of --amp-ratio-sweep "
        "(lets the sweep be split across jobs; ignored by other modes)",
    )
    parser.add_argument(
        "--stress-sweep",
        action="store_true",
        help="Stress tests (S>>K, K<=8, equal-amplitude weak sources) on task A, App. I",
    )
    parser.add_argument(
        "--arch-sweep",
        action="store_true",
        help="Architecture sensitivity sweep (token count, d_model, slot iterations) "
        "on task A, App. I",
    )
    parser.add_argument(
        "--track-dynamics",
        action="store_true",
        help="Log per-epoch slot-embedding cosine similarity and attention overlap on "
        "a fixed held-out batch (stored under 'dynamics' in each result record)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore (and overwrite) an existing results file instead of skipping runs "
        "that already have a record",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/runs/synthetic",
        help="Output directory for results",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_ablation(parse_args())
