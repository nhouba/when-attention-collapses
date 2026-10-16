#!/usr/bin/env python3
"""Loss-based baselines: can stronger regularization alone prevent collapse? (App. H, Table 10)

Vanilla or sequential SA trained with a stronger slot-diversity loss
(``w_div = 5``), an attention-orthogonality loss (``w_ortho = 3``), or both.
Architecture, data, tokenizer and evaluation are those of
``experiments/synthetic.py``; only the loss changes. Results go to
``<output-dir>/baseline_results.json`` (resume-safe).

Usage::

    python experiments/loss_baselines.py
    python experiments/loss_baselines.py --tasks sinusoids --seeds 3 --epochs 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evidence_depletion.data.synthetic import TASKS
from evidence_depletion.io import load_results
from evidence_depletion.training import LOSS_BASELINES, train_loss_baseline

ALL_METHODS = list(LOSS_BASELINES)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Loss-based baselines for slot collapse")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore (and overwrite) an existing results file instead of skipping runs "
        "that already have a record",
    )
    parser.add_argument("--output-dir", type=str, default="results/runs/loss_baselines")
    args = parser.parse_args(argv)

    methods = args.methods or ALL_METHODS
    tasks = args.tasks or list(TASKS.keys())
    seeds = list(range(args.seeds))

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "baseline_results.json"

    all_results = load_results(out_path, resume=not args.no_resume)

    def already_done(task: str, method: str, seed: int) -> bool:
        return any(
            r["task"] == task and r["method"] == method and r["seed"] == seed for r in all_results
        )

    config = {
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
    }

    total = len(methods) * len(tasks) * len(seeds)
    print(
        f"=== Baseline comparison: {len(methods)} methods x "
        f"{len(tasks)} tasks x {len(seeds)} seeds = {total} runs ===\n"
    )

    run_idx = 0
    for task in tasks:
        for method in methods:
            for seed in seeds:
                run_idx += 1
                if already_done(task, method, seed):
                    print(f"[{run_idx}/{total}] {task} {method} seed={seed} ... SKIPPED (exists)")
                    continue
                print(f"[{run_idx}/{total}] {task} {method} seed={seed} ... ", end="", flush=True)
                result = train_loss_baseline(task, method, seed, config)
                all_results.append(result)
                print(
                    f"collapse={result['collapse_rate']:.3f} "
                    f"K_acc={result['k_accuracy']:.3f} "
                    f"({result['train_time_s']:.0f}s)"
                )
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("BASELINE RESULTS (mean +/- std across seeds)")
    print("=" * 80)
    for task in tasks:
        print(f"\n--- {task} ---")
        print(f"{'Method':<25} {'Collapse':>14} {'K Acc':>14}")
        for method in methods:
            runs = [r for r in all_results if r["task"] == task and r["method"] == method]
            if not runs:
                continue
            collapses = [r["collapse_rate"] for r in runs]
            k_accs = [r["k_accuracy"] for r in runs]
            print(
                f"{method:<25} "
                f"{np.mean(collapses):.3f}+/-{np.std(collapses):.3f} "
                f"{np.mean(k_accs):.3f}+/-{np.std(k_accs):.3f}"
            )

    print(f"\nSaved {len(all_results)} results to {out_path}")
    return all_results


if __name__ == "__main__":
    main()
