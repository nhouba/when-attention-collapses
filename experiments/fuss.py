#!/usr/bin/env python3
"""FUSS benchmark: does evidence depletion reduce collapse on real audio? (Sec. 3.2, App. J)

Trains each slot-attention variant on FUSS mixtures (1-4 real sounds summed
additively) to predict per-source spectral parameters, and reports collapse,
cosine overlap, K accuracy and known-K exact-Hungarian parameter MAE;
optionally PIT SI-SDR of a parameter-free attention-to-mask readout
(``--eval-sisdr``). Results go to ``<output-dir>/fuss_results.json`` (resume-safe).

Defaults are the paper configuration (Table 1, Table 13): 4 s crops
(``--signal-length 64000``), 32 tokens, ``d = 128``, batch size 32. Resuming skips (method,
seed) pairs that already have a record without checking their budget (a
warning is printed); ``--no-resume`` starts from an empty results file::

    bash experiments/download_fuss.sh            # -> data/fuss/ssdata
    python experiments/fuss.py --fuss-dir data/fuss/ssdata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from evidence_depletion.data.fuss import FUSSDataset
from evidence_depletion.io import load_results
from evidence_depletion.training import train_fuss

METHODS = ["vanilla", "sequential", "evidence_linear", "evidence_quadratic"]


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(description="FUSS benchmark for evidence depletion")
    parser.add_argument("--fuss-dir", type=str, required=True, help="Path to FUSS ssdata directory")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--signal-length",
        type=int,
        default=64000,
        help="Audio length in samples (64000 = 4 seconds at 16 kHz)",
    )
    parser.add_argument("--max-train", type=int, default=5000, help="Max training samples")
    parser.add_argument("--max-test", type=int, default=1000, help="Max test samples")
    parser.add_argument(
        "--n-tokens", type=int, default=32, help="Number of tokens (chunks) for tokenizer"
    )
    parser.add_argument("--d-model", type=int, default=128, help="Token embedding dimension")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore (and overwrite) an existing results file instead of skipping runs "
        "that already have a record",
    )
    parser.add_argument("--output-dir", type=str, default="results/runs/fuss")
    parser.add_argument(
        "--eval-sisdr",
        action="store_true",
        help="Evaluate PIT SI-SDR via the per-slot spectral mask readout "
        "(default off; when off, si_sdr* result keys are null)",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="rank1",
        choices=["rank1", "time"],
        help="Attention->TF-mask mapping for --eval-sisdr",
    )
    args = parser.parse_args(argv)

    methods = args.methods or METHODS
    seeds = list(range(args.seeds))
    max_K = 4
    n_slots = max_K + 1

    print("Loading FUSS dataset...")
    train_ds = FUSSDataset(
        args.fuss_dir,
        split="train",
        max_K=max_K,
        signal_length=args.signal_length,
        max_samples=args.max_train,
    )
    test_ds = FUSSDataset(
        args.fuss_dir,
        split="validation",
        max_K=max_K,
        signal_length=args.signal_length,
        max_samples=args.max_test,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    sisdr_loader = None
    if args.eval_sisdr:
        sisdr_ds = FUSSDataset(
            args.fuss_dir,
            split="validation",
            max_K=max_K,
            signal_length=args.signal_length,
            max_samples=args.max_test,
            return_sources=True,
        )
        sisdr_loader = DataLoader(
            sisdr_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

    config = {
        "signal_length": args.signal_length,
        "d_model": args.d_model,
        "n_tokens": args.n_tokens,
        "lr": 3e-4,
        "epochs": args.epochs,
        "n_iter": 3,
        "gamma": 3.0,
        "tau": 0.3,
    }

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "fuss_results.json"

    all_results = load_results(out_path, resume=not args.no_resume)

    def already_done(method: str, seed: int) -> bool:
        return any(r["method"] == method and r["seed"] == seed for r in all_results)

    total = len(methods) * len(seeds)
    print(f"\n=== FUSS benchmark: {len(methods)} methods x {len(seeds)} seeds = {total} runs ===\n")

    for seed in seeds:
        for method in methods:
            if already_done(method, seed):
                print(f"  {method} seed={seed} ... SKIPPED (exists)")
                continue
            print(f"  {method} seed={seed} ... ", end="", flush=True)
            result = train_fuss(
                method,
                seed,
                config,
                train_loader,
                test_loader,
                n_slots=n_slots,
                n_params=3,
                sisdr_loader=sisdr_loader,
                mask_mode=args.mask_mode,
            )
            all_results.append(result)
            print(
                f"collapse={result['collapse_rate']:.3f} "
                f"K_acc={result['k_accuracy']:.3f} "
                f"({result['train_time_s']:.0f}s)"
            )
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("FUSS RESULTS (mean +/- std across seeds)")
    print("=" * 60)
    print(f"{'Method':<25} {'Collapse':>14} {'K Acc':>14}")
    for method in methods:
        runs = [r for r in all_results if r["method"] == method]
        if not runs:
            continue
        collapses = [r["collapse_rate"] for r in runs]
        k_accs = [r["k_accuracy"] for r in runs]
        line = (
            f"{method:<25} "
            f"{np.mean(collapses):.3f}+/-{np.std(collapses):.3f} "
            f"{np.mean(k_accs):.3f}+/-{np.std(k_accs):.3f}"
        )
        sisdris = [r["si_sdri"] for r in runs if r.get("si_sdri") is not None]
        if sisdris:
            line += f"  SI-SDRi={np.mean(sisdris):.2f}+/-{np.std(sisdris):.2f} dB"
        print(line)

    print(f"\nSaved {len(all_results)} results to {out_path}")
    return all_results


if __name__ == "__main__":
    main()
