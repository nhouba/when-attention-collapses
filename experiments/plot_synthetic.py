#!/usr/bin/env python3
"""Figures and a LaTeX table from ``ablation_results.json`` (needs the ``[plot]`` extra).

Produces in ``--output-dir``:

* ``ablation_collapse.pdf/.png`` -- peak-overlap collapse per method and task;
* ``ablation_params.pdf/.png`` -- mean parameter MAE per method and task;
* ``gamma_sensitivity.pdf/.png`` and ``tau_sensitivity.pdf/.png`` -- collapse of
  ``evidence_quadratic`` along the gamma axis (at tau = 0.3) and the tau axis
  (at gamma = 3) of the sensitivity grid (App. E), if present;
* ``ablation_table.tex`` -- per-task table of collapse, K accuracy and MAE.

Usage::

    python experiments/plot_synthetic.py --results results/runs/synthetic/ablation_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

METHOD_DISPLAY = {
    "vanilla": "Vanilla SA",
    "sequential": "Sequential SA",
    "evidence_binary": "Binary mask",
    "evidence_linear": "Linear $(1{-}\\alpha)$",
    "evidence_quadratic": "Quadratic $(1{-}\\alpha^2)$",
    "evidence_cubic": "Cubic $(1{-}\\alpha^3)$",
}
METHOD_COLORS = {
    "vanilla": "#9CA3AF",
    "sequential": "#6B7280",
    "evidence_binary": "#F59E0B",
    "evidence_linear": "#3B82F6",
    "evidence_quadratic": "#10B981",
    "evidence_cubic": "#8B5CF6",
}
METHOD_ORDER = list(METHOD_DISPLAY)
TASK_DISPLAY = {
    "sinusoids": "A: Sinusoids",
    "gaussian_bumps": "B: Gaussian Bumps",
    "multiscale": "C: Multi-scale",
}
TASK_COLORS = {"sinusoids": "#2575D1", "gaussian_bumps": "#E87A0A", "multiscale": "#10AA6E"}

Records = list[dict[str, Any]]


def _method_runs(results: Records) -> tuple[Records, list[str], list[str]]:
    method_data = [r for r in results if r["sweep"] == "method"]
    tasks = sorted({r["task"] for r in method_data})
    present = {r["method"] for r in method_data}
    return method_data, tasks, [m for m in METHOD_ORDER if m in present]


def _mean_mae(run: dict[str, Any]) -> float | None:
    vals = [v for k, v in run.items() if k.startswith("mae_") and not np.isnan(v)]
    return float(np.mean(vals)) if vals else None


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    out = output_dir / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_bars(results: Records, output_dir: Path, metric: str) -> None:
    """Bar chart per task of ``metric`` (``"collapse"`` or ``"mae"``), mean ± std over seeds."""
    method_data, tasks, methods = _method_runs(results)
    if not tasks:
        print("No method-comparison data, skipping bar charts")
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(methods))
    for ax, task in zip(axes, tasks):
        means, stds = [], []
        for method in methods:
            runs = [r for r in method_data if r["task"] == task and r["method"] == method]
            if metric == "collapse":
                vals = [r["collapse_rate"] for r in runs]
            else:
                vals = [v for v in (_mean_mae(r) for r in runs) if v is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.bar(
            x,
            means,
            yerr=stds,
            width=0.6,
            color=[METHOD_COLORS.get(m, "#999") for m in methods],
            edgecolor="white",
            lw=0.5,
            capsize=3,
            error_kw={"lw": 0.8},
        )
        ax.set_title(TASK_DISPLAY.get(task, task), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_DISPLAY.get(m, m) for m in methods], rotation=35, ha="right", fontsize=7
        )
        ax.grid(alpha=0.15, axis="y")
        if metric == "collapse":
            ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Slot Collapse Rate ↓" if metric == "collapse" else "Parameter MAE ↓")
    title = "Slot Collapse Rate by Method" if metric == "collapse" else "Parameter Recovery Error"
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, "ablation_collapse" if metric == "collapse" else "ablation_params")


def plot_sensitivity(results: Records, output_dir: Path, axis: str) -> None:
    """Collapse vs. ``axis`` (``"gamma"`` at tau = 0.3, or ``"tau"`` at gamma = 3)."""
    fixed_key, fixed_val = ("tau", 0.3) if axis == "gamma" else ("gamma", 3)
    data = [r for r in results if r["sweep"] == "sensitivity" and r[fixed_key] == fixed_val]
    if not data:
        print(f"No sensitivity data for {axis}, skipping")
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for task in sorted({r["task"] for r in data}):
        xs = sorted({r[axis] for r in data if r["task"] == task})
        vals = [
            [r["collapse_rate"] for r in data if r["task"] == task and r[axis] == x] for x in xs
        ]
        ax.errorbar(
            xs,
            [np.mean(v) for v in vals],
            yerr=[np.std(v) for v in vals],
            marker="o" if axis == "gamma" else "s",
            ms=5,
            lw=1.5,
            capsize=3,
            label=TASK_DISPLAY.get(task, task),
            color=TASK_COLORS.get(task, "#999"),
        )
    if axis == "gamma":
        ax.set_xlabel("Evidence bias strength $\\gamma$ ($\\tau = 0.3$)")
    else:
        ax.set_xlabel("Attention temperature $\\tau$ ($\\gamma = 3$)")
        ax.set_xscale("log")
    ax.set_ylabel("Slot Collapse Rate ↓")
    ax.set_title(f"Sensitivity to $\\{axis}$ (quadratic depletion)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.15)
    ax.set_ylim(-0.05, 1.05)
    _save(fig, output_dir, f"{axis}_sensitivity")


def generate_latex_table(results: Records, output_dir: Path) -> None:
    """Per-task LaTeX table; the lowest mean collapse per task is bold."""
    method_data, tasks, methods = _method_runs(results)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation results across three synthetic tasks. Mean $\\pm$ std over seeds. "
        "Collapse rate = fraction of multi-source samples where $\\geq$2 slots attend to the "
        "same source.}",
        "\\label{tab:ablation}",
        "\\small",
    ]
    for task in tasks:
        task_runs = {
            m: [r for r in method_data if r["task"] == task and r["method"] == m] for m in methods
        }
        task_runs = {m: runs for m, runs in task_runs.items() if runs}
        best = min(np.mean([r["collapse_rate"] for r in runs]) for runs in task_runs.values())
        lines += [
            f"\\textbf{{{TASK_DISPLAY.get(task, task)}}}",
            "",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Method & Collapse $\\downarrow$ & $K$ Acc $\\uparrow$ & Param MAE $\\downarrow$ \\\\",
            "\\midrule",
        ]
        for method, runs in task_runs.items():
            collapses = [r["collapse_rate"] for r in runs]
            k_accs = [r["k_accuracy"] for r in runs]
            maes = [v for v in (_mean_mae(r) for r in runs) if v is not None]
            c_str = f"${np.mean(collapses):.2f} \\pm {np.std(collapses):.2f}$"
            if np.mean(collapses) == best:
                c_str = f"\\textbf{{{c_str}}}"
            k_str = f"${np.mean(k_accs):.2f} \\pm {np.std(k_accs):.2f}$"
            m_str = f"${np.mean(maes):.3f} \\pm {np.std(maes):.3f}$" if maes else "---"
            lines.append(f"{METHOD_DISPLAY.get(method, method)} & {c_str} & {k_str} & {m_str} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\vspace{0.5em}", ""]
    lines.append("\\end{table}")

    out = output_dir / "ablation_table.tex"
    out.write_text("\n".join(lines))
    print(f"Saved {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Figures from synthetic ablation results")
    parser.add_argument("--results", type=str, required=True, help="Path to ablation_results.json")
    parser.add_argument(
        "--output-dir", type=str, default="results/figures", help="Output directory for figures"
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results\n")

    plot_bars(results, output_dir, "collapse")
    plot_bars(results, output_dir, "mae")
    plot_sensitivity(results, output_dir, "gamma")
    plot_sensitivity(results, output_dir, "tau")
    generate_latex_table(results, output_dir)
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
