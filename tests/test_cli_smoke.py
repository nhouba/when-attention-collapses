"""One-epoch end-to-end runs of every experiment CLI at tiny sizes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _helpers import REPO_ROOT, run_cli

pytestmark = pytest.mark.smoke

TINY = ["--seeds", "1", "--epochs", "1", "--n-train", "128", "--n-test", "64"]


def _records(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def test_synthetic_method_comparison_and_plot(tmp_path: Path) -> None:
    out = tmp_path / "synthetic"
    args = ["--methods", "vanilla", "evidence_linear", "sa_sh", "--output-dir", str(out)]
    run_cli("synthetic.py", *TINY, *args, cwd=tmp_path)
    records = _records(out / "ablation_results.json")
    assert len(records) == 3 * 3
    expected = {
        "task",
        "method",
        "seed",
        "train_time_s",
        "collapse_rate",
        "mean_attn_overlap",
        "k_accuracy",
        "sweep",
    }
    assert all(expected <= set(r) for r in records)

    # Resume: nothing is re-run.
    proc = run_cli("synthetic.py", *TINY, *args, cwd=tmp_path)
    assert "SKIPPED" in proc.stdout
    assert len(_records(out / "ablation_results.json")) == 9

    pytest.importorskip("matplotlib")
    figs = tmp_path / "figures"
    run_cli(
        "plot_synthetic.py",
        "--results",
        str(out / "ablation_results.json"),
        "--output-dir",
        str(figs),
        cwd=tmp_path,
    )
    assert (figs / "ablation_collapse.pdf").exists()
    assert (figs / "ablation_table.tex").exists()


def test_synthetic_sweeps(tmp_path: Path) -> None:
    out = tmp_path / "sweeps"
    run_cli(
        "synthetic.py",
        *TINY,
        "--tasks",
        "sinusoids",
        "--amp-ratio-sweep",
        "--sweep-methods",
        "evidence_linear",
        "--track-dynamics",
        "--stress-sweep",
        "--output-dir",
        str(out),
        cwd=tmp_path,
    )
    records = _records(out / "ablation_results.json")
    amp = [r for r in records if r["sweep"] == "amp_ratio"]
    stress = [r for r in records if r["sweep"] == "stress"]
    assert len(amp) == 6 and all(len(r["dynamics"]) == 1 for r in amp)
    assert len(stress) == 14
    assert all(
        r["depleted_matched"] is not None for r in stress if r["method"].startswith("evidence")
    )


def test_loss_baselines(tmp_path: Path) -> None:
    out = tmp_path / "loss"
    run_cli(
        "loss_baselines.py",
        *TINY,
        "--tasks",
        "gaussian_bumps",
        "--methods",
        "vanilla_ortho",
        "seq_divloss",
        "--output-dir",
        str(out),
        cwd=tmp_path,
    )
    records = _records(out / "baseline_results.json")
    assert [r["method"] for r in records] == ["vanilla_ortho", "seq_divloss"]
    assert records[0]["w_ortho"] == 3.0


def test_matching_pursuit(tmp_path: Path) -> None:
    out = tmp_path / "mp"
    run_cli(
        "matching_pursuit.py",
        "--seeds",
        "1",
        "--n-test",
        "16",
        "--output-dir",
        str(out),
        cwd=tmp_path,
    )
    (record,) = _records(out / "ablation_results.json")
    assert record["method"] == "matching_pursuit" and record["sweep"] == "method"


def test_order_studies(tmp_path: Path) -> None:
    out = tmp_path / "order"
    args = [
        "--seeds",
        "1",
        "--epochs",
        "1",
        "--n-train",
        "128",
        "--n-test",
        "40",
        "--n-orders",
        "2",
    ]
    run_cli("order_studies.py", *args, "--output-dir", str(out), cwd=tmp_path)
    records = _records(out / "order_studies_results.json")
    studies = {(r["method"], r["study"], r["perturbation"]) for r in records}
    assert ("evidence_linear", "order_sensitivity", None) in studies
    assert ("vanilla", "forced_error", "wrong_peak") in studies
    assert len(list((out / "checkpoints").glob("*.pt"))) == 2
    # Resume from checkpoints / records.
    proc = run_cli("order_studies.py", *args, "--output-dir", str(out), cwd=tmp_path)
    assert "SKIPPED" in proc.stdout and "WARNING" in proc.stdout

    # App. D readout check on the checkpoints just written.
    report = tmp_path / "readout.json"
    proc = run_cli(
        "app_d/readout_check.py",
        "--ckpt-dir",
        str(out / "checkpoints"),
        "--epochs",
        "1",
        "--seeds",
        "1",
        "--n-test",
        "64",
        "--output",
        str(report),
        cwd=tmp_path,
    )
    models = _records(report)["models"]
    assert set(models) == {"vanilla_seed0", "evidence_linear_seed0"}
    assert "mag_dep" in models["evidence_linear_seed0"]["median"]


def test_app_d_geometry_scripts(tmp_path: Path) -> None:
    proc = run_cli("app_d/check_bound.py", "--trials", "300", cwd=tmp_path)
    slack = float(proc.stdout.split("min slack (should be >=0):")[1].split()[0])
    assert slack >= -1e-9
    run_cli("app_d/kappa_mag.py", "--n", "16", cwd=tmp_path)
    proc = run_cli("app_d/measure_superposition.py", "--n", "32", cwd=tmp_path)
    assert "gaussian_bumps" in proc.stdout


def test_no_resume_reruns(tmp_path: Path) -> None:
    out = tmp_path / "loss"
    args = [*TINY, "--tasks", "sinusoids", "--methods", "vanilla_ortho", "--output-dir", str(out)]
    run_cli("loss_baselines.py", *args, cwd=tmp_path)
    proc = run_cli("loss_baselines.py", *args, "--no-resume", cwd=tmp_path)
    assert "SKIPPED" not in proc.stdout
    assert len(_records(out / "baseline_results.json")) == 1


def test_latency(tmp_path: Path) -> None:
    out = tmp_path / "latency"
    run_cli(
        "latency.py",
        "--batch-size",
        "2",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--output-dir",
        str(out),
        cwd=tmp_path,
    )
    records = _records(out / "latency_results.json")
    assert sum(r["record_type"] == "measurement" for r in records) >= 24
    assert any(r["record_type"] == "scaling_fit" for r in records)


def test_fuss_pipeline(tmp_path: Path, fake_fuss: Path) -> None:
    out = tmp_path / "fuss"
    run_cli(
        "fuss.py",
        "--fuss-dir",
        str(fake_fuss),
        "--methods",
        "vanilla",
        "evidence_linear",
        "--seeds",
        "1",
        "--epochs",
        "1",
        "--signal-length",
        "2400",
        "--n-tokens",
        "16",
        "--d-model",
        "32",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--eval-sisdr",
        "--output-dir",
        str(out),
        cwd=tmp_path,
    )
    records = _records(out / "fuss_results.json")
    assert [r["method"] for r in records] == ["vanilla", "evidence_linear"]
    assert all("param_mae_mean" in r and "si_sdri" in r for r in records)

    proc = run_cli(
        "fuss_param_compare.py",
        "--results-glob",
        str(out / "fuss_results.json"),
        cwd=tmp_path,
    )
    assert "evidence_linear" in proc.stdout

    fuss_args = ["--fuss-dir", str(fake_fuss), "--signal-length", "2400"]
    run_cli("fuss_param_baseline.py", *fuss_args, cwd=tmp_path)
    proc = run_cli(
        "fuss_sisdr_oracle.py", *fuss_args, "--n-mixtures", "3", "--n-chunks", "4", cwd=tmp_path
    )
    assert "rank1_model_faithful" in proc.stdout


def test_quickstart_example(tmp_path: Path) -> None:
    proc = run_cli(str(REPO_ROOT / "examples" / "quickstart.py"), cwd=tmp_path)
    assert "evidence depletion" in proc.stdout.lower()
