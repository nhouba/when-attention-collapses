"""Numerical equivalence with the original research code.

Set ``ED_REFERENCE_CODE`` to a directory holding the original scripts
(``models.py``, ``benchmarks.py``, ``ablation.py``, ``baseline_comparison.py``,
``classical_baselines.py``, ``fuss_benchmark.py``, ``order_studies.py``,
``audit_sisdr_oracle.py``, ``fuss_param_baseline.py``); without it the module is
skipped. Everything runs on CPU with fixed seeds. Tensors are compared
bit-for-bit (``torch.equal``) and result records with ``allclose(1e-6)``
(``train_time_s`` excluded).
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from _helpers import load_experiment
from torch.utils.data import DataLoader

from evidence_depletion import metrics as M
from evidence_depletion.data import fuss as fuss_data
from evidence_depletion.data import synthetic
from evidence_depletion.models import METHODS, build_model
from evidence_depletion.separation import evaluate_sisdr, slot_separation_readout
from evidence_depletion.training import (
    coverage_penalty,
    matched_param_loss,
    train_fuss,
    train_loss_baseline,
    train_synthetic,
)

REFERENCE_DIR = os.environ.get("ED_REFERENCE_CODE")

pytestmark = [
    pytest.mark.equivalence,
    pytest.mark.skipif(
        not REFERENCE_DIR or not Path(REFERENCE_DIR).is_dir(),
        reason="set ED_REFERENCE_CODE to the original research code to run equivalence tests",
    ),
]

# Methods that exist in the original code (SA-SH is reconstructed from the paper).
REFERENCE_METHODS = [m for m in METHODS if m != "sa_sh"]


@pytest.fixture(scope="module")
def ref() -> SimpleNamespace:
    sys.path.insert(0, str(Path(REFERENCE_DIR).resolve()))
    try:
        names = [
            "benchmarks",
            "models",
            "ablation",
            "baseline_comparison",
            "classical_baselines",
            "fuss_benchmark",
            "order_studies",
            "fuss_param_baseline",
            "audit_sisdr_oracle",
        ]
        return SimpleNamespace(**{n: importlib.import_module(n) for n in names})
    finally:
        sys.path.pop(0)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def assert_records_equal(a, b, path: str = "") -> None:
    """Recursive comparison: exact for non-floats, allclose(1e-6) for floats."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{path}: keys {list(a)} != {list(b)}"
        for k in a:
            if k != "train_time_s":
                assert_records_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_records_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, (float, np.floating)) and not isinstance(a, bool):
        assert isinstance(b, (float, np.floating, int)), f"{path}: {type(b)}"
        if math.isnan(a):
            assert math.isnan(b), f"{path}: {a} != {b}"
        else:
            assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def assert_tensors_equal(a, b, path: str = "") -> None:
    if a is None:
        assert b is None, path
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_tensors_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{path}: max |diff| = {(a - b).abs().max().item()}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


# ============================================================
# (a) synthetic generators
# ============================================================

GENERATOR_CASES = [
    {},
    {"K_range": (2, 6), "N": 128, "noise_std": 0.3},
    {"dominant": False, "seed": 7},
    {"seed": 123, "K_range": (6, 6)},
    {"seed": None},
]


@pytest.mark.parametrize("task", list(synthetic.TASKS))
@pytest.mark.parametrize("case", GENERATOR_CASES)
def test_generators_identical(ref, task: str, case: dict) -> None:
    case = {"seed": 3, **case}
    torch.manual_seed(99)
    out_ref = ref.benchmarks.TASKS[task]["generate"](17, **case)
    next_ref = torch.rand(3)
    torch.manual_seed(99)
    out_new = synthetic.TASKS[task]["generate"](17, **case)
    next_new = torch.rand(3)
    assert_tensors_equal(out_ref, out_new, task)
    assert torch.equal(next_ref, next_new), "RNG stream position differs after generation"
    assert ref.benchmarks.TASKS[task]["n_params"] == synthetic.TASKS[task]["n_params"]
    assert ref.benchmarks.TASKS[task]["param_names"] == synthetic.TASKS[task]["param_names"]


@pytest.mark.parametrize("ratio", [1.0, 1.5, 10.0])
def test_amp_ratio_generator_identical(ref, ratio: float) -> None:
    a = ref.benchmarks.generate_sinusoids(33, seed=5, amp_ratio=ratio)
    b = synthetic.generate_sinusoids(33, seed=5, amp_ratio=ratio)
    assert_tensors_equal(a, b)


# ============================================================
# (b) build_model: parameters, forward and gradients
# ============================================================

MODEL_CASES = [
    {},
    {"n_slots": 7, "d_model": 32, "n_tokens": 8, "n_iter": 2, "gamma": 10.0, "tau": 1.0},
    {"tokenizer_type": "direct", "n_params": 5},
]


def _forward_pair(ref_model, new_model, x, seed, **kwargs):
    torch.manual_seed(seed)
    out_ref = ref_model(x, **kwargs)
    torch.manual_seed(seed)
    out_new = new_model(x, **kwargs)
    return out_ref, out_new


@pytest.mark.parametrize("method", REFERENCE_METHODS)
@pytest.mark.parametrize("case", MODEL_CASES)
def test_build_model_identical(ref, method: str, case: dict) -> None:
    torch.manual_seed(0)
    ref_model = ref.models.build_model(method, **case)
    torch.manual_seed(0)
    new_model = build_model(method, **case)

    sd_ref, sd_new = ref_model.state_dict(), new_model.state_dict()
    assert list(sd_ref) == list(sd_new)
    for k in sd_ref:
        assert torch.equal(sd_ref[k], sd_new[k]), k

    x = torch.randn(6, 256)
    for training in (True, False):
        ref_model.train(training)
        new_model.train(training)
        out_ref, out_new = _forward_pair(ref_model, new_model, x, seed=11)
        assert_tensors_equal(out_ref, out_new, f"{method} train={training}")

    # Gradients of a loss touching every output.
    ref_model.train()
    new_model.train()
    for model in (ref_model, new_model):
        model.zero_grad()
    out_ref, out_new = _forward_pair(ref_model, new_model, x, seed=12)
    for out in (out_ref, out_new):
        slots, attn, exist, params, _ = out
        (slots.square().mean() + attn.square().mean() + exist.mean() + params.mean()).backward()
    for (n_ref, p_ref), (_, p_new) in zip(
        ref_model.named_parameters(), new_model.named_parameters()
    ):
        if p_ref.grad is None:
            assert p_new.grad is None, n_ref
        else:
            assert torch.equal(p_ref.grad, p_new.grad), n_ref


@pytest.mark.parametrize("method", ["evidence_linear", "evidence_quadratic"])
def test_inference_controls_identical(ref, method: str) -> None:
    torch.manual_seed(0)
    ref_model = ref.models.build_model(method).eval()
    torch.manual_seed(0)
    new_model = build_model(method).eval()
    x = torch.randn(9, 256)
    forced = torch.softmax(torch.randn(9, 16), dim=-1)
    for kwargs in (
        {"slot_order": [3, 1, 4, 0, 2]},
        {"forced_attn_first": forced},
        {"order_policy": "evidence_greedy"},
    ):
        out_ref, out_new = _forward_pair(ref_model, new_model, x, seed=5, **kwargs)
        assert_tensors_equal(out_ref, out_new, str(list(kwargs)))


def test_vanilla_forced_attention_identical(ref) -> None:
    torch.manual_seed(0)
    ref_model = ref.models.build_model("vanilla").eval()
    torch.manual_seed(0)
    new_model = build_model("vanilla").eval()
    forced = torch.softmax(torch.randn(4, 16), dim=-1)
    out_ref, out_new = _forward_pair(
        ref_model, new_model, torch.randn(4, 256), seed=2, forced_attn_first=forced
    )
    assert_tensors_equal(out_ref, out_new)


def test_mesh_entropy_trace_identical(ref) -> None:
    from evidence_depletion.baselines import mesh_optimize_cost

    cost = torch.randn(3, 5, 16)
    torch.manual_seed(4)
    a_ref, t_ref = ref.models.mesh_optimize_cost(cost, return_entropy_trace=True)
    torch.manual_seed(4)
    a_new, t_new = mesh_optimize_cost(cost, return_entropy_trace=True)
    assert torch.equal(a_ref, a_new) and t_ref == t_new


# ============================================================
# Metrics and losses
# ============================================================


def test_metrics_identical(ref) -> None:
    torch.manual_seed(0)
    B, S, L, P = 40, 6, 16, 3
    attn = torch.softmax(torch.randn(B, S, L) * 3, dim=-1)
    attn[::3, 1] = attn[::3, 0]
    slots = torch.randn(B, S, 8)
    K = torch.randint(0, 6, (B,))
    exist = torch.randn(B, S)
    pred, true = torch.randn(B, S, P), torch.randn(B, 5, P)
    evidence = [torch.rand(B, L) for _ in range(S + 1)]
    a = ref.ablation
    assert a.slot_collapse_rate(attn, K) == M.peak_overlap_rate(attn, K)
    assert a.mean_attn_overlap(attn, K) == M.cosine_overlap(attn, K)
    assert a.mean_slot_emb_cos(slots, K) == M.mean_slot_embedding_cosine(slots, K)
    assert a.evidence_depletion_split(evidence, exist, K) == M.evidence_depletion_split(
        evidence, exist, K
    )
    assert a.k_detection_accuracy(exist, K) == M.k_detection_accuracy(exist, K)
    assert_tensors_equal(
        a.param_recovery_error(pred, true, exist, K, P),
        M.param_recovery_error(pred, true, exist, K, P),
    )
    assert torch.equal(
        a._compute_param_loss(pred, true, exist, K, P, "cpu"),
        matched_param_loss(pred, true, exist, K, "cpu"),
    )
    assert torch.equal(a.coverage_penalty(attn), coverage_penalty(attn))
    assert_tensors_equal(
        ref.fuss_benchmark.param_mae_exact(pred, true, exist, K, P),
        M.param_mae_exact(pred, true, exist, K, P),
    )
    for exclude in (None, 0, 2):
        assert ref.order_studies.source_recovery(
            pred, true, exist, K, freq_tol=0.7, exclude_slot=exclude
        ) == M.source_recovery(pred, true, exist, K, freq_tol=0.7, exclude_slot=exclude)


# ============================================================
# (c) training runs
# ============================================================

TINY = {"n_train": 256, "n_test": 160, "epochs": 2}

TRAIN_CASES = [
    *[("sinusoids", m, {}) for m in REFERENCE_METHODS],
    ("gaussian_bumps", "vanilla", {}),
    ("gaussian_bumps", "evidence_linear", {}),
    ("multiscale", "evidence_quadratic", {}),
    ("multiscale", "sequential", {}),
    ("sinusoids", "evidence_linear", {"track_dynamics": True, "amp_ratio": 2.0}),
    ("sinusoids", "evidence_linear", {"K_range": (1, 8), "n_slots": 9, "log_depletion": True}),
    (
        "sinusoids",
        "vanilla",
        {"K_range": (6, 6), "n_slots": 7, "amp_ratio": 1.0, "log_depletion": True},
    ),
    ("sinusoids", "coverage_loss", {"w_coverage": 0.3}),
    ("sinusoids", "evidence_quadratic", {"gamma": 10, "tau": 0.1, "n_iter": 1, "n_tokens": 32}),
]


@pytest.mark.parametrize(("task", "method", "extra"), TRAIN_CASES)
def test_train_synthetic_identical(ref, task: str, method: str, extra: dict) -> None:
    config = {**TINY, **extra}
    res_ref = ref.ablation.train_one(task, method, 1, dict(config))
    res_new = train_synthetic(task, method, 1, dict(config))
    assert_records_equal(res_ref, res_new)


def test_train_synthetic_return_model_identical(ref) -> None:
    config = {**TINY, "return_model": True}
    res_ref, model_ref = ref.ablation.train_one("sinusoids", "evidence_linear", 0, dict(config))
    res_new, model_new = train_synthetic("sinusoids", "evidence_linear", 0, dict(config))
    assert_records_equal(res_ref, res_new)
    for (k, a), b in zip(model_ref.state_dict().items(), model_new.state_dict().values()):
        assert torch.equal(a, b), k


@pytest.mark.parametrize("method", ["vanilla_divloss", "vanilla_ortho", "seq_divloss_ortho"])
@pytest.mark.parametrize("task", ["sinusoids", "gaussian_bumps"])
def test_train_loss_baseline_identical(ref, task: str, method: str) -> None:
    res_ref = ref.baseline_comparison.train_one(task, method, 2, dict(TINY))
    res_new = train_loss_baseline(task, method, 2, dict(TINY))
    assert_records_equal(res_ref, res_new)


def test_matching_pursuit_identical(ref) -> None:
    mp = load_experiment("matching_pursuit")
    config = {"n_test": 60, "N": 256, "K_range": (1, 4), "amp_threshold": 0.15}
    assert_records_equal(
        ref.classical_baselines.evaluate_matching_pursuit(3, dict(config)),
        mp.evaluate_matching_pursuit(3, dict(config)),
    )


def test_order_studies_identical(ref) -> None:
    exp = load_experiment("order_studies")
    config = {**ref.order_studies.make_config(SimpleNamespace(n_train=256, n_test=150, epochs=2))}
    assert config == exp.make_config(SimpleNamespace(n_train=256, n_test=150, epochs=2))
    device = torch.device("cpu")
    for method in exp.METHODS:
        _, model_ref = ref.ablation.train_one(
            "sinusoids", method, 0, {**config, "return_model": True}
        )
        new_model = build_model(method, gamma=3.0, tau=0.3, n_iter=3)
        new_model.load_state_dict(model_ref.state_dict())
        data = synthetic.generate_sinusoids(150, seed=10000)
        assert_records_equal(
            ref.order_studies.study_forced_error(model_ref, method, 0, data, device, 5000, 0.5, 3),
            exp.study_forced_error(new_model, method, 0, data, device, 5000, 0.5, 3),
        )
        if method.startswith("evidence"):
            assert_records_equal(
                ref.order_studies.study_order_sensitivity(
                    model_ref, method, 0, data, device, 5000, 3, 3
                ),
                exp.study_order_sensitivity(new_model, method, 0, data, device, 5000, 3, 3),
            )
            assert_records_equal(
                ref.order_studies.study_evidence_ordered(
                    model_ref, method, 0, data, device, 5000, 3
                ),
                exp.study_evidence_ordered(new_model, method, 0, data, device, 5000, 3),
            )


# ============================================================
# FUSS: data, readout, training
# ============================================================


def test_fuss_dataset_identical(ref, fake_fuss: Path) -> None:
    for split in ("train", "validation", "eval"):
        for kwargs in ({}, {"return_sources": True, "max_samples": 4}, {"signal_length": 1000}):
            ds_ref = ref.fuss_benchmark.FUSSDataset(fake_fuss, split=split, **kwargs)
            ds_new = fuss_data.FUSSDataset(fake_fuss, split=split, **kwargs)
            assert ds_ref.items == ds_new.items
            for i in range(len(ds_ref)):
                assert_tensors_equal(ds_ref[i], ds_new[i], f"{split}[{i}]")
    wav = sorted((fake_fuss / "train").glob("example00001_sources/*.wav"))
    for path in wav:
        a, b = ref.fuss_benchmark.load_wav(path), fuss_data.load_wav(path)
        assert np.array_equal(a, b)
        assert np.array_equal(ref.fuss_benchmark.extract_params(a), fuss_data.extract_params(b))


def test_fuss_readout_identical(ref) -> None:
    torch.manual_seed(0)
    mix = torch.randn(3, 4000)
    attn = torch.softmax(torch.randn(3, 5, 16), dim=-1)
    exist = torch.randn(3, 5)
    exist[0] = -5.0
    for mode in ("rank1", "time"):
        assert_tensors_equal(
            ref.fuss_benchmark.slot_separation_readout(mix, attn, exist, mask_mode=mode),
            slot_separation_readout(mix, attn, exist, mask_mode=mode),
        )
    refs, ests = torch.randn(3, 4000), torch.randn(4, 4000)
    assert ref.fuss_benchmark.pit_si_sdr(ests, refs) == M.pit_si_sdr(ests, refs)
    assert ref.fuss_benchmark.pit_si_sdr(refs, ests) == M.pit_si_sdr(refs, ests)


def test_fuss_oracle_masks_identical(ref) -> None:
    oracle = load_experiment("fuss_sisdr_oracle")
    audit = ref.audit_sisdr_oracle
    torch.manual_seed(0)
    refs = torch.randn(3, 6000)
    X = audit.stft(refs.sum(0), "cpu")
    assert torch.equal(audit.irm(refs, X, "cpu"), oracle.irm(refs, X, "cpu"))
    for n_chunks in (None, 8):
        assert torch.equal(
            audit.rank1_masks(refs, X, "cpu", n_chunks),
            oracle.rank1_masks(refs, X, "cpu", n_chunks),
        )
        assert torch.equal(
            audit.rank1_model_faithful(refs, X, "cpu", n_chunks),
            oracle.rank1_model_faithful(refs, X, "cpu", n_chunks),
        )
    masks = oracle.rank1_masks(refs, X, "cpu")
    assert torch.equal(
        audit.apply_masks_sisdr(masks, X, refs.sum(0), refs, 6000, "cpu"),
        oracle.apply_masks_sisdr(masks, X, refs, 6000, "cpu"),
    )


def test_fuss_param_baseline_identical(ref, fake_fuss: Path) -> None:
    exp = load_experiment("fuss_param_baseline")
    ds = fuss_data.FUSSDataset(fake_fuss, split="train", signal_length=2000)
    assert np.array_equal(ref.fuss_param_baseline.collect_params(ds, 4), exp.collect_params(ds, 4))


@pytest.mark.parametrize("method", ["vanilla", "evidence_linear", "coverage_loss", "mesh"])
def test_train_fuss_identical(ref, fake_fuss: Path, method: str) -> None:
    kwargs = {"max_K": 4, "signal_length": 2400}
    train_ds = fuss_data.FUSSDataset(fake_fuss, split="train", **kwargs)
    test_ds = fuss_data.FUSSDataset(fake_fuss, split="validation", **kwargs)
    sisdr_ds = fuss_data.FUSSDataset(fake_fuss, split="validation", return_sources=True, **kwargs)
    config = {
        "signal_length": 2400,
        "d_model": 32,
        "n_tokens": 16,
        "lr": 3e-4,
        "epochs": 2,
        "n_iter": 3,
        "gamma": 3.0,
        "tau": 0.3,
    }

    def loaders():
        return (
            DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True),
            DataLoader(test_ds, batch_size=4, shuffle=False),
            DataLoader(sisdr_ds, batch_size=4, shuffle=False),
        )

    tr, te, si = loaders()
    res_ref = ref.fuss_benchmark.train_one_fuss(
        method, 4, dict(config), tr, te, n_slots=5, sisdr_loader=si
    )
    tr, te, si = loaders()
    res_new = train_fuss(method, 4, dict(config), tr, te, n_slots=5, sisdr_loader=si)
    assert_records_equal(res_ref, res_new)

    torch.manual_seed(0)
    model = build_model("evidence_linear", N=2400, d_model=32)
    _, _, si = loaders()
    torch.manual_seed(1)
    s_ref = ref.fuss_benchmark.evaluate_sisdr(model, si, torch.device("cpu"), mask_mode="time")
    torch.manual_seed(1)
    s_new = evaluate_sisdr(model, si, torch.device("cpu"), mask_mode="time")
    assert_records_equal(s_ref, s_new)
