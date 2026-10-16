"""Collapse metrics (Definition 1) and matching metrics on hand-built inputs."""

from __future__ import annotations

import math

import torch

from evidence_depletion.metrics import (
    cosine_overlap,
    k_detection_accuracy,
    param_mae_exact,
    param_recovery_error,
    peak_overlap_rate,
    pit_si_sdr,
    si_sdr_db,
)


def _one_hot(peaks: list[list[int]], L: int = 4) -> torch.Tensor:
    attn = torch.full((len(peaks), len(peaks[0]), L), 0.01)
    for b, row in enumerate(peaks):
        for s, p in enumerate(row):
            attn[b, s, p] = 1.0
    return attn


def test_peak_overlap_counts_shared_peaks_among_first_k_slots() -> None:
    attn = _one_hot([[0, 0, 1], [0, 1, 2], [3, 3, 3], [2, 2, 2]])
    K = torch.tensor([2, 3, 1, 3])
    # sample 0 collapses, 1 does not, 2 has K=1 (ignored), 3 collapses
    assert peak_overlap_rate(attn, K) == 2 / 3
    # slot 2 is outside the active set of sample 0 (K=2)
    assert peak_overlap_rate(_one_hot([[0, 1, 0]]), torch.tensor([2])) == 0.0
    assert peak_overlap_rate(attn, torch.ones(4, dtype=torch.long)) == 0.0


def test_cosine_overlap_is_max_offdiagonal_similarity() -> None:
    attn = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    assert cosine_overlap(attn, torch.tensor([2])) < 1e-6
    assert math.isclose(cosine_overlap(attn, torch.tensor([3])), 1 / math.sqrt(2), rel_tol=1e-5)
    assert cosine_overlap(attn, torch.tensor([1])) == 0.0


def test_param_matching_and_k_accuracy() -> None:
    pred = torch.tensor([[[1.0, 0.0], [5.0, 5.0], [9.0, 9.0]]])
    true = torch.tensor([[[5.0, 5.0], [1.0, 0.5]]])
    exist = torch.tensor([[3.0, 2.0, -1.0]])
    K = torch.tensor([2])
    mae, n = param_recovery_error(pred, true, exist, K, 2)
    assert n == 2
    torch.testing.assert_close(mae, torch.tensor([0.0, 0.25]))
    mae_exact, n_exact = param_mae_exact(pred, true, exist, K, 2)
    torch.testing.assert_close(mae_exact, mae)
    assert n_exact == 2
    assert k_detection_accuracy(exist, K) == 1.0
    assert k_detection_accuracy(exist, torch.tensor([3])) == 0.0


def test_si_sdr_and_pit() -> None:
    torch.manual_seed(0)
    refs = torch.randn(2, 1000)
    assert si_sdr_db(3 * refs[0], refs[0]) > 60
    ests = torch.stack([refs[1] + 0.1 * torch.randn(1000), refs[0]])
    _, matched = pit_si_sdr(ests, refs)
    assert sorted((r, e) for r, e, _ in matched) == [(0, 1), (1, 0)]
