"""Collapse metrics (Definition 1, App. F), parameter recovery and SI-SDR.

Collapse (Definition 1, Eq. 4) is the worst-case similarity between the
attention vectors of two active slots, averaged over multi-source inputs. On
the synthetic and FUSS benchmarks the active set is the first ``K`` slots by
index (ground-truth ``K``); because evidence depletion processes slots in index
order at inference, these are the first-processed slots. Two similarities:

* :func:`peak_overlap_rate` -- ``1[argmax_l alpha_s = argmax_l alpha_s']``; the
  average is the fraction of inputs on which two active slots share a peak
  token. Stored as ``collapse_rate`` in the result JSON files.
* :func:`cosine_overlap` -- ``cos(alpha_s, alpha_s')``, its soft analogue.
  Stored as ``mean_attn_overlap``.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def peak_overlap_rate(attn: torch.Tensor, K_true: torch.Tensor) -> float:
    """Peak overlap (Definition 1): fraction of ``K >= 2`` inputs on which two of
    the first ``min(K, S)`` slots attend most strongly to the same token.

    Args:
        attn: ``[B, S, L]`` attention.
        K_true: ``[B]`` true source counts.

    Returns:
        Rate in ``[0, 1]`` (0 if there is no multi-source input).
    """
    S = attn.shape[1]
    collapsed = 0
    for b in range(attn.shape[0]):
        K = int(K_true[b].item())
        if K <= 1:
            continue
        # With more sources than slots (K > S) only the S slots can be compared.
        n_active = min(K, S)
        peaks = attn[b, :n_active].argmax(dim=-1)
        if len(peaks.unique()) < n_active:
            collapsed += 1

    valid = (K_true > 1).sum().item()
    return collapsed / max(valid, 1)


def cosine_overlap(attn: torch.Tensor, K_true: torch.Tensor) -> float:
    """Cosine overlap (Definition 1, App. F): mean over ``K >= 2`` inputs of the
    maximum off-diagonal cosine similarity between the attention vectors of the
    first ``min(K, S)`` slots.

    Args:
        attn: ``[B, S, L]`` attention.
        K_true: ``[B]`` true source counts.

    Returns:
        Mean overlap in ``[0, 1]`` (0 if there is no multi-source input).
    """
    S = attn.shape[1]
    overlaps = []
    for b in range(attn.shape[0]):
        K = min(int(K_true[b].item()), S)
        if K < 2:
            continue
        a = attn[b, :K]
        a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.mm(a_norm, a_norm.t())
        sim.fill_diagonal_(0.0)
        overlaps.append(sim.max().item())
    return float(np.mean(overlaps)) if overlaps else 0.0


def mean_slot_embedding_cosine(slots: torch.Tensor, K_true: torch.Tensor) -> float:
    """Mean pairwise cosine similarity of the first ``K`` slot *embeddings*.

    Same sample selection as :func:`cosine_overlap`, but averages all
    off-diagonal similarities instead of taking the maximum. Used for the
    per-epoch training-dynamics log (``--track-dynamics``).

    Args:
        slots: ``[B, S, D]`` slot embeddings.
        K_true: ``[B]`` true source counts.
    """
    S = slots.shape[1]
    sims = []
    for b in range(slots.shape[0]):
        K = min(int(K_true[b].item()), S)
        if K < 2:
            continue
        s = slots[b, :K]
        s_norm = s / (s.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.mm(s_norm, s_norm.t())
        off_diag = (sim.sum() - sim.diagonal().sum()) / (K * (K - 1))
        sims.append(off_diag.item())
    return float(np.mean(sims)) if sims else 0.0


def evidence_depletion_split(
    evidence: list[torch.Tensor], exist_logits: torch.Tensor, K_true: torch.Tensor
) -> tuple[list[float], list[float]]:
    """Evidence mass depleted by matched vs. unmatched slots (App. I, over-provisioning).

    A slot is *matched* if it is among the top-``K`` slots by existence logit.
    The mass depleted by slot ``s`` is ``sum_l (e_l before s - e_l after s)``.
    Assumes the eval-mode processing order (slot index order).

    Args:
        evidence: ``S + 1`` tensors ``[B, L]`` (initial state, then after each slot).
        exist_logits: ``[B, S]``.
        K_true: ``[B]``.

    Returns:
        ``(matched_values, unmatched_values)``: per-slot depleted masses.
    """
    ev = torch.stack(evidence, dim=0)
    depleted = (ev[:-1] - ev[1:]).sum(dim=-1)
    S, B = depleted.shape

    matched_vals, unmatched_vals = [], []
    for b in range(B):
        top_k = min(max(int(K_true[b].item()), 0), S)
        matched_idx = set()
        if top_k > 0:
            _, top_indices = exist_logits[b].topk(top_k)
            matched_idx = set(top_indices.tolist())
        for s in range(S):
            if s in matched_idx:
                matched_vals.append(depleted[s, b].item())
            else:
                unmatched_vals.append(depleted[s, b].item())
    return matched_vals, unmatched_vals


def greedy_match(cost: torch.Tensor, n_pairs: int) -> list[tuple[int, int]]:
    """Greedy row-by-row assignment on a ``[n_pred, n_true]`` cost matrix.

    Row ``i`` (in order) takes the cheapest still-unused column.
    """
    matched = []
    used: set[int] = set()
    for i in range(n_pairs):
        row = cost[i].clone()
        for j in used:
            row[j] = float("inf")
        j = row.argmin().item()
        used.add(j)
        matched.append((i, j))
    return matched


def param_recovery_error(
    params_pred: torch.Tensor,
    params_true: torch.Tensor,
    exist_pred: torch.Tensor,
    K_true: torch.Tensor,
    n_params: int,
) -> tuple[torch.Tensor, int]:
    """Per-parameter MAE between the top-``K`` slots (by existence logit) and the
    ``K`` true sources, greedily matched by Euclidean parameter distance.

    Args:
        params_pred: ``[B, S, P]``.
        params_true: ``[B, max_K, P]``.
        exist_pred: ``[B, S]`` existence logits.
        K_true: ``[B]``.
        n_params: ``P``.

    Returns:
        ``(per_param_mae [P], n_matched)``.
    """
    n_slots = params_pred.shape[1]
    errors = []
    for b in range(params_pred.shape[0]):
        K = int(K_true[b].item())
        if K == 0:
            continue
        top_k = min(K, n_slots)
        _, top_indices = exist_pred[b].topk(top_k)
        pred = params_pred[b, top_indices]
        true = params_true[b, :K]
        cost = torch.cdist(pred.unsqueeze(0), true.unsqueeze(0)).squeeze(0)
        for i, j in greedy_match(cost, min(top_k, K)):
            errors.append(torch.abs(pred[i] - true[j]))

    if not errors:
        return torch.zeros(n_params), 0
    stacked = torch.stack(errors)
    return stacked.mean(dim=0), len(stacked)


def param_mae_exact(
    params_pred: torch.Tensor,
    params_true: torch.Tensor,
    exist_logits: torch.Tensor,
    K_vals: torch.Tensor,
    n_params: int,
) -> tuple[torch.Tensor, int]:
    """As :func:`param_recovery_error`, but with exact Hungarian matching
    (``scipy.optimize.linear_sum_assignment``); used for FUSS (App. J).

    Returns:
        ``(per_param_mae [P], n_matched)``.
    """
    n_slots = params_pred.shape[1]
    errors = []
    for b in range(params_pred.shape[0]):
        K = int(K_vals[b].item())
        if K == 0:
            continue
        top_k = min(K, n_slots)
        _, top_idx = exist_logits[b].topk(top_k)
        pred = params_pred[b, top_idx]
        true = params_true[b, :K]
        cost = torch.cdist(pred.unsqueeze(0), true.unsqueeze(0)).squeeze(0)
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        for i, j in zip(row, col):
            errors.append(torch.abs(pred[i] - true[j]))
    if not errors:
        return torch.zeros(n_params), 0
    e = torch.stack(errors)
    return e.mean(dim=0), len(e)


def k_detection_accuracy(
    exist_logits: torch.Tensor, K_true: torch.Tensor, threshold: float = 0.0
) -> float:
    """Fraction of inputs whose predicted count ``#{s : logit_s > threshold}`` equals ``K``."""
    K_pred = (exist_logits > threshold).sum(dim=-1)
    return (K_pred == K_true).float().mean().item()


def source_recovery(
    params_pred: torch.Tensor,
    params_true: torch.Tensor,
    exist_pred: torch.Tensor,
    K_true: torch.Tensor,
    freq_tol: float = 0.5,
    exclude_slot: int | None = None,
) -> tuple[float, float]:
    """Fraction of true sinusoids recovered, and the MAE of the matched pairs.

    Matching as in :func:`param_recovery_error`. A true source counts as
    recovered if its matched slot's frequency (parameter 0) is within
    ``freq_tol``. With ``exclude_slot`` that slot is removed from the candidate
    pool (the "remaining slots" protocol of the forced-error study, App. I).

    Returns:
        ``(recovery_rate, param_mae)``; ``param_mae`` is ``nan`` without matches.
    """
    S = params_pred.shape[1]
    recovered, total = 0, 0
    abs_errors = []
    for b in range(params_pred.shape[0]):
        K = int(K_true[b].item())
        if K == 0:
            continue
        total += K

        pool = [s for s in range(S) if s != exclude_slot]
        top_k = min(K, len(pool))
        top_local = exist_pred[b, pool].topk(top_k).indices.tolist()
        idx = [pool[i] for i in top_local]

        pred = params_pred[b, idx]
        true = params_true[b, :K]
        cost = torch.cdist(pred.unsqueeze(0), true.unsqueeze(0)).squeeze(0)
        for i, j in greedy_match(cost, min(top_k, K)):
            abs_errors.append(torch.abs(pred[i] - true[j]))
            if abs(pred[i, 0].item() - true[j, 0].item()) < freq_tol:
                recovered += 1

    mae = float(torch.stack(abs_errors).mean().item()) if abs_errors else float("nan")
    return recovered / max(total, 1), mae


def si_sdr_db(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB (Le Roux et al. 2019) of 1-D waveforms ``est``, ``ref``."""
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = (est * ref).sum() / (ref * ref).sum().clamp(min=eps)
    target = alpha * ref
    noise = est - target
    ratio = (target * target).sum() / (noise * noise).sum().clamp(min=eps)
    return 10.0 * torch.log10(ratio.clamp(min=eps))


def pit_si_sdr(
    ests: torch.Tensor, refs: torch.Tensor
) -> tuple[float, list[tuple[int, int, float]]]:
    """Permutation-invariant SI-SDR: best one-to-one assignment by brute force.

    Args:
        ests: ``[Ne, N]`` estimates.
        refs: ``[Nr, N]`` references.

    Returns:
        ``(mean_si_sdr, matched)`` with ``matched`` the ``min(Ne, Nr)`` pairs
        ``(ref_idx, est_idx, si_sdr)`` of the best assignment.
    """
    n_est, n_ref = ests.shape[0], refs.shape[0]
    table = torch.full((n_ref, n_est), float("-inf"))
    for i in range(n_ref):
        for j in range(n_est):
            table[i, j] = si_sdr_db(ests[j], refs[i])

    best_mean, best_pairs = float("-inf"), None
    if n_est >= n_ref:
        for perm in permutations(range(n_est), n_ref):
            vals = [table[i, perm[i]] for i in range(n_ref)]
            m = float(sum(vals)) / n_ref
            if m > best_mean:
                best_mean = m
                best_pairs = [(i, perm[i], float(vals[i])) for i in range(n_ref)]
    else:
        for perm in permutations(range(n_ref), n_est):
            vals = [table[perm[j], j] for j in range(n_est)]
            m = float(sum(vals)) / n_est
            if m > best_mean:
                best_mean = m
                best_pairs = [(perm[j], j, float(vals[j])) for j in range(n_est)]
    return best_mean, best_pairs
