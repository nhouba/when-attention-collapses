"""Properties of the evidence-depletion mechanism (Sec. 2.3, Algorithm 1, Proposition 2)."""

from __future__ import annotations

import pytest
import torch

from evidence_depletion import DEPLETION_FORMS, EvidenceDepletionSlotAttention, deplete
from evidence_depletion.attention import SlotAttentionOutput

B, L, D, S = 4, 12, 32, 5


def _module(form: str = "linear", **kwargs) -> EvidenceDepletionSlotAttention:
    torch.manual_seed(0)
    return EvidenceDepletionSlotAttention(n_slots=S, d_model=D, depletion=form, **kwargs)


def _run(module, tokens, seed: int = 1, **kwargs) -> SlotAttentionOutput:
    torch.manual_seed(seed)
    return module(tokens, **kwargs)


@pytest.mark.parametrize("form", DEPLETION_FORMS)
@pytest.mark.parametrize("training", [True, False])
def test_shapes_and_output_type(form: str, training: bool) -> None:
    sa = _module(form).train(training)
    out = _run(sa, torch.randn(B, L, D))
    assert isinstance(out, SlotAttentionOutput)
    assert out.slots.shape == (B, S, D)
    assert out.attn.shape == (B, S, L)
    assert len(out.evidence) == S + 1
    assert all(e.shape == (B, L) for e in out.evidence)


@pytest.mark.parametrize("form", DEPLETION_FORMS)
def test_attention_is_distribution_over_tokens(form: str) -> None:
    out = _run(_module(form).eval(), torch.randn(B, L, D))
    assert torch.all(out.attn >= 0) and torch.all(out.attn <= 1)
    torch.testing.assert_close(out.attn.sum(-1), torch.ones(B, S))


@pytest.mark.parametrize("form", DEPLETION_FORMS)
@pytest.mark.parametrize("training", [True, False])
def test_evidence_bounded_and_monotone(form: str, training: bool) -> None:
    """Proposition 2 (i): epsilon <= e^(s+1) <= e^(s) <= 1 along the processing order."""
    sa = _module(form).train(training)
    out = _run(sa, 3 * torch.randn(B, L, D))
    ev = torch.stack(out.evidence)
    assert torch.all(ev >= sa.epsilon) and torch.all(ev <= 1.0)
    assert torch.equal(ev[0], torch.ones(B, L))
    assert torch.all(ev[1:] <= ev[:-1])


def test_evidence_strictly_decreases_above_floor() -> None:
    """Proposition 2 (iii) for the linear form: softmax weights are positive."""
    out = _run(_module("linear").eval(), torch.randn(B, L, D))
    ev = torch.stack(out.evidence)
    above_floor = ev[:-1] > 1e-3
    assert torch.all(ev[1:][above_floor] < ev[:-1][above_floor])


def test_depletion_forms_match_formulas() -> None:
    torch.manual_seed(0)
    e = torch.rand(3, 7) * 0.9 + 0.1
    a = torch.rand(3, 7)
    a[0, 0], a[0, 1] = 0.5, 0.4999
    torch.testing.assert_close(deplete(e, a, "linear"), e * (1 - a))
    torch.testing.assert_close(deplete(e, a, "quadratic"), e * (1 - a**2))
    torch.testing.assert_close(deplete(e, a, "cubic"), e * (1 - a**3))
    binary = deplete(e, a, "binary")
    torch.testing.assert_close(binary, torch.where(a < 0.5, e, torch.zeros_like(e)))
    assert binary[0, 0] == 0 and binary[0, 1] == e[0, 1]
    with pytest.raises(ValueError):
        deplete(e, a, "exponential")
    with pytest.raises(ValueError):
        EvidenceDepletionSlotAttention(depletion="exponential")


@pytest.mark.parametrize("form", DEPLETION_FORMS)
def test_evidence_trajectory_follows_eq6(form: str) -> None:
    """Each recorded step equals max(e * (1 - f(alpha_s)), epsilon) for the eval order."""
    sa = _module(form).eval()
    out = _run(sa, torch.randn(B, L, D))
    for s in range(S):
        expected = deplete(out.evidence[s], out.attn[:, s], form).clamp(min=sa.epsilon)
        torch.testing.assert_close(out.evidence[s + 1], expected)


@pytest.mark.parametrize("form", DEPLETION_FORMS)
def test_permutation_equivariance_over_tokens(form: str) -> None:
    sa = _module(form).eval()
    tokens = torch.randn(B, L, D)
    perm = torch.randperm(L)
    out = _run(sa, tokens)
    out_p = _run(sa, tokens[:, perm])
    torch.testing.assert_close(out_p.slots, out.slots, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out_p.attn, out.attn[:, :, perm], rtol=1e-4, atol=1e-5)
    for e_p, e in zip(out_p.evidence, out.evidence):
        torch.testing.assert_close(e_p, e[:, perm], rtol=1e-4, atol=1e-5)


def test_eval_order_is_canonical_and_explicit_order_permutes() -> None:
    sa = _module().eval()
    tokens = torch.randn(B, L, D)
    canonical = _run(sa, tokens)
    explicit = _run(sa, tokens, slot_order=list(range(S)))
    torch.testing.assert_close(canonical.slots, explicit.slots, rtol=0, atol=0)
    reversed_order = _run(sa, tokens, slot_order=list(reversed(range(S))))
    # Slot S-1 is processed first (on full evidence) in both orders.
    last_first = _run(sa, tokens, slot_order=[S - 1, *range(S - 1)])
    torch.testing.assert_close(reversed_order.attn[:, S - 1], last_first.attn[:, S - 1])
    assert not torch.allclose(reversed_order.attn[:, 0], canonical.attn[:, 0])


def test_forced_attention_overrides_first_slot() -> None:
    sa = _module().eval()
    forced = torch.softmax(torch.randn(B, L), dim=-1)
    out = _run(sa, torch.randn(B, L, D), forced_attn_first=forced)
    torch.testing.assert_close(out.attn[:, 0], forced)
    torch.testing.assert_close(out.evidence[1], (1 - forced).clamp(min=sa.epsilon))


def test_evidence_greedy_order_is_a_permutation() -> None:
    sa = _module().eval()
    out = _run(sa, torch.randn(B, L, D), order_policy="evidence_greedy")
    assert out.slots.shape == (B, S, D)
    torch.testing.assert_close(out.attn.sum(-1), torch.ones(B, S))
    ev = torch.stack(out.evidence)
    assert torch.all(ev[1:] <= ev[:-1])
    with pytest.raises(ValueError):
        sa(torch.randn(B, L, D), order_policy="unknown")


def test_gradients_flow_to_all_parameters() -> None:
    sa = _module().train()
    out = _run(sa, torch.randn(B, L, D))
    (out.slots.pow(2).mean() + out.attn.var()).backward()
    for name, p in sa.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_depletion_reduces_attention_overlap_at_init() -> None:
    """At initialization all slots are near-identical; without evidence they
    attend to the same tokens, with evidence later slots are steered away."""
    from evidence_depletion.baselines import SequentialSA
    from evidence_depletion.metrics import cosine_overlap

    torch.manual_seed(0)
    tokens = torch.randn(64, 16, 64)
    K = torch.full((64,), S)
    torch.manual_seed(0)
    seq = SequentialSA(n_slots=S, d_model=64).eval()
    torch.manual_seed(0)
    ev = EvidenceDepletionSlotAttention(n_slots=S, d_model=64, depletion="linear").eval()
    overlap_seq = cosine_overlap(_run(seq, tokens).attn, K)
    overlap_ev = cosine_overlap(_run(ev, tokens).attn, K)
    assert overlap_ev < overlap_seq
