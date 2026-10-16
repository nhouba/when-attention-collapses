"""Shapes and basic invariants of every method returned by ``build_model``."""

from __future__ import annotations

import pytest
import torch

from evidence_depletion import METHODS, build_model
from evidence_depletion.baselines import entmax15, log_sinkhorn, mesh_optimize_cost, sparsemax

B, N, L, S = 3, 256, 16, 5


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("training", [True, False])
def test_forward_shapes(method: str, training: bool) -> None:
    torch.manual_seed(0)
    model = build_model(method, n_params=5, N=N, d_model=32, n_tokens=L, n_slots=S).train(training)
    slots, attn, exist, params, evidence = model(torch.randn(B, N))
    assert slots.shape == (B, S, 32)
    assert attn.shape == (B, S, L)
    assert exist.shape == (B, S)
    assert params.shape == (B, S, 5)
    assert torch.isfinite(attn).all() and torch.all(attn >= 0)
    if method.startswith("evidence_"):
        assert len(evidence) == S + 1
    else:
        assert evidence is None


@pytest.mark.parametrize("method", ["vanilla", "evidence_linear", "dataspace_quadratic_bias"])
def test_direct_tokenizer(method: str) -> None:
    model = build_model(method, N=N, n_tokens=L, tokenizer_type="direct")
    assert model(torch.randn(B, N))[1].shape == (B, S, L)


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        build_model("attention_is_all_you_need")


def test_hyperparameters_are_forwarded() -> None:
    model = build_model("evidence_cubic", gamma=10.0, tau=1.0, n_iter=5)
    sa = model.slot_attn
    assert (sa.gamma, sa.tau, sa.n_iter, sa.depletion) == (10.0, 1.0, 5, "cubic")
    ds = build_model("dataspace_linear_bias").slot_attn
    assert ds.depletion == "linear" and ds.use_bias


def test_sparse_normalizers_are_distributions() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 6, 9)
    for fn in (sparsemax, entmax15):
        p = fn(x, dim=1)
        torch.testing.assert_close(p.sum(1), torch.ones(4, 9))
        assert torch.all(p >= 0) and (p == 0).any()


def test_sinkhorn_columns_sum_to_one_and_mesh_lowers_entropy() -> None:
    torch.manual_seed(0)
    cost = torch.randn(2, 5, 16)
    plan = log_sinkhorn(cost)
    torch.testing.assert_close(plan.sum(1), torch.ones(2, 16))
    _, trace = mesh_optimize_cost(cost, return_entropy_trace=True)
    assert all(b <= a + 1e-6 for a, b in zip(trace, trace[1:]))


def test_sa_sh_is_noisy_sinkhorn_vanilla() -> None:
    from evidence_depletion.baselines import SASHSA, VanillaSA

    torch.manual_seed(0)
    sash = SASHSA(n_slots=S, d_model=32).eval()
    torch.manual_seed(0)
    vanilla = VanillaSA(n_slots=S, d_model=32).eval()
    # Same parameters as vanilla SA under the same seed.
    for a, b in zip(sash.state_dict().values(), vanilla.state_dict().values()):
        assert torch.equal(a, b)
    tokens = torch.randn(B, L, 32)
    torch.manual_seed(1)
    out1 = sash(tokens)
    torch.manual_seed(2)
    out2 = sash(tokens)
    # Noise is applied at inference too; columns are normalized over slots.
    assert not torch.allclose(out1.attn, out2.attn)
    torch.testing.assert_close(out1.attn.sum(1), torch.ones(B, L))

    # Without noise, one iteration equals log_sinkhorn of the vanilla logits.
    one = SASHSA(n_slots=S, d_model=32, n_iter=1, noise_std=0.0).eval()
    torch.manual_seed(3)
    slots0 = one.sample_slots(B, tokens.device)
    inp = one.norm_in(tokens)
    dots = torch.einsum("bsd,bld->bsl", one.proj_q(one.norm_s(slots0)), one.proj_k(inp)) / 32**0.5
    torch.manual_seed(3)
    torch.testing.assert_close(one(tokens).attn, log_sinkhorn(dots, n_iter=5))
    assert build_model("sa_sh", noise_std=0.1, sinkhorn_iters=7).slot_attn.noise_std == 0.1
