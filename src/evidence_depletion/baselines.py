"""Slot-attention baselines and alternative anti-collapse mechanisms.

All modules share the token -> slot interface of
:class:`~evidence_depletion.attention.EvidenceDepletionSlotAttention`: they map
tokens ``[B, L, D]`` to a :class:`~evidence_depletion.attention.SlotAttentionOutput`
and, apart from :class:`DSPNSetPredictor`, reuse the parameters of
:class:`~evidence_depletion.attention.SlotAttentionBase`.

Core comparison (Sec. 3.1, Table 1):

* :class:`VanillaSA` -- parallel slot attention, softmax over slots (Eq. 3).
* :class:`SequentialSA` -- sequential processing without evidence.
* :class:`DataSpaceDepletionSA` -- depletes the tokens themselves instead of a
  separate evidence variable ("Evidence space vs. data space").

Alternative mechanisms (Table 2, App. H):

* :class:`SparseCompetitionSA` -- sparsemax / entmax-1.5 competition over slots.
* :class:`SASHSA` -- Sinkhorn slot attention with symmetry-breaking noise (SA-SH).
* :class:`MESHSA` -- MESH: entropy-minimizing optimal-transport cost.
* :class:`CoverageSA` -- additive coverage penalty on the logits.
* :class:`StickBreakingSA` -- Genesis-v2-style stick-breaking scopes.
* :class:`DSPNSetPredictor` -- learned queries with a single cross-attention pass.

The coverage *loss* baseline uses :class:`VanillaSA` and adds
:func:`evidence_depletion.training.coverage_penalty` to the training loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from evidence_depletion.attention import DEPLETION_FORMS, SlotAttentionBase, SlotAttentionOutput

# ============================================================
# Core comparison
# ============================================================


class VanillaSA(SlotAttentionBase):
    """Standard slot attention (Locatello et al. 2020).

    All slots are updated in parallel: softmax over *slots* for each token
    (Eq. 3), then a weighted mean over tokens per slot.
    """

    def normalize(self, dots: torch.Tensor) -> torch.Tensor:
        """Slot-competition normalizer of the ``[B, S, L]`` logits (softmax over slots)."""
        return F.softmax(dots, dim=1)

    def forward(
        self, tokens: torch.Tensor, forced_attn_first: torch.Tensor | None = None
    ) -> SlotAttentionOutput:
        """Args:
        tokens: ``[B, L, D]``.
        forced_attn_first: optional ``[B, L]`` distribution overriding slot 0's
            attention in every iteration. Vanilla SA has no processing order,
            so by convention "first slot" is slot index 0 (forced-error study,
            App. I).
        """
        B, L, D = tokens.shape
        slots = self.sample_slots(B, tokens.device)
        inp = self.norm_in(tokens)
        k = self.proj_k(inp)
        v = self.proj_v(inp)

        for _ in range(self.n_iter):
            q = self.proj_q(self.norm_s(slots))
            dots = torch.einsum("bsd,bld->bsl", q, k) / (D**0.5)
            attn = self.normalize(dots)
            if forced_attn_first is not None:
                attn = attn.clone()
                attn[:, 0, :] = forced_attn_first
            # Weighted mean: renormalize over tokens per slot.
            attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            updates = torch.einsum("bsl,bld->bsd", attn / attn_sum, v)
            slots = self.update_slot(slots, updates)

        return SlotAttentionOutput(slots, attn, None)


class SequentialSA(SlotAttentionBase):
    """Sequential slot attention *without* evidence depletion.

    Slots are processed one at a time with a softmax over tokens, but claimed
    tokens are never discounted; isolates the effect of sequential processing.

    Args:
        tau: attention temperature (as in Eq. 5, without the evidence bias).
    """

    def __init__(
        self, n_slots: int = 5, d_model: int = 64, n_iter: int = 3, tau: float = 0.3
    ) -> None:
        super().__init__(n_slots=n_slots, d_model=d_model, n_iter=n_iter)
        self.tau = tau

    def logits(self, slot: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled logits ``q . k / (tau sqrt(d))``: ``[B, 1, L]``."""
        D = k.shape[-1]
        q = self.proj_q(self.norm_s(slot))
        return torch.einsum("bsd,bld->bsl", q, k) / (self.tau * D**0.5)

    def forward(self, tokens: torch.Tensor) -> SlotAttentionOutput:
        B, L, D = tokens.shape
        slots_init = self.sample_slots(B, tokens.device)
        inp = self.norm_in(tokens)

        all_slots: list[torch.Tensor | None] = [None] * self.n_slots
        all_attns: list[torch.Tensor | None] = [None] * self.n_slots
        for s_idx in self.slot_order():
            s = int(s_idx.item())
            slot = slots_init[:, s : s + 1, :]
            k = self.proj_k(inp)
            v = self.proj_v(inp)
            for _ in range(self.n_iter):
                attn = F.softmax(self.logits(slot, k), dim=-1)
                updates = torch.einsum("bsl,bld->bsd", attn, v)
                slot = self.update_slot(slot, updates)
            all_slots[s] = slot
            all_attns[s] = attn.squeeze(1)

        return SlotAttentionOutput(torch.cat(all_slots, dim=1), torch.stack(all_attns, dim=1), None)


class DataSpaceDepletionSA(SequentialSA):
    """Sequential slot attention with *data-space* depletion (architecture-matched control).

    Identical to evidence depletion except that the tokens are depleted
    directly, ``h_l <- h_l * max(1 - f(alpha_{s,l}), epsilon)``, and there is no
    separate evidence variable (Sec. 3.1, "Evidence space vs. data space").

    Args:
        tau: attention temperature.
        depletion: depletion form, one of
            :data:`~evidence_depletion.attention.DEPLETION_FORMS`.
        epsilon: floor on the per-token scale factor.
        gamma: bias strength when ``use_bias`` is set.
        use_bias: add the token-norm bias ``gamma * log(||h_l|| / max_l' ||h_l'||)``
            to the logits (controls for the bias vs. the space being depleted).
    """

    def __init__(
        self,
        n_slots: int = 5,
        d_model: int = 64,
        n_iter: int = 3,
        tau: float = 0.3,
        depletion: str = "quadratic",
        epsilon: float = 1e-4,
        gamma: float = 3.0,
        use_bias: bool = False,
    ) -> None:
        if depletion not in DEPLETION_FORMS:
            raise ValueError(f"Unknown depletion: {depletion}")
        super().__init__(n_slots=n_slots, d_model=d_model, n_iter=n_iter, tau=tau)
        self.use_bias = use_bias
        self.gamma = gamma
        self.depletion = depletion
        self.epsilon = epsilon

    def scale(self, alpha: torch.Tensor) -> torch.Tensor:
        """Per-token scale factor ``1 - f(alpha)`` (before the floor): ``[B, L]``."""
        if self.depletion == "quadratic":
            return 1 - alpha**2
        if self.depletion == "linear":
            return 1 - alpha
        if self.depletion == "cubic":
            return 1 - alpha**3
        return (alpha < 0.5).to(alpha.dtype)

    def forward(self, tokens: torch.Tensor) -> SlotAttentionOutput:
        B, L, D = tokens.shape
        slots_init = self.sample_slots(B, tokens.device)
        inp = self.norm_in(tokens)

        all_slots: list[torch.Tensor | None] = [None] * self.n_slots
        all_attns: list[torch.Tensor | None] = [None] * self.n_slots
        for s_idx in self.slot_order():
            s = int(s_idx.item())
            slot = slots_init[:, s : s + 1, :]
            k = self.proj_k(inp)
            v = self.proj_v(inp)
            for _ in range(self.n_iter):
                dots = self.logits(slot, k)
                if self.use_bias:
                    token_norm = inp.norm(dim=-1).clamp(min=1e-6)
                    max_norm = token_norm.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
                    dots = dots + (self.gamma * torch.log(token_norm / max_norm)).unsqueeze(1)
                attn = F.softmax(dots, dim=-1)
                updates = torch.einsum("bsl,bld->bsd", attn, v)
                slot = self.update_slot(slot, updates)
            all_slots[s] = slot
            all_attns[s] = attn.squeeze(1)

            inp = inp * self.scale(attn.squeeze(1)).unsqueeze(-1).clamp(min=self.epsilon)

        return SlotAttentionOutput(torch.cat(all_slots, dim=1), torch.stack(all_attns, dim=1), None)


# ============================================================
# Sparse normalizers
# ============================================================


class _SparsemaxFunction(torch.autograd.Function):
    """Sparsemax (Martins & Astudillo 2016): exact sorting-based Euclidean
    projection onto the simplex, with the closed-form Jacobian in the backward."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, dim: int) -> torch.Tensor:
        z = logits - logits.max(dim=dim, keepdim=True).values
        z_sorted, _ = torch.sort(z, dim=dim, descending=True)
        K = z.size(dim)
        shape = [1] * z.dim()
        shape[dim] = K
        rng = torch.arange(1, K + 1, device=z.device, dtype=z.dtype).view(shape)
        z_cumsum = z_sorted.cumsum(dim)
        support = rng * z_sorted > (z_cumsum - 1)
        k_support = support.sum(dim=dim, keepdim=True).clamp(min=1)
        tau = (z_cumsum.gather(dim, k_support - 1) - 1) / k_support.to(z.dtype)
        output = torch.clamp(z - tau, min=0)
        ctx.dim = dim
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (output,) = ctx.saved_tensors
        dim = ctx.dim
        supp = (output > 0).to(grad_output.dtype)
        grad = grad_output * supp
        v_hat = grad.sum(dim=dim, keepdim=True) / supp.sum(dim=dim, keepdim=True).clamp(min=1)
        return grad - v_hat * supp, None


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax along ``dim`` (Martins & Astudillo 2016)."""
    return _SparsemaxFunction.apply(logits, dim)


class _Entmax15Function(torch.autograd.Function):
    """Entmax-1.5 (Peters et al. 2019): exact sorting-based algorithm with the
    closed-form Jacobian-vector product in the backward."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, dim: int) -> torch.Tensor:
        z = (logits - logits.max(dim=dim, keepdim=True).values) / 2
        z_sorted, _ = torch.sort(z, dim=dim, descending=True)
        K = z.size(dim)
        shape = [1] * z.dim()
        shape[dim] = K
        rng = torch.arange(1, K + 1, device=z.device, dtype=z.dtype).view(shape)
        mean = z_sorted.cumsum(dim) / rng
        mean_sq = (z_sorted**2).cumsum(dim) / rng
        ss = rng * (mean_sq - mean**2)
        delta = torch.clamp((1 - ss) / rng, min=0)
        tau = mean - torch.sqrt(delta)
        support = tau <= z_sorted
        k_support = support.sum(dim=dim, keepdim=True).clamp(min=1)
        tau_star = tau.gather(dim, k_support - 1)
        output = torch.clamp(z - tau_star, min=0) ** 2
        ctx.dim = dim
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (output,) = ctx.saved_tensors
        dim = ctx.dim
        gppr = output.sqrt()
        grad = grad_output * gppr
        q = grad.sum(dim=dim, keepdim=True) / gppr.sum(dim=dim, keepdim=True).clamp(min=1e-12)
        return grad - q * gppr, None


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Entmax-1.5 along ``dim`` (Peters et al. 2019)."""
    return _Entmax15Function.apply(logits, dim)


class SparseCompetitionSA(VanillaSA):
    """Vanilla SA with the softmax over slots replaced by a sparse normalizer.

    Args:
        norm_fn: ``"sparsemax"`` (Martins & Astudillo 2016) or ``"entmax15"``
            (Peters et al. 2019); both are applied over the slot axis.
    """

    def __init__(
        self, n_slots: int = 5, d_model: int = 64, n_iter: int = 3, norm_fn: str = "sparsemax"
    ) -> None:
        super().__init__(n_slots=n_slots, d_model=d_model, n_iter=n_iter)
        if norm_fn not in ("sparsemax", "entmax15"):
            raise ValueError(f"Unknown norm_fn: {norm_fn}")
        self.norm_fn = norm_fn

    def normalize(self, dots: torch.Tensor) -> torch.Tensor:
        if self.norm_fn == "sparsemax":
            return sparsemax(dots, dim=1)
        return entmax15(dots, dim=1)


# ============================================================
# Optimal-transport attention: SA-SH and MESH
# ============================================================


def log_sinkhorn(logits: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
    """Log-domain Sinkhorn normalization of a ``[B, S, L]`` score matrix.

    Alternates normalization over tokens (dim 2) and slots (dim 1), ending with
    the slot step so that every token's column sums to one exactly (the
    softmax-over-slots convention of vanilla SA) while rows are approximately
    balanced (each slot claims about ``L / S`` tokens of mass).
    """
    log_alpha = logits
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return log_alpha.exp()


def attn_entropy(attn: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Shannon entropy ``-sum_{s,l} A_sl log A_sl`` of a ``[B, S, L]`` plan, per sample ``[B]``."""
    return -(attn * torch.log(attn.clamp(min=eps))).sum(dim=(1, 2))


def mesh_optimize_cost(
    cost: torch.Tensor,
    sinkhorn_iters: int = 5,
    mesh_steps: int = 5,
    mesh_lr: float = 0.1,
    init_noise: float = 1e-6,
    return_entropy_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
    """MESH cost optimization (Zhang et al., ICML 2023, Sec. 4, Eqs. 9-12).

    Given a raw cost / logit matrix ``cost`` of shape ``[B, S, L]``, returns the
    entropy-minimized attention ``A = sinkhorn(C'^(T))``, where ``C'`` is refined
    by ``T = mesh_steps`` per-sample normalized-gradient steps on the entropy::

        C'^(0)   = cost + eps,  eps ~ N(0, init_noise^2)   (noise applied once)
        C'^(t+1) = C'^(t) - mesh_lr * g / ||g||,  g = dH(sinkhorn(C'^(t))) / dC'^(t)

    The inner gradients are taken on a detached copy, so the per-step updates are
    constants with respect to the outer graph: ``C'^(T) = cost + offset`` and the
    training gradient flows through ``sinkhorn(cost + offset)`` into the
    projections. ``torch.enable_grad()`` lets the inner optimization run under an
    outer ``torch.no_grad()`` (evaluation).

    Returns:
        the attention ``[B, S, L]``; with ``return_entropy_trace`` also the batch-mean
        entropy before every inner step and after the last one.
    """
    C_prime = cost + init_noise * torch.randn_like(cost)

    entropy_trace = []
    for _ in range(mesh_steps):
        with torch.enable_grad():
            C_leaf = C_prime.detach().requires_grad_(True)
            attn = log_sinkhorn(C_leaf, n_iter=sinkhorn_iters)
            H_per = attn_entropy(attn)
            if return_entropy_trace:
                entropy_trace.append(H_per.detach().mean().item())
            # Per-sample entropies are independent: one grad() of the sum gives
            # every sample's gradient.
            g = torch.autograd.grad(H_per.sum(), C_leaf, create_graph=False)[0]
        gnorm = g.reshape(g.shape[0], -1).norm(dim=1).clamp(min=1e-12)
        g_unit = g / gnorm.view(-1, *([1] * (g.dim() - 1)))
        C_prime = C_prime - mesh_lr * g_unit

    attn_final = log_sinkhorn(C_prime, n_iter=sinkhorn_iters)
    if return_entropy_trace:
        entropy_trace.append(attn_entropy(attn_final).detach().mean().item())
        return attn_final, entropy_trace
    return attn_final


class MESHSA(VanillaSA):
    """MESH slot attention (Zhang et al., ICML 2023).

    Parallel structure of :class:`VanillaSA`; each iteration's attention is the
    Sinkhorn plan of an entropy-minimized transport cost
    (:func:`mesh_optimize_cost`), followed by the weighted mean over tokens.

    Args:
        sinkhorn_iters: Sinkhorn iterations per plan.
        mesh_steps: inner entropy-minimization steps ``T``.
        mesh_lr: inner step size ``lambda``.
        init_noise: std of the one-off symmetry-breaking noise on the cost.
    """

    def __init__(
        self,
        n_slots: int = 5,
        d_model: int = 64,
        n_iter: int = 3,
        sinkhorn_iters: int = 5,
        mesh_steps: int = 5,
        mesh_lr: float = 0.1,
        init_noise: float = 1e-6,
    ) -> None:
        super().__init__(n_slots=n_slots, d_model=d_model, n_iter=n_iter)
        self.sinkhorn_iters = sinkhorn_iters
        self.mesh_steps = mesh_steps
        self.mesh_lr = mesh_lr
        self.init_noise = init_noise

    def normalize(self, dots: torch.Tensor) -> torch.Tensor:
        return mesh_optimize_cost(
            dots,
            sinkhorn_iters=self.sinkhorn_iters,
            mesh_steps=self.mesh_steps,
            mesh_lr=self.mesh_lr,
            init_noise=self.init_noise,
        )


class SASHSA(VanillaSA):
    """SA-SH: Sinkhorn slot attention with symmetry-breaking noise (App. H).

    Parallel structure of :class:`VanillaSA`; the logits are perturbed by
    Gaussian noise of std ``noise_std`` (at training and at inference) and
    normalized by ``sinkhorn_iters`` log-domain Sinkhorn iterations
    (:func:`log_sinkhorn`), whose last step normalizes over slots as vanilla
    SA's softmax does. This is the Sinkhorn slot attention that MESH builds on;
    unlike :class:`MESHSA` there is no inner entropy minimization.

    This module is a reconstruction: the research code version used for the
    final runs only contains full MESH. It reproduces the configuration behind
    the SA-SH row of Table 2 (peak overlap 0.071 / 0.126 / 0.117 on sinusoids /
    Gaussians / multi-scale): 5 Sinkhorn iterations and ``N(0, 0.05^2)`` logit
    noise in training and evaluation, otherwise identical to :class:`VanillaSA`.
    Unlike the other methods it is not covered by the bit-for-bit equivalence
    tests.

    Args:
        sinkhorn_iters: Sinkhorn iterations.
        noise_std: std of the logit noise.
    """

    def __init__(
        self,
        n_slots: int = 5,
        d_model: int = 64,
        n_iter: int = 3,
        sinkhorn_iters: int = 5,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__(n_slots=n_slots, d_model=d_model, n_iter=n_iter)
        self.sinkhorn_iters = sinkhorn_iters
        self.noise_std = noise_std

    def normalize(self, dots: torch.Tensor) -> torch.Tensor:
        dots = dots + self.noise_std * torch.randn_like(dots)
        return log_sinkhorn(dots, n_iter=self.sinkhorn_iters)


# ============================================================
# Sequential alternatives: coverage and stick-breaking
# ============================================================


class CoverageSA(SequentialSA):
    """Coverage mechanism (adaptation of Tu et al. 2016).

    Sequential like :class:`SequentialSA`, with an additive coverage vector
    ``c_l = sum_{s' processed} alpha_{s',l}`` and logits ``q.k / (tau sqrt(d)) - beta c_l``.
    The penalty is purely additive: no multiplicative state and no
    log-evidence bias.

    Args:
        beta: coverage weight; 3.0 matches the scale of ``gamma = 3``.
    """

    def __init__(
        self,
        n_slots: int = 5,
        d_model: int = 64,
        n_iter: int = 3,
        tau: float = 0.3,
        beta: float = 3.0,
    ) -> None:
        super().__init__(n_slots=n_slots, d_model=d_model, n_iter=n_iter, tau=tau)
        self.beta = beta

    def forward(self, tokens: torch.Tensor) -> SlotAttentionOutput:
        B, L, D = tokens.shape
        slots_init = self.sample_slots(B, tokens.device)
        inp = self.norm_in(tokens)
        coverage = torch.zeros(B, L, device=tokens.device, dtype=tokens.dtype)

        all_slots: list[torch.Tensor | None] = [None] * self.n_slots
        all_attns: list[torch.Tensor | None] = [None] * self.n_slots
        for s_idx in self.slot_order():
            s = int(s_idx.item())
            slot = slots_init[:, s : s + 1, :]
            k = self.proj_k(inp)
            v = self.proj_v(inp)
            for _ in range(self.n_iter):
                dots = self.logits(slot, k) - self.beta * coverage.unsqueeze(1)
                attn = F.softmax(dots, dim=-1)
                updates = torch.einsum("bsl,bld->bsd", attn, v)
                slot = self.update_slot(slot, updates)
            all_slots[s] = slot
            all_attns[s] = attn.squeeze(1)
            coverage = coverage + attn.squeeze(1)

        return SlotAttentionOutput(torch.cat(all_slots, dim=1), torch.stack(all_attns, dim=1), None)


class StickBreakingSA(SequentialSA):
    """Deterministic stick-breaking scopes in the spirit of Genesis-v2.

    A stick ``m_l`` (initially 1) tracks the unassigned mass per token. Slot
    ``s`` attends over the stick-scaled inputs with scope ``sigmoid(logits)``;
    its assigned mass is ``pi_{s,l} = m_l * scope_{s,l}`` and the stick is updated
    as ``m_l <- max(m_l (1 - pi_{s,l}), epsilon)``. The returned attention is ``pi``.

    Args:
        epsilon: floor on the stick.
    """

    def __init__(
        self,
        n_slots: int = 5,
        d_model: int = 64,
        n_iter: int = 3,
        tau: float = 0.3,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__(n_slots=n_slots, d_model=d_model, n_iter=n_iter, tau=tau)
        self.epsilon = epsilon

    def forward(self, tokens: torch.Tensor) -> SlotAttentionOutput:
        B, L, D = tokens.shape
        slots_init = self.sample_slots(B, tokens.device)
        inp = self.norm_in(tokens)
        stick = torch.ones(B, L, device=tokens.device, dtype=tokens.dtype)

        all_slots: list[torch.Tensor | None] = [None] * self.n_slots
        all_attns: list[torch.Tensor | None] = [None] * self.n_slots
        for s_idx in self.slot_order():
            s = int(s_idx.item())
            slot = slots_init[:, s : s + 1, :]
            masked_inp = inp * stick.unsqueeze(-1)
            k = self.proj_k(masked_inp)
            v = self.proj_v(masked_inp)
            for _ in range(self.n_iter):
                scope = torch.sigmoid(self.logits(slot, k))
                pi = stick.unsqueeze(1) * scope
                w = pi / pi.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                updates = torch.einsum("bsl,bld->bsd", w, v)
                slot = self.update_slot(slot, updates)
            pi_final = pi.squeeze(1)
            all_slots[s] = slot
            all_attns[s] = pi_final
            stick = (stick * (1 - pi_final)).clamp(min=self.epsilon)

        return SlotAttentionOutput(torch.cat(all_slots, dim=1), torch.stack(all_attns, dim=1), None)


# ============================================================
# Non-slot-attention set predictor
# ============================================================


class DSPNSetPredictor(nn.Module):
    """DETR/DSPN-lite set predictor (non-slot-attention baseline, App. H).

    A learned set of ``S`` queries cross-attends once into the tokens with
    standard multi-head attention; there is no iterative slot competition. The
    returned attention is the head-averaged cross-attention map ``[B, S, L]``.

    Args:
        n_slots: number of queries ``S``.
        d_model: model dimension.
        n_heads: attention heads.
    """

    def __init__(self, n_slots: int = 5, d_model: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.queries = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.1)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, tokens: torch.Tensor) -> SlotAttentionOutput:
        B = tokens.shape[0]
        inp = self.norm_in(tokens)
        q = self.norm_q(self.queries.expand(B, -1, -1))
        attn_out, attn_w = self.mha(q, inp, inp, need_weights=True, average_attn_weights=True)
        slots = q + attn_out
        slots = slots + self.mlp(self.norm_mlp(slots))
        return SlotAttentionOutput(slots, attn_w, None)
