"""Sequential slot attention with evidence depletion (the paper's mechanism).

This module is self-contained (it depends only on ``torch``) and can be copied
into other code bases. The central class is :class:`EvidenceDepletionSlotAttention`,
a drop-in ``nn.Module`` mapping tokens ``[B, L, D]`` to slots ``[B, S, D]``,
per-slot attention ``[B, S, L]`` and the evidence trajectory.

Mechanism (Sec. 2.3 and Algorithm 1 of the paper). Every token ``l`` carries a
scalar evidence ``e_l`` in ``[epsilon, 1]``, initialized to 1 ("all signal
unexplained"). Slots are processed one at a time. For the current slot ``s``,
each of ``I`` refinement iterations computes (Eq. 5)::

    k_l = W_k(e_l * h_l),   v_l = W_v(e_l * h_l)
    alpha_{s,l} = softmax_l( q_s . k_l / (tau * sqrt(d)) + gamma * log e_l )
    s <- GRU(sum_l alpha_{s,l} v_l, s);   s <- s + MLP(LayerNorm(s))

and once the slot is refined the evidence is depleted multiplicatively (Eq. 6)::

    e_l <- max(e_l * (1 - f(alpha_{s,l})), epsilon)

with a depletion form ``f`` (:data:`DEPLETION_FORMS`). Proposition 2 of the
paper guarantees ``epsilon <= e^{(s+1)} <= e^{(s)}``: evidence never increases
along the processing order.

Note the change of normalization relative to standard slot attention
(Locatello et al. 2020, Eq. 3 of the paper): because slots are processed
sequentially, the softmax runs over *tokens* (each slot chooses which tokens to
read), not over slots.

Example::

    >>> import torch
    >>> from evidence_depletion import EvidenceDepletionSlotAttention
    >>> sa = EvidenceDepletionSlotAttention(n_slots=5, d_model=64, depletion="linear")
    >>> out = sa(torch.randn(8, 16, 64))
    >>> out.slots.shape, out.attn.shape, len(out.evidence)
    (torch.Size([8, 5, 64]), torch.Size([8, 5, 16]), 6)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEPLETION_FORMS: tuple[str, ...] = ("linear", "quadratic", "cubic", "binary")
"""Supported depletion forms ``f`` in ``e <- e * (1 - f(alpha))`` (Sec. 3.1, App. G):

* ``"linear"``:    ``1 - alpha``     (best on the synthetic and FUSS benchmarks)
* ``"quadratic"``: ``1 - alpha**2``  (Eq. 6; the form written in the paper)
* ``"cubic"``:     ``1 - alpha**3``
* ``"binary"``:    hard mask, evidence is zeroed where ``alpha >= 0.5``
"""


class SlotAttentionOutput(NamedTuple):
    """Return value of every slot-attention module in this package.

    Attributes:
        slots: ``[B, S, D]`` refined slot embeddings, indexed by slot identity
            (not by processing order).
        attn: ``[B, S, L]`` attention of each slot over tokens. For the
            sequential modules this is the softmax over tokens of the last
            refinement iteration; for parallel modules it is the last
            iteration's normalized map.
        evidence: for evidence depletion, a list of ``S + 1`` tensors ``[B, L]``
            (initial evidence followed by the state after each processed slot,
            in processing order); ``None`` for modules without evidence.
    """

    slots: torch.Tensor
    attn: torch.Tensor
    evidence: list[torch.Tensor] | None


def deplete(evidence: torch.Tensor, alpha: torch.Tensor, form: str) -> torch.Tensor:
    """Apply one depletion step ``e * (1 - f(alpha))`` (Eq. 6, without the floor).

    Args:
        evidence: ``[B, L]`` current evidence.
        alpha: ``[B, L]`` attention of the slot that was just refined.
        form: one of :data:`DEPLETION_FORMS`.

    Returns:
        ``[B, L]`` depleted evidence (the caller applies the ``epsilon`` floor).
    """
    if form == "quadratic":
        return evidence * (1 - alpha**2)
    if form == "linear":
        return evidence * (1 - alpha)
    if form == "cubic":
        return evidence * (1 - alpha**3)
    if form == "binary":
        return evidence * (alpha < 0.5).to(evidence.dtype)
    raise ValueError(f"Unknown depletion: {form}")


class SlotAttentionBase(nn.Module):
    """Parameters shared by all GRU-based slot-attention variants.

    Holds the learned Gaussian slot prior ``N(mu, sigma^2)``, the input / slot /
    MLP layer norms, the query-key-value projections, the GRU cell and the
    residual MLP of Locatello et al. (2020). Subclasses implement ``forward``.
    The parameters are created in a fixed order so that, under a fixed seed,
    every variant starts from the same initialization.

    Args:
        n_slots: number of slots ``S``.
        d_model: token and slot dimension ``d``.
        n_iter: refinement iterations ``I`` per slot.
    """

    def __init__(self, n_slots: int = 5, d_model: int = 64, n_iter: int = 3) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.n_iter = n_iter
        self.d = d_model

        self.slot_mu = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        self.slot_sigma = nn.Parameter(torch.ones(1, 1, d_model) * 0.1)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def sample_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Draw initial slots ``s ~ N(mu, sigma^2)`` (Algorithm 1, line 1): ``[B, S, D]``."""
        return self.slot_mu + self.slot_sigma * torch.randn(
            batch_size, self.n_slots, self.d, device=device
        )

    def slot_order(self) -> torch.Tensor:
        """Processing order: a random permutation in training, ``0..S-1`` at inference.

        Algorithm 1, line 3. The random order keeps slots from specializing to a
        position in the depletion sequence (App. I, "Slot ordering").
        """
        return torch.randperm(self.n_slots) if self.training else torch.arange(self.n_slots)

    def update_slot(self, slot: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """GRU update followed by the residual MLP (Algorithm 1, line 8)."""
        B, S, D = slot.shape
        slot = self.gru(updates.reshape(-1, D), slot.reshape(-1, D)).reshape(B, S, D)
        return slot + self.mlp(self.norm_mlp(slot))


class EvidenceDepletionSlotAttention(SlotAttentionBase):
    """Sequential slot attention with evidence depletion (Algorithm 1).

    Args:
        n_slots: number of slots ``S``.
        d_model: token and slot dimension ``d``.
        n_iter: refinement iterations ``I`` per slot.
        gamma: strength of the log-evidence attention bias ``gamma * log e_l`` (Eq. 5).
        tau: attention temperature ``tau`` (Eq. 5).
        depletion: depletion form, one of :data:`DEPLETION_FORMS`.
        epsilon: evidence floor ``epsilon`` (Eq. 6).

    Shapes:
        input ``tokens``: ``[B, L, D]``; output: :class:`SlotAttentionOutput` with
        ``slots [B, S, D]``, ``attn [B, S, L]`` and ``evidence`` a list of
        ``S + 1`` tensors ``[B, L]``.

    The paper's defaults (Sec. 3.1, App. E) are ``gamma=3``, ``tau=0.3``,
    ``n_iter=3``; ``depletion="linear"`` is the recommended starting point
    (App. J), ``"quadratic"`` is the form written in Eq. 6.
    """

    def __init__(
        self,
        n_slots: int = 5,
        d_model: int = 64,
        n_iter: int = 3,
        gamma: float = 3.0,
        tau: float = 0.3,
        depletion: str = "quadratic",
        epsilon: float = 1e-4,
    ) -> None:
        if depletion not in DEPLETION_FORMS:
            raise ValueError(f"Unknown depletion: {depletion}")
        super().__init__(n_slots=n_slots, d_model=d_model, n_iter=n_iter)
        self.gamma = gamma
        self.tau = tau
        self.depletion = depletion
        self.epsilon = epsilon

    def evidence_bias(self, evidence: torch.Tensor) -> torch.Tensor:
        """Additive attention bias ``gamma * log e_l`` (Eq. 5): ``[B, L]``."""
        return self.gamma * torch.log(evidence.clamp(min=1e-6))

    def deplete(self, evidence: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Evidence update ``max(e * (1 - f(alpha)), epsilon)`` (Eq. 6)."""
        return deplete(evidence, alpha, self.depletion).clamp(min=self.epsilon)

    def attend(self, slot: torch.Tensor, k: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Evidence-biased softmax over tokens (Eq. 5; Algorithm 1, line 7).

        Args:
            slot: ``[B, n, D]`` slots (``n = 1`` for the slot being refined).
            k: ``[B, L, D]`` keys of the evidence-scaled tokens.
            bias: ``[B, L]`` log-evidence bias.

        Returns:
            ``[B, n, L]`` attention, normalized over tokens.
        """
        D = k.shape[-1]
        q = self.proj_q(self.norm_s(slot))
        dots = torch.einsum("bsd,bld->bsl", q, k) / (self.tau * D**0.5)
        dots = dots + bias.unsqueeze(1)
        return F.softmax(dots, dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        slot_order: Sequence[int] | torch.Tensor | None = None,
        forced_attn_first: torch.Tensor | None = None,
        order_policy: str | None = None,
    ) -> SlotAttentionOutput:
        """Run Algorithm 1 on a batch of tokens.

        Args:
            tokens: ``[B, L, D]`` input tokens ``h_l``.
            slot_order: optional explicit processing order (slot indices). It
                overrides the default random (train) / canonical (eval) order;
                used by the order-sensitivity study (App. I, "Early errors and
                slot order").
            forced_attn_first: optional ``[B, L]`` token distribution that replaces
                the attention of the *first processed* slot in all of its
                refinement iterations; the slot update and the subsequent
                depletion use the forced attention (forced-error study, App. I).
            order_policy: ``"evidence_greedy"`` picks, per sample, the next slot
                greedily by captured remaining evidence; see
                :meth:`forward_evidence_greedy`. ``None`` uses the standard order.

        Returns:
            :class:`SlotAttentionOutput` ``(slots, attn, evidence)``.
        """
        if order_policy == "evidence_greedy":
            return self.forward_evidence_greedy(tokens)
        if order_policy is not None:
            raise ValueError(f"Unknown order_policy: {order_policy}")

        B, L, D = tokens.shape
        dev = tokens.device

        slots_init = self.sample_slots(B, dev)
        inp = self.norm_in(tokens)
        evidence = torch.ones(B, L, device=dev, dtype=tokens.dtype)

        if slot_order is not None:
            order = torch.as_tensor(slot_order, dtype=torch.long).cpu()
            if sorted(order.tolist()) != list(range(self.n_slots)):
                raise ValueError(
                    f"slot_order must be a permutation of 0..{self.n_slots - 1}, "
                    f"got {order.tolist()}"
                )
        else:
            order = self.slot_order()
        all_slots: list[torch.Tensor | None] = [None] * self.n_slots
        all_attns: list[torch.Tensor | None] = [None] * self.n_slots
        all_evidence = [evidence.clone()]

        for pos, s_idx in enumerate(order):
            s = int(s_idx.item())
            slot = slots_init[:, s : s + 1, :]

            # Keys and values of the evidence-scaled tokens e_l * h_l (Eq. 5).
            masked_inp = inp * evidence.unsqueeze(-1)
            k = self.proj_k(masked_inp)
            v = self.proj_v(masked_inp)

            for _ in range(self.n_iter):
                # The bias is rebuilt in every iteration (rather than hoisted) so
                # that gradients accumulate in the same order as the original
                # implementation, keeping training bit-for-bit reproducible.
                attn = self.attend(slot, k, self.evidence_bias(evidence))
                if forced_attn_first is not None and pos == 0:
                    attn = forced_attn_first.unsqueeze(1)
                updates = torch.einsum("bsl,bld->bsd", attn, v)
                slot = self.update_slot(slot, updates)

            all_slots[s] = slot
            all_attns[s] = attn.squeeze(1)

            evidence = self.deplete(evidence, attn.squeeze(1))
            all_evidence.append(evidence.clone())

        slots_out = torch.cat(all_slots, dim=1)
        attns_out = torch.stack(all_attns, dim=1)
        return SlotAttentionOutput(slots_out, attns_out, all_evidence)

    def forward_evidence_greedy(self, tokens: torch.Tensor) -> SlotAttentionOutput:
        """Inference-time greedy, evidence-guided processing order (App. I).

        Ordering slots "by remaining evidence" is degenerate, since before a slot
        runs all pending slots face the same evidence state. Instead, at each
        step every pending slot is scored by the remaining evidence its
        first-iteration attention would capture, ``score_s = sum_l alpha_{s,l} e_l``,
        and the arg-max slot is processed next. The order is chosen per sample;
        outputs are written back to each slot's index, so slot identity is
        preserved. The per-slot computation is otherwise identical to
        :meth:`forward`.
        """
        B, L, D = tokens.shape
        dev = tokens.device
        S = self.n_slots

        slots_init = self.sample_slots(B, dev)
        inp = self.norm_in(tokens)
        evidence = torch.ones(B, L, device=dev, dtype=tokens.dtype)

        slots_out = tokens.new_zeros(B, S, D)
        attns_out = tokens.new_zeros(B, S, L)
        processed = torch.zeros(B, S, dtype=torch.bool, device=dev)
        all_evidence = [evidence.clone()]
        batch_idx = torch.arange(B, device=dev)

        for _ in range(S):
            masked_inp = inp * evidence.unsqueeze(-1)
            k = self.proj_k(masked_inp)
            v = self.proj_v(masked_inp)
            bias = self.evidence_bias(evidence)

            alpha_all = self.attend(slots_init, k, bias)
            scores = (alpha_all * evidence.unsqueeze(1)).sum(dim=-1)
            scores = scores.masked_fill(processed, float("-inf"))
            next_idx = scores.argmax(dim=1)

            slot = slots_init[batch_idx, next_idx].unsqueeze(1)
            for _ in range(self.n_iter):
                attn = self.attend(slot, k, bias)
                updates = torch.einsum("bsl,bld->bsd", attn, v)
                slot = self.update_slot(slot, updates)

            slots_out[batch_idx, next_idx] = slot.squeeze(1)
            attns_out[batch_idx, next_idx] = attn.squeeze(1)
            processed[batch_idx, next_idx] = True

            evidence = self.deplete(evidence, attn.squeeze(1))
            all_evidence.append(evidence.clone())

        return SlotAttentionOutput(slots_out, attns_out, all_evidence)
