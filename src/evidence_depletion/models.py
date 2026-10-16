"""Tokenizers, prediction heads and the model factory used by all benchmarks.

Every benchmark model is ``tokenizer -> slot attention -> per-slot heads``; only
the slot-attention module varies between methods, so differences in collapse
are attributable to the attention mechanism alone (Sec. 3.1).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from evidence_depletion.attention import DEPLETION_FORMS, EvidenceDepletionSlotAttention
from evidence_depletion.baselines import (
    MESHSA,
    SASHSA,
    CoverageSA,
    DataSpaceDepletionSA,
    DSPNSetPredictor,
    SequentialSA,
    SparseCompetitionSA,
    StickBreakingSA,
    VanillaSA,
)

METHODS: tuple[str, ...] = (
    "vanilla",
    "sequential",
    *(f"evidence_{form}" for form in DEPLETION_FORMS),
    *(f"dataspace_{form}" for form in DEPLETION_FORMS),
    *(f"dataspace_{form}_bias" for form in DEPLETION_FORMS),
    "mesh",
    "sa_sh",
    "sparsemax",
    "entmax15",
    "coverage",
    "coverage_loss",
    "stickbreaking",
    "dspn",
)
"""Every method name accepted by :func:`build_model`."""


def _check_divisible(N: int, n_tokens: int) -> None:
    if N % n_tokens:
        raise ValueError(f"signal length {N} is not divisible by n_tokens={n_tokens}")


class ChunkFFTTokenizer(nn.Module):
    """Split the signal into ``n_tokens`` non-overlapping chunks, rFFT each chunk,
    and project ``[Re, Im]`` to ``d_model`` (used for sinusoids, multi-scale, FUSS).

    Shapes: ``[B, N] -> [B, n_tokens, d_model]``; ``N`` must be divisible by ``n_tokens``.
    """

    def __init__(self, N: int = 256, d_model: int = 64, n_tokens: int = 16) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        chunk_size = N // n_tokens
        n_fft = chunk_size // 2 + 1
        self.proj = nn.Sequential(nn.Linear(2 * n_fft, d_model), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        _check_divisible(N, self.n_tokens)
        chunks = x.reshape(B, self.n_tokens, N // self.n_tokens)
        fft = torch.fft.rfft(chunks, dim=-1)
        return self.proj(torch.cat([fft.real, fft.imag], dim=-1))


class ChunkTokenizer(nn.Module):
    """Split the signal into ``n_tokens`` chunks and project each chunk directly
    (no FFT; used for the Gaussian-bump task).

    Shapes: ``[B, N] -> [B, n_tokens, d_model]``; ``N`` must be divisible by ``n_tokens``.
    """

    def __init__(self, N: int = 256, d_model: int = 64, n_tokens: int = 16) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        chunk_size = N // n_tokens
        self.proj = nn.Sequential(nn.Linear(chunk_size, d_model), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        _check_divisible(N, self.n_tokens)
        return self.proj(x.reshape(B, self.n_tokens, N // self.n_tokens))


class DecompositionModel(nn.Module):
    """Tokenizer + slot attention + existence head + parameter head.

    Args:
        slot_attn_cls: slot-attention class, constructed as
            ``slot_attn_cls(n_slots=..., d_model=..., **slot_attn_kwargs)``.
        slot_attn_kwargs: extra constructor arguments of the slot-attention module.
        n_params: parameters per source (3 for sinusoids / bumps / FUSS, 5 for multi-scale).
        N: signal length.
        d_model: token / slot dimension.
        n_tokens: number of tokens ``L``.
        n_slots: number of slots ``S``.
        tokenizer_type: ``"fft"`` (:class:`ChunkFFTTokenizer`) or ``"direct"``
            (:class:`ChunkTokenizer`).
    """

    def __init__(
        self,
        slot_attn_cls: type[nn.Module],
        slot_attn_kwargs: dict[str, Any],
        n_params: int = 3,
        N: int = 256,
        d_model: int = 64,
        n_tokens: int = 16,
        n_slots: int = 5,
        tokenizer_type: str = "fft",
    ) -> None:
        super().__init__()
        if tokenizer_type == "fft":
            self.tokenizer = ChunkFFTTokenizer(N, d_model, n_tokens)
        else:
            self.tokenizer = ChunkTokenizer(N, d_model, n_tokens)

        self.slot_attn = slot_attn_cls(n_slots=n_slots, d_model=d_model, **slot_attn_kwargs)
        self.exist_head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))
        self.param_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, n_params))

    def forward(
        self, x: torch.Tensor, **sa_kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        """Args:
            x: ``[B, N]`` signals.
            **sa_kwargs: forwarded to the slot-attention module (inference-time
                controls of :class:`EvidenceDepletionSlotAttention` such as
                ``slot_order``, ``forced_attn_first`` or ``order_policy``).

        Returns:
            ``(slots [B,S,D], attn [B,S,L], exist_logits [B,S], params_pred [B,S,P], evidence)``.
        """
        tokens = self.tokenizer(x)
        slots, attn, evidence = self.slot_attn(tokens, **sa_kwargs)
        exist_logits = self.exist_head(slots).squeeze(-1)
        params_pred = self.param_head(slots)
        return slots, attn, exist_logits, params_pred, evidence


def build_model(
    method: str,
    n_params: int = 3,
    N: int = 256,
    d_model: int = 64,
    n_tokens: int = 16,
    n_slots: int = 5,
    tokenizer_type: str = "fft",
    **kwargs: Any,
) -> DecompositionModel:
    """Build a :class:`DecompositionModel` with the slot attention named by ``method``.

    Args:
        method: one of :data:`METHODS`:

            * ``"vanilla"``, ``"sequential"``;
            * ``"evidence_<form>"`` (ours) and ``"dataspace_<form>[_bias]"`` for a
              form in :data:`~evidence_depletion.attention.DEPLETION_FORMS`;
            * alternatives: ``"mesh"``, ``"sa_sh"``, ``"sparsemax"``, ``"entmax15"``,
              ``"coverage"``, ``"coverage_loss"`` (vanilla architecture; the
              penalty is added to the loss by the trainer), ``"stickbreaking"``,
              ``"dspn"``.
        n_params, N, d_model, n_tokens, n_slots, tokenizer_type: see
            :class:`DecompositionModel`.
        **kwargs: mechanism hyper-parameters, each with the paper default:
            ``n_iter=3``, ``gamma=3.0``, ``tau=0.3``, ``beta=3.0`` (coverage),
            ``sinkhorn_iters=5``, ``mesh_steps=5``, ``mesh_lr=0.1``,
            ``init_noise=1e-6`` (MESH), ``noise_std=0.05`` (SA-SH), ``n_heads=4``
            (DSPN). Keys a method does not use are ignored.
    """
    n_iter = kwargs.get("n_iter", 3)
    cls: type[nn.Module]
    if method in ("vanilla", "coverage_loss"):
        cls = VanillaSA
        sa_kwargs: dict[str, Any] = {"n_iter": n_iter}
    elif method == "sequential":
        cls = SequentialSA
        sa_kwargs = {"n_iter": n_iter, "tau": kwargs.get("tau", 0.3)}
    elif method.startswith("evidence_"):
        cls = EvidenceDepletionSlotAttention
        sa_kwargs = {
            "n_iter": n_iter,
            "gamma": kwargs.get("gamma", 3.0),
            "tau": kwargs.get("tau", 0.3),
            "depletion": method.split("_", 1)[1],
        }
    elif method.startswith("dataspace_"):
        rest = method.split("_", 1)[1]
        use_bias = rest.endswith("_bias")
        cls = DataSpaceDepletionSA
        sa_kwargs = {
            "n_iter": n_iter,
            "tau": kwargs.get("tau", 0.3),
            "depletion": rest.removesuffix("_bias"),
            "use_bias": use_bias,
            "gamma": kwargs.get("gamma", 3.0),
        }
    elif method == "mesh":
        cls = MESHSA
        sa_kwargs = {
            "n_iter": n_iter,
            "sinkhorn_iters": kwargs.get("sinkhorn_iters", 5),
            "mesh_steps": kwargs.get("mesh_steps", 5),
            "mesh_lr": kwargs.get("mesh_lr", 0.1),
            "init_noise": kwargs.get("init_noise", 1e-6),
        }
    elif method == "sa_sh":
        cls = SASHSA
        sa_kwargs = {
            "n_iter": n_iter,
            "sinkhorn_iters": kwargs.get("sinkhorn_iters", 5),
            "noise_std": kwargs.get("noise_std", 0.05),
        }
    elif method in ("sparsemax", "entmax15"):
        cls = SparseCompetitionSA
        sa_kwargs = {"n_iter": n_iter, "norm_fn": method}
    elif method == "coverage":
        cls = CoverageSA
        sa_kwargs = {
            "n_iter": n_iter,
            "tau": kwargs.get("tau", 0.3),
            "beta": kwargs.get("beta", 3.0),
        }
    elif method == "stickbreaking":
        cls = StickBreakingSA
        sa_kwargs = {"n_iter": n_iter, "tau": kwargs.get("tau", 0.3)}
    elif method == "dspn":
        cls = DSPNSetPredictor
        sa_kwargs = {"n_heads": kwargs.get("n_heads", 4)}
    else:
        raise ValueError(f"Unknown method: {method}")

    return DecompositionModel(
        slot_attn_cls=cls,
        slot_attn_kwargs=sa_kwargs,
        n_params=n_params,
        N=N,
        d_model=d_model,
        n_tokens=n_tokens,
        n_slots=n_slots,
        tokenizer_type=tokenizer_type,
    )
