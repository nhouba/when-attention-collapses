"""Evidence depletion for slot attention.

Reference implementation of "When Attention Collapses: Residual Evidence
Modeling for Compositional Inference" (Houba, NeurIPS 2026, arXiv:2605.02323).

* :class:`EvidenceDepletionSlotAttention` -- the mechanism (Algorithm 1), a
  drop-in module mapping tokens ``[B, L, D]`` to slots, attention and evidence.
* :func:`build_model` -- tokenizer + slot attention + heads for every method
  compared in the paper.
* :mod:`evidence_depletion.metrics` -- collapse metrics (Definition 1) and
  parameter-recovery metrics.
"""

from evidence_depletion import metrics
from evidence_depletion.attention import (
    DEPLETION_FORMS,
    EvidenceDepletionSlotAttention,
    SlotAttentionOutput,
    deplete,
)
from evidence_depletion.models import METHODS, DecompositionModel, build_model

__version__ = "1.0.0"

__all__ = [
    "DEPLETION_FORMS",
    "METHODS",
    "DecompositionModel",
    "EvidenceDepletionSlotAttention",
    "SlotAttentionOutput",
    "build_model",
    "deplete",
    "metrics",
]
