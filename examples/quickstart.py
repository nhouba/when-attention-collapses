"""Quickstart: slot collapse with and without evidence depletion (CPU, ~15 s).

1. Use EvidenceDepletionSlotAttention as a drop-in module on a batch of tokens.
2. Briefly train vanilla SA, sequential SA and evidence depletion on sinusoid
   mixtures with one dominant source (task A) and print the held-out collapse
   metrics of Definition 1 (peak overlap, cosine overlap; lower is better).

The paper's numbers use 5 seeds and the full budget: experiments/synthetic.py.
"""

import torch

from evidence_depletion import EvidenceDepletionSlotAttention
from evidence_depletion.data.synthetic import generate_sinusoids
from evidence_depletion.models import ChunkFFTTokenizer
from evidence_depletion.training import train_synthetic

# 1. The mechanism on its own: tokens [B, L, D] -> slots, attention, evidence.
torch.manual_seed(0)
signals, *_ = generate_sinusoids(8, K_range=(2, 4), seed=0)  # [8, 256]
tokens = ChunkFFTTokenizer(N=256, d_model=64, n_tokens=16)(signals)  # [8, 16, 64]
attention = EvidenceDepletionSlotAttention(n_slots=5, d_model=64, depletion="linear").eval()
out = attention(tokens)
print(f"slots {tuple(out.slots.shape)}, attention {tuple(out.attn.shape)}")
print("mean evidence after each slot (Prop. 2: non-increasing):")
print("  " + " -> ".join(f"{e.mean():.3f}" for e in out.evidence))

# 2. Short training runs (1024 mixtures, 15 epochs) and held-out collapse.
config = {"n_train": 1024, "n_test": 512, "epochs": 15}
print(f"\n{'method':<20}{'peak overlap':>14}{'cosine overlap':>16}")
for method in ["vanilla", "sequential", "evidence_linear"]:
    r = train_synthetic("sinusoids", method, seed=0, config=config)
    print(f"{method:<20}{r['collapse_rate']:>14.3f}{r['mean_attn_overlap']:>16.3f}")
print("\nevidence_linear = sequential slot attention with linear evidence depletion.")
