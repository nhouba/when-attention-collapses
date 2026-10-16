#!/usr/bin/env bash
# Quick sanity check (about 5-7 min on a recent laptop CPU): vanilla vs evidence_linear
# on sinusoids, 3 seeds, 40 epochs (the paper uses 5 seeds and 200 epochs).
#
# Reference run (Apple-silicon CPU, the versions pinned in requirements.txt, 5.4 min):
#   vanilla          peak overlap 0.38 +- 0.13   (seeds 0/1/2: 0.53 / 0.38 / 0.22)
#   evidence_linear  peak overlap 0.15 +- 0.11   (seeds 0/1/2: 0.31 / 0.08 / 0.07)
#   paper (200 epochs, 5 seeds): 0.32 +- 0.09 vs 0.07 +- 0.07.
# Per-seed values depend on the PyTorch version and hardware; single seeds vary a lot.
# The check passes if the mean peak overlap of evidence_linear is below half that of
# vanilla.
#
# Runs on CPU by default: the model is tiny, so a GPU brings little (a 200-epoch run
# takes ~3-6 min on a laptop CPU core and ~20-30 min on a GH200).
# Set QC_DEVICE=gpu to use CUDA if available.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

QC_EPOCHS="${QC_EPOCHS:-40}"
QC_SEEDS="${QC_SEEDS:-3}"
OUT="$OUT_ROOT/quick_check_e${QC_EPOCHS}_s${QC_SEEDS}"
if [[ "${QC_DEVICE:-cpu}" == "cpu" ]]; then export CUDA_VISIBLE_DEVICES=""; fi

run "$PYTHON" experiments/synthetic.py \
    --methods vanilla evidence_linear --tasks sinusoids \
    --seeds "$QC_SEEDS" --epochs "$QC_EPOCHS" --n-train 5000 --n-test 1000 \
    --output-dir "$OUT"

"$PYTHON" - "$OUT/ablation_results.json" <<'PY'
import json, sys
import numpy as np
recs = [r for r in json.load(open(sys.argv[1])) if r.get("sweep") == "method"]
def get(m):
    return np.array([r["collapse_rate"] for r in sorted(recs, key=lambda r: r["seed"]) if r["method"] == m])
van, evi = get("vanilla"), get("evidence_linear")
print(f"vanilla          peak overlap {van.mean():.3f} +- {van.std():.3f}  per seed {np.round(van, 3).tolist()}")
print(f"evidence_linear  peak overlap {evi.mean():.3f} +- {evi.std():.3f}  per seed {np.round(evi, 3).tolist()}")
ok = evi.mean() < 0.5 * van.mean()
print("QUICK CHECK", "PASSED" if ok else "FAILED (expected evidence_linear < 0.5 x vanilla)")
sys.exit(0 if ok else 1)
PY
