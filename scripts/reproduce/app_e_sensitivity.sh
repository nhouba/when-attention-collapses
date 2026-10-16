#!/usr/bin/env bash
# Table 7 (App. E): gamma in {1,3,10} x tau in {0.1,0.3,1.0}, sinusoids,
# evidence_quadratic, 3 seeds (0-2), 200 epochs. 27 runs, ~13 GPU-h.
# Either --gamma-sweep or --tau-sweep triggers the JOINT 3x3 grid; without
# --methods the method comparison is skipped.
# Records: sweep == "sensitivity", method == "evidence_g{gamma}_t{tau}".
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run "$PYTHON" experiments/synthetic.py --gamma-sweep --tau-sweep --tasks sinusoids \
    --seeds 3 "${SYN_COMMON[@]}" --output-dir "$OUT_ROOT/sensitivity"
run "$PYTHON" scripts/make_tables.py "$OUT_ROOT/sensitivity"
