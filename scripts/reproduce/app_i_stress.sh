#!/usr/bin/env bash
# App. I stress tests (all on sinusoids, linear depletion). Select parts with
# PARTS="amp stress arch order latency" (default: all).
#
#  amp     Table 11: --amp-ratio-sweep, ratios {1,1.5,2,3,5,10} x
#          {vanilla, sequential, evidence_linear} x 5 seeds (sequential is run but
#          not reported), one output directory per method.
#          90 runs, ~38 GPU-h (~24 GPU-h without sequential).
#  stress  slot over-provisioning S in {5,8,12,16} (K<=4), cardinality K<=8 / S=9, and
#          an unreported K=6 equal-amplitude config; 3 seeds. 42 runs, ~26 GPU-h.
#  arch    token count L in {8,16,32,64} (+ d and I axes, unreported); 3 seeds.
#          48 runs, ~20 GPU-h.
#  order   early errors / slot order (order_studies): trains vanilla + evidence_linear,
#          3 seeds, 200 epochs, saves checkpoints, then eval-only studies. ~3 GPU-h.
#  latency Table 12: attention-module forward latency, CPU + CUDA,
#          S in {5,10,20,40}, L in {16,128}, d=64, batch 32, I=3. Minutes. The paper
#          table is the CUDA, L=128 block on one GH200.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

read -r -a PART_LIST <<< "${PARTS:-amp stress arch order latency}"
for part in "${PART_LIST[@]}"; do
  case "$part" in
    amp)
      for m in vanilla sequential evidence_linear; do
          run "$PYTHON" experiments/synthetic.py --amp-ratio-sweep --track-dynamics \
              --sweep-methods "$m" --seeds 5 "${SYN_COMMON[@]}" \
              --output-dir "$OUT_ROOT/stress_amp_ratio/$m"
      done ;;
    stress)
      run "$PYTHON" experiments/synthetic.py --stress-sweep --seeds 3 "${SYN_COMMON[@]}" \
          --output-dir "$OUT_ROOT/stress_slots" ;;
    arch)
      run "$PYTHON" experiments/synthetic.py --arch-sweep --seeds 3 "${SYN_COMMON[@]}" \
          --output-dir "$OUT_ROOT/stress_arch" ;;
    order)
      run "$PYTHON" experiments/order_studies.py --seeds 3 --epochs 200 --n-train 5000 \
          --n-test 1000 --n-orders 10 --output-dir "$OUT_ROOT/stress_order" ;;
    latency)
      run "$PYTHON" experiments/latency.py --output-dir "$OUT_ROOT/stress_latency" ;;
    *) echo "unknown part: $part" >&2; exit 2 ;;
  esac
done
run "$PYTHON" scripts/make_tables.py "$OUT_ROOT/stress_amp_ratio" "$OUT_ROOT/stress_slots" \
    "$OUT_ROOT/stress_arch" "$OUT_ROOT/stress_order" "$OUT_ROOT/stress_latency"
