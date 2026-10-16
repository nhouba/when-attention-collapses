#!/usr/bin/env bash
# Table 2 and the App. H statements on cosine overlap, the set predictor and matching
# pursuit. The vanilla and evidence-depletion rows come from fig1_table1_synthetic.sh.
#
#  * sparsemax entmax15 coverage coverage_loss stickbreaking dspn: 5 seeds x 3 tasks,
#    200 epochs -> 90 runs, ~37 GPU-h.
#  * mesh (full MESH, entropy-minimizing transport cost): 15 runs, ~7 GPU-h.
#  * sa_sh (Sinkhorn x5 + N(0, 0.05^2) logit noise; reconstructed, see
#    docs/reproduce.md "Known differences"): 15 runs, ~6 GPU-h.
#  * matching pursuit: eval-only, sinusoids, 5 seeds, seconds (its collapse_rate is
#    0.0 by construction).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ALT=(sparsemax entmax15 coverage coverage_loss stickbreaking dspn)
for task in "${TASK_LIST[@]}"; do
    run "$PYTHON" experiments/synthetic.py --methods "${ALT[@]}" --tasks "$task" \
        --seeds 5 "${SYN_COMMON[@]}" --output-dir "$OUT_ROOT/alternatives/$task"
    run "$PYTHON" experiments/synthetic.py --methods mesh --tasks "$task" \
        --seeds 5 "${SYN_COMMON[@]}" --output-dir "$OUT_ROOT/alternatives_mesh/$task"
    run "$PYTHON" experiments/synthetic.py --methods sa_sh --tasks "$task" \
        --seeds 5 "${SYN_COMMON[@]}" --output-dir "$OUT_ROOT/alternatives_sash/$task"
done
# Classical matching pursuit (sinusoids only).
run "$PYTHON" experiments/matching_pursuit.py --seeds 5 --n-test 1000 \
    --output-dir "$OUT_ROOT/alternatives_matching_pursuit"
run "$PYTHON" scripts/make_tables.py "$OUT_ROOT/synthetic_main" "$OUT_ROOT/alternatives" \
    "$OUT_ROOT/alternatives_mesh" "$OUT_ROOT/alternatives_sash" "$OUT_ROOT/alternatives_matching_pursuit"
