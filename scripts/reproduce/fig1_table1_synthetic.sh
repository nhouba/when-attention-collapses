#!/usr/bin/env bash
# Figure 1 (synthetic groups), Table 1 (synthetic columns, incl. data-space rows),
# Table 8 / App. F (synthetic peak + cosine), Table 9 / App. G (all depletion forms).
#
# 8 methods x 3 tasks x 5 seeds, 200 epochs, 5000 train / 1000 test mixtures. Every
# record holds both metrics (collapse_rate = peak overlap, mean_attn_overlap = cosine).
# 120 runs, ~52 GPU-hours on one GH200 (~1550 s/run); tasks run as independent
# slices so they can be launched in parallel:  TASKS=sinusoids ./fig1_table1_synthetic.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

METHODS=(vanilla sequential evidence_binary evidence_linear evidence_quadratic
         evidence_cubic dataspace_quadratic dataspace_quadratic_bias)
for task in "${TASK_LIST[@]}"; do
    run "$PYTHON" experiments/synthetic.py --methods "${METHODS[@]}" --tasks "$task" \
        --seeds 5 "${SYN_COMMON[@]}" --output-dir "$OUT_ROOT/synthetic_main/$task"
done
run "$PYTHON" scripts/make_tables.py "$OUT_ROOT/synthetic_main"
