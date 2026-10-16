#!/usr/bin/env bash
# Table 10 (App. H) and the Sec. 3.1 "Loss-based regularization" paragraph:
# vanilla / sequential SA with w_div=5, w_ortho=3, or both; 6 variants x 3 tasks x 5 seeds,
# 200 epochs. 90 runs, ~38 GPU-h.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for task in "${TASK_LIST[@]}"; do
    run "$PYTHON" experiments/loss_baselines.py --tasks "$task" --seeds 5 "${SYN_COMMON[@]}" \
        --output-dir "$OUT_ROOT/loss_baselines/$task"
done
run "$PYTHON" scripts/make_tables.py "$OUT_ROOT/loss_baselines"
