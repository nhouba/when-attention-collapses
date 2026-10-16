#!/usr/bin/env bash
# Everything synthetic + FUSS in the paper (~265 GPU-hours if run serially on one GH200;
# every script can be sliced and run in parallel, see docs/reproduce.md).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
HERE="$REPO_ROOT/scripts/reproduce"
bash "$HERE/quick_check.sh"
bash "$HERE/fig1_table1_synthetic.sh"
bash "$HERE/table2_alternatives.sh"
bash "$HERE/app_e_sensitivity.sh"
bash "$HERE/app_h_loss_baselines.sh"
bash "$HERE/app_i_stress.sh"
bash "$HERE/fuss.sh"
run "$PYTHON" scripts/make_tables.py "$OUT_ROOT" | tee "$OUT_ROOT/paper_tables.md"
