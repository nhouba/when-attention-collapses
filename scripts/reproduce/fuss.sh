#!/usr/bin/env bash
# FUSS: Table 1 FUSS column, Figure 1 group D, Table 13, Table 8 FUSS column,
# Sec. 3.2 / App. J SI-SDR diagnostics, oracle-mask readout, parameter baseline.
#
# Data: bash experiments/download_fuss.sh   (FUSS_ssdata.tar.gz, several GB, Zenodo,
#       CC BY 4.0) -> --fuss-dir data/fuss/ssdata  (must contain train/ and validation/).
#
# Paper configuration (the defaults of experiments/fuss.py, passed explicitly):
# 4 s crops (64000 samples), 32 tokens, d_model 128, batch 32, 5000 train / 1000
# validation mixtures, 200 epochs, 5 seeds. --eval-sisdr does not change training or
# the collapse numbers, so one pass serves every FUSS artifact. One method per output
# directory, so methods can run in parallel (FUSS_METHODS="vanilla sa_sh" ...).
# ~2600-3350 s per run on a GH200: 30 runs ~ 25 GPU-h.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ ! -d "$FUSS_DIR/train" || ! -d "$FUSS_DIR/validation" ]]; then
    echo "FUSS data not found at $FUSS_DIR (expected train/ and validation/)." >&2
    echo "Run: bash experiments/download_fuss.sh   or set FUSS_DIR=..." >&2
    exit 1
fi

read -r -a FMETHODS <<< "${FUSS_METHODS:-vanilla sequential evidence_linear evidence_quadratic sparsemax sa_sh}"
FUSS_COMMON=(--fuss-dir "$FUSS_DIR" --seeds 5 --epochs 200 --signal-length 64000
             --n-tokens 32 --d-model 128 --max-train 5000 --max-test 1000 --batch-size 32)
for m in "${FMETHODS[@]}"; do
    run "$PYTHON" experiments/fuss.py "${FUSS_COMMON[@]}" --methods "$m" --eval-sisdr \
        --output-dir "$OUT_ROOT/fuss/$m"
done

# Oracle mask of the same readout family (+7.7 dB = rank1_model_faithful - floor). ~1 min.
run "$PYTHON" experiments/fuss_sisdr_oracle.py --fuss-dir "$FUSS_DIR" --signal-length 64000 \
    --n-mixtures 300 --n-chunks 32 --seed 42 | tee "$OUT_ROOT/fuss/sisdr_oracle.txt"

# Constant-predictor parameter baseline (CPU, minutes) and the 5-seed comparison.
run "$PYTHON" experiments/fuss_param_baseline.py --fuss-dir "$FUSS_DIR" --signal-length 64000 \
    --max-train 5000 --max-test 1000 | tee "$OUT_ROOT/fuss/param_baseline.txt"
run "$PYTHON" experiments/fuss_param_compare.py \
    --results-glob "$OUT_ROOT/fuss/*/fuss_results.json" | tee "$OUT_ROOT/fuss/param_compare.txt"

run "$PYTHON" scripts/make_tables.py "$OUT_ROOT/fuss"
