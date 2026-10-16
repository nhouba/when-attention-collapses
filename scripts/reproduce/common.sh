# Shared settings for scripts/reproduce/*.sh (sourced, not executed).
# Override any variable from the environment, e.g.  PYTHON=python3.11 OUT_ROOT=/data/runs ./fig1_table1_synthetic.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-results/runs}"
FUSS_DIR="${FUSS_DIR:-data/fuss/ssdata}"     # created by: bash experiments/download_fuss.sh

# The original runs used 4 BLAS threads per process (one process per GPU).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

# Paper protocol shared by every synthetic experiment (Sec. 3.1). These are also
# the script defaults; they are passed explicitly so that the commands are
# self-documenting and robust to local changes of the defaults.
SYN_COMMON=(--epochs 200 --n-train 5000 --n-test 1000)

# Synthetic task keys; seeds are always 0..N-1 (--seeds N).
ALL_TASKS=(sinusoids gaussian_bumps multiscale)
read -r -a TASK_LIST <<< "${TASKS:-${ALL_TASKS[*]}}"

run() {  # echo + run a command
    echo "+ $*" >&2
    "$@"
}

# Resume semantics: every experiment script skips (task, method, seed, sweep[, config])
# records that already exist in <output-dir>/*_results.json, IGNORING epochs/n-train.
# Never point a run with different settings at an existing output directory
# (or pass --no-resume to the experiment script).
