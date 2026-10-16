#!/usr/bin/env bash
# Download the FUSS dry mixtures (Free Universal Sound Separation; Wisdom et al., ICASSP 2021).
#
# Source:  Zenodo record 3743844, https://zenodo.org/records/3743844
# License: CC BY 4.0 (attribution required; see the record for the per-clip
#          Freesound attributions).
#
# Usage:   bash experiments/download_fuss.sh [TARGET_DIR]      (default: data/fuss)
#
# Expected layout afterwards (pass the ssdata directory as --fuss-dir):
#
#   TARGET_DIR/ssdata/
#     train/       example00000.wav, example00000_sources/{background0,foreground0,...}_sound.wav, ...
#     validation/  ...
#     eval/        ...
#
# Each mixture exampleNNNNN.wav is 10 s, 16 kHz, 16-bit mono; its sources are in
# exampleNNNNN_sources/. The archive is several GB; extraction needs about twice that.
set -euo pipefail

TARGET_DIR="${1:-data/fuss}"
ARCHIVE="FUSS_ssdata.tar.gz"
URL="https://zenodo.org/records/3743844/files/${ARCHIVE}?download=1"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

if [ ! -f "${ARCHIVE}" ]; then
  echo "Downloading ${ARCHIVE} from Zenodo record 3743844 ..."
  curl -L --fail --retry 3 -o "${ARCHIVE}.part" "${URL}"
  mv "${ARCHIVE}.part" "${ARCHIVE}"
fi

echo "Extracting ${ARCHIVE} ..."
tar -xzf "${ARCHIVE}"

SSDATA="$(find . -maxdepth 3 -type d -name ssdata | head -n 1)"
if [ -z "${SSDATA}" ] || [ ! -d "${SSDATA}/train" ] || [ ! -d "${SSDATA}/validation" ]; then
  echo "error: could not find ssdata/{train,validation,eval} after extraction" >&2
  exit 1
fi
echo "FUSS ready. Use: --fuss-dir ${TARGET_DIR}/${SSDATA#./}"
