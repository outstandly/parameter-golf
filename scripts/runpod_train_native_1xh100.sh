#!/usr/bin/env bash
set -euo pipefail

REPO_SRC="${REPO_SRC:-/tmp/pg-src}"
WORKROOT="${WORKROOT:-/tmp/pg-native-run}"
REPO_COPY="${WORKROOT}/repo"
LOG_DIR="${WORKROOT}/logs"
ARTIFACT_DIR="${WORKROOT}/artifacts"
DATA_PATH="${DATA_PATH:-/workspace/parameter-golf/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/pg-tokenizers/fineweb_1024_bpe.model}"
HF_HOME="${HF_HOME:-/tmp/hf}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
RUN_ID="${RUN_ID:-current_native_1xh100}"

mkdir -p "${WORKROOT}" "${LOG_DIR}" "${ARTIFACT_DIR}" "${HF_HOME}"
rm -rf "${REPO_COPY}"

git clone --local "${REPO_SRC}" "${REPO_COPY}" >/dev/null 2>&1 || git clone "${REPO_SRC}" "${REPO_COPY}" >/dev/null 2>&1

log_path="${LOG_DIR}/${RUN_ID}.txt"

(
  cd "${REPO_COPY}"
  env \
    HF_HOME="${HF_HOME}" \
    RUN_ID="${RUN_ID}" \
    DATA_PATH="${DATA_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    VOCAB_SIZE="${VOCAB_SIZE}" \
    VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
    TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY}" \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
    "$@" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py
) | tee "${log_path}"

if [[ -f "${REPO_COPY}/final_model.pt" ]]; then
  cp "${REPO_COPY}/final_model.pt" "${ARTIFACT_DIR}/${RUN_ID}.final_model.pt"
fi
if [[ -f "${REPO_COPY}/final_model.int8.ptz" ]]; then
  cp "${REPO_COPY}/final_model.int8.ptz" "${ARTIFACT_DIR}/${RUN_ID}.final_model.int8.ptz"
fi

echo
echo "=== Summary ${RUN_ID} ==="
grep "train_loader:dataset:" "${log_path}" | tail -n 1 || true
grep "layers:" "${log_path}" | tail -n 1 || true
grep "final_int8_zlib_roundtrip_exact" "${log_path}" | tail -n 1 || true
grep "Total submission size int8+zlib:" "${log_path}" | tail -n 1 || true
ls -lh "${ARTIFACT_DIR}/${RUN_ID}".final_model* 2>/dev/null || true
