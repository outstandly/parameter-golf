#!/usr/bin/env bash
set -euo pipefail

# Run a clean apples-to-apples 1xH100 comparison between:
# 1. OpenAI's official naive baseline script
# 2. The current working tree's train_gpt.py
#
# Intended for Runpod where:
# - repo checkout lives in /workspace/parameter-golf
# - dataset/tokenizer also live under /workspace/parameter-golf
# - training artifacts should be written to /tmp to avoid workspace quota issues

REPO_SRC="${REPO_SRC:-/workspace/parameter-golf}"
WORKROOT="${WORKROOT:-/tmp/pg-run}"
REPO_COPY="${WORKROOT}/repo"
LOG_DIR="${WORKROOT}/logs"
SHIM_DIR="${WORKROOT}/shim"
DATA_PATH="${DATA_PATH:-${REPO_SRC}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_SRC}/data/tokenizers/fineweb_1024_bpe.model}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"

OFFICIAL_RUN_ID="${OFFICIAL_RUN_ID:-official_same_setup_1xh100}"
CURRENT_RUN_ID="${CURRENT_RUN_ID:-current_same_setup_1xh100}"

mkdir -p "${WORKROOT}" "${LOG_DIR}" "${SHIM_DIR}" "${HF_HOME}"
rm -rf "${REPO_COPY}"
mkdir -p "${REPO_COPY}"

cat > "${SHIM_DIR}/sitecustomize.py" <<'PY'
import inspect

import torch.nn.functional as F

_orig = F.scaled_dot_product_attention
try:
    _has_enable_gqa = "enable_gqa" in inspect.signature(_orig).parameters
except Exception:
    _has_enable_gqa = True

if not _has_enable_gqa:
    def _compat_scaled_dot_product_attention(query, key, value, *args, **kwargs):
        enable_gqa = kwargs.pop("enable_gqa", False)
        if enable_gqa:
            q_heads = query.shape[-3]
            kv_heads = key.shape[-3]
            if q_heads % kv_heads != 0:
                raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")
            if q_heads != kv_heads:
                repeats = q_heads // kv_heads
                key = key.repeat_interleave(repeats, dim=-3)
                value = value.repeat_interleave(repeats, dim=-3)
        return _orig(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = _compat_scaled_dot_product_attention
PY

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'data/datasets' \
    --exclude 'data/tokenizers' \
    "${REPO_SRC}/" "${REPO_COPY}/"
else
  cp -R "${REPO_SRC}/." "${REPO_COPY}/"
  rm -rf "${REPO_COPY}/.git" "${REPO_COPY}/.venv" "${REPO_COPY}/data/datasets" "${REPO_COPY}/data/tokenizers"
fi

patch_attention_compat() {
  local file_path="$1"
  python3 - "${file_path}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

helper = """
def scaled_dot_product_attention_compat(query, key, value, **kwargs):
    try:
        return F.scaled_dot_product_attention(query, key, value, **kwargs)
    except TypeError:
        enable_gqa = kwargs.pop("enable_gqa", False)
        if enable_gqa:
            q_heads = query.shape[-3]
            kv_heads = key.shape[-3]
            if q_heads % kv_heads != 0:
                raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")
            if q_heads != kv_heads:
                repeats = q_heads // kv_heads
                key = key.repeat_interleave(repeats, dim=-3)
                value = value.repeat_interleave(repeats, dim=-3)
        return F.scaled_dot_product_attention(query, key, value, **kwargs)
"""

if "def scaled_dot_product_attention_compat(" not in text:
    marker = "\n\nclass CausalSelfAttention"
    if marker not in text:
        raise SystemExit(f"Could not find insertion marker in {path}")
    text = text.replace(marker, "\n\n" + helper + "\n\nclass CausalSelfAttention", 1)

text = text.replace("F.scaled_dot_product_attention(", "scaled_dot_product_attention_compat(")
path.write_text(text)
PY
}

patch_attention_compat "${REPO_COPY}/train_gpt.py"
patch_attention_compat "${REPO_COPY}/records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py"

run_job() {
  local run_id="$1"
  local script_path="$2"
  shift 2
  local log_path="${LOG_DIR}/${run_id}.txt"

  echo "=== Running ${run_id} ==="
  (
    cd "${REPO_COPY}"
    env \
      HF_HOME="${HF_HOME}" \
      PYTHONPATH="${SHIM_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      RUN_ID="${run_id}" \
      DATA_PATH="${DATA_PATH}" \
      TOKENIZER_PATH="${TOKENIZER_PATH}" \
      VOCAB_SIZE="${VOCAB_SIZE}" \
      VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
      TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY}" \
      MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
      "$@" \
      torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${script_path}"
  ) | tee "${log_path}"
  echo
}

extract_metric() {
  local label="$1"
  local log_path="$2"
  rg "${label}" "${log_path}" | tail -n 1 | sed "s/^.*${label}//"
}

run_job \
  "${OFFICIAL_RUN_ID}" \
  "records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py"

run_job \
  "${CURRENT_RUN_ID}" \
  "train_gpt.py"

printf '\n=== Summary ===\n'
for run_id in "${OFFICIAL_RUN_ID}" "${CURRENT_RUN_ID}"; do
  log_path="${LOG_DIR}/${run_id}.txt"
  printf '%s\n' "-- ${run_id} --"
  extract_metric "train_loader:dataset:" "${log_path}"
  extract_metric "layers:" "${log_path}" || true
  extract_metric "final_int8_zlib_roundtrip_exact " "${log_path}"
  extract_metric "Total submission size int8+zlib:" "${log_path}"
  printf '\n'
done
