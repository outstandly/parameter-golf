#!/usr/bin/env bash
set -euo pipefail

REPO_SRC="${REPO_SRC:-/tmp/pg-src}"
WORKROOT="${WORKROOT:-/tmp/pg-current-run}"
REPO_COPY="${WORKROOT}/repo"
LOG_DIR="${WORKROOT}/logs"
DATA_PATH="${DATA_PATH:-/workspace/parameter-golf/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/pg-tokenizers/fineweb_1024_bpe.model}"
HF_HOME="${HF_HOME:-/tmp/hf}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
RUN_ID="${RUN_ID:-current_1xh100_run}"

mkdir -p "${WORKROOT}" "${LOG_DIR}" "${HF_HOME}"
rm -rf "${REPO_COPY}"

git clone --local "${REPO_SRC}" "${REPO_COPY}" >/dev/null 2>&1 || git clone "${REPO_SRC}" "${REPO_COPY}" >/dev/null 2>&1

python3 - "${REPO_COPY}/train_gpt.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = """        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )"""

new = """        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
        )"""

if old not in text:
    raise SystemExit(f"Could not find attention pattern in {path}")
text = text.replace(old, new, 1)
path.write_text(text)
PY

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

echo
echo "=== Summary ${RUN_ID} ==="
grep "train_loader:dataset:" "${log_path}" | tail -n 1 || true
grep "layers:" "${log_path}" | tail -n 1 || true
grep "final_int8_zlib_roundtrip_exact" "${log_path}" | tail -n 1 || true
grep "Total submission size int8+zlib:" "${log_path}" | tail -n 1 || true
