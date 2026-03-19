#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_variant() {
  local run_id="$1"
  shift
  echo
  echo "=== Variant ${run_id} ==="
  RUN_ID="${run_id}" "${SCRIPT_DIR}/runpod_train_current_1xh100.sh" "$@"
}

run_variant "cur_muonwarm100" MUON_MOMENTUM_WARMUP_STEPS=100
run_variant "cur_quant32768_clip999995" INT8_KEEP_FLOAT_MAX_NUMEL=32768 INT8_CLIP_PERCENTILE=99.9995
run_variant "cur_muonwarm100_quant32768_clip999995" MUON_MOMENTUM_WARMUP_STEPS=100 INT8_KEEP_FLOAT_MAX_NUMEL=32768 INT8_CLIP_PERCENTILE=99.9995
