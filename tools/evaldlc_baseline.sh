#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA-4B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/work_dirs/dlc_bench_eval_baseline_sa2va4b}"

export PROJECT_ROOT MODEL_PATH TOKENIZER_PATH OUTPUT_DIR

bash "${PROJECT_ROOT}/tools/evaldlc_ckpt.sh" "$@"
