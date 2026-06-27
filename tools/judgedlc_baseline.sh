#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/describe-anything}"
DATA_ROOT="${DATA_ROOT:-${OFFICIAL_REPO_ROOT}/DLC-bench}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/work_dirs/dlc_bench_eval_baseline_sa2va4b}"
PRED_OUTPUT="${PRED_OUTPUT:-${OUTPUT_DIR}/pred.json}"
WHEEL_DIR="${WHEEL_DIR:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/wheels_repo}"
LLM_ENGINE="${LLM_ENGINE:-gpt-4.1-mini}"
LLM_ENGINE_PATH="${LLM_ENGINE_PATH:-https://api.openai.com/v1}"
API_KEY="${API_KEY:-YOUR_OPENAI_API_KEY_HERE}"

export PROJECT_ROOT OFFICIAL_REPO_ROOT DATA_ROOT OUTPUT_DIR PRED_OUTPUT WHEEL_DIR
export LLM_ENGINE LLM_ENGINE_PATH API_KEY

bash "${PROJECT_ROOT}/tools/judgedlc.sh" "$@"
