#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
MODEL_PATH="${MODEL_PATH:-}"
OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/describe-anything}"
DATA_ROOT="${DATA_ROOT:-${OFFICIAL_REPO_ROOT}/DLC-bench}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PRED_OUTPUT="${PRED_OUTPUT:-}"
WHEEL_DIR="${WHEEL_DIR:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/wheels_repo}"
LLM_ENGINE="${LLM_ENGINE:-gpt-4.1-mini}"
LLM_ENGINE_PATH="${LLM_ENGINE_PATH:-https://api.openai.com/v1}"
API_KEY="${API_KEY:-YOUR_OPENAI_API_KEY_HERE}"

usage() {
  cat <<EOF
Usage:
  bash tools/judgedlc_ckpt.sh --model-path PATH [options]

Required:
  --model-path PATH

Optional:
  --official-repo-root PATH
  --data-root PATH
  --output-dir PATH
  --pred-output PATH
  --wheel-dir PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --official-repo-root)
      OFFICIAL_REPO_ROOT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --pred-output)
      PRED_OUTPUT="$2"
      shift 2
      ;;
    --wheel-dir)
      WHEEL_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "--model-path is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  model_name="$(basename "${MODEL_PATH}")"
  OUTPUT_DIR="${PROJECT_ROOT}/work_dirs/dlc_bench_eval_${model_name}"
fi

if [[ -z "${PRED_OUTPUT}" ]]; then
  PRED_OUTPUT="${OUTPUT_DIR}/pred.json"
fi

export PROJECT_ROOT OFFICIAL_REPO_ROOT DATA_ROOT OUTPUT_DIR PRED_OUTPUT WHEEL_DIR
export LLM_ENGINE LLM_ENGINE_PATH API_KEY

bash "${PROJECT_ROOT}/tools/judgedlc.sh" "$@"
