#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/describe-anything}"
DATA_ROOT="${DATA_ROOT:-${OFFICIAL_REPO_ROOT}/DLC-bench}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
JOB_CPU="${JOB_CPU:-20}"
JOB_GPU="${JOB_GPU:-1}"
JOB_MEMORY="${JOB_MEMORY:-102400}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/opt/vlm/bin/python}"
LIMIT="${LIMIT:-}"
START="${START:-0}"
STEP="${STEP:-1}"
LLM_ENGINE="${LLM_ENGINE:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
LLM_ENGINE_PATH="${LLM_ENGINE_PATH:-}"
API_KEY_PATH="${API_KEY_PATH:-}"
DEFAULT_PREDICTION="${DEFAULT_PREDICTION:-}"
EVAL_SUFFIX="${EVAL_SUFFIX:-}"
VERBOSE="${VERBOSE:-0}"
QUIET="${QUIET:-0}"
CSV_ONLY="${CSV_ONLY:-0}"

usage() {
  cat <<EOF
Usage:
  bash tools/evaldlc_ckpt.sh --model-path PATH [options]

Required:
  --model-path PATH

Optional:
  --tokenizer-path PATH
  --official-repo-root PATH
  --data-root PATH
  --output-dir PATH
  --job-cpu N
  --job-gpu N
  --job-memory MB
  --cuda-devices LIST
  --device NAME
  --python-bin PATH
  --limit N
  --start N
  --step N
  --llm-engine NAME
  --llm-engine-path URL
  --api-key PATH
  --default-prediction TEXT
  --eval-suffix TEXT
  --verbose
  --quiet
  --csv-only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --tokenizer-path)
      TOKENIZER_PATH="$2"
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
    --job-cpu)
      JOB_CPU="$2"
      shift 2
      ;;
    --job-gpu)
      JOB_GPU="$2"
      shift 2
      ;;
    --job-memory)
      JOB_MEMORY="$2"
      shift 2
      ;;
    --cuda-devices)
      CUDA_DEVICES="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --start)
      START="$2"
      shift 2
      ;;
    --step)
      STEP="$2"
      shift 2
      ;;
    --llm-engine)
      LLM_ENGINE="$2"
      shift 2
      ;;
    --llm-engine-path)
      LLM_ENGINE_PATH="$2"
      shift 2
      ;;
    --api-key)
      API_KEY_PATH="$2"
      shift 2
      ;;
    --default-prediction)
      DEFAULT_PREDICTION="$2"
      shift 2
      ;;
    --eval-suffix)
      EVAL_SUFFIX="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    --csv-only)
      CSV_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "--model-path is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "${TOKENIZER_PATH}" ]]; then
  TOKENIZER_PATH="${MODEL_PATH}"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  model_name="$(basename "${MODEL_PATH}")"
  OUTPUT_DIR="${PROJECT_ROOT}/work_dirs/dlc_bench_eval_${model_name}"
fi

export PROJECT_ROOT MODEL_PATH TOKENIZER_PATH OFFICIAL_REPO_ROOT DATA_ROOT OUTPUT_DIR
export JOB_CPU JOB_GPU JOB_MEMORY CUDA_DEVICES DEVICE PYTHON_BIN
export LIMIT START STEP LLM_ENGINE LLM_ENGINE_PATH API_KEY_PATH DEFAULT_PREDICTION EVAL_SUFFIX
export VERBOSE QUIET CSV_ONLY

bash "${PROJECT_ROOT}/tools/evaldlc.sh"
