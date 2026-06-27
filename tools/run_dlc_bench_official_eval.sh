#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH=""
TOKENIZER_PATH=""
DATA_ROOT=""
PRED_OUTPUT=""
DEBUG_OUTPUT=""
DEVICE="${DEVICE:-cuda:0}"
OFFICIAL_REPO_ROOT=""
LLM_ENGINE="${LLM_ENGINE:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
LLM_ENGINE_PATH="${LLM_ENGINE_PATH:-}"
LLM_ENGINE_KWARGS="${LLM_ENGINE_KWARGS:-}"
API_KEY_PATH="${API_KEY_PATH:-}"
DEFAULT_PREDICTION="${DEFAULT_PREDICTION:-}"
EVAL_SUFFIX="${EVAL_SUFFIX:-}"
VERBOSE="${VERBOSE:-0}"
QUIET="${QUIET:-0}"
CSV_ONLY="${CSV_ONLY:-0}"
LIMIT="${LIMIT:-}"
START="${START:-0}"
STEP="${STEP:-1}"
STUDENT_QUESTION="${STUDENT_QUESTION:-$'\nDescribe the masked region in detail.'}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_INDEX_URL="${PIP_INDEX_URL:-http://mirrors.h.pjlab.org.cn/pypi/web/simple}"
WHEEL_DIR="${WHEEL_DIR:-}"

usage() {
  cat <<EOF
Usage:
  bash tools/run_dlc_bench_official_eval.sh --model-path PATH --data-root PATH --pred-output PATH --official-repo-root PATH [options]

This is a convenience wrapper that runs:
  1. export only
  2. judge only

Judge dependency options:
  --wheel-dir PATH           Offline wheel directory for judge dependencies
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
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --pred-output)
      PRED_OUTPUT="$2"
      shift 2
      ;;
    --debug-output)
      DEBUG_OUTPUT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --official-repo-root)
      OFFICIAL_REPO_ROOT="$2"
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
    --llm-engine-kwargs)
      LLM_ENGINE_KWARGS="$2"
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
    --student-question)
      STUDENT_QUESTION="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
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
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" || -z "${DATA_ROOT}" || -z "${PRED_OUTPUT}" || -z "${OFFICIAL_REPO_ROOT}" ]]; then
  echo "--model-path, --data-root, --pred-output, and --official-repo-root are required." >&2
  usage >&2
  exit 1
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"

EXPORT_CMD=(
  bash "${ROOT_DIR}/tools/run_dlc_bench_export_only.sh"
  --python-bin "${PYTHON_BIN}"
  --model-path "${MODEL_PATH}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --data-root "${DATA_ROOT}"
  --pred-output "${PRED_OUTPUT}"
  --device "${DEVICE}"
  --start "${START}"
  --step "${STEP}"
  --student-question "${STUDENT_QUESTION}"
)
if [[ -n "${LIMIT}" ]]; then
  EXPORT_CMD+=(--limit "${LIMIT}")
fi
if [[ -n "${DEBUG_OUTPUT}" ]]; then
  EXPORT_CMD+=(--debug-output "${DEBUG_OUTPUT}")
fi

JUDGE_CMD=(
  bash "${ROOT_DIR}/tools/run_dlc_bench_judge_only.sh"
  --python-bin "${PYTHON_BIN}"
  --pred-output "${PRED_OUTPUT}"
  --data-root "${DATA_ROOT}"
  --official-repo-root "${OFFICIAL_REPO_ROOT}"
  --llm-engine "${LLM_ENGINE}"
)
if [[ -n "${WHEEL_DIR}" ]]; then
  JUDGE_CMD+=(--wheel-dir "${WHEEL_DIR}")
fi
if [[ -n "${LLM_ENGINE_PATH}" ]]; then
  JUDGE_CMD+=(--llm-engine-path "${LLM_ENGINE_PATH}")
fi
if [[ -n "${API_KEY_PATH}" ]]; then
  JUDGE_CMD+=(--api-key "${API_KEY_PATH}")
fi
if [[ -n "${DEFAULT_PREDICTION}" ]]; then
  JUDGE_CMD+=(--default-prediction "${DEFAULT_PREDICTION}")
fi
if [[ -n "${EVAL_SUFFIX}" ]]; then
  JUDGE_CMD+=(--eval-suffix "${EVAL_SUFFIX}")
fi
if [[ "${VERBOSE}" == "1" ]]; then
  JUDGE_CMD+=(--verbose)
fi
if [[ "${QUIET}" == "1" ]]; then
  JUDGE_CMD+=(--quiet)
fi
if [[ "${CSV_ONLY}" == "1" ]]; then
  JUDGE_CMD+=(--csv-only)
fi
if [[ -n "${LLM_ENGINE_KWARGS}" ]]; then
  JUDGE_CMD+=(--llm-engine-kwargs "${LLM_ENGINE_KWARGS}")
fi

echo "Step 1/2: export pred.json"
"${EXPORT_CMD[@]}"

echo "Step 2/2: run official judge"
"${JUDGE_CMD[@]}"
