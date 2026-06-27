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
LIMIT="${LIMIT:-}"
START="${START:-0}"
STEP="${STEP:-1}"
STUDENT_QUESTION="${STUDENT_QUESTION:-$'\nDescribe the masked region in detail.'}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<EOF
Usage:
  bash tools/run_dlc_bench_export_only.sh --model-path PATH --data-root PATH --pred-output PATH [options]

Options:
  --model-path PATH
  --tokenizer-path PATH
  --data-root PATH
  --pred-output PATH
  --debug-output PATH
  --device NAME               Default: cuda:0
  --limit N
  --start N                   Default: 0
  --step N                    Default: 1
  --student-question TEXT     Default: Describe the masked region in detail.
  --python-bin PATH           Default: python3
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

if [[ -z "${MODEL_PATH}" || -z "${DATA_ROOT}" || -z "${PRED_OUTPUT}" ]]; then
  echo "--model-path, --data-root, and --pred-output are required." >&2
  usage >&2
  exit 1
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"

EXPORT_CMD=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/tools/eval_dlc_bench_official.py"
  --model-path "${MODEL_PATH}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --data-root "${DATA_ROOT}"
  --output "${PRED_OUTPUT}"
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

echo "Generating official DLC-Bench prediction file only..."
"${EXPORT_CMD[@]}"
