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
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.h.pjlab.org.cn/pypi/web/simple}"

usage() {
  echo "Usage:"
  echo "  bash tools/run_dlc_bench_official_eval.sh --model-path PATH --data-root PATH --pred-output PATH --official-repo-root PATH [options]"
  echo
  echo "Options:"
  echo "  --model-path PATH"
  echo "  --tokenizer-path PATH"
  echo "  --data-root PATH"
  echo "  --pred-output PATH"
  echo "  --debug-output PATH"
  echo "  --device NAME               Default: cuda:0"
  echo "  --official-repo-root PATH   Path to cloned NVlabs/describe-anything repo."
  echo "  --llm-engine NAME           Passed as --model to official eval."
  echo "  --llm-engine-path URL       Passed as --base-url to official eval."
  echo "  --llm-engine-kwargs TEXT    Accepted for compatibility and ignored by the official script."
  echo "  --api-key PATH              Passed to official eval."
  echo "  --default-prediction TEXT   Passed to official eval."
  echo "  --eval-suffix TEXT          Passed to official eval."
  echo "  --verbose                   Passed to official eval."
  echo "  --quiet                     Passed to official eval."
  echo "  --csv-only                  Passed to official eval."
  echo "  --limit N"
  echo "  --start N                   Default: 0"
  echo "  --step N                    Default: 1"
  echo "  --student-question TEXT     Default: Describe the masked region in detail."
  echo "  --python-bin PATH           Default: python3"
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

if [[ ! -d "${OFFICIAL_REPO_ROOT}" ]]; then
  echo "Official repo root does not exist: ${OFFICIAL_REPO_ROOT}" >&2
  exit 1
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
EVAL_SCRIPT="${OFFICIAL_REPO_ROOT}/evaluation/eval_model_outputs.py"
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Missing official eval script: ${EVAL_SCRIPT}" >&2
  exit 1
fi

ensure_python_dep() {
  local module_name="$1"
  local package_name="${2:-$1}"
  if ! "${PYTHON_BIN}" -c "import ${module_name}" >/dev/null 2>&1; then
    echo "Installing missing evaluation dependency: ${package_name}" >&2
    "${PYTHON_BIN}" -m pip install -i "${PIP_INDEX_URL}" "${package_name}"
  fi
}

ensure_python_dep "inflect" "inflect"
ensure_python_dep "tqdm" "tqdm"
ensure_python_dep "openai" "openai"

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

echo "Generating official DLC-Bench prediction file..."
"${EXPORT_CMD[@]}"

EVAL_CMD=(
  "${PYTHON_BIN}"
  "${EVAL_SCRIPT}"
  --pred "${PRED_OUTPUT}"
  --qa "${DATA_ROOT}/qa.json"
  --class-names "${DATA_ROOT}/class_names.json"
  --model "${LLM_ENGINE}"
)
if [[ -n "${LLM_ENGINE_PATH}" ]]; then
  EVAL_CMD+=(--base-url "${LLM_ENGINE_PATH}")
fi
if [[ -n "${API_KEY_PATH}" ]]; then
  EVAL_CMD+=(--api-key "${API_KEY_PATH}")
fi
if [[ -n "${DEFAULT_PREDICTION}" ]]; then
  EVAL_CMD+=(--default-prediction "${DEFAULT_PREDICTION}")
fi
if [[ -n "${EVAL_SUFFIX}" ]]; then
  EVAL_CMD+=(--suffix "${EVAL_SUFFIX}")
fi
if [[ "${VERBOSE}" == "1" ]]; then
  EVAL_CMD+=(--verbose)
fi
if [[ "${QUIET}" == "1" ]]; then
  EVAL_CMD+=(--quiet)
fi
if [[ "${CSV_ONLY}" == "1" ]]; then
  EVAL_CMD+=(--csv)
fi
if [[ -n "${LLM_ENGINE_KWARGS}" ]]; then
  echo "Ignoring --llm-engine-kwargs because official eval_model_outputs.py does not accept it." >&2
fi

echo "Running official NVlabs/describe-anything evaluation..."
"${EVAL_CMD[@]}"
