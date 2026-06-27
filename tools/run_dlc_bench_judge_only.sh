#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PRED_OUTPUT=""
DATA_ROOT=""
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
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_INDEX_URL="${PIP_INDEX_URL:-http://mirrors.h.pjlab.org.cn/pypi/web/simple}"

usage() {
  cat <<EOF
Usage:
  bash tools/run_dlc_bench_judge_only.sh --pred-output PATH --data-root PATH --official-repo-root PATH [options]

Options:
  --pred-output PATH
  --data-root PATH
  --official-repo-root PATH
  --llm-engine NAME
  --llm-engine-path URL
  --llm-engine-kwargs TEXT
  --api-key PATH
  --default-prediction TEXT
  --eval-suffix TEXT
  --verbose
  --quiet
  --csv-only
  --python-bin PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pred-output)
      PRED_OUTPUT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
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

if [[ -z "${PRED_OUTPUT}" || -z "${DATA_ROOT}" || -z "${OFFICIAL_REPO_ROOT}" ]]; then
  echo "--pred-output, --data-root, and --official-repo-root are required." >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${PRED_OUTPUT}" ]]; then
  echo "Missing pred file: ${PRED_OUTPUT}" >&2
  exit 1
fi

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
    "${PYTHON_BIN}" -m pip install \
      -i "${PIP_INDEX_URL}" \
      --trusted-host "mirrors.h.pjlab.org.cn" \
      "${package_name}"
  fi
}

ensure_python_dep "inflect" "inflect"
ensure_python_dep "tqdm" "tqdm"
ensure_python_dep "openai" "openai"

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

echo "Running official NVlabs/describe-anything evaluation on an existing pred.json..."
"${EVAL_CMD[@]}"
