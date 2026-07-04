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
RESUME_ATTEMPTS="${RESUME_ATTEMPTS:-8}"
RESUME_SLEEP_SECONDS="${RESUME_SLEEP_SECONDS:-3}"
VERBOSE="${VERBOSE:-0}"
QUIET="${QUIET:-0}"
CSV_ONLY="${CSV_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_INDEX_URL="${PIP_INDEX_URL:-http://mirrors.h.pjlab.org.cn/pypi/web/simple}"
WHEEL_DIR="${WHEEL_DIR:-}"

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
  --resume-attempts N
  --resume-sleep-seconds N
  --verbose
  --quiet
  --csv-only
  --python-bin PATH
  --wheel-dir PATH
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
    --resume-attempts)
      RESUME_ATTEMPTS="$2"
      shift 2
      ;;
    --resume-sleep-seconds)
      RESUME_SLEEP_SECONDS="$2"
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

if [[ -z "${PRED_OUTPUT}" || -z "${DATA_ROOT}" || -z "${OFFICIAL_REPO_ROOT}" ]]; then
  echo "--pred-output, --data-root, and --official-repo-root are required." >&2
  usage >&2
  exit 1
fi

if [[ -n "${WHEEL_DIR}" && ! -d "${WHEEL_DIR}" ]]; then
  echo "Missing wheel directory: ${WHEEL_DIR}" >&2
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

ensure_python_deps() {
  local missing=0
  local dep_specs=("inflect" "typeguard" "more_itertools" "tqdm" "openai")

  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1 || missing=1
import importlib
required = ["inflect", "typeguard", "more_itertools", "tqdm", "openai"]
for name in required:
    importlib.import_module(name)
PY

  if [[ "${missing}" != "1" ]]; then
    return 0
  fi

  echo "Installing missing evaluation dependencies..." >&2
  if [[ -n "${WHEEL_DIR}" ]]; then
    "${PYTHON_BIN}" -m pip install \
      --no-index \
      --find-links "${WHEEL_DIR}" \
      "${dep_specs[@]}"
  else
    "${PYTHON_BIN}" -m pip install \
      -i "${PIP_INDEX_URL}" \
      --trusted-host "mirrors.h.pjlab.org.cn" \
      "${dep_specs[@]}"
  fi
}

ensure_python_deps

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

EVAL_FILE="${PRED_OUTPUT%.*}_eval${EVAL_SUFFIX}.json"

count_completed_items() {
  if [[ ! -f "${EVAL_FILE}" ]]; then
    echo 0
    return 0
  fi
  "${PYTHON_BIN}" - "${EVAL_FILE}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
done = sum(1 for value in data.values() if isinstance(value, dict) and "score_pos" in value)
print(done)
PY
}

count_total_items() {
  "${PYTHON_BIN}" - "${DATA_ROOT}/qa.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data))
PY
}

TOTAL_ITEMS="$(count_total_items)"
ATTEMPT=1

while (( ATTEMPT <= RESUME_ATTEMPTS )); do
  CURRENT_DONE="$(count_completed_items)"
  if (( CURRENT_DONE >= TOTAL_ITEMS )); then
    echo "Evaluation already complete: ${CURRENT_DONE}/${TOTAL_ITEMS} items in ${EVAL_FILE}"
    break
  fi

  echo "Running official NVlabs/describe-anything evaluation on an existing pred.json..."
  echo "Resume attempt ${ATTEMPT}/${RESUME_ATTEMPTS}: ${CURRENT_DONE}/${TOTAL_ITEMS} completed"
  set +e
  "${EVAL_CMD[@]}"
  CMD_EXIT=$?
  set -e

  NEW_DONE="$(count_completed_items)"
  echo "Attempt ${ATTEMPT} finished with exit code ${CMD_EXIT}; progress is now ${NEW_DONE}/${TOTAL_ITEMS}"

  if (( NEW_DONE >= TOTAL_ITEMS )); then
    break
  fi

  if (( ATTEMPT >= RESUME_ATTEMPTS )); then
    echo "Evaluation still incomplete after ${RESUME_ATTEMPTS} attempts: ${NEW_DONE}/${TOTAL_ITEMS}" >&2
    exit 1
  fi

  sleep "${RESUME_SLEEP_SECONDS}"
  ATTEMPT=$((ATTEMPT + 1))
done
