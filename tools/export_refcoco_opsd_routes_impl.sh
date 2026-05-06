#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_CONFIG="${SA2VA_REFCOCO_OPSD_CONFIG:?SA2VA_REFCOCO_OPSD_CONFIG must be set by the entry wrapper.}"
DEFAULT_WORK_DIR="${SA2VA_REFCOCO_OPSD_DEFAULT_WORK_DIR:?SA2VA_REFCOCO_OPSD_DEFAULT_WORK_DIR must be set by the entry wrapper.}"
DEFAULT_MODEL_PATH="${SA2VA_REFCOCO_OPSD_DEFAULT_MODEL_PATH:?SA2VA_REFCOCO_OPSD_DEFAULT_MODEL_PATH must be set by the entry wrapper.}"
DEFAULT_TOKENIZER_PATH="${SA2VA_REFCOCO_OPSD_DEFAULT_TOKENIZER_PATH:-${DEFAULT_MODEL_PATH}}"
MODEL_FLAVOR="${SA2VA_REFCOCO_OPSD_MODEL_FLAVOR:-custom}"
ENTRY_NAME="${SA2VA_REFCOCO_OPSD_ENTRY_NAME:-$(basename "$0")}"

CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-.venv/bin/activate}"
DATA_ROOT="${DATA_ROOT:-/data/xiaoyicheng/refcoco}"
IMAGE_ROOT="${IMAGE_ROOT:-/data/xiaoyicheng/refcoco/train2014}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-train}"
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${DEFAULT_TOKENIZER_PATH}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ROUTE_MODEL="${ROUTE_MODEL:-teacher}"
GLOBAL_STEP="${GLOBAL_STEP:-0}"
LIMIT="${LIMIT:-}"
DEVICE="${DEVICE:-}"

usage() {
  echo "Usage:"
  echo "  bash tools/${ENTRY_NAME} [options]"
  echo
  echo "Options:"
  echo "  --model-path PATH       Base model path. Default: ${DEFAULT_MODEL_PATH}"
  echo "  --tokenizer-path PATH   Tokenizer path. Default: ${DEFAULT_TOKENIZER_PATH}"
  echo "  --checkpoint PATH       Optional checkpoint used to estimate initial routes."
  echo "  --activate-script PATH  Activation script. Default: .venv/bin/activate"
  echo "  --data-root PATH        RefCOCO annotation root or its parent directory. Default: /data/xiaoyicheng/refcoco"
  echo "  --image-root PATH       train2014 image directory. Default: /data/xiaoyicheng/refcoco/train2014"
  echo "  --dataset NAME          refcoco | refcoco_plus | refcoco+ | refcocog. Default: refcoco"
  echo "  --split NAME            Dataset split. Default: train"
  echo "  --work-dir PATH         Output work dir. Default: ${DEFAULT_WORK_DIR}"
  echo "  --route-model NAME      teacher | student. Default: teacher"
  echo "  --global-step N         Recorded global step. Default: 0"
  echo "  --limit N               Optional sample cap for dry run."
  echo "  --device STR            Optional device override, e.g. cuda:0"
  echo "  -h, --help              Show this help."
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
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --activate-script)
      ACTIVATE_SCRIPT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --image-root)
      IMAGE_ROOT="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --route-model)
      ROUTE_MODEL="$2"
      shift 2
      ;;
    --global-step)
      GLOBAL_STEP="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
  echo "Activate script does not exist: ${ACTIVATE_SCRIPT}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ACTIVATE_SCRIPT}"

REFCOCO_ROOT="${DATA_ROOT}"
if [[ "$(basename "${DATA_ROOT}")" == "refcoco" ]]; then
  REFCOCO_ROOT="${DATA_ROOT}"
  DATA_ROOT="$(dirname "${DATA_ROOT}")"
else
  REFCOCO_ROOT="${DATA_ROOT}/refcoco"
fi

if [[ ! -d "${REFCOCO_ROOT}" ]]; then
  echo "Expected RefCOCO annotations under: ${REFCOCO_ROOT}" >&2
  echo "Current DATA_ROOT=${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${IMAGE_ROOT}" ]]; then
  echo "Image root does not exist: ${IMAGE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -d "${TOKENIZER_PATH}" ]]; then
  echo "Tokenizer path does not exist: ${TOKENIZER_PATH}" >&2
  exit 1
fi

if [[ -n "${CHECKPOINT_PATH}" && ! -e "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ "${ROUTE_MODEL}" != "teacher" && "${ROUTE_MODEL}" != "student" ]]; then
  echo "--route-model must be teacher or student, got: ${ROUTE_MODEL}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}/route_cache"
OUT_PATH="${WORK_DIR}/route_cache/routes_step_$(printf '%07d' "${GLOBAL_STEP}").jsonl"

CMD=(
  python tools/export_opsd_routes.py
  "${CONFIG}"
  --out "${OUT_PATH}"
  --route-model "${ROUTE_MODEL}"
  --global-step "${GLOBAL_STEP}"
  --image-root "${IMAGE_ROOT}"
  --update-latest
  --cfg-options
  "path=${MODEL_PATH}"
  "tokenizer_path=${TOKENIZER_PATH}"
  "model.model_path=${MODEL_PATH}"
  "model.tokenizer_path=${TOKENIZER_PATH}"
  "train_dataset.data_root=${DATA_ROOT}"
  "train_dataset.dataset_name=${DATASET}"
  "train_dataset.split=${SPLIT}"
  "train_dataset.image_root=${IMAGE_ROOT}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT_PATH}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ -n "${DEVICE}" ]]; then
  CMD+=(--device "${DEVICE}")
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "Exporting RefCOCO OPSD routes with:"
echo "  MODEL_FLAVOR=${MODEL_FLAVOR}"
echo "  ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT}"
echo "  PYTHON=$(command -v python)"
echo "  CONFIG=${CONFIG}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "  REFCOCO_ROOT=${REFCOCO_ROOT}"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  IMAGE_ROOT=${IMAGE_ROOT}"
echo "  DATASET=${DATASET}"
echo "  SPLIT=${SPLIT}"
echo "  WORK_DIR=${WORK_DIR}"
echo "  CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "  ROUTE_MODEL=${ROUTE_MODEL}"
echo "  GLOBAL_STEP=${GLOBAL_STEP}"
echo "  LIMIT=${LIMIT}"
echo "  DEVICE=${DEVICE}"
echo "  OUT_PATH=${OUT_PATH}"

"${CMD[@]}"
