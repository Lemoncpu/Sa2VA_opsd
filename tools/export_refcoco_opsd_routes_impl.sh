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
GPUS="${GPUS:-}"
PORT="${PORT:-$((29500 + RANDOM % 1000))}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
CUDA_DEVICE_IDS="${CUDA_VISIBLE_DEVICES:-}"

DATA_ROOT="${DATA_ROOT:-/data/xiaoyicheng/refcoco}"
IMAGE_ROOT="${IMAGE_ROOT:-/data/xiaoyicheng/refcoco/train2014}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-train}"
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${DEFAULT_TOKENIZER_PATH}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ROUTE_MODEL="${ROUTE_MODEL:-student}"
GLOBAL_STEP="${GLOBAL_STEP:-0}"
LIMIT="${LIMIT:-}"
ONLY_MISSING_FROM_MANIFEST="${ONLY_MISSING_FROM_MANIFEST:-0}"
DEEPSPEED="${DEEPSPEED:-deepspeed_zero2}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"
DEFAULT_GPUS=8

count_csv_items() {
  local csv="${1// /}"
  local count=0
  local rest
  local item

  if [[ -z "${csv}" ]]; then
    echo 0
    return
  fi

  rest="${csv},"
  while [[ -n "${rest}" ]]; do
    item="${rest%%,*}"
    rest="${rest#*,}"
    count=$((count + 1))
  done

  echo "${count}"
}

build_cuda_device_ids() {
  local gpu_count="$1"
  local ids=()
  local idx

  for ((idx = 0; idx < gpu_count; idx++)); do
    ids+=("${idx}")
  done

  (
    IFS=','
    echo "${ids[*]}"
  )
}

validate_gpu_count() {
  local gpu_count="$1"

  if [[ ! "${gpu_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU count must be a positive integer, got: ${gpu_count}" >&2
    exit 1
  fi
}

validate_positive_int() {
  local option_name="$1"
  local value="$2"

  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${option_name} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
}

validate_cuda_device_ids() {
  local csv="${1// /}"
  local rest
  local item
  declare -A seen=()

  if [[ -z "${csv}" ]]; then
    echo "CUDA device ids cannot be empty." >&2
    exit 1
  fi

  rest="${csv},"
  while [[ -n "${rest}" ]]; do
    item="${rest%%,*}"
    rest="${rest#*,}"

    if [[ ! "${item}" =~ ^[0-9]+$ ]]; then
      echo "Invalid CUDA device id: ${item}. Use comma-separated integers like 0,1,3." >&2
      exit 1
    fi
    if [[ -n "${seen[${item}]:-}" ]]; then
      echo "Duplicate CUDA device id: ${item}" >&2
      exit 1
    fi
    seen["${item}"]=1
  done
}

usage() {
  echo "Usage:"
  echo "  bash tools/${ENTRY_NAME} [options] [-- extra export args]"
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
  echo "  --gpus N                Number of GPUs. If omitted, infer from --cuda-devices."
  echo "  --cuda-count N          Alias of --gpus."
  echo "  --cuda-devices IDS      Comma-separated CUDA ids, e.g. 0,1,3. If omitted, uses 0..N-1."
  echo "  --port N                torchrun master port. Default: random port in [29500, 30499]"
  echo "  --deepspeed NAME        Compatibility arg forwarded through tools/dist.sh. Default: deepspeed_zero2"
  echo "  --batch-size N          Override per-device route export batch size."
  echo "  --route-model NAME      teacher | student. Default: student"
  echo "  --global-step N         Recorded global step. Default: 0"
  echo "  --limit N               Optional sample cap."
  echo "  --only-missing          Only export sample_keys missing from the existing manifest, then merge them back."
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
    --gpus|--cuda-count)
      GPUS="$2"
      shift 2
      ;;
    --cuda-devices|--gpu-ids)
      CUDA_DEVICE_IDS="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --deepspeed)
      DEEPSPEED="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE_OVERRIDE="$2"
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
    --only-missing|--only-missing-from-manifest)
      ONLY_MISSING_FROM_MANIFEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

EXTRA_ARGS=("$@")

if [[ -n "${LIMIT}" ]]; then
  validate_positive_int "--limit" "${LIMIT}"
fi
if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
  validate_positive_int "--batch-size" "${BATCH_SIZE_OVERRIDE}"
fi

if [[ -n "${CUDA_DEVICE_IDS}" ]]; then
  CUDA_DEVICE_IDS="${CUDA_DEVICE_IDS// /}"
  validate_cuda_device_ids "${CUDA_DEVICE_IDS}"
fi

if [[ -z "${GPUS}" && -z "${CUDA_DEVICE_IDS}" ]]; then
  GPUS=1
  CUDA_DEVICE_IDS="$(build_cuda_device_ids "${GPUS}")"
elif [[ -n "${CUDA_DEVICE_IDS}" ]]; then
  CUDA_DEVICE_COUNT="$(count_csv_items "${CUDA_DEVICE_IDS}")"
  if [[ -z "${GPUS}" ]]; then
    GPUS="${CUDA_DEVICE_COUNT}"
  fi
fi

validate_gpu_count "${GPUS}"

if [[ -z "${CUDA_DEVICE_IDS}" ]]; then
  CUDA_DEVICE_IDS="$(build_cuda_device_ids "${GPUS}")"
fi

CUDA_DEVICE_COUNT="$(count_csv_items "${CUDA_DEVICE_IDS}")"
if [[ "${GPUS}" -ne "${CUDA_DEVICE_COUNT}" ]]; then
  echo "--gpus/--cuda-count (${GPUS}) does not match --cuda-devices count (${CUDA_DEVICE_COUNT})." >&2
  exit 1
fi

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

EFFECTIVE_BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-1}"

mkdir -p "${WORK_DIR}/route_cache"
OUT_PATH="${WORK_DIR}/route_cache/routes_step_$(printf '%07d' "${GLOBAL_STEP}").jsonl"

EXPORT_ARGS=(
  --out "${OUT_PATH}"
  --batch-size "${EFFECTIVE_BATCH_SIZE}"
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
  EXPORT_ARGS+=(--checkpoint "${CHECKPOINT_PATH}")
fi
if [[ -n "${LIMIT}" ]]; then
  EXPORT_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${ONLY_MISSING_FROM_MANIFEST}" == "1" ]]; then
  EXPORT_ARGS+=(--only-missing-from-manifest)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  EXPORT_ARGS+=("${EXTRA_ARGS[@]}")
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
echo "  BATCH_SIZE_OVERRIDE=${BATCH_SIZE_OVERRIDE}"
echo "  EFFECTIVE_BATCH_SIZE=${EFFECTIVE_BATCH_SIZE}"
echo "  ONLY_MISSING_FROM_MANIFEST=${ONLY_MISSING_FROM_MANIFEST}"
echo "  GPUS=${GPUS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_IDS}"
echo "  PORT=${PORT}"
echo "  DEEPSPEED=${DEEPSPEED}"
echo "  OUT_PATH=${OUT_PATH}"

if [[ "${GPUS}" -eq 1 ]]; then
  PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_IDS}" \
  python tools/export_opsd_routes.py "${CONFIG}" "${EXPORT_ARGS[@]}"
else
  PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_IDS}" \
  PORT="${PORT}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  DEEPSPEED="${DEEPSPEED}" \
  bash tools/dist.sh export_opsd_routes "${CONFIG}" "${GPUS}" "${EXPORT_ARGS[@]}"
fi
