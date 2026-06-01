#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-}"
DATA_ROOT="${DATA_ROOT:-/data/xiaoyicheng/refcoco}"
IMAGE_ROOT="${IMAGE_ROOT:-/data/xiaoyicheng/refcoco/train2014}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-train}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/work_dirs/refcoco_sam_confuser_pool}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${ROOT_DIR}/pretrained/sam2/sam21L/sam2.1_hiera_large.pt}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
OVERWRITE="${OVERWRITE:-0}"
PYTHON_BIN=""

usage() {
  echo "Usage:"
  echo "  bash tools/conf.sh [options] [-- extra exporter args]"
  echo
  echo "Options:"
  echo "  --activate-script PATH   Optional activation script."
  echo "  --data-root PATH         RefCOCO root or its parent directory. Default: /data/xiaoyicheng/refcoco"
  echo "  --image-root PATH        train2014 image directory. Default: /data/xiaoyicheng/refcoco/train2014"
  echo "  --dataset NAME           refcoco | refcoco_plus | refcoco+ | refcocog. Default: refcoco"
  echo "  --split NAME             Dataset split. Default: train"
  echo "  --out-dir PATH           Export root. Default: ${ROOT_DIR}/work_dirs/refcoco_sam_confuser_pool"
  echo "  --sam2-config NAME       SAM2 Hydra config name or yaml path. Default: configs/sam2/sam2_hiera_l.yaml"
  echo "  --sam2-checkpoint PATH   SAM2 checkpoint path."
  echo "  --device NAME            Torch device, e.g. cuda:0 or cpu. Default: cuda:0"
  echo "  --limit N                Optional cap on unique images."
  echo "  --shard-index N          0-based shard index. Default: 0"
  echo "  --num-shards N           Total shard count. Default: 1"
  echo "  --overwrite              Overwrite existing per-image outputs."
  echo "  -h, --help               Show this help."
}

validate_positive_int() {
  local option_name="$1"
  local value="$2"

  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${option_name} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
}

validate_nonnegative_int() {
  local option_name="$1"
  local value="$2"

  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${option_name} must be a non-negative integer, got: ${value}" >&2
    exit 1
  fi
}

realign_venv_from_activate() {
  local activate_path="$1"
  local activate_dir
  local expected_venv

  activate_dir="$(cd "$(dirname "${activate_path}")" && pwd)"
  expected_venv="$(cd "${activate_dir}/.." && pwd)"

  export VIRTUAL_ENV="${expected_venv}"
  case ":${PATH}:" in
    *":${expected_venv}/bin:"*) ;;
    *)
      PATH="${expected_venv}/bin:${PATH}"
      export PATH
      ;;
  esac

  hash -r 2>/dev/null || true
}

resolve_python_bin() {
  local candidate
  local candidates=()
  local context_message="${1:-}"

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidates+=("${VIRTUAL_ENV}/bin/python" "${VIRTUAL_ENV}/bin/python3")
  fi
  if [[ -n "${SA2VA_PYTHON:-}" ]]; then
    candidates=("${SA2VA_PYTHON}" "${candidates[@]}")
  fi
  candidates+=("$(command -v python 2>/dev/null || true)")
  candidates+=("$(command -v python3 2>/dev/null || true)")

  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      PYTHON_BIN="${candidate}"
      export SA2VA_PYTHON="${PYTHON_BIN}"
      return 0
    fi
  done

  if [[ -n "${context_message}" ]]; then
    echo "${context_message}" >&2
  fi
  echo "No usable python interpreter found." >&2
  exit 127
}

validate_environment() {
  "${PYTHON_BIN}" - <<'PY'
import sys

errors = []

for module_name in ("torch", "PIL", "pycocotools", "hydra"):
    try:
        __import__(module_name)
    except Exception as exc:  # pragma: no cover - shell-time validation only
        errors.append(f"{module_name} import failed: {exc}")

if errors:
    raise SystemExit(
        "Active environment is missing RefCOCO SAM confuser export dependencies. "
        f"python={sys.executable}. "
        + " ".join(errors)
    )

print(f"conf export env OK: {sys.executable}")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --sam2-config)
      SAM2_CONFIG="$2"
      shift 2
      ;;
    --sam2-checkpoint)
      SAM2_CHECKPOINT="$2"
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
    --shard-index)
      SHARD_INDEX="$2"
      shift 2
      ;;
    --num-shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
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
validate_nonnegative_int "--shard-index" "${SHARD_INDEX}"
validate_positive_int "--num-shards" "${NUM_SHARDS}"
if (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "--shard-index must be smaller than --num-shards, got ${SHARD_INDEX} and ${NUM_SHARDS}." >&2
  exit 1
fi

if [[ -n "${ACTIVATE_SCRIPT}" ]]; then
  if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
    echo "Activate script does not exist: ${ACTIVATE_SCRIPT}" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
  realign_venv_from_activate "${ACTIVATE_SCRIPT}"
  resolve_python_bin "No usable python interpreter found after activating ${ACTIVATE_SCRIPT}."
else
  resolve_python_bin
fi

validate_environment

REFCOCO_ROOT="${DATA_ROOT}"
if [[ "$(basename "${DATA_ROOT}")" == "refcoco" ]]; then
  REFCOCO_ROOT="${DATA_ROOT}"
else
  REFCOCO_ROOT="${DATA_ROOT}/refcoco"
fi

if [[ ! -d "${REFCOCO_ROOT}" ]]; then
  echo "Expected RefCOCO annotations under: ${REFCOCO_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${IMAGE_ROOT}" ]]; then
  echo "Image root does not exist: ${IMAGE_ROOT}" >&2
  exit 1
fi

if [[ ! -e "${SAM2_CHECKPOINT}" ]]; then
  echo "SAM2 checkpoint does not exist: ${SAM2_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

EXPORT_ARGS=(
  --data-root "${DATA_ROOT}"
  --image-root "${IMAGE_ROOT}"
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --sam2-config "${SAM2_CONFIG}"
  --sam2-checkpoint "${SAM2_CHECKPOINT}"
  --out-dir "${OUT_DIR}"
  --device "${DEVICE}"
  --shard-index "${SHARD_INDEX}"
  --num-shards "${NUM_SHARDS}"
)

if [[ -n "${LIMIT}" ]]; then
  EXPORT_ARGS+=(--limit "${LIMIT}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  EXPORT_ARGS+=(--overwrite)
fi

echo "Exporting RefCOCO SAM confuser pool with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  REFCOCO_ROOT=${REFCOCO_ROOT}"
echo "  IMAGE_ROOT=${IMAGE_ROOT}"
echo "  DATASET=${DATASET}"
echo "  SPLIT=${SPLIT}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  SAM2_CONFIG=${SAM2_CONFIG}"
echo "  SAM2_CHECKPOINT=${SAM2_CHECKPOINT}"
echo "  DEVICE=${DEVICE}"
echo "  SHARD_INDEX=${SHARD_INDEX}"
echo "  NUM_SHARDS=${NUM_SHARDS}"
if [[ -n "${LIMIT}" ]]; then
  echo "  LIMIT=${LIMIT}"
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  echo "  OVERWRITE=1"
fi

"${PYTHON_BIN}" tools/export_refcoco_sam_confuser_pool.py "${EXPORT_ARGS[@]}" "${EXTRA_ARGS[@]}"
