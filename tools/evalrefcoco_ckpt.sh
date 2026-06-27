#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATA_ROOT}/train2014}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/projects/sa2va/configs/refcoco_caption_to_mask_eval_4b_local.py}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
JOB_CPU="${JOB_CPU:-20}"
JOB_GPU="${JOB_GPU:-1}"
JOB_MEMORY="${JOB_MEMORY:-102400}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/opt/vlm/bin/python}"
LIMIT="${LIMIT:-}"

usage() {
  cat <<EOF
Usage:
  bash tools/evalrefcoco_ckpt.sh --model-path PATH [options]

Required:
  --model-path PATH

Optional:
  --tokenizer-path PATH
  --data-root PATH
  --image-root PATH
  --config PATH
  --output PATH
  --job-cpu N
  --job-gpu N
  --job-memory MB
  --cuda-devices LIST
  --device NAME
  --python-bin PATH
  --limit N
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
    --image-root)
      IMAGE_ROOT="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
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

if [[ -z "${OUTPUT_PATH}" ]]; then
  model_name="$(basename "${MODEL_PATH}")"
  OUTPUT_PATH="${PROJECT_ROOT}/work_dirs/refcoco_caption_to_mask_eval_${model_name}.json"
fi

rjob submit \
  --cpu="${JOB_CPU}" \
  --gpu="${JOB_GPU}" \
  --memory="${JOB_MEMORY}" \
  --charged-group=ai4ls_gpu \
  --private-machine=group \
  --mount=gpfs://gpfs1/dnacoding:/mnt/shared-storage-user/dnacoding \
  --mount=gpfs://gpfs1/wuyucheng:/mnt/shared-storage-user/wuyucheng \
  --image registry.h.pjlab.org.cn/ailab-dnacoding/wuyucheng:test1 \
  --custom-resources brainpp.cn/fuse=1 \
  --enable-sshd \
-- env \
  JOB_GPU="${JOB_GPU}" \
  CUDA_DEVICES="${CUDA_DEVICES}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  MODEL_PATH="${MODEL_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  DATA_ROOT="${DATA_ROOT}" \
  IMAGE_ROOT="${IMAGE_ROOT}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  OUTPUT_PATH="${OUTPUT_PATH}" \
  DEVICE="${DEVICE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  LIMIT="${LIMIT}" \
  bash -lc '
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?}"
MODEL_PATH="${MODEL_PATH:?}"
TOKENIZER_PATH="${TOKENIZER_PATH:?}"
DATA_ROOT="${DATA_ROOT:?}"
IMAGE_ROOT="${IMAGE_ROOT:?}"
CONFIG_PATH="${CONFIG_PATH:?}"
OUTPUT_PATH="${OUTPUT_PATH:?}"
DEVICE="${DEVICE:?}"
PYTHON_BIN="${PYTHON_BIN:?}"
JOB_GPU="${JOB_GPU:?}"
LOG_DIR="$(dirname "${OUTPUT_PATH}")"
LOG_FILE="${LOG_DIR}/eval_refcoco_${JOB_GPU}gpu.log"

mkdir -p "${LOG_DIR}"
: >"${LOG_FILE}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export SA2VA_REFCOCO_EVAL_MODEL_PATH="${MODEL_PATH}"
export SA2VA_REFCOCO_EVAL_TOKENIZER_PATH="${TOKENIZER_PATH}"
export SA2VA_REFCOCO_EVAL_DATA_ROOT="${DATA_ROOT}"
export SA2VA_REFCOCO_EVAL_IMAGE_ROOT="${IMAGE_ROOT}"
export SA2VA_REFCOCO_EVAL_DEVICE="${DEVICE}"
export SA2VA_REFCOCO_EVAL_OUTPUT="${OUTPUT_PATH}"
if [[ -n "${LIMIT:-}" ]]; then
  export SA2VA_REFCOCO_EVAL_LIMIT="${LIMIT}"
fi

cd /opt
tar -xzf vlm_env.tar.gz -C /opt/vlm
rm vlm_env.tar.gz
/opt/vlm/bin/python /opt/vlm/bin/conda-unpack

cat > /etc/apt/sources.list <<EOF
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-security main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-backports main restricted universe multiverse
EOF

apt update
apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
/opt/vlm/bin/python -c "import torch, transformers; print(\"ok\")"

cd "${PROJECT_ROOT}"

EVAL_CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/tools/eval_refcoco_caption_to_mask.py"
  --config "${CONFIG_PATH}"
  --image-root "${IMAGE_ROOT}"
  --output "${OUTPUT_PATH}"
  --device "${DEVICE}"
)

if [[ -n "${LIMIT:-}" ]]; then
  EVAL_CMD+=(--limit "${LIMIT}")
fi

stdbuf -oL -eL "${EVAL_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
'
