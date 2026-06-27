#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/projects/sa2va/configs/refcoco_caption_to_mask_eval_4b_local.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA-4B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATA_ROOT}/train2014}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
JOB_CPU="${JOB_CPU:-20}"
JOB_GPU="${JOB_GPU:-1}"
JOB_MEMORY="${JOB_MEMORY:-102400}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/opt/vlm/bin/python}"
LIMIT="${LIMIT:-}"
INSTALL_SYSTEM_LIBS="${INSTALL_SYSTEM_LIBS:-1}"

usage() {
  cat <<EOF
Usage:
  bash tools/evalrefcoco_pth.sh --checkpoint PATH [options]

Required:
  --checkpoint PATH

Optional:
  --config PATH
  --base-model-path PATH
  --tokenizer-path PATH
  --data-root PATH
  --image-root PATH
  --output PATH
  --job-cpu N
  --job-gpu N
  --job-memory MB
  --cuda-devices LIST
  --device NAME
  --python-bin PATH
  --limit N
  --install-system-libs 0|1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --base-model-path)
      BASE_MODEL_PATH="$2"
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
    --install-system-libs)
      INSTALL_SYSTEM_LIBS="$2"
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

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "--checkpoint is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "${TOKENIZER_PATH}" ]]; then
  TOKENIZER_PATH="${BASE_MODEL_PATH}"
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
  ckpt_name="$(basename "${CHECKPOINT_PATH}")"
  ckpt_stem="${ckpt_name%.*}"
  OUTPUT_PATH="${PROJECT_ROOT}/work_dirs/refcoco_caption_to_mask_eval_${ckpt_stem}.json"
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
  PROJECT_ROOT="${PROJECT_ROOT}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  DATA_ROOT="${DATA_ROOT}" \
  IMAGE_ROOT="${IMAGE_ROOT}" \
  OUTPUT_PATH="${OUTPUT_PATH}" \
  DEVICE="${DEVICE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  LIMIT="${LIMIT}" \
  INSTALL_SYSTEM_LIBS="${INSTALL_SYSTEM_LIBS}" \
  bash -lc '
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?}"
CONFIG_PATH="${CONFIG_PATH:?}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?}"
TOKENIZER_PATH="${TOKENIZER_PATH:?}"
DATA_ROOT="${DATA_ROOT:?}"
IMAGE_ROOT="${IMAGE_ROOT:?}"
OUTPUT_PATH="${OUTPUT_PATH:?}"
DEVICE="${DEVICE:?}"
PYTHON_BIN="${PYTHON_BIN:?}"
INSTALL_SYSTEM_LIBS="${INSTALL_SYSTEM_LIBS:-1}"
LOG_FILE="$(dirname "${OUTPUT_PATH}")/eval_refcoco_pth.log"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
: >"${LOG_FILE}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export SA2VA_REFCOCO_EVAL_DATA_ROOT="${DATA_ROOT}"
export SA2VA_REFCOCO_EVAL_IMAGE_ROOT="${IMAGE_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd /opt
tar -xzf vlm_env.tar.gz -C /opt/vlm
rm vlm_env.tar.gz
/opt/vlm/bin/python /opt/vlm/bin/conda-unpack
/opt/vlm/bin/python -c "import sys; print(sys.version)"
if [[ "${INSTALL_SYSTEM_LIBS}" == "1" ]]; then
  cat > /etc/apt/sources.list <<EOF
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-security main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-backports main restricted universe multiverse
EOF
  apt update
  apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
fi
/opt/vlm/bin/python -c "import cv2; import torch; from transformers import PreTrainedModel; print(\"ok\")"

cd "${PROJECT_ROOT}"

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/tools/eval_refcoco_caption_to_mask_pth.py"
  --config "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --base-model-path "${BASE_MODEL_PATH}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --image-root "${IMAGE_ROOT}"
  --output "${OUTPUT_PATH}"
  --device "${DEVICE}"
)

if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

stdbuf -oL -eL "${CMD[@]}"
'
