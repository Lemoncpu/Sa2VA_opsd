#!/usr/bin/env bash

set -euo pipefail

JOB_CPU="${JOB_CPU:-80}"
JOB_GPU="${JOB_GPU:-4}"
JOB_MEMORY="${JOB_MEMORY:-409600}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATA_ROOT}/train2014}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA-4B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/work_dirs/sa2va_opsd_refcoco_internvl3_4b_v3_manifest}"
RESUME_PATH="${RESUME_PATH:-}"

if [[ -z "${RESUME_PATH}" ]]; then
  echo "RESUME_PATH is required for tools/train_resume.sh" >&2
  exit 1
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
  DATA_ROOT="${DATA_ROOT}" \
  IMAGE_ROOT="${IMAGE_ROOT}" \
  MODEL_PATH="${MODEL_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  WORK_DIR="${WORK_DIR}" \
  RESUME_PATH="${RESUME_PATH}" \
  bash -lc '
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?}"
DATA_ROOT="${DATA_ROOT:?}"
IMAGE_ROOT="${IMAGE_ROOT:?}"
MODEL_PATH="${MODEL_PATH:?}"
TOKENIZER_PATH="${TOKENIZER_PATH:?}"
WORK_DIR="${WORK_DIR:?}"
RESUME_PATH="${RESUME_PATH:?}"

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

TRAIN_CMD=(
  bash "${PROJECT_ROOT}/tools/train_refcoco_opsd_4b.sh"
  --gpus "${JOB_GPU}"
  --cuda-devices "${CUDA_DEVICES}"
  --data-root "${DATA_ROOT}"
  --image-root "${IMAGE_ROOT}"
  --model-path "${MODEL_PATH}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --work-dir "${WORK_DIR}"
  --route-mode manifest
  --resume "${RESUME_PATH}"
)

"${TRAIN_CMD[@]}"
'
