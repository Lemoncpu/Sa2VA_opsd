#!/usr/bin/env bash

set -euo pipefail

JOB_CPU="${JOB_CPU:-20}"
JOB_GPU="${JOB_GPU:-1}"
JOB_MEMORY="${JOB_MEMORY:-102400}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATA_ROOT}/train2014}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA-4B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/work_dirs/sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3_manifest}"
SAM_CONFUSER_POOL_DIR="${SAM_CONFUSER_POOL_DIR:-${WORK_DIR}/sam_confuser_pool}"
RESUME_PATH="${RESUME_PATH:-}"
LOAD_FROM_PATH="${LOAD_FROM_PATH:-}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${PROJECT_ROOT}/pretrained/sam2/sam21L/sam2_hiera_large.pt}"
CONF_OVERWRITE="${CONF_OVERWRITE:-0}"

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
  SAM_CONFUSER_POOL_DIR="${SAM_CONFUSER_POOL_DIR}" \
  RESUME_PATH="${RESUME_PATH}" \
  LOAD_FROM_PATH="${LOAD_FROM_PATH}" \
  SAM2_CONFIG="${SAM2_CONFIG}" \
  SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" \
  CONF_OVERWRITE="${CONF_OVERWRITE}" \
  bash -lc '
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?}"
JOB_GPU="${JOB_GPU:?}"
DATA_ROOT="${DATA_ROOT:?}"
IMAGE_ROOT="${IMAGE_ROOT:?}"
MODEL_PATH="${MODEL_PATH:?}"
TOKENIZER_PATH="${TOKENIZER_PATH:?}"
WORK_DIR="${WORK_DIR:?}"
SAM_CONFUSER_POOL_DIR="${SAM_CONFUSER_POOL_DIR:?}"
LOG_FILE="${WORK_DIR}/train_${JOB_GPU}gpu.log"
SAM2_CONFIG="${SAM2_CONFIG:?}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:?}"
CONF_OVERWRITE="${CONF_OVERWRITE:-0}"

mkdir -p "${WORK_DIR}"
: >"${LOG_FILE}"
export PYTHONUNBUFFERED=1

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

if [[ ! -d "${SAM_CONFUSER_POOL_DIR}" ]]; then
  echo "SAM confuser pool dir missing, generating: ${SAM_CONFUSER_POOL_DIR}"
  mkdir -p "${SAM_CONFUSER_POOL_DIR}"

  IFS="," read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_DEVICES}"
  if [[ "${#CUDA_DEVICE_ARRAY[@]}" -ne "${JOB_GPU}" ]]; then
    echo "JOB_GPU (${JOB_GPU}) does not match CUDA_DEVICES count (${#CUDA_DEVICE_ARRAY[@]})." >&2
    exit 1
  fi

  CONF_PIDS=()
  CONF_LOGS=()
  for ((idx = 0; idx < JOB_GPU; idx++)); do
    physical_device="${CUDA_DEVICE_ARRAY[$idx]}"
    shard_log="${WORK_DIR}/conf_shard${idx}_of_${JOB_GPU}.log"
    conf_cmd=(
      bash "${PROJECT_ROOT}/tools/conf.sh"
      --data-root "${DATA_ROOT}"
      --image-root "${IMAGE_ROOT}"
      --dataset refcoco
      --split train
      --out-dir "${SAM_CONFUSER_POOL_DIR}"
      --sam2-config "${SAM2_CONFIG}"
      --sam2-checkpoint "${SAM2_CHECKPOINT}"
      --device cuda:0
      --shard-index "${idx}"
      --num-shards "${JOB_GPU}"
    )
    if [[ "${CONF_OVERWRITE}" == "1" ]]; then
      conf_cmd+=(--overwrite)
    fi
    (
      export CUDA_VISIBLE_DEVICES="${physical_device}"
      "${conf_cmd[@]}"
    ) >"${shard_log}" 2>&1 &
    CONF_PIDS+=("$!")
    CONF_LOGS+=("${shard_log}")
  done

  conf_status=0
  for idx in "${!CONF_PIDS[@]}"; do
    if ! wait "${CONF_PIDS[$idx]}"; then
      conf_status=1
      echo "Confuser export shard ${idx} failed. Log: ${CONF_LOGS[$idx]}" >&2
      tail -n 200 "${CONF_LOGS[$idx]}" >&2 || true
    fi
  done
  if (( conf_status != 0 )); then
    exit "${conf_status}"
  fi
fi

PLOT_CMD=(
  /opt/vlm/bin/python
  "${PROJECT_ROOT}/tools/plot_training_metrics.py"
  "${WORK_DIR}"
  --watch
  --interval-seconds 10
  --smooth 1
)

"${PLOT_CMD[@]}" &
PLOT_PID=$!
cleanup() {
  if [[ -n "${PLOT_PID:-}" ]]; then
    kill "${PLOT_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

TRAIN_CMD=(
  bash "${PROJECT_ROOT}/tools/train_refcoco_opsd_4b.sh"
  --gpus "${JOB_GPU}"
  --cuda-devices "${CUDA_DEVICES}"
  --data-root "${DATA_ROOT}"
  --image-root "${IMAGE_ROOT}"
  --model-path "${MODEL_PATH}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --work-dir "${WORK_DIR}"
  --batch-size 1
  --sam-confuser-pool-dir "${SAM_CONFUSER_POOL_DIR}"
  --route-mode manifest
)

if [[ -n "${LOAD_FROM_PATH:-}" ]]; then
  TRAIN_CMD+=(--load-from "${LOAD_FROM_PATH}")
fi

if [[ -n "${RESUME_PATH:-}" ]]; then
  TRAIN_CMD+=(--resume "${RESUME_PATH}")
fi

stdbuf -oL -eL "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
'
