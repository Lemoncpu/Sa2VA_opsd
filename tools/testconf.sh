#!/usr/bin/env bash

set -euo pipefail

JOB_CPU="${JOB_CPU:-80}"
JOB_GPU="${JOB_GPU:-4}"
JOB_MEMORY="${JOB_MEMORY:-409600}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATA_ROOT}/train2014}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/work_dirs/sa2va_opsd_refcoco_internvl3_4b_v3_manifest}"
SAM_CONFUSER_POOL_DIR="${SAM_CONFUSER_POOL_DIR:-${WORK_DIR}/sam_confuser_pool}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${PROJECT_ROOT}/pretrained/sam2/sam21L/sam2.1_hiera_large.pt}"
DATASET="${DATASET:-refcoco}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"

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
  WORK_DIR="${WORK_DIR}" \
  SAM_CONFUSER_POOL_DIR="${SAM_CONFUSER_POOL_DIR}" \
  SAM2_CONFIG="${SAM2_CONFIG}" \
  SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" \
  DATASET="${DATASET}" \
  SPLIT="${SPLIT}" \
  LIMIT="${LIMIT}" \
  OVERWRITE="${OVERWRITE}" \
  bash -lc '
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?}"
JOB_GPU="${JOB_GPU:?}"
CUDA_DEVICES="${CUDA_DEVICES:?}"
DATA_ROOT="${DATA_ROOT:?}"
IMAGE_ROOT="${IMAGE_ROOT:?}"
WORK_DIR="${WORK_DIR:?}"
SAM_CONFUSER_POOL_DIR="${SAM_CONFUSER_POOL_DIR:?}"
SAM2_CONFIG="${SAM2_CONFIG:?}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:?}"
DATASET="${DATASET:?}"
SPLIT="${SPLIT:?}"

mkdir -p "${WORK_DIR}"

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
/opt/vlm/bin/python - <<'PY'
import importlib

for module_name in ("torch", "transformers", "PIL", "pycocotools", "hydra"):
    importlib.import_module(module_name)

print("python env ok")
PY

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/tools/conf.sh" ]]; then
  echo "conf.sh does not exist under PROJECT_ROOT: ${PROJECT_ROOT}/tools/conf.sh" >&2
  exit 1
fi

REFCOCO_ROOT="${DATA_ROOT}"
if [[ "$(basename "${DATA_ROOT}")" != "refcoco" ]]; then
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

IFS="," read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_DEVICES}"
if [[ "${#CUDA_DEVICE_ARRAY[@]}" -ne "${JOB_GPU}" ]]; then
  echo "JOB_GPU (${JOB_GPU}) does not match CUDA_DEVICES count (${#CUDA_DEVICE_ARRAY[@]})." >&2
  exit 1
fi

mkdir -p "${SAM_CONFUSER_POOL_DIR}"

PIDS=()
SHARD_LOGS=()
SHARD_DEVICE_IDS=()
for ((idx = 0; idx < JOB_GPU; idx++)); do
  physical_device="${CUDA_DEVICE_ARRAY[$idx]}"
  shard_log="${WORK_DIR}/conf_shard${idx}_of_${JOB_GPU}.log"
  cmd=(
    bash "${PROJECT_ROOT}/tools/conf.sh"
    --data-root "${DATA_ROOT}"
    --image-root "${IMAGE_ROOT}"
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --out-dir "${SAM_CONFUSER_POOL_DIR}"
    --sam2-config "${SAM2_CONFIG}"
    --sam2-checkpoint "${SAM2_CHECKPOINT}"
    --device cuda:0
    --shard-index "${idx}"
    --num-shards "${JOB_GPU}"
  )
  if [[ -n "${LIMIT:-}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [[ "${OVERWRITE:-0}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  (
    export CUDA_VISIBLE_DEVICES="${physical_device}"
    "${cmd[@]}"
  ) >"${shard_log}" 2>&1 &
  PIDS+=("$!")
  SHARD_LOGS+=("${shard_log}")
  SHARD_DEVICE_IDS+=("${physical_device}")
done

status=0
FAILED_SHARDS=()
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if ! wait "${pid}"; then
    status=1
    FAILED_SHARDS+=("${idx}")
  fi
done

if (( status != 0 )); then
  for idx in "${FAILED_SHARDS[@]}"; do
    shard_log="${SHARD_LOGS[$idx]}"
    physical_device="${SHARD_DEVICE_IDS[$idx]}"
    echo "Shard ${idx}/${JOB_GPU} failed on CUDA_VISIBLE_DEVICES=${physical_device}. Log: ${shard_log}" >&2
    if [[ -f "${shard_log}" ]]; then
      echo "----- tail ${shard_log} -----" >&2
      tail -n 200 "${shard_log}" >&2 || true
      echo "----- end tail ${shard_log} -----" >&2
    else
      echo "Shard log not found: ${shard_log}" >&2
    fi
  done
fi

exit "${status}"
'
