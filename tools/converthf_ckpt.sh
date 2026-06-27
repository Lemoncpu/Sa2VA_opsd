#!/usr/bin/env bash

set -euo pipefail

JOB_CPU="${JOB_CPU:-16}"
JOB_GPU="${JOB_GPU:-1}"
JOB_MEMORY="${JOB_MEMORY:-65536}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/projects/sa2va/configs/sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3.py}"
PTH_MODEL="${PTH_MODEL:-}"
SAVE_PATH="${SAVE_PATH:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA-4B}"
PYTHON_BIN="${PYTHON_BIN:-/opt/vlm/bin/python}"
SKIP_APT_INSTALL="${SKIP_APT_INSTALL:-1}"

usage() {
  cat <<EOF
Usage:
  bash tools/converthf_ckpt.sh --pth-model PATH [options]

Required:
  --pth-model PATH

Optional:
  --config PATH
  --save-path PATH
  --base-model-path PATH
  --job-cpu N
  --job-gpu N
  --job-memory MB
  --cuda-devices LIST
  --python-bin PATH
  --skip-apt-install 0|1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pth-model)
      PTH_MODEL="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --save-path)
      SAVE_PATH="$2"
      shift 2
      ;;
    --base-model-path)
      BASE_MODEL_PATH="$2"
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
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-apt-install)
      SKIP_APT_INSTALL="$2"
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

if [[ -z "${PTH_MODEL}" ]]; then
  echo "--pth-model is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "${SAVE_PATH}" ]]; then
  ckpt_name="$(basename "${PTH_MODEL}")"
  ckpt_stem="${ckpt_name%.*}"
  SAVE_PATH="${PROJECT_ROOT}/work_dirs/hf_${ckpt_stem}"
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
  CONFIG_PATH="${CONFIG_PATH}" \
  PTH_MODEL="${PTH_MODEL}" \
  SAVE_PATH="${SAVE_PATH}" \
  BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  SKIP_APT_INSTALL="${SKIP_APT_INSTALL}" \
  bash -lc '
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?}"
CONFIG_PATH="${CONFIG_PATH:?}"
PTH_MODEL="${PTH_MODEL:?}"
SAVE_PATH="${SAVE_PATH:?}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?}"
PYTHON_BIN="${PYTHON_BIN:?}"
SKIP_APT_INSTALL="${SKIP_APT_INSTALL:-1}"
LOG_DIR="$(dirname "${SAVE_PATH}")"
LOG_FILE="${LOG_DIR}/convert_$(basename "${SAVE_PATH}").log"
TMP_CONFIG="${LOG_DIR}/.$(basename "${SAVE_PATH}").convert_config.py"

mkdir -p "${LOG_DIR}"
: >"${LOG_FILE}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
exec > >(tee -a "${LOG_FILE}") 2>&1
cleanup() {
  rm -f "${TMP_CONFIG}"
}
trap cleanup EXIT

echo "[convert] remote_job_started"
echo "[convert] log_file=${LOG_FILE}"
echo "[convert] skip_apt_install=${SKIP_APT_INSTALL}"

cd /opt
tar -xzf vlm_env.tar.gz -C /opt/vlm
rm vlm_env.tar.gz
/opt/vlm/bin/python /opt/vlm/bin/conda-unpack

if [[ "${SKIP_APT_INSTALL}" != "1" ]]; then
  cat > /etc/apt/sources.list <<EOF
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-security main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-backports main restricted universe multiverse
EOF

  apt update
  apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
else
  echo "[convert] skipping apt install and using image-provided system libraries"
fi
/opt/vlm/bin/python -c "import torch, transformers; print(\"ok\")"

cd "${PROJECT_ROOT}"

cat > "${TMP_CONFIG}" <<EOF
from pathlib import Path
_src = Path(r"${CONFIG_PATH}")
_text = _src.read_text()
_text = _text.replace("path = \"./pretrained/Sa2VA-4B\"", "path = r\"${BASE_MODEL_PATH}\"")
_text = _text.replace("tokenizer_path = path", "tokenizer_path = path")
exec(compile(_text, str(_src), "exec"))
EOF

CONVERT_CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/tools/convert_to_hf.py"
  "${TMP_CONFIG}"
  "${PTH_MODEL}"
  --save-path "${SAVE_PATH}"
)

echo "[convert] config=${CONFIG_PATH}" | tee -a "${LOG_FILE}"
echo "[convert] pth_model=${PTH_MODEL}" | tee -a "${LOG_FILE}"
echo "[convert] base_model_path=${BASE_MODEL_PATH}" | tee -a "${LOG_FILE}"
echo "[convert] save_path=${SAVE_PATH}" | tee -a "${LOG_FILE}"

set +e
stdbuf -oL -eL "${CONVERT_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
convert_status=${PIPESTATUS[0]}
set -e

if [[ "${convert_status}" -ne 0 ]]; then
  echo "[convert] FAILED exit_code=${convert_status}" | tee -a "${LOG_FILE}" >&2
  exit "${convert_status}"
fi

if [[ ! -d "${SAVE_PATH}" ]]; then
  echo "[convert] FAILED missing_output_dir=${SAVE_PATH}" | tee -a "${LOG_FILE}" >&2
  exit 1
fi

echo "[convert] SUCCESS output_dir=${SAVE_PATH}" | tee -a "${LOG_FILE}"
'
