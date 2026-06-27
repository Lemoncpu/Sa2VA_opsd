#!/usr/bin/env bash

set -euo pipefail

JOB_CPU="${JOB_CPU:-20}"
JOB_GPU="${JOB_GPU:-1}"
JOB_MEMORY="${JOB_MEMORY:-102400}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA-4B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/describe-anything}"
DATA_ROOT="${DATA_ROOT:-${OFFICIAL_REPO_ROOT}/DLC-bench}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/work_dirs/dlc_bench_eval}"
PRED_OUTPUT="${PRED_OUTPUT:-${OUTPUT_DIR}/pred.json}"
DEBUG_OUTPUT="${DEBUG_OUTPUT:-${OUTPUT_DIR}/debug.json}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/opt/vlm/bin/python}"
LIMIT="${LIMIT:-}"
START="${START:-0}"
STEP="${STEP:-1}"
LLM_ENGINE="${LLM_ENGINE:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
LLM_ENGINE_PATH="${LLM_ENGINE_PATH:-}"
API_KEY_PATH="${API_KEY_PATH:-}"
DEFAULT_PREDICTION="${DEFAULT_PREDICTION:-}"
EVAL_SUFFIX="${EVAL_SUFFIX:-}"
VERBOSE="${VERBOSE:-0}"
QUIET="${QUIET:-0}"
CSV_ONLY="${CSV_ONLY:-0}"

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
  OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PRED_OUTPUT="${PRED_OUTPUT}" \
  DEBUG_OUTPUT="${DEBUG_OUTPUT}" \
  DEVICE="${DEVICE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  LIMIT="${LIMIT}" \
  START="${START}" \
  STEP="${STEP}" \
  LLM_ENGINE="${LLM_ENGINE}" \
  LLM_ENGINE_PATH="${LLM_ENGINE_PATH}" \
  API_KEY_PATH="${API_KEY_PATH}" \
  DEFAULT_PREDICTION="${DEFAULT_PREDICTION}" \
  EVAL_SUFFIX="${EVAL_SUFFIX}" \
  VERBOSE="${VERBOSE}" \
  QUIET="${QUIET}" \
  CSV_ONLY="${CSV_ONLY}" \
  bash -lc '
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?}"
MODEL_PATH="${MODEL_PATH:?}"
TOKENIZER_PATH="${TOKENIZER_PATH:?}"
OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:?}"
DATA_ROOT="${DATA_ROOT:?}"
OUTPUT_DIR="${OUTPUT_DIR:?}"
PRED_OUTPUT="${PRED_OUTPUT:?}"
DEBUG_OUTPUT="${DEBUG_OUTPUT:?}"
DEVICE="${DEVICE:?}"
PYTHON_BIN="${PYTHON_BIN:?}"
JOB_GPU="${JOB_GPU:?}"
LOG_FILE="${OUTPUT_DIR}/eval_dlc_${JOB_GPU}gpu.log"

mkdir -p "${OUTPUT_DIR}"
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
/opt/vlm/bin/python -m pip install -q inflect tqdm openai

cd "${PROJECT_ROOT}"

EVAL_CMD=(
  bash "${PROJECT_ROOT}/tools/run_dlc_bench_official_eval.sh"
  --python-bin "${PYTHON_BIN}"
  --model-path "${MODEL_PATH}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --data-root "${DATA_ROOT}"
  --pred-output "${PRED_OUTPUT}"
  --debug-output "${DEBUG_OUTPUT}"
  --official-repo-root "${OFFICIAL_REPO_ROOT}"
  --device "${DEVICE}"
  --start "${START}"
  --step "${STEP}"
  --llm-engine "${LLM_ENGINE}"
)

if [[ -n "${LIMIT:-}" ]]; then
  EVAL_CMD+=(--limit "${LIMIT}")
fi
if [[ -n "${LLM_ENGINE_PATH:-}" ]]; then
  EVAL_CMD+=(--llm-engine-path "${LLM_ENGINE_PATH}")
fi
if [[ -n "${API_KEY_PATH:-}" ]]; then
  EVAL_CMD+=(--api-key "${API_KEY_PATH}")
fi
if [[ -n "${DEFAULT_PREDICTION:-}" ]]; then
  EVAL_CMD+=(--default-prediction "${DEFAULT_PREDICTION}")
fi
if [[ -n "${EVAL_SUFFIX:-}" ]]; then
  EVAL_CMD+=(--eval-suffix "${EVAL_SUFFIX}")
fi
if [[ "${VERBOSE}" == "1" ]]; then
  EVAL_CMD+=(--verbose)
fi
if [[ "${QUIET}" == "1" ]]; then
  EVAL_CMD+=(--quiet)
fi
if [[ "${CSV_ONLY}" == "1" ]]; then
  EVAL_CMD+=(--csv-only)
fi

stdbuf -oL -eL "${EVAL_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
'
