#!/usr/bin/env bash

set -euo pipefail

JOB_CPU="${JOB_CPU:-8}"
JOB_GPU="${JOB_GPU:-0}"
JOB_MEMORY="${JOB_MEMORY:-32768}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/describe-anything}"
DATA_ROOT="${DATA_ROOT:-${OFFICIAL_REPO_ROOT}/DLC-bench}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/work_dirs/dlc_bench_eval}"
PRED_OUTPUT="${PRED_OUTPUT:-${OUTPUT_DIR}/pred.json}"
WHEEL_DIR="${WHEEL_DIR:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/wheels_repo}"
PYTHON_BIN="${PYTHON_BIN:-/opt/vlm/bin/python}"
LLM_ENGINE="${LLM_ENGINE:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
LLM_ENGINE_PATH="${LLM_ENGINE_PATH:-https://api.openai.com/v1}"
API_KEY="${API_KEY:-sk-e3xpxVZMhL4I0iaZwurunj8yXK0Zaei25JOV0TP5QKdND04m}"
DEFAULT_PREDICTION="${DEFAULT_PREDICTION:-}"
EVAL_SUFFIX="${EVAL_SUFFIX:-}"
VERBOSE="${VERBOSE:-0}"
QUIET="${QUIET:-0}"
CSV_ONLY="${CSV_ONLY:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-http://mirrors.h.pjlab.org.cn/pypi/web/simple}"

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
  OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PRED_OUTPUT="${PRED_OUTPUT}" \
  WHEEL_DIR="${WHEEL_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  LLM_ENGINE="${LLM_ENGINE}" \
  LLM_ENGINE_PATH="${LLM_ENGINE_PATH}" \
  API_KEY="${API_KEY}" \
  DEFAULT_PREDICTION="${DEFAULT_PREDICTION}" \
  EVAL_SUFFIX="${EVAL_SUFFIX}" \
  VERBOSE="${VERBOSE}" \
  QUIET="${QUIET}" \
  CSV_ONLY="${CSV_ONLY}" \
  PIP_INDEX_URL="${PIP_INDEX_URL}" \
  bash -lc '
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?}"
OFFICIAL_REPO_ROOT="${OFFICIAL_REPO_ROOT:?}"
DATA_ROOT="${DATA_ROOT:?}"
OUTPUT_DIR="${OUTPUT_DIR:?}"
PRED_OUTPUT="${PRED_OUTPUT:?}"
WHEEL_DIR="${WHEEL_DIR:?}"
PYTHON_BIN="${PYTHON_BIN:?}"
LOG_FILE="${OUTPUT_DIR}/judge_dlc.log"

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

cd "${PROJECT_ROOT}"

JUDGE_CMD=(
  bash "${PROJECT_ROOT}/tools/run_dlc_bench_judge_only.sh"
  --python-bin "${PYTHON_BIN}"
  --pred-output "${PRED_OUTPUT}"
  --data-root "${DATA_ROOT}"
  --official-repo-root "${OFFICIAL_REPO_ROOT}"
  --llm-engine "${LLM_ENGINE}"
  --wheel-dir "${WHEEL_DIR}"
)

if [[ -n "${LLM_ENGINE_PATH:-}" ]]; then
  JUDGE_CMD+=(--llm-engine-path "${LLM_ENGINE_PATH}")
fi
if [[ -n "${API_KEY:-}" && "${API_KEY}" != "YOUR_OPENAI_API_KEY_HERE" ]]; then
  JUDGE_CMD+=(--api-key "${API_KEY}")
fi
if [[ -n "${DEFAULT_PREDICTION:-}" ]]; then
  JUDGE_CMD+=(--default-prediction "${DEFAULT_PREDICTION}")
fi
if [[ -n "${EVAL_SUFFIX:-}" ]]; then
  JUDGE_CMD+=(--eval-suffix "${EVAL_SUFFIX}")
fi
if [[ "${VERBOSE:-0}" == "1" ]]; then
  JUDGE_CMD+=(--verbose)
fi
if [[ "${QUIET:-0}" == "1" ]]; then
  JUDGE_CMD+=(--quiet)
fi
if [[ "${CSV_ONLY:-0}" == "1" ]]; then
  JUDGE_CMD+=(--csv-only)
fi

stdbuf -oL -eL "${JUDGE_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
'
