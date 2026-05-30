#!/usr/bin/env bash

set -x

FILE=$1
CONFIG=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((18500 + RANDOM % 2000))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
DEEPSPEED=${DEEPSPEED:-deepspeed_zero2}
PYTHON_BIN=${SA2VA_PYTHON:-}

if [[ $FILE == *.py ]]; then
    FILE=${FILE}
else
    FILE=tools/${FILE}.py
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No usable python interpreter found. Set SA2VA_PYTHON or activate a working virtual environment before calling tools/dist.sh." >&2
  exit 127
fi

if "${PYTHON_BIN}" -c "import torch.distributed.run" >/dev/null 2>&1
then
  echo "Using torch.distributed.run mode."
  PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    --nproc_per_node=${GPUS} \
    ${FILE} ${CONFIG} --launcher pytorch --deepspeed "${DEEPSPEED}" "${@:4}"
else
  echo "Using launch mode."
  PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${PYTHON_BIN}" -m torch.distributed.launch \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    --nproc_per_node=${GPUS} \
    ${FILE} ${CONFIG} --launcher pytorch --deepspeed "${DEEPSPEED}" "${@:4}"
fi
