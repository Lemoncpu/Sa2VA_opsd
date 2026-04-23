#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG="${CONFIG:-projects/sa2va/configs/refcoco_teacher_context_validation_2b.py}"

ARGS=(
  tools/eval_teacher_privileged_diagnosis_refcoco.py
  --config "${CONFIG}"
)

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

if [[ -n "${IMAGE_ROOT:-}" ]]; then
  ARGS+=(--image-root "${IMAGE_ROOT}")
fi

if [[ -n "${OUTPUT:-}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${ARGS[@]}" \
  "$@"
