#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec env \
  SA2VA_REFCOCO_OPSD_CONFIG="${SA2VA_REFCOCO_OPSD_CONFIG:-projects/sa2va/configs/sa2va_opsd_refcoco_internvl3_4b_v3.py}" \
  SA2VA_REFCOCO_OPSD_ONLINE_CONFIG="${SA2VA_REFCOCO_OPSD_ONLINE_CONFIG:-projects/sa2va/configs/sa2va_opsd_refcoco_internvl3_4b_v3_online.py}" \
  SA2VA_REFCOCO_OPSD_DEFAULT_WORK_DIR="${SA2VA_REFCOCO_OPSD_DEFAULT_WORK_DIR:-${ROOT_DIR}/work_dirs/sa2va_opsd_refcoco_internvl3_4b_v3}" \
  SA2VA_REFCOCO_OPSD_DEFAULT_MODEL_PATH="${SA2VA_REFCOCO_OPSD_DEFAULT_MODEL_PATH:-${ROOT_DIR}/pretrained/Sa2VA-4B}" \
  SA2VA_REFCOCO_OPSD_DEFAULT_TOKENIZER_PATH="${SA2VA_REFCOCO_OPSD_DEFAULT_TOKENIZER_PATH:-${ROOT_DIR}/pretrained/Sa2VA-4B}" \
  SA2VA_REFCOCO_OPSD_MODEL_FLAVOR="${SA2VA_REFCOCO_OPSD_MODEL_FLAVOR:-4b}" \
  SA2VA_REFCOCO_OPSD_ENTRY_NAME="${SA2VA_REFCOCO_OPSD_ENTRY_NAME:-train_refcoco_opsd_4b.sh}" \
  bash "${ROOT_DIR}/tools/train_refcoco_opsd_impl.sh" "$@"
