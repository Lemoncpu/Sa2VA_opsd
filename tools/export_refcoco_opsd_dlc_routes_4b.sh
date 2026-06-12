#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SA2VA_REFCOCO_OPSD_CONFIG="${SA2VA_REFCOCO_OPSD_CONFIG:-${ROOT_DIR}/projects/sa2va/configs/sa2va_opsd_combine_4b_dlc.py}" \
SA2VA_REFCOCO_OPSD_DEFAULT_WORK_DIR="${SA2VA_REFCOCO_OPSD_DEFAULT_WORK_DIR:-${ROOT_DIR}/work_dirs/sa2va_opsd_combine_4b_dlc}" \
SA2VA_REFCOCO_OPSD_DEFAULT_MODEL_PATH="${SA2VA_REFCOCO_OPSD_DEFAULT_MODEL_PATH:-${ROOT_DIR}/pretrained/Sa2VA-4B}" \
SA2VA_REFCOCO_OPSD_DEFAULT_TOKENIZER_PATH="${SA2VA_REFCOCO_OPSD_DEFAULT_TOKENIZER_PATH:-${ROOT_DIR}/pretrained/Sa2VA-4B}" \
SA2VA_REFCOCO_OPSD_MODEL_FLAVOR="${SA2VA_REFCOCO_OPSD_MODEL_FLAVOR:-sa2va4b}" \
SA2VA_REFCOCO_OPSD_ENTRY_NAME="${SA2VA_REFCOCO_OPSD_ENTRY_NAME:-export_refcoco_opsd_dlc_routes_4b.sh}" \
  bash "${ROOT_DIR}/tools/export_refcoco_opsd_dlc_routes_impl.sh" "$@"
