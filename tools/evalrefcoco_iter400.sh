#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/work_dirs/hf_iter_400}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATA_ROOT}/train2014}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/work_dirs/refcoco_caption_to_mask_eval_hf_iter_400.json}"

export PROJECT_ROOT MODEL_PATH TOKENIZER_PATH DATA_ROOT IMAGE_ROOT OUTPUT_PATH

bash "${PROJECT_ROOT}/tools/evalrefcoco_ckpt.sh" "$@"
