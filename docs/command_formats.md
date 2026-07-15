# Command Formats

This file is the source of truth for command templates that Codex should follow when giving training, route-export, confuser-export, plotting, and evaluation commands in this repository.

## General Rules
- Always `cd /data5/qianshan.wei/Sa2VA_opsd` before running project commands on the target server.
- Always activate the runtime environment first:
  - `source /data5/qianshan.wei/sa2va4b_host_venv/bin/activate`
- Always export:
  - `export PYTHONPATH=/data5/qianshan.wei/Sa2VA_opsd:$PYTHONPATH`
- For commands that write logs or checkpoints, always create the target `WORK_DIR` first with `mkdir -p`.
- For background commands, redirect both stdout and stderr into a log file under `WORK_DIR`.
- When giving an online-routing training command, explicitly set:
  - `ROUTE_MODE=online`
- When giving a manifest-routing training command, explicitly set:
  - `ROUTE_MODE=manifest`
- Do not omit `ROUTE_MODE` when presenting final commands.

## Required Variables
- `MODEL_PATH`: base Sa2VA checkpoint path
- `TOKENIZER_PATH`: tokenizer path, usually same as `MODEL_PATH`
- `DATA_ROOT`: RefCOCO annotation root, expected to be the final dataset directory
- `IMAGE_ROOT`: RefCOCO image directory, usually `train2014`
- `SAM_CONFUSER_POOL_DIR`: exported SAM confuser pool root
- `WORK_DIR`: logs / checkpoints / route cache directory
- `SA2VA_TRAIN_FALLBACK_CHAIN=refcoco_opsd_4b`: required for the current host fallback training path

## Short Referring Training

### 1 GPU Online Routing
Use this for short referring caption training without a precomputed route manifest.

```bash
cd /data5/qianshan.wei/Sa2VA_opsd
source /data5/qianshan.wei/sa2va4b_host_venv/bin/activate
export PYTHONPATH=/data5/qianshan.wei/Sa2VA_opsd:$PYTHONPATH

WORK_DIR=/data5/qianshan.wei/Sa2VA_opsd/work_dirs/<run_name>
mkdir -p "${WORK_DIR}"

CUDA_VISIBLE_DEVICES=0 \
ACTIVATE_SCRIPT=/data5/qianshan.wei/sa2va4b_host_venv/bin/activate \
GPUS=1 \
ROUTE_MODE=online \
MODEL_PATH=/data5/qianshan.wei/Sa2VA-4B \
TOKENIZER_PATH=/data5/qianshan.wei/Sa2VA-4B \
DATA_ROOT=/data5/qianshan.wei/refcoco/snapshots/5daa32cfe87ea355fab400b5ec5a8a9bb476ffd2 \
IMAGE_ROOT=/data5/qianshan.wei/refcoco/snapshots/5daa32cfe87ea355fab400b5ec5a8a9bb476ffd2/train2014 \
SAM_CONFUSER_POOL_DIR=/data5/qianshan.wei/Sa2VA_opsd/work_dirs/refcoco_sam_confuser_pool_8gpu \
WORK_DIR="${WORK_DIR}" \
SA2VA_TRAIN_FALLBACK_CHAIN=refcoco_opsd_4b \
bash /data5/qianshan.wei/Sa2VA_opsd/tools/train_refcoco_opsd_4b_referring.sh \
> "${WORK_DIR}/train.log" 2>&1 &
```

### 1 GPU Manifest Routing
Use this only when a full route manifest already exists in `WORK_DIR/route_cache/`.

```bash
cd /data5/qianshan.wei/Sa2VA_opsd
source /data5/qianshan.wei/sa2va4b_host_venv/bin/activate
export PYTHONPATH=/data5/qianshan.wei/Sa2VA_opsd:$PYTHONPATH

WORK_DIR=/data5/qianshan.wei/Sa2VA_opsd/work_dirs/<run_name>
mkdir -p "${WORK_DIR}"

CUDA_VISIBLE_DEVICES=0 \
ACTIVATE_SCRIPT=/data5/qianshan.wei/sa2va4b_host_venv/bin/activate \
GPUS=1 \
ROUTE_MODE=manifest \
MODEL_PATH=/data5/qianshan.wei/Sa2VA-4B \
TOKENIZER_PATH=/data5/qianshan.wei/Sa2VA-4B \
DATA_ROOT=/data5/qianshan.wei/refcoco/snapshots/5daa32cfe87ea355fab400b5ec5a8a9bb476ffd2 \
IMAGE_ROOT=/data5/qianshan.wei/refcoco/snapshots/5daa32cfe87ea355fab400b5ec5a8a9bb476ffd2/train2014 \
SAM_CONFUSER_POOL_DIR=/data5/qianshan.wei/Sa2VA_opsd/work_dirs/refcoco_sam_confuser_pool_8gpu \
WORK_DIR="${WORK_DIR}" \
SA2VA_TRAIN_FALLBACK_CHAIN=refcoco_opsd_4b \
bash /data5/qianshan.wei/Sa2VA_opsd/tools/train_refcoco_opsd_4b_referring.sh \
> "${WORK_DIR}/train.log" 2>&1 &
```

## Referring Route Export

### 4 GPU Referring Route Export
```bash
cd /data5/qianshan.wei/Sa2VA_opsd
source /data5/qianshan.wei/sa2va4b_host_venv/bin/activate
export PYTHONPATH=/data5/qianshan.wei/Sa2VA_opsd:$PYTHONPATH

WORK_DIR=/data5/qianshan.wei/Sa2VA_opsd/work_dirs/<route_work_dir>
mkdir -p "${WORK_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACTIVATE_SCRIPT=/data5/qianshan.wei/sa2va4b_host_venv/bin/activate \
GPUS=4 \
MODEL_PATH=/data5/qianshan.wei/Sa2VA-4B \
TOKENIZER_PATH=/data5/qianshan.wei/Sa2VA-4B \
DATA_ROOT=/data5/qianshan.wei/refcoco/snapshots/5daa32cfe87ea355fab400b5ec5a8a9bb476ffd2 \
IMAGE_ROOT=/data5/qianshan.wei/refcoco/snapshots/5daa32cfe87ea355fab400b5ec5a8a9bb476ffd2/train2014 \
SAM_CONFUSER_POOL_DIR=/data5/qianshan.wei/Sa2VA_opsd/work_dirs/refcoco_sam_confuser_pool_8gpu \
WORK_DIR="${WORK_DIR}" \
bash /data5/qianshan.wei/Sa2VA_opsd/tools/export_refcoco_opsd_routes_4b_referring.sh \
> "${WORK_DIR}/export_routes.log" 2>&1 &
```

## Training Curve Plot
Use positional `log_path`; do not use `--log-file`.

```bash
/data5/qianshan.wei/sa2va4b_host_venv/bin/python \
  /data5/qianshan.wei/Sa2VA_opsd/tools/plot_training_metrics.py \
  /data5/qianshan.wei/Sa2VA_opsd/work_dirs/<run_name>/train.log \
  --output /data5/qianshan.wei/Sa2VA_opsd/work_dirs/<run_name>/training_curve.png \
  --watch
```

## Notes
- `tools/train_refcoco_opsd_4b_referring.sh` chooses the online config only when `ROUTE_MODE=online` is explicitly provided.
- If `ROUTE_MODE` is omitted, the wrapper defaults to manifest mode, which is wrong for online-routing training.
- When giving commands to users, prefer complete copy-pasteable blocks from this file instead of reconstructing them from memory.
