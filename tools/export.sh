#!/usr/bin/env bash

set -euo pipefail

rjob submit \
  --cpu=20 \
  --gpu=1 \
  --memory=102400 \
  --charged-group=ai4ls_gpu \
  --private-machine=group \
  --mount=gpfs://gpfs1/dnacoding:/mnt/shared-storage-user/dnacoding \
  --mount=gpfs://gpfs1/wuyucheng:/mnt/shared-storage-user/wuyucheng \
  --image registry.h.pjlab.org.cn/ailab-dnacoding/wuyucheng:test1 \
  --custom-resources brainpp.cn/fuse=1 \
  --enable-sshd \
  -- bash -lc '
set -euo pipefail

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

bash /mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd/tools/export_refcoco_opsd_routes_4b.sh \
  --gpus 1 \
  --cuda-devices 0 \
  --data-root /mnt/shared-storage-user/wuyucheng/dataset/refcoco \
  --image-root /mnt/shared-storage-user/wuyucheng/dataset/refcoco/train2014 \
  --model-path /mnt/shared-storage-user/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA-4B \
  --tokenizer-path /mnt/shared-storage-user/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA-4B \
  --work-dir /mnt/shared-storage-user/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd/work_dirs/sa2va_opsd_refcoco_internvl3_4b_v3_manifest \
  --global-step 0 \
  --route-model teacher \
  --batch-size 2 \
  -- \
  --cfg-options model.grpo_group_size=8
'
