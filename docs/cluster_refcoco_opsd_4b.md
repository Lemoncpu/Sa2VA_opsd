# RefCOCO OPSD 4B 集群训练交付模板

这份模板固定对应当前项目的这次训练需求：

- 代码仓库：当前 `Sa2VA` 项目
- 依赖环境：`uv sync --extra=legacy`
- 训练脚本：`bash tools/train_refcoco_opsd_4b.sh`
- GPU：单卡
- 恢复训练：必须带 `--resume`
- 恢复点：`iter_800.pth`
- 本地数据：`/data/xiaoyicheng/refcoco`

## 1. 交付物划分

这次需要分成 4 类资产，不要混在一起：

1. GitHub：只放代码和文档
2. DockerHub：只放环境镜像
3. checkpoint：单独传 `iter_800.pth`
4. 数据：单独准备 `refcoco` 目录

不要把下面这些内容 push 到 GitHub：

- `pretrained/`
- `work_dirs/`
- `*.pth`
- 数据目录

原因：

- GitHub 普通仓库对单文件大于 100 MiB 直接拒绝
- Git LFS 也有单文件上限，你当前的 `iter_800.pth` 远超这个范围

GitHub 官方文档：

- 普通仓库大文件限制：<https://docs.github.com/repositories/working-with-files/managing-large-files/about-large-files-on-github>
- Git LFS 限制：<https://docs.github.com/repositories/working-with-files/managing-large-files/about-git-large-file-storage>

## 2. GitHub 代码仓库模板

如果你要推到一个新的 GitHub 仓库，建议按下面做。

先检查当前工作区，确认只提交你想交付的代码：

```bash
git status --short
```

如果原来的 `origin` 不是你的 GitHub 仓库，可以先保留原远端，再加一个新的 GitHub 远端：

```bash
git remote rename origin upstream
git remote add github https://github.com/<your-name>/<your-repo>.git
```

创建一个交付分支：

```bash
git checkout -b cluster/refcoco-opsd-4b
```

提交代码：

```bash
git add .
git commit -m "Prepare RefCOCO OPSD 4B cluster training handoff"
git push -u github cluster/refcoco-opsd-4b
```

如果你不想把本地草稿文件一起提交，先用下面命令做二次确认：

```bash
git diff --name-only --cached
```

## 3. Docker 镜像模板

仓库根目录已经提供了：

- `Dockerfile.cluster`
- `.dockerignore`

这个镜像模板的策略是：

- 基础镜像：CUDA 12.4 + cuDNN + Ubuntu 22.04
- Python：3.11
- 依赖管理：`uv`
- 安装命令：`uv sync --extra=legacy --frozen`

本地构建：

```bash
docker build -f Dockerfile.cluster -t <dockerhub-user>/sa2va-refcoco-opsd:legacy-cu124 .
```

推到 DockerHub：

```bash
docker login
docker push <dockerhub-user>/sa2va-refcoco-opsd:legacy-cu124
```

### 3.1 如果不用 Docker：可以，但不要直接打包当前 `.venv`

这套项目本身不强依赖 Docker。

当前训练脚本默认就是：

- 进入仓库根目录
- `source .venv/bin/activate`
- 再启动训练

所以在集群上完全可以走：

1. 代码压缩包
2. `pyproject.toml` + `uv.lock`
3. 集群本地执行 `uv sync --extra=legacy --frozen`

也就是说，不用 Docker 是可行的。

但不建议把你当前机器上已经创建好的 `.venv` 直接压缩后扔到集群，原因是：

- `uv` 创建的虚拟环境通常会引用宿主机上的 Python 安装位置
- 虚拟环境里的部分可执行脚本会写入绝对路径 shebang
- 只要解压路径、宿主机路径或 Python 位置变了，就可能直接失效

更稳的交付方式是：打包源码和依赖描述文件，在集群上重建 `.venv`，而不是搬运现成 `.venv`。

#### 3.1.1 无 Docker 方案的前提

集群侧至少要满足下面几点：

- Linux x86_64
- Python 3.11
- NVIDIA 驱动能够兼容 `torch` 对应的 CUDA 12.4 用户态依赖
- 能运行 `uv sync --extra=legacy --frozen`

如果集群不能联网，额外还要准备：

- `uv` 缓存压缩包
- 可用的 Python 3.11 解释器目录，或者集群本身提供 Python 3.11 module

另外，集群基础环境最好具备和 `Dockerfile.cluster` 等价的系统依赖，至少关注：

- `ffmpeg`
- `libgl1`
- `libglib2.0-0`

如果后续发生本地编译，再补：

- `build-essential`
- `ninja-build`

#### 3.1.2 推荐打包内容

推荐把交付物拆成下面几类：

1. `sa2va_code_uv.tar.gz`
2. `uv_cache.tar.gz`（仅在集群不能联网时需要）
3. `python311.tar.gz`（仅在集群没有可用 Python 3.11 时需要）
4. `iter_800.pth`
5. `refcoco` 数据压缩包或目录

代码压缩包建议只包含源码和锁文件，不要包含 `.venv`、数据、checkpoint、输出目录：

```bash
tar   --exclude='.git'   --exclude='.venv'   --exclude='pretrained'   --exclude='work_dirs'   --exclude='logs'   --exclude='*.pth'   -czf sa2va_code_uv.tar.gz   .
```

如果集群不能联网，可额外打包本机 `uv` 缓存：

```bash
tar -C "${XDG_CACHE_HOME:-$HOME/.cache}" -czf uv_cache.tar.gz uv
```

如果集群没有 Python 3.11，但允许你自带解释器，可额外打包 `uv` 管理的 Python 目录：

```bash
tar -C "$HOME/.local/share/uv/python"   -czf python311.tar.gz   cpython-3.11.13-linux-x86_64-gnu
```

#### 3.1.3 集群侧恢复命令模板

假设代码解压到 `/mnt/work/Sa2VA`。

如果集群已经有可用的 Python 3.11：

```bash
mkdir -p /mnt/work/Sa2VA
tar -xzf sa2va_code_uv.tar.gz -C /mnt/work/Sa2VA
cd /mnt/work/Sa2VA

uv sync --extra=legacy --frozen   --python /usr/bin/python3.11   --no-managed-python
```

如果集群不能联网，并且你已经额外带了 `uv_cache.tar.gz` 和 `python311.tar.gz`：

```bash
mkdir -p /mnt/work/Sa2VA /mnt/runtime
tar -xzf sa2va_code_uv.tar.gz -C /mnt/work/Sa2VA
tar -xzf uv_cache.tar.gz -C /mnt/runtime
tar -xzf python311.tar.gz -C /mnt/runtime
cd /mnt/work/Sa2VA

export UV_CACHE_DIR=/mnt/runtime/uv

uv sync --extra=legacy --frozen --offline   --python /mnt/runtime/cpython-3.11.13-linux-x86_64-gnu/bin/python3.11
```

上面这条路的核心是：

- 允许不用 Docker
- 但默认不要搬运当前 `.venv`
- 要搬的是源码、锁文件，以及离线所需缓存/解释器

## 4. 必须单独传的 checkpoint

这次恢复点不是代码，也不是镜像，必须单独放到集群可访问路径。

当前本地 checkpoint 路径：

```bash
/data/xiaoyicheng/Sa2va_opsd/Sa2VA/work_dirs/sa2va_opsd_refcoco_internvl3_4b_v3/iter_800.pth
```

建议传到集群上的固定位置，例如：

```bash
/mnt/checkpoints/iter_800.pth
```

可选传输方式：

- `scp`
- `rsync`
- 集群共享盘
- 对象存储
- 私有模型仓库

不建议：

- push 到 GitHub
- 放进 Docker 镜像

## 5. HF 上需要下载的内容

### 5.1 必下：模型

当前训练代码直接从 Hugging Face 格式目录加载模型，因此集群上至少需要下载：

- `ByteDance/Sa2VA-4B`

来源：

- <https://huggingface.co/ByteDance/Sa2VA-4B>

下载命令：

```bash
huggingface-cli download   ByteDance/Sa2VA-4B   --local-dir /mnt/models/Sa2VA-4B   --local-dir-use-symlinks False
```

训练时：

- `--model-path /mnt/models/Sa2VA-4B`
- `--tokenizer-path /mnt/models/Sa2VA-4B`

### 5.2 数据说明

当前这套代码不是直接读取 Hugging Face `datasets` 的 parquet/json 格式，而是读取 RefCOCO 的原始目录结构。

代码实际要求的目录结构是：

```text
/mnt/data/refcoco/
├── instances.json
├── refs(unc).p
├── refs(google).p
└── train2014/
```

也就是说，这次“可直接跑”的方案里：

- 模型可以从 HF 直接下载
- 数据最好直接拷贝你现有的 `/data/xiaoyicheng/refcoco`

如果师兄强制要求“数据也必须从 HF 下载”，有两种做法：

1. 你把现有 `/data/xiaoyicheng/refcoco` 打包上传到一个私有 HF Dataset repo，然后集群侧 `snapshot_download`
2. 先从别的 HF RefCOCO 数据集下载，再自行转换成上面的原始目录结构

第二种并不是当前仓库的即插即用路径，不建议写成默认方案。

可参考的 HF RefCOCO 数据集页面：

- `jxu124/refcoco`：<https://huggingface.co/datasets/jxu124/refcoco>
- `PaDT-MLLM/RefCOCO`：<https://huggingface.co/datasets/PaDT-MLLM/RefCOCO>

上面两个页面存在，但我这里给你的判断是：它们不是当前训练脚本的直接输入格式，这一点是基于当前仓库的数据读取代码做出的结论。

### 5.3 推荐：把 checkpoint 和 RefCOCO 上传到你自己的私有 Hugging Face

这次任务里，把 `checkpoint` 和 `refcoco` 放到你自己的 Hugging Face 是可行的，而且是一个比较干净的交付方式。

更推荐的落地方式是：

1. `ByteDance/Sa2VA-4B` 继续直接从公开 HF 下载
2. `iter_800.pth` 上传到你自己的私有 HF `bucket`
3. `refcoco` 也上传到你自己的私有 HF `bucket`

推荐 `bucket` 的原因：

- HF 官方把 `bucket` 定义为更适合存放 checkpoint、logs、intermediate artifacts 的地方
- `bucket` 不是 Git-backed repo，更适合大文件和目录同步
- 你这次的 `iter_800.pth` 是 52G，`refcoco` 约 26G，这类文件更适合 `bucket` 而不是普通 dataset repo

参考：

- Storage Buckets：<https://huggingface.co/docs/hub/storage-buckets>
- Storage limits：<https://huggingface.co/docs/hub/storage-limits>
- Upload large folder：<https://huggingface.co/docs/huggingface_hub/guides/upload>

额外限制说明：

- HF 免费账号私有存储默认是 100GB
- 你当前 `iter_800.pth` 约 52G，`refcoco` 约 26G，总体约 78G
- 只上传这两样通常够，但空间已经比较紧，不建议再往同一个私有空间里堆太多额外大文件

参考：

- 私有存储配额：<https://huggingface.co/docs/hub/storage-limits>

#### 5.3.1 为什么默认要设成 private

建议默认使用私有存储：

- `iter_800.pth` 是你的训练恢复点，不适合公开
- `refcoco` 里包含 COCO 图像，是否适合公开再分发不建议拍脑袋处理
- HF 支持私有 repo / 私有 bucket，私有后外部访问会返回 404

参考：

- Repository visibility：<https://huggingface.co/docs/hub/repositories-settings>

#### 5.3.2 推荐上传结构

建议你在自己的 HF 账号下建一个私有 bucket，例如：

```text
hf://buckets/<your-hf-name>/sa2va-private/
├── checkpoints/
│   └── iter_800.pth
└── refcoco/
    ├── instances.json
    ├── refs(unc).p
    ├── refs(google).p
    └── train2014.zip
```

这里建议上传 `train2014.zip`，而不是直接上传 `train2014/` 目录下的所有图片，原因是：

- 原始图片文件太多
- 普通 Git-backed dataset repo 对目录规模不友好
- 你的当前训练代码只要求最终在集群上解压出 `train2014/` 即可

#### 5.3.3 上传命令模板

先登录：

```bash
hf auth login
```

创建私有 bucket：

```bash
hf buckets create <your-hf-name>/sa2va-private --private
```

上传 checkpoint：

```bash
hf buckets cp \
  /data/xiaoyicheng/Sa2va_opsd/Sa2VA/work_dirs/sa2va_opsd_refcoco_internvl3_4b_v3/iter_800.pth \
  hf://buckets/<your-hf-name>/sa2va-private/checkpoints/iter_800.pth
```

上传 RefCOCO：

```bash
hf buckets cp \
  /data/xiaoyicheng/refcoco/train2014.zip \
  hf://buckets/<your-hf-name>/sa2va-private/refcoco/train2014.zip

hf buckets cp \
  /data/xiaoyicheng/refcoco/instances.json \
  hf://buckets/<your-hf-name>/sa2va-private/refcoco/instances.json

hf buckets cp \
  '/data/xiaoyicheng/refcoco/refs(unc).p' \
  'hf://buckets/<your-hf-name>/sa2va-private/refcoco/refs(unc).p'

hf buckets cp \
  '/data/xiaoyicheng/refcoco/refs(google).p' \
  'hf://buckets/<your-hf-name>/sa2va-private/refcoco/refs(google).p'
```

#### 5.3.4 集群侧下载命令模板

先登录：

```bash
hf auth login
```

下载 checkpoint：

```bash
hf buckets cp \
  hf://buckets/<your-hf-name>/sa2va-private/checkpoints/iter_800.pth \
  /mnt/checkpoints/iter_800.pth
```

下载 RefCOCO：

```bash
hf buckets sync \
  hf://buckets/<your-hf-name>/sa2va-private/refcoco \
  /mnt/data/refcoco

cd /mnt/data/refcoco
unzip train2014.zip
```

#### 5.3.5 如果你坚持用 private dataset repo

也不是不行，但建议只把下面这些文件传进去：

- `train2014.zip`
- `instances.json`
- `refs(unc).p`
- `refs(google).p`

不要直接把 `train2014/` 整个图片目录原样按 Git 目录结构推上去。

如果你确实要用 dataset repo，可参考：

```bash
hf repo create <your-hf-name>/refcoco-private --repo-type dataset --private

hf upload <your-hf-name>/refcoco-private /data/xiaoyicheng/refcoco/train2014.zip train2014.zip --repo-type dataset
hf upload <your-hf-name>/refcoco-private /data/xiaoyicheng/refcoco/instances.json instances.json --repo-type dataset
hf upload <your-hf-name>/refcoco-private '/data/xiaoyicheng/refcoco/refs(unc).p' 'refs(unc).p' --repo-type dataset
hf upload <your-hf-name>/refcoco-private '/data/xiaoyicheng/refcoco/refs(google).p' 'refs(google).p' --repo-type dataset
```

集群侧下载：

```bash
hf download <your-hf-name>/refcoco-private \
  --repo-type dataset \
  --local-dir /mnt/data/refcoco

cd /mnt/data/refcoco
unzip train2014.zip
```

#### 5.3.6 这套方案下训练命令不变

无论你最终把 `checkpoint` 和 `refcoco` 放在共享盘、私有 HF bucket 还是私有 dataset repo，只要最后在集群上整理成下面的本地路径，训练命令都不需要改：

- `/mnt/checkpoints/iter_800.pth`
- `/mnt/data/refcoco/instances.json`
- `/mnt/data/refcoco/refs(unc).p`
- `/mnt/data/refcoco/refs(google).p`
- `/mnt/data/refcoco/train2014/`

## 6. 集群侧目录模板

建议在集群上统一成下面的布局：

```text
/mnt/
├── data/
│   └── refcoco/
│       ├── instances.json
│       ├── refs(unc).p
│       ├── refs(google).p
│       └── train2014/
├── models/
│   └── Sa2VA-4B/
├── checkpoints/
│   └── iter_800.pth
└── output/
```

## 7. 容器启动模板

下面是推荐的启动方式，默认只使用第 0 张卡：

```bash
docker run --rm -it   --gpus '"device=0"'   --ipc=host   --shm-size=64g   --ulimit memlock=-1   --ulimit stack=67108864   -v /mnt/data/refcoco:/mnt/data/refcoco:ro   -v /mnt/models/Sa2VA-4B:/mnt/models/Sa2VA-4B:ro   -v /mnt/checkpoints/iter_800.pth:/mnt/checkpoints/iter_800.pth:ro   -v /mnt/output:/mnt/output   <dockerhub-user>/sa2va-refcoco-opsd:legacy-cu124   bash
```

进入容器后，切到项目目录执行训练。

### 7.1 如果不用 Docker

假设你已经：

- 把代码解压到 `/mnt/work/Sa2VA`
- 在该目录执行过 `uv sync --extra=legacy --frozen`
- 把模型、checkpoint、RefCOCO 整理到了文档前面约定的位置

那么可以直接在宿主机上运行：

```bash
cd /mnt/work/Sa2VA

bash tools/train_refcoco_opsd_4b.sh   --gpus 1   --cuda-devices 0   --data-root /mnt/data/refcoco   --image-root /mnt/data/refcoco/train2014   --model-path /mnt/models/Sa2VA-4B   --tokenizer-path /mnt/models/Sa2VA-4B   --work-dir /mnt/output/sa2va_opsd_refcoco_internvl3_4b_v3_resume_from_iter800   --resume /mnt/checkpoints/iter_800.pth
```

如果你没有把环境建在仓库根目录的 `.venv`，可以显式指定激活脚本：

```bash
bash tools/train_refcoco_opsd_4b.sh   --activate-script /path/to/venv/bin/activate   --gpus 1   --cuda-devices 0   --data-root /mnt/data/refcoco   --image-root /mnt/data/refcoco/train2014   --model-path /mnt/models/Sa2VA-4B   --tokenizer-path /mnt/models/Sa2VA-4B   --work-dir /mnt/output/sa2va_opsd_refcoco_internvl3_4b_v3_resume_from_iter800   --resume /mnt/checkpoints/iter_800.pth
```

## 8. 单卡恢复训练命令

这就是这次交付里最关键的实际运行命令：

```bash
cd /workspace/Sa2VA

bash tools/train_refcoco_opsd_4b.sh   --gpus 1   --cuda-devices 0   --data-root /mnt/data/refcoco   --image-root /mnt/data/refcoco/train2014   --model-path /mnt/models/Sa2VA-4B   --tokenizer-path /mnt/models/Sa2VA-4B   --work-dir /mnt/output/sa2va_opsd_refcoco_internvl3_4b_v3_resume_from_iter800   --resume /mnt/checkpoints/iter_800.pth
```

## 9. 一条龙执行版本

如果你想把容器启动和训练合在一条命令里，可以直接用这个：

```bash
docker run --rm -it   --gpus '"device=0"'   --ipc=host   --shm-size=64g   --ulimit memlock=-1   --ulimit stack=67108864   -v /mnt/data/refcoco:/mnt/data/refcoco:ro   -v /mnt/models/Sa2VA-4B:/mnt/models/Sa2VA-4B:ro   -v /mnt/checkpoints/iter_800.pth:/mnt/checkpoints/iter_800.pth:ro   -v /mnt/output:/mnt/output   <dockerhub-user>/sa2va-refcoco-opsd:legacy-cu124   bash -lc '
    cd /workspace/Sa2VA &&     bash tools/train_refcoco_opsd_4b.sh       --gpus 1       --cuda-devices 0       --data-root /mnt/data/refcoco       --image-root /mnt/data/refcoco/train2014       --model-path /mnt/models/Sa2VA-4B       --tokenizer-path /mnt/models/Sa2VA-4B       --work-dir /mnt/output/sa2va_opsd_refcoco_internvl3_4b_v3_resume_from_iter800       --resume /mnt/checkpoints/iter_800.pth
  '
```

## 10. 交给师兄的最短说明模板

可以直接把下面这段发给师兄：

```text
代码我会放到 GitHub，环境我会打成 Docker 镜像推到 DockerHub。

这次训练使用：
- 环境依赖：uv sync --extra=legacy
- 模型：ByteDance/Sa2VA-4B
- 数据目录：refcoco 原始目录结构（instances.json / refs(unc).p / refs(google).p / train2014）
- 训练脚本：bash tools/train_refcoco_opsd_4b.sh
- GPU：1 卡
- 恢复训练：--resume /mnt/checkpoints/iter_800.pth

实际运行命令：
bash tools/train_refcoco_opsd_4b.sh   --gpus 1   --cuda-devices 0   --data-root /mnt/data/refcoco   --image-root /mnt/data/refcoco/train2014   --model-path /mnt/models/Sa2VA-4B   --tokenizer-path /mnt/models/Sa2VA-4B   --work-dir /mnt/output/sa2va_opsd_refcoco_internvl3_4b_v3_resume_from_iter800   --resume /mnt/checkpoints/iter_800.pth
```
