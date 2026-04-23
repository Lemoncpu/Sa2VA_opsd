from os import getenv

from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import OptimWrapper
from torch.optim import AdamW

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop

from projects.sa2va.hooks.ema_teacher_hook import EMATeacherHook
from projects.sa2va.datasets.data_utils_opsd_v2 import sa2va_opsd_collect_fn_v2
from projects.sa2va.datasets.refcoco_opsd import Sa2VAOpsdRefCocoDataset
from projects.sa2va.models.sa2va_opsd_v2 import Sa2VAOPSDModelV2


path = getenv("SA2VA_PUBLIC_2B_MODEL_PATH", "/data/xyc/Sa2va_opsd/Sa2VA/pretrained/Sa2VA-4B")
tokenizer_path = getenv("SA2VA_PUBLIC_2B_TOKENIZER_PATH", path)
data_root = getenv("SA2VA_PUBLIC_2B_REFCOCO_DATA_ROOT", "/data/xyc")
dataset_name = getenv("SA2VA_PUBLIC_2B_REFCOCO_DATASET", "refcoco")
split = getenv("SA2VA_PUBLIC_2B_REFCOCO_SPLIT", "val")
image_root = getenv("SA2VA_PUBLIC_2B_REFCOCO_IMAGE_ROOT", "/data/xyc/refcoco/refcoco_val_100")
device = getenv("SA2VA_PUBLIC_2B_DEVICE", "cuda:0")

batch_size = 1
accumulative_counts = 1
dataloader_num_workers = 0
max_epochs = 1
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
save_steps = 1000000
save_total_limit = 1

student_question = (
    "<image>Can you provide me with a concise description of the region in the picture marked by region1? "
    "Answer with a short referring expression in RefCOCO style, usually 2 to 6 words, naming the target itself "
    "with only the most necessary attribute or location cue. Prefer forms like category plus color, size, "
    "left/right, top/bottom, or a nearby relation. Do not describe the whole scene. Do not write a full sentence "
    "and do not start with 'it is'. Do not output [SEG], masks, tags, or placeholder tokens."
)

model = dict(
    type=Sa2VAOPSDModelV2,
    model_path=path,
    enable_teacher=True,
    teacher_ema_alpha=0.999,
    tokenizer_path=tokenizer_path,
    device=device,
    torch_dtype="auto",
    use_flash_attn=True,
    teacher_temperature=1.0,
    jsd_beta=0.5,
    iou_low_threshold=-1.0,
    iou_high_threshold=0.0,
    mid_iou_alpha=1.0,
    entropy_weight_beta=1.0,
    grpo_group_size=2,
    grpo_clip_eps=0.2,
    grpo_advantage_eps=1e-6,
    grpo_sample_temperature=1.0,
    grpo_sample_top_p=1.0,
    grpo_sample_max_new_tokens=48,
    low_iou_regen_max_new_tokens=48,
    min_caption_tokens=4,
    reconstruct_question_template="<image>Please segment the region described as: {caption}",
)

train_dataset = dict(
    type=Sa2VAOpsdRefCocoDataset,
    data_root=data_root,
    dataset_name=dataset_name,
    split=split,
    image_root=image_root,
    image_root_candidates=[
        image_root,
        "/data/xyc/refcoco/refcoco_val_100",
        "/data/xyc/refcoco/train2014",
        "/data/xyc/images/mscoco/images/train2014",
        "/data/coco/train2014",
    ],
    repeats=1,
    shuffle=False,
    skip_empty_masks=True,
    student_question=student_question,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=sa2va_opsd_collect_fn_v2),
)

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
)

param_scheduler = []

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

custom_hooks = [
    dict(type=EMATeacherHook),
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=1),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

visualizer = None
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
work_dir = getenv("SA2VA_PUBLIC_2B_WORK_DIR", "./work_dirs/sa2va_opsd_refval100_v2_grpo_smoke")
