from mmengine.config import read_base

with read_base():
    from .sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3_referring import *  # noqa: F401,F403


route_mode = "online"
use_manifest_routes = False

model["use_online_route_for_loss"] = True
model["referring_hard_loss_weight_base"] = 1.2
model["referring_hard_loss_weight_compound"] = 1.4
model["referring_hard_loss_weight_max"] = 1.55
model["enable_referring_direct_mask_loss"] = True
model["referring_direct_mask_loss_weight"] = 0.7
model["referring_teacher_direct_mask_loss_weight"] = 0.35
model["referring_confuser_separation_loss_weight"] = 0.15
model["referring_direct_mask_loss_min_iou_gate"] = 0.0

train_dataset["route_manifest_path"] = None
train_dataset["route_manifest_required"] = False
train_dataset["skip_route_manifest_skip_samples"] = False

train_dataloader["dataset"]["route_manifest_path"] = None
train_dataloader["dataset"]["route_manifest_required"] = False
train_dataloader["dataset"]["skip_route_manifest_skip_samples"] = False

train_dataloader["sampler"] = dict(
    type=DefaultSampler,
    shuffle=True,
)

custom_hooks = [dict(type=EMATeacherHook)]
