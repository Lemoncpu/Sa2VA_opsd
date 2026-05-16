from mmengine.config import read_base

with read_base():
    from .sa2va_opsd_refcoco_internvl3_2b_v3 import *  # noqa: F401,F403


route_mode = "online"
use_manifest_routes = False

model["use_online_route_for_loss"] = True

train_dataset["route_manifest_path"] = None
train_dataset["route_manifest_latest_path"] = None
train_dataset["route_manifest_required"] = False
train_dataset["skip_route_manifest_skip_samples"] = False

train_dataloader["dataset"]["route_manifest_path"] = None
train_dataloader["dataset"]["route_manifest_latest_path"] = None
train_dataloader["dataset"]["route_manifest_required"] = False
train_dataloader["dataset"]["skip_route_manifest_skip_samples"] = False

train_dataloader["sampler"] = dict(
    type=DefaultSampler,
    shuffle=True,
)

custom_hooks = [dict(type=EMATeacherHook)]
