from projects.sa2va.configs.sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3 import *  # noqa: F401,F403

from projects.sa2va.datasets.common import DEFAULT_MASK_TO_REFERRING_QUESTION
from projects.sa2va.models.sa2va_opsd_referring_v3 import Sa2VAOPSDReferringModelV3


route_cache_dir = "./work_dirs/sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3_referring/route_cache"
route_manifest_path = f"{route_cache_dir}/routes_step_0000000.jsonl"
sam_confuser_pool_dir = "./work_dirs/sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3_referring/sam_confuser_pool"
work_dir = "./work_dirs/sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3_referring"

model["type"] = Sa2VAOPSDReferringModelV3
model["description_max_new_tokens"] = 24
model["grpo_sample_max_new_tokens"] = 24
model["min_caption_tokens"] = 2
model["caption_low_density_length_threshold"] = 12

train_dataset["student_question"] = DEFAULT_MASK_TO_REFERRING_QUESTION
train_dataset["route_manifest_path"] = route_manifest_path if use_manifest_routes else None
train_dataset["sam_confuser_pool_dir"] = sam_confuser_pool_dir

train_dataloader["dataset"]["student_question"] = DEFAULT_MASK_TO_REFERRING_QUESTION
train_dataloader["dataset"]["route_manifest_path"] = route_manifest_path if use_manifest_routes else None
train_dataloader["dataset"]["sam_confuser_pool_dir"] = sam_confuser_pool_dir

