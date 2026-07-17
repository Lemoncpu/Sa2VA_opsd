from projects.sa2va.configs.sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3 import *  # noqa: F401,F403

from projects.sa2va.datasets.referring_prompts import DEFAULT_MASK_TO_REFERRING_QUESTION
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
model["referring_hard_loss_weight_base"] = 1.2
model["referring_hard_loss_weight_compound"] = 1.4
model["referring_hard_loss_weight_max"] = 1.55
model["enable_referring_direct_mask_loss"] = True
model["referring_direct_mask_loss_weight"] = 0.7
model["referring_teacher_direct_mask_loss_weight"] = 0.35
model["referring_confuser_separation_loss_weight"] = 0.15
model["referring_direct_mask_loss_min_iou_gate"] = 0.0
model["teacher_update_mode"] = "frozen_snapshot"
model["referring_type_conditioned_candidate_count_per_type"] = 2
model["enable_referring_onpolicy_type_guidance"] = True
model["referring_onpolicy_min_posterior_gain"] = 0.08
model["referring_onpolicy_min_posterior_iou"] = 0.55
model["referring_onpolicy_min_token_overlap"] = 0.5
model["referring_onpolicy_drop_token_weight"] = 1.6
model["referring_onpolicy_keep_token_weight"] = 1.3
model["referring_onpolicy_position_token_weight"] = 1.4
model["referring_enable_posterior_type_explanation"] = True

train_dataset["student_question"] = DEFAULT_MASK_TO_REFERRING_QUESTION
train_dataset["route_manifest_path"] = route_manifest_path if use_manifest_routes else None
train_dataset["sam_confuser_pool_dir"] = sam_confuser_pool_dir

train_dataloader["dataset"]["student_question"] = DEFAULT_MASK_TO_REFERRING_QUESTION
train_dataloader["dataset"]["route_manifest_path"] = route_manifest_path if use_manifest_routes else None
train_dataloader["dataset"]["sam_confuser_pool_dir"] = sam_confuser_pool_dir
