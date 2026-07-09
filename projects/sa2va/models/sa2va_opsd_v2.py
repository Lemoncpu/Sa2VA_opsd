import inspect
import random
import re
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModel
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from projects.sa2va.datasets.common import SEG_QUESTIONS
from projects.sa2va.evaluation.teacher_diagnosis_common import (
    GRPO_POSITIVE_ROUTE,
    ON_POLICY_DISTILL_ROUTE,
    TEACHER_REGENERATE_ROUTE,
    build_mask_relation_context,
    build_teacher_regenerate_difference_context,
    classify_teacher_route,
)


if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

try:
    torch.is_autocast_enabled("cuda")
except TypeError:
    _orig_torch_is_autocast_enabled = torch.is_autocast_enabled

    def _compat_is_autocast_enabled(device_type=None):
        del device_type
        return _orig_torch_is_autocast_enabled()

    torch.is_autocast_enabled = _compat_is_autocast_enabled


@dataclass
class DescriptionResult:
    raw_prediction: str
    clean_caption: str
    completion_ids: torch.Tensor
    status: str
    raw_failure_mode: str = ""
    clean_status: str = ""


@dataclass
class ReconstructionResult:
    pred_mask: object
    question: object
    raw_prediction: str
    prediction_masks_count: int
    seg_token_count: int
    status: str


@dataclass
class ConfuserSelectionResult:
    option_probs: torch.Tensor
    predicted_option_idx: int
    correct_option_idx: int
    reward: float
    reward_raw: float
    selected_correct: bool
    correct_option_prob: float
    confuser_iou_weights: torch.Tensor
    confuser_penalty: float


@dataclass
class TeacherRegeneratePipelineResult:
    diagnosis_raw: str = ""
    structured_diagnosis_raw: str = ""
    single_stage_raw: str = ""
    problem_raw: str = ""
    direction_raw: str = ""
    reason_raw: str = ""
    dlc_raw: str = ""
    verification_raw: str = ""
    repair_raw: str = ""
    target_summary: str = ""
    distractor_summary: str = ""
    shared_evidence: str = ""
    target_only_evidence: str = ""
    distractor_only_evidence: str = ""
    difference_focus: str = ""
    target_localization_hint: str = ""
    distractor_localization_hint: str = ""
    likely_drift_reason: str = ""
    caption_problem: str = ""
    correction_direction: str = ""
    reason: str = ""
    confidence: str = ""
    target_anchor: str = ""
    distractor_anchor: str = ""
    detailed_caption: str = ""
    verification_caption: str = ""
    detailed_completion_ids: torch.Tensor = None
    difference_context_nontrivial: bool = False
    diagnosis_valid: bool = False
    problem_valid: bool = False
    direction_valid: bool = False
    reason_valid: bool = False
    detailed_status: str = "empty"
    verification_status: str = "empty"
    verification_iou: float = 0.0
    gate_passed: bool = False
    reason_is_coarse: bool = False
    teacher_pipeline_mode: str = "single_stage"
    teacher_diagnosis_retry_count: int = 0
    teacher_dlc_candidate_count: int = 0
    teacher_dlc_valid_candidate_count: int = 0
    teacher_dlc_selected_by: str = ""
    teacher_verification_used: bool = False
    teacher_fallback_used: bool = False
    teacher_fallback_reason: str = ""
    teacher_selected_caption_source: str = ""
    teacher_dlc_candidate_scores: tuple = ()
    difference_context_failure_reason: str = ""
    diagnosis_failure_reason: str = ""
    problem_failure_reason: str = ""
    direction_failure_reason: str = ""
    reason_failure_reason: str = ""
    detailed_failure_reason: str = ""
    verification_failure_reason: str = ""
    stop_stage: str = "difference_context"


class Sa2VAOPSDModelV2(BaseModel):
    """Standalone OPSD implementation aligned to official sample.py usage."""

    def __init__(
        self,
        model_path,
        teacher_model_path=None,
        enable_teacher=True,
        teacher_ema_alpha=0.999,
        tokenizer_path=None,
        torch_dtype="auto",
        teacher_temperature=1.0,
        jsd_beta=0.5,
        privileged_iou_precision=4,
        iou_low_threshold=0.5,
        iou_high_threshold=0.9,
        mid_iou_alpha=1.0,
        entropy_weight_beta=1.0,
        grpo_group_size=2,
        grpo_clip_eps=0.2,
        grpo_advantage_eps=1e-6,
        grpo_sample_temperature=1.0,
        grpo_sample_top_p=1.0,
        grpo_confuser_num_options=4,
        grpo_confuser_num_negatives=3,
        grpo_confuser_min_candidates=3,
        grpo_confuser_duplicate_iou_threshold=0.7,
        grpo_confuser_min_area_ratio=0.001,
        grpo_confuser_max_area_ratio=0.95,
        grpo_confuser_nearby_center_weight=0.25,
        grpo_confuser_overlap_weight=1.0,
        grpo_confuser_answer_max_new_tokens=1,
        grpo_confuser_zero_reward_on_wrong=True,
        description_max_new_tokens=96,
        description_repetition_penalty=1.1,
        description_no_repeat_ngram_size=4,
        grpo_sample_max_new_tokens=48,
        low_iou_regen_max_new_tokens=48,
        teacher_summary_template=None,
        reconstruct_question_template=None,
        reconstruct_question_templates=None,
        min_caption_tokens=4,
        device="cuda:0",
        use_flash_attn=True,
        use_mask_focused_caption_image=True,
        mask_focused_context_mode="grayscale",
        caption_quality_reward_weight=0.0,
        caption_reward_valid_bonus=0.05,
        caption_reward_sufficient_bonus=0.1,
        caption_reward_generic_penalty=0.25,
        caption_reward_repetition_penalty=0.2,
        caption_reward_truncated_penalty=0.35,
        caption_reward_empty_penalty=0.5,
        caption_reward_scene_spill_penalty=0.2,
        caption_reward_low_density_penalty=0.2,
        caption_low_density_length_threshold=28,
        grpo_low_iou_penalty=0.0,
        grpo_very_low_iou_penalty=0.0,
        grpo_zero_iou_penalty=0.0,
        grpo_missing_mask_penalty=0.0,
        enable_invalid_caption_recovery=True,
        use_online_route_for_loss=True,
        max_teacher_regenerate_fraction=0.2,
        max_recovery_fraction=0.1,
        enable_debug_sample_logging=False,
        student_freeze_llm=True,
        student_freeze_visual_encoder=True,
        student_llm_lora=None,
        rolling_metric_window_iters=10,
    ):
        super().__init__()
        if teacher_model_path == "__skip__":
            raise ValueError(
                'teacher_model_path="__skip__" has been removed. '
                "Use enable_teacher=False for teacher-free evaluation, or omit "
                "teacher_model_path to use the student EMA teacher."
            )
        self.model_path = model_path
        self.enable_teacher = bool(enable_teacher)
        self.teacher_ema_alpha = float(teacher_ema_alpha)
        if not (0.0 < self.teacher_ema_alpha <= 1.0):
            raise ValueError(
                f"teacher_ema_alpha must be in (0, 1], got {self.teacher_ema_alpha}."
            )
        if self.enable_teacher and teacher_model_path not in {None, "", model_path}:
            raise ValueError(
                "External teacher_model_path is no longer supported. "
                "The teacher must be the EMA version of the student, so omit "
                "teacher_model_path or set it to model_path."
            )
        self.tokenizer_path = tokenizer_path or model_path
        self.teacher_temperature = teacher_temperature
        self.jsd_beta = jsd_beta
        self.privileged_iou_precision = privileged_iou_precision
        self.iou_low_threshold = float(iou_low_threshold)
        self.iou_high_threshold = float(iou_high_threshold)
        self.mid_iou_alpha = float(mid_iou_alpha)
        self.entropy_weight_beta = float(entropy_weight_beta)
        self.grpo_group_size = max(int(grpo_group_size), 0)
        self.grpo_clip_eps = float(grpo_clip_eps)
        self.grpo_advantage_eps = float(grpo_advantage_eps)
        self.grpo_sample_temperature = float(grpo_sample_temperature)
        self.grpo_sample_top_p = float(grpo_sample_top_p)
        self.grpo_confuser_num_options = max(int(grpo_confuser_num_options), 2)
        self.grpo_confuser_num_negatives = max(int(grpo_confuser_num_negatives), 1)
        self.grpo_confuser_min_candidates = max(int(grpo_confuser_min_candidates), self.grpo_confuser_num_negatives)
        self.grpo_confuser_duplicate_iou_threshold = float(grpo_confuser_duplicate_iou_threshold)
        self.grpo_confuser_min_area_ratio = float(grpo_confuser_min_area_ratio)
        self.grpo_confuser_max_area_ratio = float(grpo_confuser_max_area_ratio)
        self.grpo_confuser_nearby_center_weight = float(grpo_confuser_nearby_center_weight)
        self.grpo_confuser_overlap_weight = float(grpo_confuser_overlap_weight)
        self.grpo_confuser_answer_max_new_tokens = max(int(grpo_confuser_answer_max_new_tokens), 1)
        self.grpo_confuser_zero_reward_on_wrong = bool(grpo_confuser_zero_reward_on_wrong)
        self.description_max_new_tokens = max(int(description_max_new_tokens), 1)
        self.description_repetition_penalty = float(description_repetition_penalty)
        self.description_no_repeat_ngram_size = max(int(description_no_repeat_ngram_size), 0)
        self.grpo_sample_max_new_tokens = max(int(grpo_sample_max_new_tokens), 1)
        self.low_iou_regen_max_new_tokens = max(int(low_iou_regen_max_new_tokens), 1)
        if self.grpo_confuser_num_options != self.grpo_confuser_num_negatives + 1:
            raise ValueError(
                "grpo_confuser_num_options must equal grpo_confuser_num_negatives + 1, got "
                f"{self.grpo_confuser_num_options} and {self.grpo_confuser_num_negatives}."
            )
        if self.iou_low_threshold > self.iou_high_threshold:
            raise ValueError(
                f"iou_low_threshold must be <= iou_high_threshold, got "
                f"{self.iou_low_threshold} > {self.iou_high_threshold}."
            )
        if self.enable_teacher and not self._teacher_routes_require_teacher_model():
            print(
                "[Sa2VA_OPSD_V2] enable_teacher=True but the configured IoU thresholds "
                "can only route to GRPO. Skipping teacher model allocation."
            )
            self.enable_teacher = False
        self.teacher_model_path = None if not self.enable_teacher else (teacher_model_path or model_path)
        self.device = torch.device(device)
        self.use_flash_attn = use_flash_attn
        self.min_caption_tokens = max(int(min_caption_tokens), 1)
        self.use_mask_focused_caption_image = bool(use_mask_focused_caption_image)
        self.mask_focused_context_mode = str(mask_focused_context_mode).lower().strip()
        del caption_quality_reward_weight
        self.caption_reward_valid_bonus = float(caption_reward_valid_bonus)
        self.caption_reward_sufficient_bonus = float(caption_reward_sufficient_bonus)
        self.caption_reward_generic_penalty = float(caption_reward_generic_penalty)
        self.caption_reward_repetition_penalty = float(caption_reward_repetition_penalty)
        self.caption_reward_truncated_penalty = float(caption_reward_truncated_penalty)
        self.caption_reward_empty_penalty = float(caption_reward_empty_penalty)
        self.caption_reward_scene_spill_penalty = float(caption_reward_scene_spill_penalty)
        self.caption_reward_low_density_penalty = float(caption_reward_low_density_penalty)
        self.caption_low_density_length_threshold = max(int(caption_low_density_length_threshold), 1)
        del grpo_low_iou_penalty, grpo_very_low_iou_penalty, grpo_zero_iou_penalty, grpo_missing_mask_penalty
        self.enable_invalid_caption_recovery = bool(enable_invalid_caption_recovery)
        self.use_online_route_for_loss = bool(use_online_route_for_loss)
        self.max_teacher_regenerate_fraction = float(max_teacher_regenerate_fraction)
        self.max_recovery_fraction = float(max_recovery_fraction)
        self.enable_debug_sample_logging = bool(enable_debug_sample_logging)
        self.student_freeze_llm = bool(student_freeze_llm)
        self.student_freeze_visual_encoder = bool(student_freeze_visual_encoder)
        self.student_llm_lora = student_llm_lora
        self.rolling_metric_window_iters = max(int(rolling_metric_window_iters), 1)
        self._cumulative_valid_count = 0
        self._cumulative_description_ok_count = 0
        self._cumulative_description_empty_count = 0
        self._cumulative_description_truncated_count = 0
        self._cumulative_description_seg_style_count = 0
        self._cumulative_reconstruct_ok_count = 0
        self._cumulative_reconstruct_failed_count = 0
        self._cumulative_reconstruct_skip_count = 0
        self._cumulative_reconstruct_invalid_caption_skip_count = 0
        self._cumulative_reconstruct_empty_prediction_masks_count = 0
        self._cumulative_empty_gt_mask_count = 0
        self._cumulative_seg_correct_count = 0
        self._cumulative_iou_sum = 0.0
        self._cumulative_nonempty_gt_count = 0
        self._cumulative_nonempty_caption_count = 0
        self._cumulative_caption_token_sum = 0.0
        self._cumulative_loss_count = 0
        self._cumulative_teacher_regenerate_count = 0
        self._cumulative_on_policy_distill_count = 0
        self._cumulative_grpo_positive_count = 0
        self._cumulative_regen_loss_count = 0
        self._cumulative_onpolicy_loss_count = 0
        self._cumulative_grpo_loss_count = 0
        self._cumulative_total_loss_sum = 0.0
        self._cumulative_regen_ce_sum = 0.0
        self._cumulative_onpolicy_jsd_sum = 0.0
        self._cumulative_grpo_sum = 0.0
        self._cumulative_grpo_reward_sum = 0.0
        self._cumulative_grpo_reward_count = 0
        self._cumulative_grpo_mcq_correct_count = 0
        self._cumulative_grpo_mcq_count = 0
        self._cumulative_grpo_mcq_correct_conf_sum = 0.0
        self._cumulative_recovery_caption_count = 0
        self._cumulative_invalid_caption_penalty_count = 0
        self._cumulative_hard_reconstruct_failure_count = 0
        self._cumulative_teacher_regenerate_ce_applied_count = 0
        self._cumulative_teacher_regenerate_suppressed_count = 0
        self._cumulative_teacher_regenerate_verified_count = 0
        self._cumulative_teacher_regenerate_rejected_count = 0
        self._cumulative_teacher_regenerate_verified_iou_sum = 0.0
        self._cumulative_teacher_regenerate_analysis_count = 0
        self._cumulative_teacher_difference_context_nontrivial_count = 0
        self._cumulative_teacher_problem_valid_count = 0
        self._cumulative_teacher_direction_valid_count = 0
        self._cumulative_teacher_reason_valid_count = 0
        self._cumulative_teacher_reason_coarse_count = 0
        self._cumulative_teacher_diagnosis_valid_count = 0
        self._cumulative_teacher_dlc_valid_count = 0
        self._cumulative_teacher_regenerate_verification_valid_count = 0
        self._cumulative_teacher_regenerate_verification_iou_sum = 0.0
        self._cumulative_recovery_ce_applied_count = 0
        self._cumulative_recovery_suppressed_count = 0
        self._cumulative_scene_spill_caption_count = 0
        self._cumulative_low_density_long_caption_count = 0
        self._cumulative_detail_sufficient_caption_count = 0
        self._cumulative_generic_caption_count = 0
        self._cumulative_repetitive_caption_count = 0
        self._metric_window = deque(maxlen=self.rolling_metric_window_iters)
        self._caption_bad_words_ids = None

        del teacher_summary_template
        self.reconstruct_question_template = reconstruct_question_template or (
            "<image>\n"
            "Return the segmentation mask for the target region referred to by the description below.\n"
            "Use nearby objects, scene cues, or local relations only to identify the target region.\n"
            "Do not include those contextual regions in the mask unless they are explicitly part of the described target.\n"
            "Description: {caption}"
        )
        self.reconstruct_question_templates = tuple(
            reconstruct_question_templates
            or (
                self.reconstruct_question_template,
                *[f"<image>{template}" for template in SEG_QUESTIONS],
            )
        )

        self._validate_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False,
        )
        self._caption_bad_words_ids = self._build_caption_bad_words_ids()
        self._grpo_option_letters = tuple(chr(ord("A") + idx) for idx in range(self.grpo_confuser_num_options))
        self._grpo_option_token_ids = self._resolve_grpo_option_token_ids()
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        except Exception:
            self.processor = None
        self.model_dtype = self._resolve_torch_dtype(torch_dtype)
        self.student_model = self._load_model(self.model_path)
        object.__setattr__(self, "old_policy_model", self._load_model(self.model_path))
        self._sync_old_policy()
        self.teacher_model = None
        if self.enable_teacher:
            self.teacher_model = self._load_model(self.teacher_model_path)
            self._sync_teacher()

    def _validate_device(self):
        if self.device.type != "cuda":
            raise ValueError(f"Sa2VAOPSDModelV2 only supports CUDA. Got {self.device}.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Sa2VA_OPSD V2 training.")
        if torch.cuda.device_count() <= (self.device.index or 0):
            raise RuntimeError(
                f"Configured device {self.device} is not visible. "
                f"Visible device count={torch.cuda.device_count()}."
            )

    def _teacher_routes_require_teacher_model(self):
        # Teacher is only needed when either the low-IoU regeneration route or
        # the mid-band on-policy distillation route can be reached for IoU in [0, 1].
        teacher_regenerate_possible = self.iou_low_threshold >= 0.0
        onpolicy_lower = max(self.iou_low_threshold, 0.0)
        onpolicy_upper = min(self.iou_high_threshold, 1.0)
        onpolicy_possible = onpolicy_lower < onpolicy_upper
        return teacher_regenerate_possible or onpolicy_possible

    @staticmethod
    def _resolve_torch_dtype(torch_dtype):
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        if torch_dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch_dtype

    def _load_model(self, model_path):
        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=self.model_dtype,
            use_flash_attn=self.use_flash_attn,
        )

        def _is_all_tied_weights_keys_error(exc):
            return isinstance(exc, AttributeError) and "all_tied_weights_keys" in str(exc)

        try:
            model = AutoModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **load_kwargs,
            )
            model.to(self.device)
        except AttributeError as exc:
            if not _is_all_tied_weights_keys_error(exc):
                raise
            model = AutoModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=False,
                **load_kwargs,
            )
            model.to(self.device)
        except NotImplementedError as exc:
            if "meta tensor" not in str(exc):
                raise
            try:
                model = AutoModel.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    device_map={"": str(self.device)},
                    **load_kwargs,
                )
            except AttributeError as attr_exc:
                if not _is_all_tied_weights_keys_error(attr_exc):
                    raise
                model = AutoModel.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=False,
                    **load_kwargs,
                )
                model.to(self.device)
        self._prefer_non_reentrant_gradient_checkpointing(model)
        self._ensure_runtime_state(model)
        self._configure_student_trainability(model)
        self._ensure_generation_ready(model)
        return model

    def _configure_student_trainability(self, model):
        language_model = getattr(model, "language_model", None)
        visual_model = getattr(model, "visual_model", None)
        if visual_model is None:
            visual_model = getattr(model, "visual", None)

        if self.student_freeze_llm and language_model is not None:
            language_model.requires_grad_(False)
        if self.student_freeze_visual_encoder and visual_model is not None:
            visual_model.requires_grad_(False)

        if self.student_llm_lora:
            self._apply_student_llm_lora(model)

    def _apply_student_llm_lora(self, model):
        language_model = getattr(model, "language_model", None)
        if language_model is None:
            raise ValueError("student_llm_lora requires the model to expose language_model.")

        lora_config = self.student_llm_lora
        if isinstance(lora_config, dict):
            lora_config = dict(lora_config)
            lora_config.setdefault("task_type", "CAUSAL_LM")
            lora_config.setdefault("bias", "none")
            lora_config.setdefault("target_modules", None)
            lora_config.setdefault("modules_to_save", ["lm_head", "embed_tokens"])
            lora_config = LoraConfig(**lora_config)
        if getattr(lora_config, "target_modules", None) is None:
            target_modules = []
            for name, module in language_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.append(name)
            lora_config.target_modules = target_modules

        language_model = get_peft_model(language_model, lora_config)
        enable_input_require_grads = getattr(language_model, "enable_input_require_grads", None)
        if callable(enable_input_require_grads):
            enable_input_require_grads()
        model.language_model = language_model

    @staticmethod
    def _prefer_non_reentrant_gradient_checkpointing(model):
        candidate_modules = Sa2VAOPSDModelV2._gradient_checkpointing_candidates(model)
        for candidate in candidate_modules:
            enable_fn = getattr(candidate, "gradient_checkpointing_enable", None)
            if not callable(enable_fn):
                continue
            if not bool(getattr(candidate, "supports_gradient_checkpointing", False)):
                continue
            try:
                is_enabled = bool(getattr(candidate, "is_gradient_checkpointing", False))
            except Exception:
                is_enabled = False
            if not is_enabled:
                continue
            try:
                enable_fn(gradient_checkpointing_kwargs={"use_reentrant": False})
            except (TypeError, ValueError):
                continue
        Sa2VAOPSDModelV2._patch_legacy_gradient_checkpointing_modules(model)

    @staticmethod
    def _patch_legacy_gradient_checkpointing_modules(model):
        for module in model.modules():
            if type(module).__name__ == "InternVisionEncoder":
                Sa2VAOPSDModelV2._patch_intern_vision_encoder_forward(module)

    @staticmethod
    def _resolve_non_reentrant_checkpoint_fn(module):
        checkpoint_fn = getattr(module, "_gradient_checkpointing_func", None)
        if callable(checkpoint_fn):
            return checkpoint_fn

        def _checkpoint(function, *args):
            return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False)

        return _checkpoint

    @staticmethod
    def _patch_intern_vision_encoder_forward(module):
        if getattr(module, "_opsd_non_reentrant_checkpoint_patched", False):
            return

        def _forward(self, inputs_embeds, output_hidden_states=None, return_dict=None):
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            encoder_states = () if output_hidden_states else None
            hidden_states = inputs_embeds
            checkpoint_fn = Sa2VAOPSDModelV2._resolve_non_reentrant_checkpoint_fn(self)

            for encoder_layer in self.layers:
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                if self.gradient_checkpointing and self.training:
                    layer_outputs = checkpoint_fn(encoder_layer, hidden_states)
                else:
                    layer_outputs = encoder_layer(hidden_states)
                hidden_states = layer_outputs

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, encoder_states] if v is not None)
            return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)

        module.forward = MethodType(_forward, module)
        module._opsd_non_reentrant_checkpoint_patched = True

    @staticmethod
    def _gradient_checkpointing_candidates(model):
        candidate_modules = []
        for candidate in (
            model,
            getattr(model, "model", None),
            getattr(model, "language_model", None),
            getattr(model, "vision_model", None),
        ):
            if candidate is not None and all(id(candidate) != id(existing) for existing in candidate_modules):
                candidate_modules.append(candidate)
        return candidate_modules

    @staticmethod
    def _ensure_runtime_state(model):
        if not hasattr(model, "_count"):
            model._count = 0

    @staticmethod
    def _patch_sa2va_chat_generate_dtype(model):
        generate_fn = getattr(model, "generate", None)
        language_model = getattr(model, "language_model", None)
        extract_feature = getattr(model, "extract_feature", None)
        if not callable(generate_fn) or language_model is None or not callable(extract_feature):
            return
        if getattr(model, "_opsd_generate_dtype_patched", False):
            return

        def _generate(
            self,
            pixel_values=None,
            input_ids=None,
            attention_mask=None,
            visual_features=None,
            generation_config=None,
            output_hidden_states=None,
            return_dict=None,
            prompt_masks=None,
            vp_overall_mask=None,
            **generate_kwargs,
        ):
            device = self.device
            assert self.img_context_token_id is not None
            lm_compute_dtype = None
            try:
                lm_compute_dtype = next(self.language_model.parameters()).dtype
            except StopIteration:
                lm_compute_dtype = None

            if pixel_values is not None:
                if visual_features is not None:
                    vit_embeds = visual_features
                else:
                    if type(pixel_values) is list or pixel_values.ndim == 5:
                        if type(pixel_values) is list:
                            pixel_values = [
                                x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                            ]
                        pixel_values = torch.cat(
                            [image.to(self.vision_model.dtype) for image in pixel_values], dim=0
                        )

                    vit_embeds = self.extract_feature(pixel_values.to(device))
                image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
                image_flags = image_flags.long()
                vit_embeds = vit_embeds[image_flags == 1]

                input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)

                if vp_overall_mask is not None and prompt_masks is not None:
                    vp_embeds = []
                    vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
                    prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

                    vp_overall_mask = vp_overall_mask[image_flags == 1]
                    overall_tile_vit_embeds = vit_embeds[vp_overall_mask]

                    i_vp_img = 0
                    for i_img in range(len(vit_embeds)):
                        vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                        if vp_overall_mask[i_img]:
                            tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)
                            objects_prompt_masks = prompt_masks[i_vp_img]
                            n_obj = len(objects_prompt_masks)
                            tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                            objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                            vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                            i_vp_img += 1

                    vp_embeds = torch.cat(vp_embeds, dim=0)
                else:
                    vp_embeds = None

                input_ids = input_ids.reshape(B * N)
                selected = input_ids == self.img_context_token_id
                assert selected.sum() != 0
                if vp_embeds is None:
                    input_embeds[selected] = vit_embeds.reshape(-1, C).to(
                        device=input_embeds.device,
                        dtype=input_embeds.dtype,
                    )
                else:
                    reshaped_vp_embeds = vp_embeds.reshape(-1, C).to(
                        device=input_embeds.device,
                        dtype=input_embeds.dtype,
                    )
                    if len(input_embeds[selected]) != len(reshaped_vp_embeds):
                        print(
                            "Shape mismatch, selected is {}, vp embeds is {} !!!".format(
                                len(input_embeds[selected]), len(reshaped_vp_embeds)
                            )
                        )
                        min_tokens = min(len(input_embeds[selected]), len(reshaped_vp_embeds))
                        input_embeds[selected][:min_tokens] = reshaped_vp_embeds[:min_tokens]
                    else:
                        input_embeds[selected] = reshaped_vp_embeds

                input_embeds = input_embeds.reshape(B, N, C)
            else:
                input_embeds = self.language_model.get_input_embeddings()(input_ids)

            if lm_compute_dtype in {torch.float16, torch.bfloat16} and input_embeds.dtype != lm_compute_dtype:
                input_embeds = input_embeds.to(dtype=lm_compute_dtype)

            outputs = self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask.to(device),
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                **generate_kwargs,
            )

            return outputs

        model.generate = MethodType(_generate, model)
        model._opsd_generate_dtype_patched = True

    def _ensure_generation_ready(self, model):
        self._ensure_runtime_state(model)
        self._patch_sa2va_chat_generate_dtype(model)
        prepare_fn = getattr(model, "preparing_for_generation", None)
        if callable(prepare_fn):
            if not getattr(model, "init_prediction_config", False) or not hasattr(model, "stop_criteria"):
                prepare_fn(tokenizer=self.tokenizer)
        self._ensure_runtime_state(model)
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map is None:
            model.to(self.device)

    def has_teacher_model(self):
        return self.enable_teacher and self.teacher_model is not None

    def has_old_policy_model(self):
        return getattr(self, "old_policy_model", None) is not None

    def require_teacher_model(self, context="this operation"):
        if self.has_teacher_model():
            return self.teacher_model
        raise RuntimeError(
            f"{context} requires enable_teacher=True. "
            "Teacher-free evaluation does not construct teacher_model."
        )

    def require_old_policy_model(self, context="this operation"):
        old_policy_model = getattr(self, "old_policy_model", None)
        if old_policy_model is not None:
            return old_policy_model
        raise RuntimeError(f"{context} requires old_policy_model to be initialized.")

    def train(self, mode=True):
        super().train(mode)
        if self.has_old_policy_model():
            self.old_policy_model.eval()
        if self.has_teacher_model():
            self.teacher_model.eval()
        return self

    def to(self, *args, **kwargs):
        target_dtype = kwargs.get("dtype")
        if target_dtype is None and len(args) == 1 and isinstance(args[0], torch.dtype):
            target_dtype = args[0]
        if target_dtype is not None:
            self.student_model.to(dtype=target_dtype)
            if self.has_old_policy_model():
                self.old_policy_model.to(dtype=target_dtype)
            if self.has_teacher_model():
                self.teacher_model.to(dtype=target_dtype)
        self.student_model.to(self.device)
        if self.has_old_policy_model():
            self.old_policy_model.to(self.device)
            self.old_policy_model.eval()
        if self.has_teacher_model():
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
        return self

    def _sync_old_policy(self):
        if not self.has_old_policy_model():
            return False
        self.old_policy_model.load_state_dict(self.student_model.state_dict(), strict=False)
        self.old_policy_model.to(self.device)
        self.old_policy_model.requires_grad_(False)
        self.old_policy_model.eval()
        return True

    def _sync_teacher(self):
        if not self.has_teacher_model():
            return
        self.teacher_model.load_state_dict(self.student_model.state_dict(), strict=False)
        self.teacher_model.to(self.device)
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

    @torch.no_grad()
    def update_teacher_ema(self, alpha=None):
        if not self.has_teacher_model():
            return False

        ema_alpha = self.teacher_ema_alpha if alpha is None else float(alpha)
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"EMA alpha must be in (0, 1], got {ema_alpha}.")

        teacher_params = dict(self.teacher_model.named_parameters())
        for name, student_param in self.student_model.named_parameters():
            teacher_param = teacher_params.get(name)
            if teacher_param is None:
                continue
            student_data = student_param.detach()
            if teacher_param.is_floating_point():
                teacher_param.mul_(ema_alpha).add_(student_data, alpha=1.0 - ema_alpha)
            else:
                teacher_param.copy_(student_data)

        teacher_buffers = dict(self.teacher_model.named_buffers())
        for name, student_buffer in self.student_model.named_buffers():
            teacher_buffer = teacher_buffers.get(name)
            if teacher_buffer is None:
                continue
            teacher_buffer.copy_(student_buffer.detach())

        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        return True

    def _metric_tensor(self, value, dtype):
        return torch.tensor(float(value), device=self.device, dtype=dtype)

    def state_dict(self, *args, **kwargs):
        full_state = super().state_dict(*args, **kwargs)
        student_only_state = {
            k: v
            for k, v in full_state.items()
            if k.startswith("student_model.")
        }
        return student_only_state

    def load_state_dict(self, state_dict, strict=True):
        has_teacher_state = any(k.startswith("teacher_model.") for k in state_dict)
        filtered_state = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith(("old_policy_model.",))
        }

        if not self.has_teacher_model():
            student_only_state = {
                k: v
                for k, v in filtered_state.items()
                if not k.startswith(("teacher_model.",))
            }
            result = super().load_state_dict(student_only_state, strict=strict)
            self._sync_old_policy()
            return result

        if has_teacher_state:
            result = super().load_state_dict(filtered_state, strict=strict)
            self._sync_old_policy()
            return result

        student_only_state = {
            k: v
            for k, v in filtered_state.items()
            if not k.startswith(("teacher_model.",))
        }
        result = super().load_state_dict(student_only_state, strict=False)
        self._sync_teacher()
        self._sync_old_policy()
        return result

    @staticmethod
    def _to_numpy_mask(mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        return (np.asarray(mask) > 0).astype(np.uint8)

    @staticmethod
    def _normalize_prompt_masks_array(prompt_masks):
        if isinstance(prompt_masks, np.ndarray):
            masks = prompt_masks.astype(np.float32)
            if masks.ndim == 2:
                masks = np.expand_dims(masks, axis=0)
            if masks.ndim != 3:
                raise ValueError(f"prompt_masks must have shape (n_prompts, h, w), got {masks.shape}")
            return masks

        masks = [np.asarray(item, dtype=np.float32) for item in prompt_masks]
        if not masks:
            raise ValueError("prompt_masks is empty.")
        return np.stack(masks, axis=0)

    def _build_mask_focused_image(self, image, prompt_masks):
        if not self.use_mask_focused_caption_image:
            return image

        mask_stack = self._normalize_prompt_masks_array(prompt_masks)
        union_mask = (mask_stack > 0).any(axis=0).astype(np.uint8)
        target_h, target_w = image.size[1], image.size[0]
        if union_mask.shape != (target_h, target_w):
            union_mask_t = torch.from_numpy(union_mask[None, None].astype(np.float32))
            union_mask = F.interpolate(union_mask_t, size=(target_h, target_w), mode="nearest")[0, 0].numpy()
            union_mask = (union_mask > 0).astype(np.uint8)

        image_np = np.asarray(image.convert("RGB")).copy()
        if self.mask_focused_context_mode == "black":
            focused = np.zeros_like(image_np)
        else:
            grayscale = np.asarray(image.convert("L").convert("RGB"), dtype=np.uint8)
            focused = grayscale.copy()
        focused[union_mask.astype(bool)] = image_np[union_mask.astype(bool)]
        return Image.fromarray(focused, mode="RGB")

    @staticmethod
    def _clean_caption_text(caption):
        caption = "" if caption is None else str(caption)
        caption = caption.replace("<|im_end|>", "")
        caption = caption.replace("<|end|>", "")
        caption = caption.replace("<|endoftext|>", "")
        caption = re.sub(r"\s+", " ", caption).strip()
        for prefix in ("Sure, ", "Sure. ", "Certainly, "):
            if caption.startswith(prefix):
                caption = caption[len(prefix):].strip()
        caption = re.sub(r"^(sure|certainly|okay|ok|yes)[,:\.\s]+", "", caption, flags=re.IGNORECASE)
        # Keep the original sentence structure while removing formatting tags
        # emitted by the interleaved caption+segmentation output format.
        caption = re.sub(r"</?p>", "", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\[SEG\]\.?", "", caption, flags=re.IGNORECASE)
        # Generated EOS/control markers can be truncated at max_new_tokens and
        # leave tail fragments like "<|end" or a bare "<|".
        caption = re.sub(r"<\|[^>\n]*\|>", " ", caption)
        caption = re.sub(r"<\|.*$", "", caption)
        caption = re.sub(r"<[^>]+>", " ", caption)
        caption = re.sub(r"<[^>\n]*$", "", caption)
        caption = re.sub(r"(assistant|bot)\s*[:：]\s*", "", caption, flags=re.IGNORECASE)
        # Normalize referential scaffolds so training captions match DLC-style
        # direct descriptions instead of "the target / the region / region1 ..."
        # lead-ins.
        caption = re.sub(
            r"^(?:the\s+)?target(?:\s+(?:object|person|item|area|region))?"
            r"(?:\s+in\s+region\d+)?\s+(?:is|appears\s+to\s+be|looks\s+like)\s+",
            "",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(
            r"^(?:the\s+)?(?:object|person|item|area)\s+in\s+region\d+\s+"
            r"(?:is|appears\s+to\s+be|looks\s+like)\s+",
            "",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(
            r"^(?:the\s+)?region\d+\s+(?:shows|contains|is|appears\s+to\s+be|looks\s+like)\s+",
            "",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(
            r"^(?:the\s+)?region\s+(?:shows|contains)\s+",
            "",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(
            r"^(?:the\s+)?(?:target|object|person|item|area|region)\s*[:：]\s*",
            "",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(r"\s+", " ", caption)
        caption = re.sub(r"\s+([,.;:!?])", r"\1", caption)
        caption = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", caption)
        caption = caption.strip(" .,")
        return caption

    @staticmethod
    def _infer_description_status(caption):
        if caption is None:
            return "decode_error"
        normalized = caption.strip()
        if not normalized:
            return "empty"
        lowered = re.sub(r"[\s\.\!\?]+", " ", normalized.lower()).strip()
        generic_acknowledgements = {
            "sure",
            "certainly",
            "okay",
            "ok",
            "yes",
            "it",
            "this",
            "that",
            "the object",
            "the region",
        }
        if lowered in generic_acknowledgements:
            return "empty"
        if lowered in {"[seg]", "it is [seg]", "the segmentation result is [seg]", "segmentation result is [seg]"}:
            return "seg_style_answer"
        words = re.findall(r"[a-z0-9']+", lowered)
        if not words:
            return "empty"
        truncated_phrases = {
            "it is",
            "this is",
            "that is",
            "there is",
            "there are",
            "these are",
            "those are",
            "he is",
            "she is",
            "they are",
            "sure",
            "certainly",
            "okay",
            "ok",
            "yes",
            "a",
            "an",
            "the",
        }
        if lowered in truncated_phrases:
            return "truncated_caption"
        trailing_incomplete_tokens = {
            "a",
            "an",
            "the",
            "on",
            "in",
            "at",
            "with",
            "of",
            "to",
            "by",
            "from",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
        }
        if len(words) <= 3 and words[0] in {"it", "this", "that", "there", "he", "she", "they"}:
            return "truncated_caption"
        if len(words) <= 5 and words[-1] in trailing_incomplete_tokens:
            return "truncated_caption"
        return "ok"

    @staticmethod
    def _caption_token_count(caption):
        return len(re.findall(r"[A-Za-z0-9']+", caption or ""))

    def _is_caption_content_sufficient(self, caption):
        token_count = self._caption_token_count(caption)
        if token_count >= 1 and self._is_np_like_caption(caption):
            return True
        if token_count < self.min_caption_tokens:
            return False
        lowered = re.sub(r"\s+", " ", (caption or "").strip().lower())
        weak_prefixes = (
            "it is ",
            "this is ",
            "that is ",
            "there is ",
            "there are ",
            "he is ",
            "she is ",
            "they are ",
        )
        for prefix in weak_prefixes:
            remainder = lowered[len(prefix):].strip() if lowered.startswith(prefix) else None
            if remainder is not None and len(re.findall(r"[a-z0-9']+", remainder)) < max(self.min_caption_tokens - 1, 2):
                return False
        detail_patterns = (
            r"\b(wearing|holding|sitting|standing|lying|parked|placed|next to|on top of|near|with)\b",
            r"\b(red|blue|green|yellow|black|white|brown|pink|purple|orange|gray|grey|wooden|metal|striped|plaid)\b",
            r"\b(head|face|hair|shirt|jacket|pants|hat|helmet|glasses|bag|phone|book|plate|cup|bench|chair|table)\b",
        )
        detail_hit_count = sum(bool(re.search(pattern, lowered)) for pattern in detail_patterns)
        if detail_hit_count == 0 and token_count < max(self.min_caption_tokens + 2, 6):
            return False
        return True

    @staticmethod
    def _caption_scene_spill_hit_count(caption):
        lowered = re.sub(r"\s+", " ", (caption or "").strip().lower())
        if not lowered:
            return 0
        spill_phrases = (
            "in the image",
            "in this image",
            "in the picture",
            "in this picture",
            "in the photo",
            "in this photo",
            "in the scene",
            "another person",
            "other people",
            "another object",
            "other objects",
            "someone else",
            "group of people",
            "surrounded by",
            "in the background",
            "appears to be",
            "seems to be",
            "possibly",
            "might be",
            "may be",
            "likely",
            "probably",
            "suggesting that",
            "suggests that",
            "engaged in",
            "participating in",
            "watching an event",
            "while another",
            "next to another",
            "standing next to another",
            "part of a larger",
            "one of several",
            "among other",
            "with other",
            "this scene suggests",
            "adds a sense of",
            "providing a sense of",
        )
        return sum(1 for phrase in spill_phrases if phrase in lowered)

    def _is_low_density_long_caption(self, caption):
        token_count = self._caption_token_count(caption)
        if token_count < self.caption_low_density_length_threshold:
            return False
        if self._has_repetitive_caption_pattern(caption):
            return True
        if self._is_overly_generic_caption(caption):
            return True
        if self._caption_scene_spill_hit_count(caption) >= 2:
            return True
        return False

    def _description_quality_score(self, raw_prediction, clean_caption, status):
        token_count = self._caption_token_count(clean_caption)
        has_seg_markup = int(bool(re.search(r"\[SEG\]|</?p>", raw_prediction or "", flags=re.IGNORECASE)))
        np_like = int(self._is_np_like_caption(clean_caption))
        overly_generic = int(self._is_overly_generic_caption(clean_caption))
        spill_hits = self._caption_scene_spill_hit_count(clean_caption)
        return (
            int(status == "ok"),
            -spill_hits,
            np_like,
            min(token_count, 16),
            -overly_generic,
            -has_seg_markup,
            -abs(token_count - 6),
            len(clean_caption or ""),
        )

    def _caption_quality_reward(self, clean_caption, status):
        reward = 0.0
        if status == "empty":
            return -self.caption_reward_empty_penalty
        if status in {"truncated_caption", "seg_style_answer", "decode_error"}:
            return -self.caption_reward_truncated_penalty
        if status == "ok":
            spill_hit_count = self._caption_scene_spill_hit_count(clean_caption)
            has_scene_spill = spill_hit_count >= 1
            reward += self.caption_reward_valid_bonus
            if (
                self._is_caption_content_sufficient(clean_caption)
                and not self._is_overly_generic_caption(clean_caption)
                and not has_scene_spill
            ):
                reward += self.caption_reward_sufficient_bonus
            if self._is_overly_generic_caption(clean_caption):
                reward -= self.caption_reward_generic_penalty
            if self._has_repetitive_caption_pattern(clean_caption):
                reward -= self.caption_reward_repetition_penalty
            if spill_hit_count >= 4:
                reward -= self.caption_reward_scene_spill_penalty + 0.25
            elif spill_hit_count >= 2:
                reward -= self.caption_reward_scene_spill_penalty + 0.1
            elif spill_hit_count >= 1:
                reward -= self.caption_reward_scene_spill_penalty
            if self._is_low_density_long_caption(clean_caption):
                reward -= self.caption_reward_low_density_penalty
        return reward

    @staticmethod
    def _format_float_list(values):
        return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"

    @staticmethod
    def _has_repetitive_caption_pattern(caption):
        normalized = re.sub(r"\s+", " ", (caption or "").strip().lower())
        if not normalized:
            return False
        repeated_phrases = (
            "sure it is",
            "it is it is",
            "sure sure",
            "region1 region1",
            "the region marked as",
        )
        if any(normalized.count(phrase) >= 2 for phrase in repeated_phrases):
            return True
        tokens = re.findall(r"[a-z0-9']+", normalized)
        if len(tokens) >= 24:
            unique_ratio = len(set(tokens)) / max(len(tokens), 1)
            if unique_ratio < 0.45:
                return True
        for n in (2, 3):
            if len(tokens) < n * 3:
                continue
            counts = {}
            for idx in range(len(tokens) - n + 1):
                ngram = tuple(tokens[idx: idx + n])
                counts[ngram] = counts.get(ngram, 0) + 1
            if any(count >= 3 for count in counts.values()):
                return True
        return False

    @staticmethod
    def _is_np_like_caption(caption):
        lowered = re.sub(r"\s+", " ", (caption or "").strip().lower())
        if not lowered:
            return False
        return not bool(re.search(r"\b(is|are|was|were|be|being|been)\b", lowered))

    @staticmethod
    def _is_overly_generic_caption(caption):
        lowered = re.sub(r"\s+", " ", (caption or "").strip().lower())
        if not lowered:
            return True
        generic_singletons = {
            "people",
            "person",
            "man",
            "woman",
            "child",
            "crowd",
            "group",
            "wall",
            "building",
            "store",
            "street",
            "room",
            "table",
            "object",
            "area",
            "background",
        }
        generic_prefixes = (
            "a person",
            "the person",
            "a man",
            "the man",
            "a woman",
            "the woman",
            "people",
            "the people",
            "a crowd",
            "the crowd",
            "a group",
            "the group",
        )
        words = re.findall(r"[a-z0-9']+", lowered)
        if len(words) == 1 and lowered in generic_singletons:
            return True
        if len(words) <= 3 and lowered.startswith(generic_prefixes):
            return True
        generic_phrases = (
            "seems interested",
            "looks calm",
            "appears calm",
            "kindness and care",
            "attention to detail",
            "enjoying time",
            "spending time",
            "having fun",
            "leisure time",
            "social interactions",
            "culture and",
            "local customs",
            "take care of",
            "taking care of",
            "staying hydrated",
        )
        generic_phrase_hits = sum(1 for phrase in generic_phrases if phrase in lowered)
        if generic_phrase_hits >= 2:
            return True
        if len(words) >= 28:
            generic_token_count = sum(
                1
                for token in words
                if token
                in {
                    "someone",
                    "something",
                    "person",
                    "people",
                    "man",
                    "woman",
                    "child",
                    "time",
                    "thing",
                    "activity",
                    "activities",
                    "interest",
                    "interested",
                    "calm",
                    "care",
                    "kindness",
                    "culture",
                    "social",
                    "outdoors",
                    "nature",
                }
            )
            if generic_token_count >= max(6, len(words) // 5):
                return True
        return False

    def _resolve_reconstruct_questions(self, caption):
        primary_template = self.reconstruct_question_templates[0]
        return [primary_template.format(caption=caption, class_name=caption)]

    @staticmethod
    def _teacher_regenerate_iou_improvement(student_iou, teacher_iou):
        return float(teacher_iou) - float(student_iou)

    def _teacher_regenerate_gate_passed(self, student_iou, teacher_iou):
        student_iou = float(student_iou)
        teacher_iou = float(teacher_iou)
        iou_gain = teacher_iou - student_iou
        if student_iou >= float(self.iou_high_threshold):
            return iou_gain > 0.0
        return iou_gain > 0.5 or (teacher_iou >= 0.6 and iou_gain >= 0.1)

    @staticmethod
    def _window_metric_counts():
        return (
            "valid_count",
            "loss_count",
            "nonempty_gt_count",
            "nonempty_caption_count",
            "description_ok_count",
            "description_empty_count",
            "description_truncated_count",
            "description_seg_style_count",
            "description_raw_seg_style_count",
            "reconstruct_ok_count",
            "reconstruct_failed_count",
            "reconstruct_skip_count",
            "reconstruct_invalid_caption_skip_count",
            "reconstruct_empty_prediction_masks_count",
            "empty_gt_mask_count",
            "seg_correct_count",
            "teacher_regenerate_count",
            "on_policy_distill_count",
            "grpo_positive_count",
            "regen_loss_count",
            "onpolicy_loss_count",
            "grpo_loss_count",
            "grpo_reward_count",
            "grpo_mcq_correct_count",
            "grpo_mcq_count",
            "grpo_reward_raw_count",
            "recovery_caption_count",
            "invalid_caption_penalty_count",
            "hard_reconstruct_failure_count",
            "teacher_regenerate_ce_applied_count",
            "teacher_regenerate_suppressed_count",
            "teacher_regenerate_verified_count",
            "teacher_regenerate_rejected_count",
            "teacher_reconstruct_ok_count",
            "teacher_positive_gain_count",
            "teacher_regenerate_analysis_count",
            "caption_mode_failure_count",
            "onpolicy_blocked_by_seg_style_count",
            "grpo_blocked_by_seg_style_count",
            "teacher_recovery_seg_style_count",
            "teacher_recovery_seg_style_success_count",
            "teacher_difference_context_nontrivial_count",
            "teacher_problem_valid_count",
            "teacher_direction_valid_count",
            "teacher_reason_valid_count",
            "teacher_reason_coarse_count",
            "teacher_regenerate_verification_valid_count",
            "teacher_diagnosis_valid_count",
            "teacher_dlc_valid_count",
            "grpo_zero_reward_variance_count",
            "grpo_nonzero_reward_count",
            "grpo_missing_confuser_count",
            "recovery_ce_applied_count",
            "recovery_suppressed_count",
            "scene_spill_caption_count",
            "low_density_long_caption_count",
            "detail_sufficient_caption_count",
            "generic_caption_count",
            "repetitive_caption_count",
        )

    @staticmethod
    def _window_metric_sums():
        return (
            "iou_sum",
            "caption_token_sum",
            "total_loss_sum",
            "regen_ce_sum",
            "onpolicy_jsd_sum",
            "grpo_sum",
            "grpo_reward_sum",
            "grpo_reward_raw_sum",
            "grpo_gt_prob_sum",
            "grpo_confuser_penalty_sum",
            "grpo_mcq_correct_conf_sum",
            "teacher_regenerate_verified_iou_sum",
            "teacher_regenerate_verification_iou_sum",
            "teacher_iou_gain_sum",
        )

    def _append_window_metrics(self, payload):
        self._metric_window.append(payload)

    def _aggregate_window_metrics(self):
        totals = {
            key: 0
            for key in (*self._window_metric_counts(), *self._window_metric_sums())
        }
        for payload in self._metric_window:
            for key in totals:
                totals[key] += payload.get(key, 0)
        return totals

    def _grpo_reward_from_iou(self, *, iou, pred_mask_missing):
        if pred_mask_missing:
            return -2.0
        iou = float(iou)
        if iou >= self.iou_high_threshold:
            return 1.0
        if iou >= self.iou_low_threshold:
            return -1.0 + (iou - self.iou_low_threshold) / (self.iou_high_threshold - self.iou_low_threshold)
        return -2.0

    @staticmethod
    def _invalid_reconstruction_placeholder(status):
        return ReconstructionResult(
            pred_mask=None,
            question=None,
            raw_prediction="",
            prediction_masks_count=0,
            seg_token_count=0,
            status=status,
        )

    @staticmethod
    def _canonicalize_referring_expression(caption):
        caption = re.sub(r"\s+", " ", (caption or "").strip())
        if not caption:
            return caption
        caption = re.sub(r"^(a|an)\s+", "the ", caption, flags=re.IGNORECASE)
        subject_predicate = re.match(r"^(the\s+.+?)\s+is\s+(.+)$", caption, flags=re.IGNORECASE)
        if subject_predicate:
            subject = subject_predicate.group(1).strip(" ,.")
            predicate = subject_predicate.group(2).strip(" ,.")
            pp_match = re.search(
                r"\b(in|on|at|with|near|under|over|behind|beside|by|inside|outside|next to|in front of)\b.+$",
                predicate,
                flags=re.IGNORECASE,
            )
            if pp_match:
                caption = f"{subject} {pp_match.group(0)}"
            else:
                caption = subject
        caption = re.sub(r"\bis\s+([a-z]+ing)\b", r"\1", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\bare\s+([a-z]+ing)\b", r"\1", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\bis on\b", " on", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\bis in\b", " in", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\bis at\b", " at", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\bis with\b", " with", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\s+", " ", caption).strip(" .,")
        return caption

    @staticmethod
    def _subject_only_referring_expression(caption):
        caption = re.sub(r"\s+", " ", (caption or "").strip())
        if not caption:
            return caption
        caption = re.sub(r"^(a|an)\s+", "the ", caption, flags=re.IGNORECASE)
        match = re.match(r"^(the\s+.+?)\s+\b(is|are|was|were)\b", caption, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ,.")
        return caption.strip(" ,.")

    @staticmethod
    def _append_spatial_hint_to_question(question, spatial_hint):
        spatial_hint = re.sub(r"\s+", " ", (spatial_hint or "").strip())
        if not spatial_hint:
            return question
        question = question.rstrip()
        if not question.endswith((".", "?", "!")):
            question = question + "."
        return f"{question}\nLocalization hint: {spatial_hint}"

    @staticmethod
    def _coarse_spatial_hint(mask):
        mask = np.asarray(mask)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return ""
        h, w = mask.shape
        cx = float(xs.mean()) / max(w, 1)
        cy = float(ys.mean()) / max(h, 1)
        horiz = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
        vert = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
        area_ratio = float(mask.sum()) / float(h * w)
        size = "small" if area_ratio < 0.08 else "large" if area_ratio > 0.28 else "medium-sized"
        if horiz == "center" and vert == "middle":
            loc = "near the center"
        elif horiz == "center":
            loc = f"near the {vert}"
        elif vert == "middle":
            loc = f"on the {horiz} side"
        else:
            loc = f"in the {vert} {horiz}"
        return f"The target is {size} and located {loc}."

    def _encode_completion_from_caption(self, caption, *, model=None):
        del model
        return self.tokenizer(
            caption,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

    def _build_caption_bad_words_ids(self):
        phrases = (
            "[SEG]",
            "segmentation",
            "segmentation result",
            "the segmentation result",
        )
        bad_words_ids = []
        seen = set()
        for phrase in phrases:
            for candidate in (phrase, f" {phrase}"):
                token_ids = self.tokenizer(candidate, add_special_tokens=False).input_ids
                if not token_ids:
                    continue
                key = tuple(int(token_id) for token_id in token_ids)
                if key in seen:
                    continue
                seen.add(key)
                bad_words_ids.append(list(key))
        return bad_words_ids

    @staticmethod
    def _infer_raw_caption_failure_mode(raw_prediction):
        normalized = "" if raw_prediction is None else str(raw_prediction).strip()
        if not normalized:
            return ""
        lowered = normalized.lower()
        compact = re.sub(r"\s+", " ", lowered)
        if "[seg]" in compact:
            return "seg_style_answer"
        if "segmentation result" in compact:
            return "seg_style_answer"
        if re.search(r"</?p>", compact) and "[seg]" in compact:
            return "seg_style_answer"
        short_seg_patterns = (
            "sure, [seg].",
            "sure, it is [seg].",
            "it is [seg].",
            "[seg].",
        )
        if compact.strip() in short_seg_patterns:
            return "seg_style_answer"
        return ""

    @staticmethod
    def _is_caption_mode_failure_status(status):
        return status in {"seg_style_answer", "truncated_caption", "empty", "decode_error"}

    def _is_caption_trainable_for_student_losses(self, description):
        if description is None:
            return False
        if self._is_caption_mode_failure_status(description.status):
            return False
        if description.raw_failure_mode == "seg_style_answer":
            return False
        return True

    def _finalize_description_result(self, *, raw_prediction, clean_caption, completion_ids):
        raw_failure_mode = self._infer_raw_caption_failure_mode(raw_prediction)
        clean_status = self._infer_description_status(clean_caption)
        status = clean_status
        if raw_failure_mode == "seg_style_answer":
            status = "seg_style_answer"
        elif status == "ok" and not self._is_caption_content_sufficient(clean_caption):
            status = "truncated_caption"
        clean_caption, completion_ids, was_truncated = self._truncate_caption_completion(
            clean_caption,
            completion_ids,
            max_tokens=self.description_max_new_tokens,
        )
        if status != "seg_style_answer" and was_truncated:
            status = "truncated_caption"
        return DescriptionResult(
            raw_prediction=raw_prediction,
            clean_caption=clean_caption,
            completion_ids=completion_ids,
            status=status,
            raw_failure_mode=raw_failure_mode,
            clean_status=clean_status,
        )

    @staticmethod
    def _extract_labeled_teacher_text(raw_prediction, label):
        if raw_prediction is None:
            return ""
        pattern = rf"(?is)(?:^|\n)\s*{re.escape(label)}\s*:\s*(.*?)(?=\n\s*[A-Z_]+\s*:|\Z)"
        match = re.search(pattern, str(raw_prediction))
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _normalize_teacher_field_text(value):
        normalized = "" if value is None else str(value).strip()
        return re.sub(r"\s+", " ", normalized)

    def _teacher_field_is_effective(self, value, *, invalid_markers=("none", "unknown", "")):
        normalized = self._normalize_teacher_field_text(value).lower()
        return normalized not in {marker.lower() for marker in invalid_markers}

    @staticmethod
    def _split_teacher_field_list(value):
        if value is None:
            return []
        items = []
        for part in re.split(r"[,\n]", str(value)):
            item = re.sub(r"\s+", " ", part).strip()
            if item:
                items.append(item)
        return items

    def _clean_teacher_phrase_list_text(self, value):
        items = self._split_teacher_field_list(value)
        if not items:
            return ""
        return ", ".join(items)

    def _teacher_text_overlap_ratio(self, text_a, text_b):
        tokens_a = set(re.findall(r"[a-z0-9']+", self._normalize_teacher_field_text(text_a).lower()))
        tokens_b = set(re.findall(r"[a-z0-9']+", self._normalize_teacher_field_text(text_b).lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return float(len(tokens_a & tokens_b)) / float(max(min(len(tokens_a), len(tokens_b)), 1))

    def _build_difference_field_phrase_hints(self, text):
        if not self._teacher_field_is_effective(text):
            return []
        normalized = self._normalize_teacher_field_text(text)
        fragments = []
        for piece in re.split(r"[;,]", normalized):
            piece = re.sub(r"\s+", " ", piece).strip(" .")
            if len(piece.split()) >= 2:
                fragments.append(piece)
        if not fragments and normalized:
            fragments.append(normalized)
        return fragments[:4]

    def _difference_text_has_semantic_anchor(self, text):
        normalized = self._normalize_teacher_field_text(text).lower()
        if not normalized:
            return False
        anchor_patterns = (
            "left side",
            "right side",
            "upper area",
            "lower area",
            "middle width",
            "middle height",
            "broad area",
            "small patch",
            "local offset",
            "size contrast",
            "target-specific",
            "distractor",
            "target-only",
            "distractor-only",
            "shared evidence",
            "shared overlap",
            "overlap",
            "boundary",
            "edge",
            "centered",
            "shifted",
            "size",
            "shape",
        )
        return any(pattern in normalized for pattern in anchor_patterns)

    def _teacher_reason_is_coarse(self, text):
        normalized = self._normalize_teacher_field_text(text).lower()
        if not normalized:
            return True
        coarse_hits = any(
            token in normalized
            for token in (
                "broad area",
                "left side",
                "right side",
                "upper area",
                "lower area",
                "middle height",
                "middle width",
            )
        )
        fine_hits = any(
            token in normalized
            for token in (
                "small",
                "tiny",
                "medium",
                "large",
                "shifted away",
                "not centered near",
                "more to the",
                "higher toward",
                "lower toward",
                "competing region",
                "still matches",
                "still fits",
                "boundary",
                "edge",
                "centered",
                "shifted",
                "size",
                "shape",
                "overlap",
            )
        )
        return coarse_hits and not fine_hits

    @staticmethod
    def _teacher_failure_type_set():
        return {
            "too_generic",
            "wrong_attribute",
            "wrong_part_focus",
            "wrong_spatial_anchor",
            "distractor_leak",
            "scene_spill",
            "mixed_target",
            "underspecified_local_detail",
            "unknown",
        }

    def _build_empty_teacher_regenerate_pipeline_result(self):
        completion_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)
        return TeacherRegeneratePipelineResult(detailed_completion_ids=completion_ids)

    @staticmethod
    def _teacher_problem_labels():
        return ("CAPTION_PROBLEM",)

    @staticmethod
    def _teacher_direction_labels():
        return ("CORRECTION_DIRECTION",)

    @staticmethod
    def _teacher_reason_labels():
        return ("REASON",)

    @staticmethod
    def _teacher_diagnosis_labels():
        return ("CAPTION_PROBLEM", "CORRECTION_DIRECTION", "REASON")

    @staticmethod
    def _teacher_structured_diagnosis_labels():
        return (
            "CAPTION_PROBLEM",
            "CORRECTION_DIRECTION",
            "REASON",
            "CONFIDENCE",
            "TARGET_ANCHOR",
            "DISTRACTOR_ANCHOR",
        )

    def validate_teacher_problem_identification(self, result):
        if not self._teacher_field_is_effective(result.caption_problem, invalid_markers=("",)):
            return False, "problem_invalid:missing_caption_problem"
        if len(result.caption_problem.split()) < 4:
            return False, "problem_invalid:too_short"
        normalized_problem = result.caption_problem.lower()
        if not any(
            token in normalized_problem
            for token in (
                "too generic",
                "underspecified",
                "does not specify",
                "fails to distinguish",
                "missing",
                "does not separate",
                "does not isolate",
                "does not accurately describe",
                "fails to capture",
                "under-describes",
                "still fits",
                "drifts",
                "pulled toward",
            )
        ):
            return False, "problem_invalid:generic_problem"
        semantic_anchor_ok = (
            self._difference_text_has_semantic_anchor(result.caption_problem)
            or any(
                token in normalized_problem
                for token in (
                    "size",
                    "shape",
                    "boundary",
                    "edge",
                    "centered",
                    "shifted",
                    "overlap",
                    "shared",
                    "target-only",
                    "distractor",
                    "left",
                    "right",
                    "top",
                    "bottom",
                    "center",
                    "middle",
                    "region",
                    "drift",
                    "fit",
                )
            )
        )
        if not semantic_anchor_ok:
            weak_anchor_ok = any(
                token in normalized_problem
                for token in (
                    "left",
                    "right",
                    "top",
                    "bottom",
                    "center",
                    "middle",
                    "target-only",
                    "distractor-only",
                    "missing",
                    "still fits",
                    "drifts",
                )
            )
            if not weak_anchor_ok:
                return False, "problem_invalid:lacks_fine_grained_anchor"
        return True, ""

    def validate_teacher_correction_direction(self, result):
        if not self._teacher_field_is_effective(result.correction_direction, invalid_markers=("",)):
            return False, "direction_invalid:missing_direction"
        if len(result.correction_direction.split()) < 5:
            return False, "direction_invalid:too_short"
        normalized_direction = result.correction_direction.lower()
        has_add_action = any(
            token in normalized_direction
            for token in (
                "add",
                "strengthen",
                "highlight",
                "specify",
                "focus on",
                "make explicit",
            )
        )
        has_avoid_action = any(
            token in normalized_direction
            for token in (
                "avoid",
                "suppress",
                "separate",
                "stop matching",
                "not fit",
                "not match",
                "downplay",
                "remove",
            )
        )
        if not has_add_action:
            return False, "direction_invalid:missing_add_action"
        distractor_strength = self._caption_token_count(result.distractor_only_evidence or "")
        if (
            self._teacher_field_is_effective(result.distractor_only_evidence)
            and distractor_strength >= 8
            and not has_avoid_action
        ):
            return False, "direction_invalid:missing_avoid_action"
        if (
            self._teacher_text_overlap_ratio(result.correction_direction, result.target_summary) >= 0.75
            and not (has_add_action and (has_avoid_action or not self._teacher_field_is_effective(result.distractor_only_evidence)))
        ):
            return False, "direction_invalid:summary_repetition"
        if (
            any(
                token in normalized_direction
                for token in ("does not accurately describe", "missing details", "fails to describe")
            )
            and not (has_add_action or has_avoid_action)
        ):
            return False, "direction_invalid:problem_restatement"
        return True, ""

    def validate_teacher_reason_explanation(self, result):
        if not self._teacher_field_is_effective(result.reason, invalid_markers=("",)):
            return False, "reason_invalid:missing_reason"
        if len(result.reason.split()) < 5:
            return False, "reason_invalid:too_short"
        if self._teacher_reason_is_coarse(result.reason):
            return False, "reason_invalid:too_coarse"
        normalized_reason = result.reason.lower()
        if not any(
            token in normalized_reason
            for token in ("misses", "still fits", "still matches", "pulls reconstruction", "drifts toward")
        ):
            return False, "reason_invalid:missing_failure_mechanism"
        if self._teacher_field_is_effective(result.target_only_evidence) and not any(
            token in normalized_reason
            for token in ("target-only", "shared", "overlap", "missing", "under-describes")
        ) and not self._difference_text_has_semantic_anchor(result.reason):
            weak_target_anchor_ok = any(
                token in normalized_reason
                for token in ("left", "right", "top", "bottom", "center", "middle", "target", "region")
            )
            if not weak_target_anchor_ok:
                return False, "reason_invalid:missing_target_difference_anchor"
        if self._teacher_field_is_effective(result.distractor_only_evidence) and not any(
            token in normalized_reason
            for token in ("distractor", "extra region", "still fits", "still matches", "drifts toward")
        ):
            weak_distractor_anchor_ok = any(
                token in normalized_reason
                for token in ("left", "right", "top", "bottom", "center", "middle", "other region")
            )
            if not weak_distractor_anchor_ok:
                return False, "reason_invalid:missing_distractor_difference_anchor"
        return True, ""

    def _parse_teacher_light_diagnosis(self, raw_prediction, base_result=None):
        result = base_result or self._build_empty_teacher_regenerate_pipeline_result()
        result.diagnosis_raw = "" if raw_prediction is None else str(raw_prediction)
        sections = self._parse_teacher_labeled_sections(raw_prediction, self._teacher_diagnosis_labels())
        result.caption_problem = self._normalize_teacher_field_text(sections.get("CAPTION_PROBLEM", ""))
        result.correction_direction = self._normalize_teacher_field_text(
            sections.get("CORRECTION_DIRECTION", "")
        )
        result.reason = self._normalize_teacher_field_text(sections.get("REASON", ""))
        if not result.reason and "reason" in result.correction_direction.lower():
            split_match = re.match(r"(?is)(.*?)(?:\bREASON\b[:\s-]+)(.+)", result.correction_direction)
            if split_match:
                result.correction_direction = self._normalize_teacher_field_text(split_match.group(1))
                result.reason = self._normalize_teacher_field_text(split_match.group(2))
        if (
            not result.reason
            and self._teacher_field_is_effective(result.likely_drift_reason, invalid_markers=("",))
        ):
            result.reason = self._normalize_teacher_field_text(result.likely_drift_reason)
        if not self._teacher_field_is_effective(result.correction_direction, invalid_markers=("",)):
            if (
                self._teacher_field_is_effective(result.target_only_evidence)
                and self._teacher_field_is_effective(result.distractor_only_evidence)
            ):
                result.correction_direction = (
                    "Strengthen the target-only evidence and suppress the distractor-only evidence "
                    "so the caption separates the gtmask from the reconstructed distractor."
                )
            elif self._teacher_field_is_effective(result.target_only_evidence):
                result.correction_direction = (
                    "Strengthen the target-only evidence by making the missing target detail explicit."
                )
            elif self._teacher_field_is_effective(result.distractor_only_evidence):
                result.correction_direction = (
                    "Suppress the distractor-only evidence by avoiding wording that still fits the distractor region."
                )
        result.reason_is_coarse = self._teacher_reason_is_coarse(result.reason)
        return result

    def validate_teacher_light_diagnosis(self, result):
        if not result.difference_context_nontrivial:
            return False, "difference_context_invalid:trivial"
        if not self._teacher_field_is_effective(result.caption_problem, invalid_markers=("",)):
            return False, "diagnosis_invalid:missing_caption_problem"
        if not self._teacher_field_is_effective(result.correction_direction, invalid_markers=("",)):
            return False, "diagnosis_invalid:missing_correction_direction"
        if not self._teacher_field_is_effective(result.reason, invalid_markers=("",)):
            return False, "diagnosis_invalid:missing_reason"
        if self._teacher_reason_is_coarse(result.reason):
            return False, "diagnosis_invalid:reason_too_coarse"
        semantic_anchor_ok = (
            self._difference_text_has_semantic_anchor(result.caption_problem)
            or self._difference_text_has_semantic_anchor(result.reason)
            or self._difference_text_has_semantic_anchor(result.correction_direction)
        )
        phrase_level_problem_ok = any(
            token in result.caption_problem.lower()
            for token in (
                "too generic",
                "underspecified",
                "wrong anchor",
                "pulled toward",
                "missing target",
                "target-only",
                "distractor-only",
                "shared overlap",
            )
        )
        if not semantic_anchor_ok and not phrase_level_problem_ok:
            return False, "diagnosis_invalid:missing_difference_evidence"
        if len(result.caption_problem.split()) < 6:
            return False, "diagnosis_invalid:caption_problem_too_short"
        if len(result.reason.split()) < 6:
            return False, "diagnosis_invalid:reason_too_short"
        if not any(
            token in (result.caption_problem + " " + result.reason).lower()
            for token in ("miss", "generic", "underspecified", "pulled toward", "still fits", "still matches")
        ):
            return False, "diagnosis_invalid:missing_failure_mechanism"
        if any(
            result.caption_problem.lower().strip() == generic
            for generic in (
                "the caption is vague",
                "the caption is too generic",
                "the caption is wrong",
            )
        ):
            return False, "diagnosis_invalid:generic_caption_problem"
        if (
            any(
                token in result.caption_problem.lower()
                for token in ("too generic", "lacks specific details", "lacks details", "does not specify")
            )
            and not self._difference_text_has_semantic_anchor(result.caption_problem)
        ):
            return False, "diagnosis_invalid:generic_caption_problem"
        if any(
            result.reason.lower().strip() == generic
            for generic in (
                "the target-only evidence matters because",
                "the distractor-only evidence matters because",
            )
        ):
            return False, "diagnosis_invalid:generic_reason"
        return True, ""

    def _parse_teacher_structured_diagnosis(self, raw_prediction, base_result=None):
        result = base_result or self._build_empty_teacher_regenerate_pipeline_result()
        result.structured_diagnosis_raw = "" if raw_prediction is None else str(raw_prediction)
        result.diagnosis_raw = result.structured_diagnosis_raw
        sections = self._parse_teacher_labeled_sections(
            raw_prediction,
            self._teacher_structured_diagnosis_labels(),
        )
        result.problem_raw = result.structured_diagnosis_raw
        result.direction_raw = result.structured_diagnosis_raw
        result.reason_raw = result.structured_diagnosis_raw
        result.caption_problem = self._normalize_teacher_field_text(sections.get("CAPTION_PROBLEM", ""))
        result.correction_direction = self._normalize_teacher_field_text(
            sections.get("CORRECTION_DIRECTION", "")
        )
        result.reason = self._normalize_teacher_field_text(sections.get("REASON", ""))
        result.confidence = self._normalize_teacher_field_text(sections.get("CONFIDENCE", "")).lower()
        result.target_anchor = self._normalize_teacher_field_text(sections.get("TARGET_ANCHOR", ""))
        result.distractor_anchor = self._normalize_teacher_field_text(sections.get("DISTRACTOR_ANCHOR", ""))
        if not self._teacher_field_is_effective(result.target_anchor, invalid_markers=("",)):
            result.target_anchor = self._normalize_teacher_field_text(result.target_only_evidence)
        if not self._teacher_field_is_effective(result.distractor_anchor, invalid_markers=("",)):
            result.distractor_anchor = self._normalize_teacher_field_text(result.distractor_only_evidence)
        if result.confidence not in {"high", "medium", "low"}:
            result.confidence = "medium"
        result.reason_is_coarse = self._teacher_reason_is_coarse(result.reason)
        if not self._teacher_field_is_effective(result.caption_problem, invalid_markers=("",)) and self._teacher_field_is_effective(
            result.likely_drift_reason,
            invalid_markers=("",),
        ):
            result.caption_problem = self._normalize_teacher_field_text(result.likely_drift_reason)
        if not self._teacher_field_is_effective(result.correction_direction, invalid_markers=("",)):
            if self._teacher_field_is_effective(result.target_only_evidence) and self._teacher_field_is_effective(
                result.distractor_only_evidence
            ):
                result.correction_direction = (
                    "Add the target-only cue and avoid wording that still fits the distractor-side cue."
                )
            elif self._teacher_field_is_effective(result.target_only_evidence):
                result.correction_direction = "Add the missing target-only cue more explicitly."
            elif self._teacher_field_is_effective(result.distractor_only_evidence):
                result.correction_direction = "Avoid wording that still fits the distractor-side cue."
        if not self._teacher_field_is_effective(result.reason, invalid_markers=("",)) and self._teacher_field_is_effective(
            result.likely_drift_reason,
            invalid_markers=("",),
        ):
            result.reason = self._normalize_teacher_field_text(result.likely_drift_reason)
        result.problem_valid, result.problem_failure_reason = self.validate_teacher_problem_identification(result)
        result.direction_valid, result.direction_failure_reason = self.validate_teacher_correction_direction(result)
        result.reason_valid, result.reason_failure_reason = self.validate_teacher_reason_explanation(result)
        return result

    def validate_teacher_structured_diagnosis(self, result):
        if not result.difference_context_nontrivial:
            has_minimal_signal = any(
                self._teacher_field_is_effective(value, invalid_markers=("",))
                for value in (
                    result.caption_problem,
                    result.correction_direction,
                    result.reason,
                    result.likely_drift_reason,
                )
            )
            if not has_minimal_signal:
                return False, "structured_diagnosis_invalid:trivial_difference_context"
        if not result.problem_valid:
            return False, result.problem_failure_reason or "structured_diagnosis_invalid:problem"
        if not result.direction_valid:
            return False, result.direction_failure_reason or "structured_diagnosis_invalid:direction"
        if not result.reason_valid:
            return False, result.reason_failure_reason or "structured_diagnosis_invalid:reason"
        if self._teacher_text_overlap_ratio(result.caption_problem, result.correction_direction) >= 0.9:
            return False, "structured_diagnosis_invalid:problem_direction_duplicate"
        if not any(
            self._teacher_field_is_effective(value, invalid_markers=("",))
            for value in (result.target_anchor, result.distractor_anchor)
        ):
            return False, "structured_diagnosis_invalid:missing_anchor"
        result.problem_valid = True
        result.direction_valid = True
        result.reason_valid = True
        result.diagnosis_valid = True
        return True, ""

    def _teacher_has_minimal_diagnosis_signal(self, result):
        return any(
            self._teacher_field_is_effective(value, invalid_markers=("",))
            for value in (
                result.caption_problem,
                result.correction_direction,
                result.reason,
                result.likely_drift_reason,
                result.target_only_evidence,
                result.distractor_only_evidence,
            )
        )

    def _teacher_has_actionable_diagnosis_signal(self, result):
        if result.diagnosis_valid:
            return True
        if not self._teacher_has_minimal_diagnosis_signal(result):
            return False
        has_difference_anchor = any(
            self._teacher_field_is_effective(value, invalid_markers=("",))
            for value in (
                result.target_only_evidence,
                result.distractor_only_evidence,
                result.target_anchor,
                result.distractor_anchor,
            )
        )
        has_rewrite_intent = any(
            self._teacher_field_is_effective(value, invalid_markers=("",))
            for value in (
                result.caption_problem,
                result.correction_direction,
                result.reason,
                result.likely_drift_reason,
            )
        )
        return bool(has_difference_anchor and has_rewrite_intent)

    @staticmethod
    def _teacher_template_prefix_hit_count(text):
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not normalized:
            return 0
        template_prefixes = (
            "the target ",
            "target ",
            "the region ",
            "region1 ",
            "region 1 ",
            "the masked region ",
            "the highlighted region ",
            "this region ",
            "this target ",
        )
        return sum(int(normalized.startswith(prefix)) for prefix in template_prefixes)

    def _clean_teacher_dlc_caption_text(self, caption):
        caption = self._clean_caption_text(caption)
        caption = re.sub(
            r"^(?:the\s+)?(?:target|region|masked region|highlighted region|selected region)\b(?:\s*[:,.-]\s*|\s+)",
            "",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(
            r"^(?:caption|description|dlc|verification caption)\s*[:,.-]\s*",
            "",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(
            r"^(?:it\s+is|this\s+is|that\s+is)\s+",
            "",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(r"\s+", " ", caption).strip(" .,")
        return caption

    def _teacher_candidate_length_score(self, caption):
        token_count = self._caption_token_count(caption)
        if 8 <= token_count <= 24:
            return 0.5
        if 5 <= token_count <= 28:
            return 0.2
        return -0.2

    def _score_teacher_dlc_candidate(self, candidate):
        caption = candidate.get("caption", "")
        status = candidate.get("status", "empty")
        failure_reason = candidate.get("failure_reason", "")
        reconstruct_iou = float(candidate.get("reconstruct_iou", 0.0) or 0.0)
        student_iou = float(candidate.get("student_iou", 0.0) or 0.0)
        iou_gain = reconstruct_iou - student_iou
        score = 0.0
        if status == "ok" and not failure_reason:
            score += 2.5
        elif status == "ok":
            score += 0.5
        else:
            score -= 1.0
        if failure_reason == "teacher_dlc_invalid:missing_target_only_evidence":
            score -= 1.0
        elif failure_reason == "teacher_dlc_invalid:distractor_overlap":
            score -= 1.4
        if self._teacher_field_is_effective(candidate.get("target_only_evidence", "")):
            if (
                self._contains_any_phrase(caption, candidate.get("target_only_evidence", ""))
                or self._difference_text_has_semantic_anchor(caption)
            ):
                score += 1.6
            else:
                score -= 1.5
        if self._teacher_field_is_effective(candidate.get("distractor_only_evidence", "")) and self._contains_any_phrase(
            caption,
            candidate.get("distractor_only_evidence", ""),
        ):
            score -= 1.2
        score -= 0.5 * float(self._teacher_template_prefix_hit_count(caption))
        if self._is_overly_generic_caption(caption):
            score -= 0.8
        score += self._teacher_candidate_length_score(caption)
        score += 1.2 * reconstruct_iou
        if iou_gain > 0.0:
            score += 8.0 * iou_gain
        else:
            score += 4.0 * iou_gain
        return float(score)

    def _teacher_candidate_is_usable(self, candidate):
        if candidate.get("status") != "ok":
            return False
        failure_reason = str(candidate.get("failure_reason", "") or "")
        if failure_reason in {
            "",
            "teacher_dlc_invalid:missing_target_only_evidence",
            "teacher_dlc_invalid:distractor_overlap",
        }:
            return True
        return False

    def _materialize_teacher_caption_result(self, result, caption, source):
        caption = self._clean_teacher_dlc_caption_text(caption)
        completion_ids = self._encode_completion_from_caption(caption)
        caption, completion_ids, was_truncated = self._truncate_caption_completion(
            caption,
            completion_ids,
            max_tokens=self.description_max_new_tokens,
        )
        status = self._infer_description_status(caption)
        if status == "ok" and not self._is_caption_content_sufficient(caption):
            status = "truncated_caption"
        if status != "seg_style_answer" and was_truncated:
            status = "truncated_caption"
        result.detailed_caption = caption
        result.detailed_completion_ids = completion_ids
        result.detailed_status = status
        result.teacher_selected_caption_source = source
        return result

    @staticmethod
    def _teacher_fault_report_labels():
        return (
            "PRIMARY_FAILURE_TYPE",
            "SECONDARY_FAILURE_TYPES",
            "FAILURE_CONFIDENCE",
            "TARGET_SUMMARY",
            "REF_SUMMARY",
            "MISSING_EVIDENCE",
            "DISTRACTOR_EVIDENCE",
            "BAD_PHRASES_IN_STUDENT",
            "MISSING_PHRASES_NEEDED",
            "KEEPABLE_PHRASES",
            "EVIDENCE_FOR_FAILURE",
        )

    def _parse_teacher_labeled_sections(self, raw_prediction, labels):
        text = "" if raw_prediction is None else str(raw_prediction)
        if not text:
            return {label: "" for label in labels}
        label_pattern = "|".join(re.escape(label) for label in labels)
        pattern = re.compile(rf"(?is)\b({label_pattern})\s*:\s*")
        matches = list(pattern.finditer(text))
        values = {label: "" for label in labels}
        for idx, match in enumerate(matches):
            label = match.group(1)
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            values[label] = text[start:end].strip()
        return values

    def _parse_teacher_fault_report(self, raw_prediction):
        result = self._build_empty_teacher_regenerate_pipeline_result()
        result.fault_report_raw = "" if raw_prediction is None else str(raw_prediction)
        sections = self._parse_teacher_labeled_sections(raw_prediction, self._teacher_fault_report_labels())
        result.primary_failure_type = self._normalize_teacher_field_text(
            sections.get("PRIMARY_FAILURE_TYPE", "")
        ).lower()
        secondary_raw = self._normalize_teacher_field_text(sections.get("SECONDARY_FAILURE_TYPES", ""))
        secondary_items = []
        for item in self._split_teacher_field_list(secondary_raw):
            item_lower = item.lower()
            if item_lower and item_lower != "none":
                secondary_items.append(item_lower)
        result.secondary_failure_types = tuple(secondary_items)
        result.failure_confidence = self._normalize_teacher_field_text(
            sections.get("FAILURE_CONFIDENCE", "")
        ).lower()
        result.target_summary = self._normalize_teacher_field_text(sections.get("TARGET_SUMMARY", ""))
        result.ref_summary = self._normalize_teacher_field_text(sections.get("REF_SUMMARY", ""))
        result.missing_evidence = self._normalize_teacher_field_text(sections.get("MISSING_EVIDENCE", ""))
        result.distractor_evidence = self._normalize_teacher_field_text(sections.get("DISTRACTOR_EVIDENCE", ""))
        result.bad_phrases_in_student = self._clean_teacher_phrase_list_text(
            sections.get("BAD_PHRASES_IN_STUDENT", "")
        )
        result.missing_phrases_needed = self._clean_teacher_phrase_list_text(
            sections.get("MISSING_PHRASES_NEEDED", "")
        )
        result.keepable_phrases = self._clean_teacher_phrase_list_text(
            sections.get("KEEPABLE_PHRASES", "")
        )
        result.evidence_for_failure = self._normalize_teacher_field_text(
            sections.get("EVIDENCE_FOR_FAILURE", "")
        )
        return result

    def _backfill_teacher_fault_report(
        self,
        result,
        *,
        relation_context,
        student_caption,
    ):
        if result.primary_failure_type not in self._teacher_failure_type_set():
            result.primary_failure_type = "unknown"
        if result.failure_confidence not in {"high", "medium", "low"}:
            result.failure_confidence = "medium"
        if not self._teacher_field_is_effective(result.target_summary, invalid_markers=("",)):
            result.target_summary = self._normalize_teacher_field_text(relation_context.get("gt_summary", ""))
        if not self._teacher_field_is_effective(result.ref_summary, invalid_markers=("",)):
            result.ref_summary = self._normalize_teacher_field_text(relation_context.get("ref_summary", ""))
        if not self._teacher_field_is_effective(result.missing_evidence):
            result.missing_evidence = self._normalize_teacher_field_text(relation_context.get("gt_only_summary", ""))
        if not self._teacher_field_is_effective(result.distractor_evidence):
            result.distractor_evidence = self._normalize_teacher_field_text(relation_context.get("ref_only_summary", ""))
        if not self._teacher_field_is_effective(result.bad_phrases_in_student):
            trimmed_caption = self._clean_caption_text(student_caption)
            result.bad_phrases_in_student = trimmed_caption if trimmed_caption else "unknown"
        if not self._teacher_field_is_effective(result.missing_phrases_needed):
            if self._teacher_field_is_effective(result.missing_evidence):
                result.missing_phrases_needed = result.missing_evidence
            else:
                result.missing_phrases_needed = "none"
        if not self._teacher_field_is_effective(result.keepable_phrases):
            result.keepable_phrases = "none"
        if not self._teacher_field_is_effective(result.evidence_for_failure, invalid_markers=("",)):
            missing_text = result.missing_evidence if self._teacher_field_is_effective(result.missing_evidence) else "no clear missing target evidence"
            distractor_text = (
                result.distractor_evidence if self._teacher_field_is_effective(result.distractor_evidence) else "no clear distractor evidence"
            )
            result.evidence_for_failure = (
                f"The student caption drifts because the reconstruction misses {missing_text} "
                f"and instead includes {distractor_text}."
            )
        return result

    def validate_teacher_fault_report(self, result):
        valid_types = self._teacher_failure_type_set()
        if result.primary_failure_type not in valid_types:
            return False, "fault_report_invalid:bad_primary_type"
        if result.failure_confidence not in {"high", "medium", "low"}:
            return False, "fault_report_invalid:bad_confidence"
        if not self._teacher_field_is_effective(result.target_summary, invalid_markers=("",)):
            return False, "fault_report_invalid:missing_target_summary"
        if not self._teacher_field_is_effective(result.ref_summary, invalid_markers=("",)):
            return False, "fault_report_invalid:missing_ref_summary"
        if not self._teacher_field_is_effective(result.evidence_for_failure, invalid_markers=("",)):
            return False, "fault_report_invalid:missing_evidence_for_failure"
        has_failure_evidence = (
            self._teacher_field_is_effective(result.missing_evidence)
            or self._teacher_field_is_effective(result.distractor_evidence)
        )
        if not has_failure_evidence:
            return False, "fault_report_invalid:no_failure_evidence"
        has_phrase_level_diagnosis = any(
            self._teacher_field_is_effective(value)
            for value in (
                result.bad_phrases_in_student,
                result.missing_phrases_needed,
                result.keepable_phrases,
            )
        )
        if not has_phrase_level_diagnosis:
            return False, "fault_report_invalid:no_phrase_level_diagnosis"
        return True, ""

    def _parse_teacher_repair_plan(self, raw_prediction, base_result):
        result = base_result
        result.repair_plan_raw = "" if raw_prediction is None else str(raw_prediction)
        result.remove_phrases = self._clean_teacher_phrase_list_text(
            self._extract_labeled_teacher_text(raw_prediction, "REMOVE_PHRASES")
        )
        result.add_phrases = self._clean_teacher_phrase_list_text(
            self._extract_labeled_teacher_text(raw_prediction, "ADD_PHRASES")
        )
        result.emphasize_phrases = self._clean_teacher_phrase_list_text(
            self._extract_labeled_teacher_text(raw_prediction, "EMPHASIZE_PHRASES")
        )
        result.deemphasize_phrases = self._clean_teacher_phrase_list_text(
            self._extract_labeled_teacher_text(raw_prediction, "DEEMPHASIZE_PHRASES")
        )
        result.localize_with = self._normalize_teacher_field_text(
            self._extract_labeled_teacher_text(raw_prediction, "LOCALIZE_WITH")
        )
        result.avoid_phrases = self._clean_teacher_phrase_list_text(
            self._extract_labeled_teacher_text(raw_prediction, "AVOID_PHRASES")
        )
        result.rewrite_strategy = self._normalize_teacher_field_text(
            self._extract_labeled_teacher_text(raw_prediction, "REWRITE_STRATEGY")
        )
        return result

    def validate_teacher_repair_plan(self, result):
        has_core_actions = any(
            self._teacher_field_is_effective(value)
            for value in (result.add_phrases, result.remove_phrases, result.emphasize_phrases)
        )
        if not has_core_actions:
            return False, "repair_plan_invalid:empty_core_actions"
        if not self._teacher_field_is_effective(result.localize_with, invalid_markers=("",)):
            return False, "repair_plan_invalid:missing_localize_with"
        if not self._teacher_field_is_effective(result.rewrite_strategy, invalid_markers=("",)):
            return False, "repair_plan_invalid:missing_strategy"
        add_set = {item.lower() for item in self._split_teacher_field_list(result.add_phrases)}
        avoid_set = {item.lower() for item in self._split_teacher_field_list(result.avoid_phrases)}
        if add_set and avoid_set and (add_set & avoid_set):
            return False, "repair_plan_invalid:conflicting_actions"
        remove_set = {item.lower() for item in self._split_teacher_field_list(result.remove_phrases)}
        keep_set = {item.lower() for item in self._split_teacher_field_list(result.keepable_phrases)}
        if remove_set and keep_set and remove_set == keep_set:
            return False, "repair_plan_invalid:conflicting_actions"
        return True, ""

    def _contains_any_phrase(self, text, phrases_text):
        normalized_text = self._normalize_teacher_field_text(text).lower()
        if not normalized_text:
            return False
        for phrase in self._split_teacher_field_list(phrases_text):
            phrase = phrase.lower()
            if phrase and phrase in normalized_text:
                return True
        return False

    def _validate_teacher_dlc(self, result):
        if not self._teacher_field_is_effective(result.detailed_caption, invalid_markers=("",)):
            return "teacher_dlc_invalid:empty"
        detailed_caption_lower = result.detailed_caption.lower()
        if (
            self._teacher_text_overlap_ratio(result.detailed_caption, result.distractor_summary) >= 0.75
            and self._teacher_field_is_effective(result.target_only_evidence)
            and not self._difference_text_has_semantic_anchor(result.detailed_caption)
        ):
            return "teacher_dlc_invalid:distractor_overlap"
        if self._teacher_field_is_effective(result.target_only_evidence) and not (
            self._difference_text_has_semantic_anchor(result.detailed_caption)
            or any(
                token in detailed_caption_lower
                for token in (
                    "target-only",
                    "shared overlap",
                    "boundary",
                    "edge",
                    "centered",
                    "shifted",
                    "size",
                    "shape",
                )
            )
        ):
            return "teacher_dlc_invalid:missing_target_only_evidence"
        return ""

    def _validate_teacher_verification_caption(self, result):
        if not self._teacher_field_is_effective(result.verification_caption, invalid_markers=("",)):
            return "verification_invalid:empty"
        verification_tokens = self._caption_token_count(result.verification_caption)
        detailed_tokens = self._caption_token_count(result.detailed_caption)
        if detailed_tokens > 0 and verification_tokens >= detailed_tokens:
            return "verification_invalid:not_shorter_than_dlc"
        if self._is_overly_generic_caption(result.verification_caption):
            return "verification_invalid:too_generic"
        verification_caption_lower = result.verification_caption.lower()
        if self._teacher_field_is_effective(result.target_only_evidence) and not (
            self._difference_text_has_semantic_anchor(result.verification_caption)
            or any(
                token in verification_caption_lower
                for token in (
                    "target-only",
                    "boundary",
                    "edge",
                    "centered",
                    "shifted",
                    "size",
                    "shape",
                )
            )
        ):
            return "verification_invalid:lost_target_only_evidence"
        if self._teacher_text_overlap_ratio(result.verification_caption, result.distractor_summary) >= 0.75:
            return "verification_invalid:distractor_overlap"
        return ""

    def _truncate_caption_completion(self, caption, completion_ids, *, max_tokens):
        max_tokens = max(int(max_tokens), 1)
        if completion_ids.ndim != 2 or completion_ids.shape[0] != 1:
            return caption, completion_ids, False
        if int(completion_ids.shape[1]) <= max_tokens:
            return caption, completion_ids, False
        truncated_ids = completion_ids[:, :max_tokens]
        truncated_caption = self.tokenizer.decode(
            truncated_ids[0],
            skip_special_tokens=False,
        ).strip()
        truncated_caption = self._clean_caption_text(truncated_caption)
        return truncated_caption, truncated_ids, True

    @staticmethod
    def _mask_bbox(mask):
        if mask is None:
            return None
        mask = np.asarray(mask)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def _resolve_grpo_option_token_ids(self):
        option_token_ids = []
        for option_text in self._grpo_option_letters:
            tokenized = self.tokenizer(option_text, add_special_tokens=False).input_ids
            if len(tokenized) != 1:
                spaced_option = f" {option_text}"
                tokenized = self.tokenizer(spaced_option, add_special_tokens=False).input_ids
            if len(tokenized) != 1:
                raise ValueError(f"Unable to resolve single-token answer id for option {option_text!r}.")
            option_token_ids.append(int(tokenized[0]))
        return tuple(option_token_ids)

    def _should_debug_print(self):
        if not self.enable_debug_sample_logging:
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    @staticmethod
    def _dist_is_initialized():
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @staticmethod
    def _dist_rank():
        if Sa2VAOPSDModelV2._dist_is_initialized():
            return int(torch.distributed.get_rank())
        return 0

    def _cuda_memory_stats(self):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {
                "allocated": 0,
                "reserved": 0,
                "max_allocated": 0,
                "max_reserved": 0,
                "free": 0,
                "total": 0,
            }
        device = self.device
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        max_allocated = int(torch.cuda.max_memory_allocated(device))
        max_reserved = int(torch.cuda.max_memory_reserved(device))
        try:
            free, total = torch.cuda.mem_get_info(device)
            free = int(free)
            total = int(total)
        except Exception:
            free = 0
            total = 0
        return {
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_allocated,
            "max_reserved": max_reserved,
            "free": free,
            "total": total,
        }

    @staticmethod
    def _format_memory_value(num_bytes):
        gib = float(num_bytes) / float(1024 ** 3)
        return f"{gib:.2f}GiB"

    def _format_cuda_memory_stats(self):
        stats = self._cuda_memory_stats()
        return (
            "cuda_mem("
            f"alloc={self._format_memory_value(stats['allocated'])}, "
            f"reserved={self._format_memory_value(stats['reserved'])}, "
            f"max_alloc={self._format_memory_value(stats['max_allocated'])}, "
            f"max_reserved={self._format_memory_value(stats['max_reserved'])}, "
            f"free={self._format_memory_value(stats['free'])}, "
            f"total={self._format_memory_value(stats['total'])})"
        )

    def _log_realtime_memory(self, stage, *, extra=None):
        if not self._should_debug_print():
            return
        suffix = "" if not extra else f" {extra}"
        print(
            "[Sa2VA_OPSD_V2_MEM] "
            f"rank={self._dist_rank()} stage={stage} "
            f"{self._format_cuda_memory_stats()}{suffix}",
            flush=True,
        )

    def _log_ddp_route_alignment_debug(self, payload):
        if not self.enable_debug_sample_logging:
            return
        if not self._dist_is_initialized():
            print(
                "[Sa2VA_OPSD_V2_DDP_LOCAL_DEBUG] "
                f"rank={payload.get('rank')} batch_route={payload.get('batch_route')} "
                f"loss_family={payload.get('loss_family')} "
                f"entry_count={payload.get('entry_count')} "
                f"regen_entries={payload.get('regen_entry_count')} "
                f"onpolicy_entries={payload.get('onpolicy_entry_count')} "
                f"grpo_entries={payload.get('grpo_entry_count')} "
                f"optimized_count={payload.get('optimized_count')} "
                f"{self._format_cuda_memory_stats()}",
                flush=True,
            )
            return
        print(
            "[Sa2VA_OPSD_V2_DDP_LOCAL_DEBUG] "
            f"rank={payload.get('rank')} batch_route={payload.get('batch_route')} "
            f"loss_family={payload.get('loss_family')} "
            f"entry_count={payload.get('entry_count')} "
            f"real_entry_count={payload.get('real_entry_count')} "
            f"dummy_entry_count={payload.get('dummy_entry_count')} "
            f"regen_entries={payload.get('regen_entry_count')} "
            f"onpolicy_entries={payload.get('onpolicy_entry_count')} "
            f"grpo_entries={payload.get('grpo_entry_count')} "
            f"regen_loss_count={payload.get('regen_loss_count')} "
            f"onpolicy_loss_count={payload.get('onpolicy_loss_count')} "
            f"grpo_loss_count={payload.get('grpo_loss_count')} "
            f"optimized_count={payload.get('optimized_count')} "
            f"{self._format_cuda_memory_stats()}",
            flush=True,
        )
        gathered_payloads = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_payloads, payload)
        if self._dist_rank() != 0:
            return

        manifest_route_signatures = {
            tuple(record.get("manifest_route") for record in item.get("records", []))
            for item in gathered_payloads
        }
        loss_family_signatures = {
            str(item.get("loss_family"))
            for item in gathered_payloads
        }
        entry_count_signatures = {
            (
                int(item.get("entry_count", 0)),
                int(item.get("real_entry_count", 0)),
                int(item.get("dummy_entry_count", 0)),
                int(item.get("optimized_count", 0)),
            )
            for item in gathered_payloads
        }

        suspicious_reasons = []
        if len(manifest_route_signatures) > 1:
            suspicious_reasons.append("manifest-route-mismatch")
        if len(loss_family_signatures) > 1:
            suspicious_reasons.append("loss-family-mismatch")
        if len(entry_count_signatures) > 1:
            suspicious_reasons.append("entry-count-mismatch")
        if not suspicious_reasons:
            suspicious_reasons.append("no-obvious-mismatch-detected")

        print(
            "[Sa2VA_OPSD_V2_DDP_DEBUG] "
            f"potential_issue={','.join(suspicious_reasons)} "
            f"world_size={len(gathered_payloads)}",
            flush=True,
        )
        for item in gathered_payloads:
            records_text = []
            for record in item.get("records", []):
                teacher_iou_plain = record.get("teacher_iou_plain")
                teacher_iou_text = "None" if teacher_iou_plain is None else f"{float(teacher_iou_plain):.4f}"
                iou_value = record.get("iou")
                iou_text = "None" if iou_value is None else f"{float(iou_value):.4f}"
                records_text.append(
                    "sample_key={sample_key} manifest_route={manifest_route} loss_family={loss_family} "
                    "online_route={online_route} reconstruct_status={reconstruct_status} iou={iou} "
                    "allow_teacher_ce={allow_teacher_ce} teacher_reconstruct_ok={teacher_reconstruct_ok} "
                    "teacher_gate_passed={teacher_gate_passed} teacher_iou_plain={teacher_iou_plain} "
                    "teacher_completion_len={teacher_completion_len} is_dummy={is_dummy} dummy_reason={dummy_reason} "
                    "entry_added={entry_added} loss_branch={loss_branch} grpo_skip_reason={grpo_skip_reason} "
                    "confuser_candidate_count={confuser_candidate_count} "
                    "selected_confuser_count={selected_confuser_count} "
                    "scored_confuser_count={scored_confuser_count}".format(
                        sample_key=record.get("sample_key"),
                        manifest_route=record.get("manifest_route"),
                        loss_family=record.get("loss_family"),
                        online_route=record.get("online_route"),
                        reconstruct_status=record.get("reconstruct_status"),
                        iou=iou_text,
                        allow_teacher_ce=record.get("allow_teacher_ce"),
                        teacher_reconstruct_ok=record.get("teacher_reconstruct_ok"),
                        teacher_gate_passed=record.get("teacher_gate_passed"),
                        teacher_iou_plain=teacher_iou_text,
                        teacher_completion_len=record.get("teacher_completion_len"),
                        is_dummy=record.get("is_dummy"),
                        dummy_reason=record.get("dummy_reason"),
                        entry_added=record.get("entry_added"),
                        loss_branch=record.get("loss_branch"),
                        grpo_skip_reason=record.get("grpo_skip_reason"),
                        confuser_candidate_count=record.get("confuser_candidate_count"),
                        selected_confuser_count=record.get("selected_confuser_count"),
                        scored_confuser_count=record.get("scored_confuser_count"),
                    )
                )
            print(
                "[Sa2VA_OPSD_V2_DDP_DEBUG] "
                f"rank={item.get('rank')} batch_route={item.get('batch_route')} "
                f"loss_family={item.get('loss_family')} "
                f"entry_count={item.get('entry_count')} "
                f"real_entry_count={item.get('real_entry_count')} "
                f"dummy_entry_count={item.get('dummy_entry_count')} "
                f"dummy_reasons={item.get('dummy_reasons')} "
                f"regen_entries={item.get('regen_entry_count')} "
                f"onpolicy_entries={item.get('onpolicy_entry_count')} "
                f"grpo_entries={item.get('grpo_entry_count')} "
                f"regen_loss_count={item.get('regen_loss_count')} "
                f"onpolicy_loss_count={item.get('onpolicy_loss_count')} "
                f"grpo_loss_count={item.get('grpo_loss_count')} "
                f"optimized_count={item.get('optimized_count')} "
                f"records=[{' ; '.join(records_text)}]",
                flush=True,
            )

    def _log_pre_return_debug(
        self,
        *,
        batch_route,
        last_route,
        optimized_count,
        regen_loss_count,
        onpolicy_loss_count,
        grpo_loss_count,
        total_loss,
        total_regen_ce,
        total_onpolicy_jsd,
        total_grpo,
        rank_debug_records,
    ):
        if not self.enable_debug_sample_logging:
            return
        total_loss_value = None if total_loss is None else float(total_loss.detach().item())
        total_regen_value = None if total_regen_ce is None else float(total_regen_ce.detach().item())
        total_onpolicy_value = None if total_onpolicy_jsd is None else float(total_onpolicy_jsd.detach().item())
        total_grpo_value = None if total_grpo is None else float(total_grpo.detach().item())
        records_text = []
        for record in rank_debug_records:
            records_text.append(
                "sample_key={sample_key} loss_branch={loss_branch} online_route={online_route} "
                "iou={iou:.4f} is_dummy={is_dummy} dummy_reason={dummy_reason} "
                "grpo_skip_reason={grpo_skip_reason} confuser_candidate_count={confuser_candidate_count} "
                "selected_confuser_count={selected_confuser_count} "
                "scored_confuser_count={scored_confuser_count} "
                "teacher_verification_caption_status={teacher_verification_caption_status} "
                "teacher_verification_caption={teacher_verification_caption} "
                "teacher_dlc={teacher_dlc} "
                "target_summary={target_summary} "
                "distractor_summary={distractor_summary} "
                "shared_evidence={shared_evidence} "
                "target_only_evidence={target_only_evidence} "
                "distractor_only_evidence={distractor_only_evidence} "
                "difference_focus={difference_focus} "
                "likely_drift_reason={likely_drift_reason} "
                "caption_problem={caption_problem} "
                "correction_direction={correction_direction} "
                "reason={reason} "
                "single_stage_raw={single_stage_raw} "
                "structured_diagnosis_raw={structured_diagnosis_raw} "
                "repair_raw={repair_raw} "
                "problem_raw={problem_raw} "
                "direction_raw={direction_raw} "
                "reason_raw={reason_raw} "
                "teacher_problem_valid={teacher_problem_valid} "
                "teacher_direction_valid={teacher_direction_valid} "
                "teacher_reason_valid={teacher_reason_valid} "
                "teacher_reason_is_coarse={teacher_reason_is_coarse} "
                "teacher_problem_failure_reason={teacher_problem_failure_reason} "
                "teacher_direction_failure_reason={teacher_direction_failure_reason} "
                "teacher_reason_failure_reason={teacher_reason_failure_reason} "
                "teacher_diagnosis_failure_reason={teacher_diagnosis_failure_reason} "
                "teacher_pipeline_mode={teacher_pipeline_mode} "
                "teacher_diagnosis_retry_count={teacher_diagnosis_retry_count} "
                "teacher_dlc_candidate_count={teacher_dlc_candidate_count} "
                "teacher_dlc_valid_candidate_count={teacher_dlc_valid_candidate_count} "
                "teacher_dlc_selected_by={teacher_dlc_selected_by} "
                "teacher_verification_used={teacher_verification_used} "
                "teacher_fallback_used={teacher_fallback_used} "
                "teacher_fallback_reason={teacher_fallback_reason} "
                "teacher_selected_caption_source={teacher_selected_caption_source} "
                "teacher_dlc_candidate_scores={teacher_dlc_candidate_scores} "
                "teacher_pipeline_stop_stage={teacher_pipeline_stop_stage} "
                "teacher_pipeline_failure_reason={teacher_pipeline_failure_reason}".format(
                    sample_key=record.get("sample_key"),
                    loss_branch=record.get("loss_branch"),
                    online_route=record.get("online_route"),
                    iou=float(record.get("iou", 0.0)),
                    is_dummy=record.get("is_dummy"),
                    dummy_reason=record.get("dummy_reason"),
                    grpo_skip_reason=record.get("grpo_skip_reason"),
                    confuser_candidate_count=record.get("confuser_candidate_count"),
                    selected_confuser_count=record.get("selected_confuser_count"),
                    scored_confuser_count=record.get("scored_confuser_count"),
                    teacher_verification_caption_status=record.get("teacher_verification_caption_status"),
                    teacher_verification_caption=repr(record.get("teacher_verification_caption", "")),
                    teacher_dlc=repr(record.get("teacher_dlc", "")),
                    target_summary=repr(record.get("target_summary", "")),
                    distractor_summary=repr(record.get("distractor_summary", "")),
                    shared_evidence=repr(record.get("shared_evidence", "")),
                    target_only_evidence=repr(record.get("target_only_evidence", "")),
                    distractor_only_evidence=repr(record.get("distractor_only_evidence", "")),
                    difference_focus=repr(record.get("difference_focus", "")),
                    likely_drift_reason=repr(record.get("likely_drift_reason", "")),
                    caption_problem=repr(record.get("caption_problem", "")),
                    correction_direction=repr(record.get("correction_direction", "")),
                    reason=repr(record.get("reason", "")),
                    single_stage_raw=repr(record.get("single_stage_raw", "")),
                    structured_diagnosis_raw=repr(record.get("structured_diagnosis_raw", "")),
                    repair_raw=repr(record.get("repair_raw", "")),
                    problem_raw=repr(record.get("problem_raw", "")),
                    direction_raw=repr(record.get("direction_raw", "")),
                    reason_raw=repr(record.get("reason_raw", "")),
                    teacher_problem_valid=record.get("teacher_problem_valid"),
                    teacher_direction_valid=record.get("teacher_direction_valid"),
                    teacher_reason_valid=record.get("teacher_reason_valid"),
                    teacher_reason_is_coarse=record.get("teacher_reason_is_coarse"),
                    teacher_problem_failure_reason=record.get("teacher_problem_failure_reason"),
                    teacher_direction_failure_reason=record.get("teacher_direction_failure_reason"),
                    teacher_reason_failure_reason=record.get("teacher_reason_failure_reason"),
                    teacher_diagnosis_failure_reason=record.get("teacher_diagnosis_failure_reason"),
                    teacher_pipeline_mode=record.get("teacher_pipeline_mode"),
                    teacher_diagnosis_retry_count=record.get("teacher_diagnosis_retry_count"),
                    teacher_dlc_candidate_count=record.get("teacher_dlc_candidate_count"),
                    teacher_dlc_valid_candidate_count=record.get("teacher_dlc_valid_candidate_count"),
                    teacher_dlc_selected_by=record.get("teacher_dlc_selected_by"),
                    teacher_verification_used=record.get("teacher_verification_used"),
                    teacher_fallback_used=record.get("teacher_fallback_used"),
                    teacher_fallback_reason=record.get("teacher_fallback_reason"),
                    teacher_selected_caption_source=record.get("teacher_selected_caption_source"),
                    teacher_dlc_candidate_scores=repr(record.get("teacher_dlc_candidate_scores", ())),
                    teacher_pipeline_stop_stage=record.get("teacher_pipeline_stop_stage"),
                    teacher_pipeline_failure_reason=record.get("teacher_pipeline_failure_reason"),
                )
            )
        print(
            "[Sa2VA_OPSD_V2_PRE_RETURN_DEBUG] "
            f"rank={self._dist_rank()} batch_route={batch_route} last_route={last_route} "
            f"optimized_count={optimized_count} regen_loss_count={regen_loss_count} "
            f"onpolicy_loss_count={onpolicy_loss_count} grpo_loss_count={grpo_loss_count} "
            f"total_loss={total_loss_value} total_regen_ce={total_regen_value} "
            f"total_onpolicy_jsd={total_onpolicy_value} total_grpo={total_grpo_value} "
            f"{self._format_cuda_memory_stats()} "
            f"records=[{' ; '.join(records_text)}]",
            flush=True,
        )

    def _debug_sample(
        self,
        *,
        sample_key,
        route,
        student_question,
        raw_prediction,
        caption,
        description_status,
        raw_caption_failure_mode="",
        clean_description_status="",
        student_caption_trainable=None,
        reconstruct_question,
        raw_reconstruct_prediction,
        reconstruct_status,
        seg_token_count,
        prediction_masks_count,
        pred_mask,
        gt_mask,
        iou,
        empty_gt_mask,
    ):
        if not self._should_debug_print():
            return
        pred_sum = None if pred_mask is None else int(np.asarray(pred_mask).sum())
        gt_sum = None if gt_mask is None else int(np.asarray(gt_mask).sum())
        resize_info = ""
        if pred_mask is not None and gt_mask is not None:
            _, resized, pred_shape_before_resize, pred_shape_after_resize = self._prepare_pred_mask_for_iou(
                gt_mask, pred_mask
            )
            resize_info = (
                f"\n[Sa2VA_OPSD_V2_DEBUG] pred_mask_shape_before_resize={pred_shape_before_resize} "
                f"pred_mask_shape_after_resize={pred_shape_after_resize} resized_for_iou={resized}"
            )
        print(
            f"sample_key={sample_key!r}\n"
            f"route={route} low_iou_threshold={self.iou_low_threshold:.4f} "
            f"high_iou_threshold={self.iou_high_threshold:.4f}\n"
            f"student_question={student_question!r}\n"
            f"student_caption={caption!r}\n"
            f"raw_prediction={raw_prediction!r}\n"
            f"description_status={description_status}\n"
            f"raw_caption_failure_mode={raw_caption_failure_mode}\n"
            f"clean_description_status={clean_description_status}\n"
            f"student_caption_trainable={student_caption_trainable}\n"
            f"reconstruct_question={reconstruct_question!r}\n"
            f"raw_reconstruct_prediction={raw_reconstruct_prediction!r}\n"
            f"reconstruct_status={reconstruct_status}\n"
            f"seg_token_count={seg_token_count}\n"
            f"prediction_masks_count={prediction_masks_count}\n"
            f"pred_mask_sum={pred_sum} gt_mask_sum={gt_sum}\n"
            f"pred_mask_shape={None if pred_mask is None else tuple(np.asarray(pred_mask).shape)} "
            f"gt_mask_shape={None if gt_mask is None else tuple(np.asarray(gt_mask).shape)}\n"
            f"pred_bbox={self._mask_bbox(pred_mask)} gt_bbox={self._mask_bbox(gt_mask)}\n"
            f"empty_gt_mask={empty_gt_mask} iou={iou:.4f}"
            f"{resize_info}"
        )

    @staticmethod
    @contextmanager
    def _temporary_eval_model(model):
        was_training = model.training
        model.eval()
        try:
            yield model
        finally:
            model.train(was_training)

    def _predict_forward_eval(self, model, **kwargs):
        with self._temporary_eval_model(model):
            with torch.inference_mode():
                signature = inspect.signature(model.predict_forward)
                accepted_kwargs = dict(kwargs)
                prompt_text = accepted_kwargs.get("text")
                if isinstance(prompt_text, str):
                    has_visual_input = accepted_kwargs.get("image") is not None or accepted_kwargs.get("video") is not None
                    uses_mask_prompts = accepted_kwargs.get("mask_prompts") is not None
                    if (has_visual_input or uses_mask_prompts) and "<image>" not in prompt_text:
                        accepted_kwargs["text"] = f"<image>\n{prompt_text.lstrip()}"
                generation_override_keys = (
                    "max_new_tokens",
                    "do_sample",
                    "temperature",
                    "top_p",
                    "repetition_penalty",
                    "no_repeat_ngram_size",
                    "bad_words_ids",
                )
                if "processor" in signature.parameters and "processor" not in kwargs:
                    accepted_kwargs["processor"] = self.processor
                dropped_generation_overrides = {}
                if not any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                ):
                    dropped_generation_overrides = {
                        key: accepted_kwargs[key]
                        for key in generation_override_keys
                        if key in accepted_kwargs and key not in signature.parameters
                    }
                    accepted_kwargs = {
                        key: value
                        for key, value in accepted_kwargs.items()
                        if key in signature.parameters
                    }
                generation_config = getattr(model, "gen_config", None)
                original_generation_values = None
                if generation_config is not None and dropped_generation_overrides:
                    original_generation_values = {
                        key: getattr(generation_config, key, None)
                        for key in dropped_generation_overrides
                    }
                    for key, value in dropped_generation_overrides.items():
                        setattr(generation_config, key, value)
                try:
                    return model.predict_forward(**accepted_kwargs)
                finally:
                    if original_generation_values is not None:
                        for key, value in original_generation_values.items():
                            setattr(generation_config, key, value)

    def predict_text_with_masks(
        self,
        model,
        *,
        image,
        text,
        mask_prompts=None,
        apply_mask_focus=False,
    ):
        self._ensure_generation_ready(model)
        formatted_mask_prompts = None
        if mask_prompts is not None:
            formatted_mask_prompts = self._format_mask_prompts_for_predict_forward(mask_prompts)
        prompt_image = self._build_mask_focused_image(image, mask_prompts) if (apply_mask_focus and mask_prompts is not None) else image
        predict_dict = self._predict_forward_eval(
            model,
            image=prompt_image,
            text=text,
            past_text="",
            mask_prompts=formatted_mask_prompts,
            tokenizer=self.tokenizer,
        )
        return {"prediction": predict_dict.get("prediction", "")}

    @staticmethod
    def _format_mask_prompts_for_predict_forward(mask_prompts):
        mask_prompts = np.asarray(mask_prompts, dtype=np.float32)
        if mask_prompts.ndim == 2:
            mask_prompts = np.expand_dims(mask_prompts, axis=0)
        if mask_prompts.ndim != 3:
            raise ValueError(f"mask_prompts must have shape (n_prompts, h, w), got {mask_prompts.shape}")
        # HF predict_forward expects an iterable whose items are (n_prompts, h, w),
        # even for a single image.
        return [mask_prompts]

    @staticmethod
    def _to_teacher_prompt_masks(prompt_masks):
        if isinstance(prompt_masks, np.ndarray):
            if prompt_masks.ndim == 2:
                return [torch.from_numpy(prompt_masks.astype(np.float32))]
            if prompt_masks.ndim == 3:
                return [torch.from_numpy(item.astype(np.float32)) for item in prompt_masks]
        return [torch.as_tensor(item, dtype=torch.float32) for item in prompt_masks]

    @staticmethod
    def _strip_image_placeholder(text):
        return text.replace("<image>\n", "").replace("<image>", "").strip()

    def _normalize_student_question(self, student_question):
        return self._strip_image_placeholder(student_question)

    def _create_region_prompt(self, model, prompt_masks):
        stacked_masks = torch.stack(
            [torch.as_tensor(item, dtype=torch.float32, device=self.device) for item in prompt_masks],
            dim=0,
        )
        target_size = int(model.image_size // model.patch_size * model.downsample_ratio)
        resized = F.interpolate(
            stacked_masks.unsqueeze(0),
            size=(target_size, target_size),
            mode="nearest",
        ).squeeze(0)
        region_pixels = [int(mask.bool().sum().item()) for mask in resized]
        vp_token_str = "\nThere are {} part regions in the picture: ".format(len(region_pixels))
        for idx, pixels in enumerate(region_pixels):
            vp_token_str += (
                f"region{idx + 1}{model.VP_START_TOKEN}"
                f"{model.IMG_CONTEXT_TOKEN * pixels}"
                f"{model.VP_END_TOKEN}"
            )
            vp_token_str += ".\n" if idx == len(region_pixels) - 1 else ", "
        return [resized], vp_token_str

    def _build_forward_inputs(self, model, image, prompt_masks, question_text):
        self._ensure_generation_ready(model)
        ori_image_size = image.size
        if hasattr(model, "dynamic_preprocess"):
            images = model.dynamic_preprocess(
                image,
                model.min_dynamic_patch,
                model.max_dynamic_patch,
                model.image_size,
                model.use_thumbnail,
            )
        else:
            from projects.sa2va.hf.models.modeling_sa2va_chat import dynamic_preprocess

            images = dynamic_preprocess(
                image,
                model.min_dynamic_patch,
                model.max_dynamic_patch,
                model.image_size,
                model.use_thumbnail,
            )
        pixel_values = torch.stack([model.transformer(item) for item in images]).to(
            device=self.device,
            dtype=model.torch_dtype,
        )
        if prompt_masks is not None:
            prompt_masks, vp_token_str = self._create_region_prompt(model, prompt_masks)
            vp_overall_mask = torch.tensor([False] * (len(images) - 1) + [True], device=self.device)
        else:
            vp_token_str = ""
            vp_overall_mask = None
        clean_question = self._normalize_student_question(question_text)
        full_human_prompt = "<image>\n" + vp_token_str + clean_question
        num_image_tokens = pixel_values.shape[0] * model.patch_token
        image_token_str = (
            f"{model.IMG_START_TOKEN}"
            f"{model.IMG_CONTEXT_TOKEN * num_image_tokens}"
            f"{model.IMG_END_TOKEN}\n"
        )
        input_text = full_human_prompt.replace("<image>\n", image_token_str, 1)
        input_text = model.template["INSTRUCTION"].format(
            input=input_text,
            round=1,
            bot_name=model.bot_name,
        )
        ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)
        attention_mask = torch.ones_like(ids, dtype=torch.bool)
        position_ids = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt_masks": prompt_masks,
            "vp_overall_mask": vp_overall_mask,
            "ori_image_size": ori_image_size,
        }

    def _extract_vit_embeds(self, model, pixel_values):
        return model.extract_feature(pixel_values.to(self.device))

    def _compose_inputs_embeds(self, model, mm_inputs, input_ids=None):
        input_ids = mm_inputs["input_ids"] if input_ids is None else input_ids.to(self.device)
        pixel_values = mm_inputs["pixel_values"]
        prompt_masks = mm_inputs["prompt_masks"]
        vp_overall_mask = mm_inputs["vp_overall_mask"]
        input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
        batch_size, seq_len, hidden_dim = input_embeds.shape
        flat_input_embeds = input_embeds.reshape(batch_size * seq_len, hidden_dim)
        vit_embeds = self._extract_vit_embeds(model, pixel_values)
        image_flags = (torch.sum(pixel_values, dim=(1, 2, 3)) != 0).to(self.device).long()
        vit_embeds = vit_embeds[image_flags == 1]
        if prompt_masks is None or vp_overall_mask is None:
            vp_embeds = vit_embeds.reshape(-1, hidden_dim)
        else:
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(self.device).bool()[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask]
            vp_img_idx = 0
            for image_idx in range(len(vit_embeds)):
                vp_embeds.append(vit_embeds[image_idx].reshape(-1, hidden_dim))
                if vp_overall_mask[image_idx]:
                    tile_vit_embeds = overall_tile_vit_embeds[vp_img_idx].reshape(-1, hidden_dim)
                    object_masks = prompt_masks[vp_img_idx].to(self.device).bool()
                    num_objects = len(object_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(num_objects, 1, 1)
                    object_masks = object_masks.reshape(num_objects, -1)
                    vp_embeds.append(tile_vit_embeds[object_masks])
                    vp_img_idx += 1
            vp_embeds = torch.cat(vp_embeds, dim=0)
        selected = input_ids.reshape(batch_size * seq_len) == model.img_context_token_id
        expected_tokens = int(selected.sum().item())
        if vp_embeds.shape[0] < expected_tokens:
            raise RuntimeError(
                f"VP embed count mismatch for {type(model).__name__}: "
                f"expected {expected_tokens}, got {vp_embeds.shape[0]}."
            )
        flat_input_embeds[selected] = vp_embeds[:expected_tokens]
        return flat_input_embeds.reshape(batch_size, seq_len, hidden_dim)

    def _forward_sequence_with_model(
        self,
        model,
        image,
        prompt_masks,
        prompt_text,
        completion_ids,
        apply_mask_focus=True,
    ):
        teacher_prompt_masks = None if prompt_masks is None else self._to_teacher_prompt_masks(prompt_masks)
        caption_image = self._build_mask_focused_image(image, prompt_masks) if apply_mask_focus and prompt_masks is not None else image
        mm_inputs = self._build_forward_inputs(model, caption_image, teacher_prompt_masks, prompt_text)
        prompt_len = mm_inputs["input_ids"].shape[1]
        completion_len = int(completion_ids.shape[1])
        full_ids = torch.cat([mm_inputs["input_ids"], completion_ids.to(self.device)], dim=1)
        full_attention_mask = torch.ones_like(full_ids, dtype=torch.bool)
        full_position_ids = torch.arange(full_ids.shape[1], device=self.device).unsqueeze(0)
        inputs_embeds = self._compose_inputs_embeds(model, mm_inputs, input_ids=full_ids)
        forward_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            position_ids=full_position_ids,
            use_cache=False,
            return_dict=True,
        )
        requested_logits_to_keep = None
        try:
            forward_signature = inspect.signature(model.language_model.forward)
            if completion_len > 0 and "logits_to_keep" in forward_signature.parameters:
                requested_logits_to_keep = completion_len + 1
                forward_kwargs["logits_to_keep"] = requested_logits_to_keep
        except (TypeError, ValueError):
            pass

        outputs = model.language_model(**forward_kwargs)
        if requested_logits_to_keep is not None and outputs.logits.shape[1] == requested_logits_to_keep:
            return outputs.logits[:, :-1, :]
        return outputs.logits[:, prompt_len - 1: -1, :]

    @staticmethod
    def _pad_tensor_rows(tensors, *, pad_value, dtype=None, device=None):
        if not tensors:
            raise ValueError("tensors must not be empty.")
        target_device = device if device is not None else tensors[0].device
        target_dtype = dtype if dtype is not None else tensors[0].dtype
        max_len = max(int(tensor.shape[1]) for tensor in tensors)
        padded = torch.full(
            (len(tensors), max_len),
            pad_value,
            dtype=target_dtype,
            device=target_device,
        )
        valid_mask = torch.zeros((len(tensors), max_len), dtype=torch.bool, device=target_device)
        for row_idx, tensor in enumerate(tensors):
            if tensor.ndim != 2 or tensor.shape[0] != 1:
                raise ValueError(f"Expected tensors with shape (1, seq_len), got {tuple(tensor.shape)}.")
            row = tensor[0].to(device=target_device, dtype=target_dtype)
            row_len = int(row.shape[0])
            padded[row_idx, :row_len] = row
            valid_mask[row_idx, :row_len] = True
        return padded, valid_mask

    @staticmethod
    def _pad_sequence_batch(sequences, *, pad_value, dtype=None, device=None):
        if not sequences:
            raise ValueError("sequences must not be empty.")
        target_device = device if device is not None else sequences[0].device
        target_dtype = dtype if dtype is not None else sequences[0].dtype
        tail_shape = tuple(sequences[0].shape[1:])
        max_len = max(int(sequence.shape[0]) for sequence in sequences)
        padded = torch.full(
            (len(sequences), max_len, *tail_shape),
            pad_value,
            dtype=target_dtype,
            device=target_device,
        )
        valid_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=target_device)
        for row_idx, sequence in enumerate(sequences):
            if tuple(sequence.shape[1:]) != tail_shape:
                raise ValueError(
                    f"Expected all sequences to share tail shape {tail_shape}, got {tuple(sequence.shape[1:])}."
                )
            row = sequence.to(device=target_device, dtype=target_dtype)
            row_len = int(row.shape[0])
            padded[row_idx, :row_len] = row
            valid_mask[row_idx, :row_len] = True
        return padded, valid_mask

    def _build_full_sequence_batch(self, model, samples):
        if not samples:
            raise ValueError("samples must not be empty.")

        prompt_sequences = []
        prompt_lengths = []
        completion_rows = []
        completion_lengths = []
        full_input_ids_rows = []
        full_embed_rows = []
        embedding_layer = model.language_model.get_input_embeddings()

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        for sample in samples:
            image = sample["image"]
            prompt_masks = sample.get("prompt_masks")
            prompt_text = sample["prompt_text"]
            apply_mask_focus = bool(sample.get("apply_mask_focus", True))
            completion_ids = sample["completion_ids"]
            if completion_ids.ndim == 1:
                completion_ids = completion_ids.unsqueeze(0)
            if completion_ids.ndim != 2 or completion_ids.shape[0] != 1:
                raise ValueError(
                    f"Expected completion_ids with shape (1, seq_len), got {tuple(completion_ids.shape)}."
                )
            completion_ids = completion_ids.to(self.device)
            prompt_image = self._build_mask_focused_image(image, prompt_masks) if apply_mask_focus and prompt_masks is not None else image
            prompt_masks_for_forward = None if prompt_masks is None else self._to_teacher_prompt_masks(prompt_masks)
            mm_inputs = self._build_forward_inputs(model, prompt_image, prompt_masks_for_forward, prompt_text)
            prompt_embeds = self._compose_inputs_embeds(model, mm_inputs)
            if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
                raise ValueError(
                    f"Expected prompt_embeds with shape (1, seq_len, hidden), got {tuple(prompt_embeds.shape)}."
                )
            completion_embeds = embedding_layer(completion_ids)
            full_input_ids = torch.cat([mm_inputs["input_ids"], completion_ids], dim=1)
            full_embeds = torch.cat([prompt_embeds, completion_embeds], dim=1)

            prompt_sequences.append(mm_inputs)
            prompt_lengths.append(int(prompt_embeds.shape[1]))
            completion_rows.append(completion_ids[0])
            completion_lengths.append(int(completion_ids.shape[1]))
            full_input_ids_rows.append(full_input_ids[0])
            full_embed_rows.append(full_embeds[0])

        completion_ids_batch, completion_mask = self._pad_sequence_batch(
            completion_rows,
            pad_value=pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        full_input_ids_batch, _ = self._pad_sequence_batch(
            full_input_ids_rows,
            pad_value=pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        full_inputs_embeds, attention_mask = self._pad_sequence_batch(
            full_embed_rows,
            pad_value=0.0,
            dtype=full_embed_rows[0].dtype,
            device=self.device,
        )
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(~attention_mask, 0)
        prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long, device=self.device)
        completion_lengths = torch.tensor(completion_lengths, dtype=torch.long, device=self.device)
        return {
            "samples": prompt_sequences,
            "input_ids": full_input_ids_batch,
            "inputs_embeds": full_inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "completion_ids": completion_ids_batch,
            "completion_mask": completion_mask,
            "prompt_lengths": prompt_lengths,
            "completion_lengths": completion_lengths,
        }

    def _forward_sequence_multi_sample_with_model(self, model, samples, output_hidden_states=False):
        batch_inputs = self._build_full_sequence_batch(model, samples)
        outputs = model.language_model(
            inputs_embeds=batch_inputs["inputs_embeds"],
            attention_mask=batch_inputs["attention_mask"],
            position_ids=batch_inputs["position_ids"],
            use_cache=False,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )
        logits = outputs.logits
        completion_len = int(batch_inputs["completion_ids"].shape[1])
        vocab_size = logits.shape[-1]
        if completion_len == 0:
            completion_logits = logits[:, 0:0, :]
            gather_positions = torch.zeros((logits.shape[0], 0), dtype=torch.long, device=logits.device)
        else:
            gather_positions = batch_inputs["prompt_lengths"].unsqueeze(1) - 1 + torch.arange(
                completion_len, device=logits.device
            ).unsqueeze(0)
            gather_positions = gather_positions.clamp(min=0, max=logits.shape[1] - 1)
            completion_logits = logits.gather(
                dim=1,
                index=gather_positions.unsqueeze(-1).expand(-1, -1, vocab_size),
            )
        result = {
            "batch_inputs": batch_inputs,
            "logits": completion_logits,
            "completion_ids": batch_inputs["completion_ids"],
            "completion_mask": batch_inputs["completion_mask"],
            "gather_positions": gather_positions,
        }
        if output_hidden_states:
            last_hidden_states = outputs.hidden_states[-1]
            hidden_dim = last_hidden_states.shape[-1]
            completion_hidden_states = last_hidden_states.gather(
                dim=1,
                index=gather_positions.unsqueeze(-1).expand(-1, -1, hidden_dim),
            )
            result["completion_hidden_states"] = completion_hidden_states
            result["outputs"] = outputs
        return result

    @staticmethod
    def _masked_token_mean(values, valid_mask):
        weights = valid_mask.to(dtype=values.dtype)
        return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    @staticmethod
    def _sequence_cross_entropy_batch_from_logits(logits, completion_ids, completion_mask):
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            completion_ids.reshape(-1).to(logits.device),
            reduction="none",
        ).reshape_as(completion_ids)
        return Sa2VAOPSDModelV2._masked_token_mean(token_losses, completion_mask)

    def _forward_sequence_batch_with_model(
        self,
        model,
        image,
        prompt_masks,
        prompt_text,
        completion_ids_batch,
        apply_mask_focus=True,
    ):
        if completion_ids_batch.ndim != 2:
            raise ValueError(
                f"completion_ids_batch must have shape (batch, seq_len), got {tuple(completion_ids_batch.shape)}."
            )
        teacher_prompt_masks = None if prompt_masks is None else self._to_teacher_prompt_masks(prompt_masks)
        caption_image = self._build_mask_focused_image(image, prompt_masks) if apply_mask_focus and prompt_masks is not None else image
        mm_inputs = self._build_forward_inputs(model, caption_image, teacher_prompt_masks, prompt_text)
        prompt_len = mm_inputs["input_ids"].shape[1]
        completion_len = int(completion_ids_batch.shape[1])
        prompt_embeds = self._compose_inputs_embeds(model, mm_inputs)
        completion_ids_batch = completion_ids_batch.to(self.device)
        completion_embeds = model.language_model.get_input_embeddings()(completion_ids_batch)
        full_inputs_embeds = torch.cat(
            [prompt_embeds.repeat(completion_ids_batch.shape[0], 1, 1), completion_embeds],
            dim=1,
        )
        full_attention_mask = torch.ones(
            full_inputs_embeds.shape[:2],
            dtype=torch.bool,
            device=self.device,
        )
        full_position_ids = torch.arange(full_inputs_embeds.shape[1], device=self.device).unsqueeze(0).expand(
            completion_ids_batch.shape[0], -1
        )
        forward_kwargs = dict(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            position_ids=full_position_ids,
            use_cache=False,
            return_dict=True,
        )
        requested_logits_to_keep = None
        try:
            forward_signature = inspect.signature(model.language_model.forward)
            if completion_len > 0 and "logits_to_keep" in forward_signature.parameters:
                requested_logits_to_keep = completion_len + 1
                forward_kwargs["logits_to_keep"] = requested_logits_to_keep
        except (TypeError, ValueError):
            pass

        outputs = model.language_model(**forward_kwargs)
        if requested_logits_to_keep is not None and outputs.logits.shape[1] == requested_logits_to_keep:
            return outputs.logits[:, :-1, :]
        return outputs.logits[:, prompt_len - 1: -1, :]
    def predict_teacher_caption_on_student_trajectory(
        self,
        *,
        image,
        mask_prompts,
        student_question,
        student_caption,
        apply_mask_focus=True,
    ):
        student_caption = self._clean_caption_text(student_caption or "")
        student_completion_ids = self._encode_completion_from_caption(student_caption)
        if student_completion_ids.shape[1] == 0:
            return DescriptionResult(
                raw_prediction="",
                clean_caption="",
                completion_ids=student_completion_ids,
                status="empty",
                raw_failure_mode="",
                clean_status="empty",
            )

        teacher_model = self.require_teacher_model("Teacher trajectory prediction")
        with torch.inference_mode():
            teacher_logits = self._forward_sequence_with_model(
                teacher_model,
                image,
                mask_prompts,
                student_question,
                student_completion_ids,
                apply_mask_focus=apply_mask_focus,
            )
        predicted_ids = teacher_logits.argmax(dim=-1)
        raw_prediction = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=False).strip()
        clean_caption = self._clean_caption_text(raw_prediction)
        return self._finalize_description_result(
            raw_prediction=raw_prediction,
            clean_caption=clean_caption,
            completion_ids=predicted_ids,
        )

    def generate_description_with_model(
        self,
        model,
        *,
        image,
        mask_prompts,
        student_question,
        apply_mask_focus=True,
    ):
        formatted_mask_prompts = self._format_mask_prompts_for_predict_forward(mask_prompts)
        prompt_image = self._build_mask_focused_image(image, mask_prompts) if apply_mask_focus else image
        predict_dict = self._predict_forward_eval(
            model,
            image=prompt_image,
            text=student_question,
            past_text="",
            mask_prompts=formatted_mask_prompts,
            tokenizer=self.tokenizer,
            max_new_tokens=self.description_max_new_tokens,
            do_sample=False,
            repetition_penalty=self.description_repetition_penalty,
            no_repeat_ngram_size=self.description_no_repeat_ngram_size,
            bad_words_ids=self._caption_bad_words_ids,
        )
        raw_prediction = predict_dict.get("prediction", "")
        clean_caption = self._clean_caption_text(raw_prediction)
        completion_ids = self._encode_completion_from_caption(clean_caption, model=model)
        return self._finalize_description_result(
            raw_prediction=raw_prediction,
            clean_caption=clean_caption,
            completion_ids=completion_ids,
        )

    def generate_description(self, image, mask_prompts, student_question):
        return self.generate_description_with_model(
            self.student_model,
            image=image,
            mask_prompts=mask_prompts,
            student_question=student_question,
            apply_mask_focus=True,
        )

    def _predict_teacher_privileged_text(
        self,
        *,
        image,
        teacher_prompt_masks,
        teacher_prompt,
        max_new_tokens=None,
        do_sample=False,
        temperature=None,
        top_p=None,
    ):
        teacher_model = self.require_teacher_model("Teacher privileged regeneration")
        formatted_mask_prompts = self._format_mask_prompts_for_predict_forward(teacher_prompt_masks)
        prompt_image = self._build_mask_focused_image(image, teacher_prompt_masks)
        predict_kwargs = dict(
            image=prompt_image,
            text=teacher_prompt,
            past_text="",
            mask_prompts=formatted_mask_prompts,
            tokenizer=self.tokenizer,
            max_new_tokens=self.description_max_new_tokens if max_new_tokens is None else int(max_new_tokens),
            do_sample=bool(do_sample),
            repetition_penalty=self.description_repetition_penalty,
            no_repeat_ngram_size=self.description_no_repeat_ngram_size,
            bad_words_ids=self._caption_bad_words_ids,
        )
        if temperature is not None:
            predict_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            predict_kwargs["top_p"] = float(top_p)
        predict_dict = self._predict_forward_eval(
            teacher_model,
            **predict_kwargs,
        )
        return "" if predict_dict is None else str(predict_dict.get("prediction", ""))

    def reconstruct_mask(self, image, caption, description_status, spatial_hint="", gt_mask=None):
        del spatial_hint
        if description_status != "ok":
            return ReconstructionResult(
                pred_mask=None,
                question=None,
                raw_prediction="",
                prediction_masks_count=0,
                seg_token_count=0,
                status="skipped_invalid_description",
            )
        reconstruct_question = self._resolve_reconstruct_questions(caption)[0]
        predict_dict = self._predict_forward_eval(
            self.student_model,
            image=image,
            text=reconstruct_question,
            past_text="",
            mask_prompts=None,
            tokenizer=self.tokenizer,
            max_new_tokens=self.low_iou_regen_max_new_tokens,
            do_sample=False,
        )
        raw_prediction = predict_dict.get("prediction", "")
        prediction_masks = predict_dict.get("prediction_masks")
        prediction_masks_count = 0 if prediction_masks is None else len(prediction_masks)
        seg_token_count = int(predict_dict.get("seg_token_count", 0) or 0)
        if not prediction_masks:
            return ReconstructionResult(
                pred_mask=None,
                question=reconstruct_question,
                raw_prediction=raw_prediction,
                prediction_masks_count=prediction_masks_count,
                seg_token_count=seg_token_count,
                status="empty_prediction_masks",
            )
        first_mask = prediction_masks[0]
        if isinstance(first_mask, torch.Tensor):
            first_mask = first_mask.detach().cpu().numpy()
        first_mask = np.asarray(first_mask)
        if first_mask.ndim == 3 and first_mask.shape[0] == 1:
            first_mask = first_mask[0]
        pred_mask = self._to_numpy_mask(first_mask)
        status = "ok" if pred_mask.sum() > 0 else "zero_area_mask"
        return ReconstructionResult(
            pred_mask=pred_mask,
            question=reconstruct_question,
            raw_prediction=raw_prediction,
            prediction_masks_count=prediction_masks_count,
            seg_token_count=seg_token_count,
            status=status,
        )

    def _compute_iou(self, gt_mask, pred_mask):
        if pred_mask is None:
            return 0.0
        gt_mask = self._to_numpy_mask(gt_mask)
        pred_mask = self._to_numpy_mask(pred_mask)
        if gt_mask.shape != pred_mask.shape:
            pred_mask_t = torch.from_numpy(pred_mask[None, None].astype(np.float32))
            pred_mask_t = F.interpolate(pred_mask_t, size=gt_mask.shape, mode="nearest")[0, 0]
            pred_mask = (pred_mask_t.numpy() > 0).astype(np.uint8)
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    def _prepare_pred_mask_for_iou(self, gt_mask, pred_mask):
        gt_mask = self._to_numpy_mask(gt_mask)
        pred_mask = self._to_numpy_mask(pred_mask)
        resized = False
        pred_shape_before_resize = tuple(pred_mask.shape)
        if gt_mask.shape != pred_mask.shape:
            pred_mask_t = torch.from_numpy(pred_mask[None, None].astype(np.float32))
            pred_mask_t = F.interpolate(pred_mask_t, size=gt_mask.shape, mode="nearest")[0, 0]
            pred_mask = (pred_mask_t.numpy() > 0).astype(np.uint8)
            resized = True
        return pred_mask, resized, pred_shape_before_resize, tuple(pred_mask.shape)

    @staticmethod
    def _mask_summary(mask):
        if mask is None:
            return "empty mask"
        mask = np.asarray(mask)
        ys, xs = np.where(mask > 0)
        h, w = mask.shape
        area = int(mask.sum())
        area_ratio = float(area) / float(max(h * w, 1))
        if len(xs) == 0 or len(ys) == 0:
            return f"empty mask, area_ratio={area_ratio:.4f}"
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        center = [round(float(xs.mean()), 2), round(float(ys.mean()), 2)]
        return f"area_ratio={area_ratio:.4f}, bbox={bbox}, center={center}"

    def build_teacher_privileged_prompt_v3(
        self,
        *,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        generation_mode="trajectory_guidance",
    ):
        clean_question = self._strip_image_placeholder(student_question)
        relation_context = build_mask_relation_context(
            model=self,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
        )
        seg_correct = bool(teacher_fields.get("caption_to_mask_seg_correct", False))
        route = teacher_fields.get("teacher_route", ON_POLICY_DISTILL_ROUTE)
        if route == TEACHER_REGENERATE_ROUTE:
            route_guidance = (
                f"The current IoU is below {self.iou_low_threshold:.2f}, so the student's caption leads the reconstruction too far away from the gtmask. "
                "The teacher must diagnose why the current caption causes the refmask to differ from the gtmask, identify which visual descriptions are wrong, missing, too generic, or overemphasized, and then regenerate a better caption for the gtmask."
            )
        elif route == ON_POLICY_DISTILL_ROUTE:
            route_guidance = (
                f"The current IoU is between {self.iou_low_threshold:.2f} and {self.iou_high_threshold:.2f}, so this sample enters the on-policy correction branch. "
                "The teacher must first compare the two masks in detail, identify what content is shared, what target evidence is missing from the refmask, and what distractor evidence is wrongly included in the refmask. "
                "Then the teacher must analyze the student caption and infer which words, attributes, parts, or local relations likely caused those errors. "
                "Use that diagnosis to supervise the student's token trajectory: increase probability on tokens that better explain the gtmask and suppress tokens that explain refmask-only distractor regions."
            )
        else:
            route_guidance = (
                f"The current IoU is at least {self.iou_high_threshold:.2f}, so the reconstruction already matches the gtmask well. "
                "Large corrections are likely harmful; keep any remaining guidance minimal."
            )
        prompt_intro = (
            "<image>\n"
            "You are a teacher supervising a mask-to-caption task. The goal is to optimize a model that takes a target mask as input and generates a caption as output. "
            "The evaluation criterion is whether the generated caption is precise and clear enough to reconstruct the original ground-truth mask. "
            "You are given privileged access to the target mask (region1 = gtmask) and the reconstructed mask (region2 = refmask). "
            "Your job is to analyze, at pixel and region level, why the current student caption produces the current reconstructed mask, and then provide the correct supervision for this route.\n"
        )
        prompt = prompt_intro + (
            f"Teacher route: {route}\n"
            f"Student prompt: {clean_question}\n"
            f"Student caption: {student_caption}\n"
            f"Description status: {description_status}\n"
            f"Reconstruction status: {reconstruction.status}\n"
            f"Reconstruction question: {reconstruction.question or ''}\n"
            f"caption_to_mask_seg_correct: {'true' if seg_correct else 'false'}\n"
            "IoU is the intersection-over-union between gtmask and refmask: intersection / union.\n"
            f"Current IoU between gtmask(region1) and refmask(region2): {iou:.4f}\n"
            f"Shared overlap summary between region1 and region2: {relation_context['overlap_summary']}\n"
            f"Unique non-overlap area in gtmask (region1-only pixels, missing from refmask): {relation_context['gt_only_summary']}\n"
            f"Unique non-overlap area in refmask (region2-only pixels, erroneous distractor pixels): {relation_context['ref_only_summary']}\n"
            "Required reasoning order:\n"
            "1. Analyze region1 (gtmask) carefully and summarize what object content it truly contains.\n"
            "2. Analyze region2 (refmask) carefully and summarize what object content it currently captures.\n"
            "3. Compare the two masks pixel by pixel and region by region: identify what target evidence is missing from region2 and what extra distractor evidence appears only in region2.\n"
            "4. Read the student caption and explain why that wording leads the model toward region2 instead of region1. Identify which descriptions are wrong, missing, too vague, misleading, or overemphasized.\n"
            "5. Use that diagnosis to decide the correct supervision for this route.\n"
            f"{route_guidance}\n"
            "Do not rely on any pre-labeled failure category beyond the route. Base your supervision on the actual visual content of region1 and region2, their pixel-level differences, and the failure mode implied by the student caption."
        )
        if generation_mode == "teacher_problem_identification":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"Current IoU between region1 and region2: {iou:.4f}\n"
                f"Target summary: {teacher_fields.get('target_summary', '')}\n"
                f"Distractor summary: {teacher_fields.get('distractor_summary', '')}\n"
                f"Shared evidence: {teacher_fields.get('shared_evidence', '')}\n"
                f"Target-only evidence: {teacher_fields.get('target_only_evidence', '')}\n"
                f"Distractor-only evidence: {teacher_fields.get('distractor_only_evidence', '')}\n"
                f"Target localization hint: {teacher_fields.get('target_localization_hint', '')}\n"
                f"Distractor localization hint: {teacher_fields.get('distractor_localization_hint', '')}\n"
                "Identify only the concrete caption problem. Do not generate a fix or a reason.\n"
                "Output exactly one natural-language sentence and nothing else.\n"
                "Rules:\n"
                "- The sentence must say which target-specific feature, overlap-vs-only difference, or local distinction the failed caption does not express.\n"
                "- Use the target summary, distractor summary, and shared evidence to ground the statement in visible content.\n"
                "- Use target-only evidence and distractor-only evidence to point to the specific missing distinction.\n"
                "- The sentence must not give a fix and must not explain why.\n"
                "- The sentence must not stop at broad area, left, right, top, or bottom alone.\n"
                "- Do not output bullets, markdown, extra labels, analysis preambles, or [SEG]."
            )
        elif generation_mode == "teacher_correction_direction":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"CAPTION_PROBLEM: {teacher_fields.get('caption_problem', '')}\n"
                f"Target summary: {teacher_fields.get('target_summary', '')}\n"
                f"Distractor summary: {teacher_fields.get('distractor_summary', '')}\n"
                f"Shared evidence: {teacher_fields.get('shared_evidence', '')}\n"
                f"Target-only evidence: {teacher_fields.get('target_only_evidence', '')}\n"
                f"Distractor-only evidence: {teacher_fields.get('distractor_only_evidence', '')}\n"
                f"Difference focus: {teacher_fields.get('difference_focus', '')}\n"
                "Write only an edit instruction for the caption. Do not restate the problem and do not explain why.\n"
                "Output exactly one natural-language sentence and nothing else.\n"
                "Rules:\n"
                "- Do not restate the target summary or distractor summary.\n"
                "- Do not describe the object again from scratch.\n"
                "- The sentence must say what target-side detail should be added, strengthened, specified, or made explicit.\n"
                "- If distractor-only evidence exists, the sentence must also say what distractor-compatible wording should be avoided, suppressed, separated, or removed.\n"
                "- When both target-only and distractor-only differences exist, the sentence must contain one add action and one avoid action.\n"
                "- Use the difference focus as the main edit target.\n"
                "- Prefer sentence shapes like: Add the target-side detail about ..., and avoid wording that still fits ... .\n"
                "- Do not output bullets, markdown, extra labels, analysis preambles, or [SEG]."
            )
        elif generation_mode == "teacher_reason_explanation":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"CAPTION_PROBLEM: {teacher_fields.get('caption_problem', '')}\n"
                f"CORRECTION_DIRECTION: {teacher_fields.get('correction_direction', '')}\n"
                f"Target summary: {teacher_fields.get('target_summary', '')}\n"
                f"Distractor summary: {teacher_fields.get('distractor_summary', '')}\n"
                f"Shared evidence: {teacher_fields.get('shared_evidence', '')}\n"
                f"Target-only evidence: {teacher_fields.get('target_only_evidence', '')}\n"
                f"Distractor-only evidence: {teacher_fields.get('distractor_only_evidence', '')}\n"
                f"Target localization hint: {teacher_fields.get('target_localization_hint', '')}\n"
                f"Distractor localization hint: {teacher_fields.get('distractor_localization_hint', '')}\n"
                f"Likely drift reason: {teacher_fields.get('likely_drift_reason', '')}\n"
                "Explain only why the failed caption drifts and why the correction direction is needed.\n"
                "Output exactly one natural-language sentence and nothing else.\n"
                "Rules:\n"
                "- The sentence must explain why the failed caption misses target-only evidence relative to the shared overlap.\n"
                "- When distractor-only evidence exists, the sentence must also explain why the failed caption still fits that distractor-side evidence.\n"
                "- The sentence must use finer-grained visible differences and must not stop at broad area, left, right, top, or bottom alone.\n"
                "- Prefer the sentence shape: The caption misses ... so it does not isolate the target; it still fits ... which pulls reconstruction toward the distractor.\n"
                "- Do not output bullets, markdown, extra labels, analysis preambles, or [SEG]."
            )
        elif generation_mode == "teacher_structured_diagnosis":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"Current IoU between region1 and region2: {iou:.4f}\n"
                f"Target summary: {teacher_fields.get('target_summary', '')}\n"
                f"Distractor summary: {teacher_fields.get('distractor_summary', '')}\n"
                f"Shared evidence: {teacher_fields.get('shared_evidence', '')}\n"
                f"Target-only evidence: {teacher_fields.get('target_only_evidence', '')}\n"
                f"Distractor-only evidence: {teacher_fields.get('distractor_only_evidence', '')}\n"
                f"Difference focus: {teacher_fields.get('difference_focus', '')}\n"
                f"Likely drift reason: {teacher_fields.get('likely_drift_reason', '')}\n"
                "Diagnose the failure in a structured way before proposing a corrected caption.\n"
                "Output exactly these 6 lines in LABEL: value format:\n"
                "CAPTION_PROBLEM:\n"
                "CORRECTION_DIRECTION:\n"
                "REASON:\n"
                "CONFIDENCE:\n"
                "TARGET_ANCHOR:\n"
                "DISTRACTOR_ANCHOR:\n"
                "Rules:\n"
                "- CAPTION_PROBLEM must identify the concrete missing distinction or misleading phrasing in the failed caption, not the fix.\n"
                "- CORRECTION_DIRECTION must state how to revise the caption, including an add action and, when distractor-only evidence exists, an avoid action.\n"
                "- REASON must explain why the failed caption still matches region2 or misses region1.\n"
                "- CONFIDENCE must be exactly high, medium, or low.\n"
                "- TARGET_ANCHOR must name the strongest target-side distinguishing cue.\n"
                "- DISTRACTOR_ANCHOR must name the strongest distractor-side misleading cue, or none.\n"
                "- Do not output a DLC, JSON, bullets, markdown, or any labels beyond the required six."
            )
            retry_reason = self._normalize_teacher_field_text(teacher_fields.get("diagnosis_retry_reason", ""))
            if retry_reason:
                prompt += (
                    f"\nPrevious diagnosis failed local validation because: {retry_reason}.\n"
                    "Repair the six fields directly instead of repeating the same generic wording."
                )
        elif generation_mode == "teacher_dlc_from_diagnosis":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"Target summary: {teacher_fields.get('target_summary', '')}\n"
                f"Distractor summary: {teacher_fields.get('distractor_summary', '')}\n"
                f"Target-only evidence: {teacher_fields.get('target_only_evidence', '')}\n"
                f"Distractor-only evidence: {teacher_fields.get('distractor_only_evidence', '')}\n"
                f"Target localization hint: {teacher_fields.get('target_localization_hint', '')}\n"
                f"CAPTION_PROBLEM: {teacher_fields.get('caption_problem', '')}\n"
                f"CORRECTION_DIRECTION: {teacher_fields.get('correction_direction', '')}\n"
                f"REASON: {teacher_fields.get('reason', '')}\n"
                "Write one corrected detailed localized caption for the target mask.\n"
                "Output exactly one line:\n"
                "DLC: <one natural and complete detailed localized caption>\n"
                "Rules:\n"
                "- The DLC must target the target summary, not the distractor summary.\n"
                "- It must absorb the target-only evidence when that evidence is available.\n"
                "- It must avoid turning distractor-only evidence into the main anchor.\n"
                "- It must stay natural and specific, and must not output [SEG], bullets, or extra labels."
            )
        elif generation_mode == "teacher_dlc_candidate":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"Target summary: {teacher_fields.get('target_summary', '')}\n"
                f"Distractor summary: {teacher_fields.get('distractor_summary', '')}\n"
                f"Target-only evidence: {teacher_fields.get('target_only_evidence', '')}\n"
                f"Distractor-only evidence: {teacher_fields.get('distractor_only_evidence', '')}\n"
                f"Difference focus: {teacher_fields.get('difference_focus', '')}\n"
                f"CAPTION_PROBLEM: {teacher_fields.get('caption_problem', '')}\n"
                f"CORRECTION_DIRECTION: {teacher_fields.get('correction_direction', '')}\n"
                f"REASON: {teacher_fields.get('reason', '')}\n"
                f"TARGET_ANCHOR: {teacher_fields.get('target_anchor', '')}\n"
                f"DISTRACTOR_ANCHOR: {teacher_fields.get('distractor_anchor', '')}\n"
                "Write one corrected detailed localized caption for region1.\n"
                "Output exactly one line:\n"
                "DLC: <one natural and complete detailed localized caption>\n"
                "Rules:\n"
                "- The DLC must directly describe the target object or region, not 'the target' or 'the region'.\n"
                "- Keep at least one target-only cue explicit.\n"
                "- Avoid distractor-compatible anchor wording when distractor-only evidence exists.\n"
                "- Prefer a clean DLC-bench style caption instead of a templated explanation.\n"
                "- Do not output analysis, lists, markdown, [SEG], or extra labels."
            )
            candidate_style = self._normalize_teacher_field_text(teacher_fields.get("candidate_style_hint", ""))
            if candidate_style:
                prompt += f"\nStyle hint for this candidate: {candidate_style}\n"
        elif generation_mode == "teacher_dlc_repair":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"Target summary: {teacher_fields.get('target_summary', '')}\n"
                f"Distractor summary: {teacher_fields.get('distractor_summary', '')}\n"
                f"Target-only evidence: {teacher_fields.get('target_only_evidence', '')}\n"
                f"Distractor-only evidence: {teacher_fields.get('distractor_only_evidence', '')}\n"
                f"CAPTION_PROBLEM: {teacher_fields.get('caption_problem', '')}\n"
                f"CORRECTION_DIRECTION: {teacher_fields.get('correction_direction', '')}\n"
                f"REASON: {teacher_fields.get('reason', '')}\n"
                f"TARGET_ANCHOR: {teacher_fields.get('target_anchor', '')}\n"
                f"DISTRACTOR_ANCHOR: {teacher_fields.get('distractor_anchor', '')}\n"
                f"Previous DLC failure code: {teacher_fields.get('dlc_failure_code', '')}\n"
                f"Rejected candidate summary: {teacher_fields.get('rejected_dlc_summary', '')}\n"
                "Repair the failed DLC generation and write one better caption for region1.\n"
                "Output exactly one line:\n"
                "DLC: <one natural and complete detailed localized caption>\n"
                "Rules:\n"
                "- Fix the reported failure instead of repeating the same template.\n"
                "- Preserve target-only distinguishing detail.\n"
                "- Avoid sentence openings like 'the target', 'the region', or 'region1'.\n"
                "- Do not output explanations, bullets, markdown, [SEG], or extra labels."
            )
        elif generation_mode == "teacher_verification_caption_from_diagnosis":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"DLC: {teacher_fields.get('detailed_caption', '')}\n"
                f"Target-only evidence: {teacher_fields.get('target_only_evidence', '')}\n"
                f"Distractor-only evidence: {teacher_fields.get('distractor_only_evidence', '')}\n"
                f"Target localization hint: {teacher_fields.get('target_localization_hint', '')}\n"
                f"CORRECTION_DIRECTION: {teacher_fields.get('correction_direction', '')}\n"
                "Write a shorter verification caption for reconstruction gating.\n"
                "Output exactly one line:\n"
                "VERIFICATION_CAPTION: <one shorter verifier-friendly caption>\n"
                "Rules:\n"
                "- It must be shorter than the DLC.\n"
                "- It must keep at least one target-only difference detail when target-only evidence exists.\n"
                "- It must not repeat the distractor-only evidence as the main anchor.\n"
                "- It must not collapse into a generic category phrase and must not output [SEG]."
            )
        if generation_mode == "teacher_fault_report":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"Description status: {description_status}\n"
                f"Reconstruction status: {reconstruction.status}\n"
                f"Current IoU between region1 and region2: {iou:.4f}\n"
                f"Target mask summary: {relation_context['gt_summary']}\n"
                f"Reconstructed mask summary: {relation_context['ref_summary']}\n"
                f"Shared overlap summary: {relation_context['overlap_summary']}\n"
                f"Region1-only summary (missing target pixels): {relation_context['gt_only_summary']}\n"
                f"Region2-only summary (distractor pixels wrongly predicted): {relation_context['ref_only_summary']}\n"
                f"{route_guidance}\n"
                "You must diagnose the failure before any rewrite. Do not generate a caption.\n"
                "Output exactly these 11 lines. Use one line per field. Never leave a field blank. If unsure, use unknown or none.\n"
                "PRIMARY_FAILURE_TYPE:\n"
                "SECONDARY_FAILURE_TYPES:\n"
                "FAILURE_CONFIDENCE:\n"
                "TARGET_SUMMARY:\n"
                "REF_SUMMARY:\n"
                "MISSING_EVIDENCE:\n"
                "DISTRACTOR_EVIDENCE:\n"
                "BAD_PHRASES_IN_STUDENT:\n"
                "MISSING_PHRASES_NEEDED:\n"
                "KEEPABLE_PHRASES:\n"
                "EVIDENCE_FOR_FAILURE:\n"
                "Rules:\n"
                "- PRIMARY_FAILURE_TYPE must be exactly one of: too_generic, wrong_attribute, wrong_part_focus, wrong_spatial_anchor, distractor_leak, scene_spill, mixed_target, underspecified_local_detail, unknown.\n"
                "- SECONDARY_FAILURE_TYPES must contain zero to three labels from the same set, comma separated, or none.\n"
                "- FAILURE_CONFIDENCE must be exactly high, medium, or low.\n"
                "- TARGET_SUMMARY and REF_SUMMARY must describe only visible content.\n"
                "- MISSING_EVIDENCE must state what gtmask contains but refmask misses.\n"
                "- DISTRACTOR_EVIDENCE must state what refmask wrongly includes.\n"
                "- BAD_PHRASES_IN_STUDENT must name the student phrases most likely causing the drift, comma separated, or unknown.\n"
                "- MISSING_PHRASES_NEEDED must name the missing caption phrases needed to recover gtmask, comma separated, or none.\n"
                "- KEEPABLE_PHRASES must name student phrases that are still correct, comma separated, or none.\n"
                "- EVIDENCE_FOR_FAILURE must explain why the student caption leads to region2 instead of region1 and must reference the missing and distractor evidence.\n"
                "- Write every field on its own single line in LABEL: value format.\n"
                "- Do not output a caption, explanation preamble, bullets, [SEG], markdown, or any labels beyond the required field names."
            )
        elif generation_mode == "teacher_repair_plan":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"Current IoU between region1 and region2: {iou:.4f}\n"
                f"PRIMARY_FAILURE_TYPE: {teacher_fields.get('primary_failure_type', '')}\n"
                f"SECONDARY_FAILURE_TYPES: {teacher_fields.get('secondary_failure_types', '')}\n"
                f"FAILURE_CONFIDENCE: {teacher_fields.get('failure_confidence', '')}\n"
                f"TARGET_SUMMARY: {teacher_fields.get('target_summary', '')}\n"
                f"REF_SUMMARY: {teacher_fields.get('ref_summary', '')}\n"
                f"MISSING_EVIDENCE: {teacher_fields.get('missing_evidence', '')}\n"
                f"DISTRACTOR_EVIDENCE: {teacher_fields.get('distractor_evidence', '')}\n"
                f"BAD_PHRASES_IN_STUDENT: {teacher_fields.get('bad_phrases_in_student', '')}\n"
                f"MISSING_PHRASES_NEEDED: {teacher_fields.get('missing_phrases_needed', '')}\n"
                f"KEEPABLE_PHRASES: {teacher_fields.get('keepable_phrases', '')}\n"
                f"EVIDENCE_FOR_FAILURE: {teacher_fields.get('evidence_for_failure', '')}\n"
                "Generate a minimal rewrite plan. Do not generate the final caption yet.\n"
                "Output exactly the following fields in this exact order:\n"
                "REMOVE_PHRASES:\n"
                "ADD_PHRASES:\n"
                "EMPHASIZE_PHRASES:\n"
                "DEEMPHASIZE_PHRASES:\n"
                "LOCALIZE_WITH:\n"
                "AVOID_PHRASES:\n"
                "REWRITE_STRATEGY:\n"
                "Rules:\n"
                "- REMOVE_PHRASES must name misleading phrases to exclude from the new DLC, comma separated, or none.\n"
                "- ADD_PHRASES must name missing target evidence phrases to add, sourced from the diagnosis only.\n"
                "- EMPHASIZE_PHRASES must name the most target-specific visible cues to foreground.\n"
                "- DEEMPHASIZE_PHRASES must name descriptions that may remain but should not anchor the caption.\n"
                "- LOCALIZE_WITH must be one sentence describing the minimum localization strategy.\n"
                "- AVOID_PHRASES must include generic or distractor-prone phrases that would likely recreate the failure.\n"
                "- REWRITE_STRATEGY must be one sentence describing how to fix the caption without merely making it longer.\n"
                "- Do not output bullets, explanations, a caption, [SEG], or any labels beyond the required field names."
            )
        elif generation_mode == "teacher_dlc":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"PRIMARY_FAILURE_TYPE: {teacher_fields.get('primary_failure_type', '')}\n"
                f"SECONDARY_FAILURE_TYPES: {teacher_fields.get('secondary_failure_types', '')}\n"
                f"MISSING_EVIDENCE: {teacher_fields.get('missing_evidence', '')}\n"
                f"DISTRACTOR_EVIDENCE: {teacher_fields.get('distractor_evidence', '')}\n"
                f"BAD_PHRASES_IN_STUDENT: {teacher_fields.get('bad_phrases_in_student', '')}\n"
                f"MISSING_PHRASES_NEEDED: {teacher_fields.get('missing_phrases_needed', '')}\n"
                f"KEEPABLE_PHRASES: {teacher_fields.get('keepable_phrases', '')}\n"
                f"REMOVE_PHRASES: {teacher_fields.get('remove_phrases', '')}\n"
                f"ADD_PHRASES: {teacher_fields.get('add_phrases', '')}\n"
                f"EMPHASIZE_PHRASES: {teacher_fields.get('emphasize_phrases', '')}\n"
                f"DEEMPHASIZE_PHRASES: {teacher_fields.get('deemphasize_phrases', '')}\n"
                f"LOCALIZE_WITH: {teacher_fields.get('localize_with', '')}\n"
                f"AVOID_PHRASES: {teacher_fields.get('avoid_phrases', '')}\n"
                f"REWRITE_STRATEGY: {teacher_fields.get('rewrite_strategy', '')}\n"
                "Write the corrected detailed localized caption now.\n"
                "Output exactly one line:\n"
                "DLC: <one natural and complete detailed localized caption>\n"
                "Rules:\n"
                "- The DLC must be one complete natural sentence.\n"
                "- It must include the critical ADD_PHRASES evidence when provided.\n"
                "- It must not include REMOVE_PHRASES or AVOID_PHRASES.\n"
                "- It may retain KEEPABLE_PHRASES when they remain correct.\n"
                "- It must follow LOCALIZE_WITH and REWRITE_STRATEGY.\n"
                "- It must not invent attributes unsupported by the diagnosis.\n"
                "- Do not output explanations, extra labels, bullets, or [SEG]."
            )
        elif generation_mode == "teacher_verification_caption":
            prompt = prompt_intro + (
                f"Student prompt: {clean_question}\n"
                f"DLC: {teacher_fields.get('detailed_caption', '')}\n"
                f"PRIMARY_FAILURE_TYPE: {teacher_fields.get('primary_failure_type', '')}\n"
                f"MISSING_EVIDENCE: {teacher_fields.get('missing_evidence', '')}\n"
                f"EMPHASIZE_PHRASES: {teacher_fields.get('emphasize_phrases', '')}\n"
                f"ADD_PHRASES: {teacher_fields.get('add_phrases', '')}\n"
                f"LOCALIZE_WITH: {teacher_fields.get('localize_with', '')}\n"
                f"AVOID_PHRASES: {teacher_fields.get('avoid_phrases', '')}\n"
                "Write a shorter verifier-friendly caption derived from the DLC and gtmask.\n"
                "Output exactly one line:\n"
                "VERIFICATION_CAPTION: <one shorter verifier-friendly caption>\n"
                "Rules:\n"
                "- The verification caption must be shorter than the DLC.\n"
                "- It must preserve at least one target-specific distinguishing trait.\n"
                "- It must not collapse into a generic phrase like the man, the person, or the object.\n"
                "- It must not introduce any attribute not already supported by the DLC and diagnosis.\n"
                "- It must not include AVOID_PHRASES, explanations, extra labels, bullets, or [SEG]."
            )
        return prompt

    def build_teacher_regenerate_privileged_context_prompt(
        self,
        *,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        teacher_fields,
    ):
        clean_question = self._strip_image_placeholder(student_question)
        return (
            "<image>\n"
            "You are a teacher supervising a mask-to-caption task with privileged access to region1 (gtmask) and "
            "region2 (the current reconstructed refmask). Your job is to compare the two masks carefully, infer why "
            "the student caption drifts, and then write a better target caption plus a shorter verification caption.\n"
            f"Student prompt: {clean_question}\n"
            f"Failed student caption: {student_caption}\n"
            f"Description status: {description_status}\n"
            f"Reconstruction status: {reconstruction.status}\n"
            f"Reconstruction question: {reconstruction.question or ''}\n"
            f"Current IoU between region1 and region2: {iou:.4f}\n"
            f"Target summary: {teacher_fields.get('target_summary', '')}\n"
            f"Distractor summary: {teacher_fields.get('distractor_summary', '')}\n"
            f"Shared evidence: {teacher_fields.get('shared_evidence', '')}\n"
            f"Target-only evidence: {teacher_fields.get('target_only_evidence', '')}\n"
            f"Distractor-only evidence: {teacher_fields.get('distractor_only_evidence', '')}\n"
            f"Difference focus: {teacher_fields.get('difference_focus', '')}\n"
            f"Likely drift reason: {teacher_fields.get('likely_drift_reason', '')}\n"
            f"Target localization hint: {teacher_fields.get('target_localization_hint', '')}\n"
            f"Distractor localization hint: {teacher_fields.get('distractor_localization_hint', '')}\n"
        )

    def build_teacher_regenerate_single_prompt(
        self,
        *,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        teacher_fields,
    ):
        context_prompt = self.build_teacher_regenerate_privileged_context_prompt(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            teacher_fields=teacher_fields,
        )
        return context_prompt + (
            "Follow this workflow internally before writing your answer:\n"
            "1. Understand region1 / gtmask precisely.\n"
            "2. Understand region2 / refmask precisely.\n"
            "3. Compare the shared evidence, the target-only missing evidence, and the distractor-only extra evidence.\n"
            "4. Infer why the student caption drifts toward region2 instead of isolating region1.\n"
            "5. Write one corrected detailed localized caption for region1.\n"
            "Write only one corrected detailed localized caption for region1.\n"
            "Your answer must start immediately with 'DLC:' on the first line.\n"
            "Do not write any preface such as 'Sure', 'The task is', 'The answer is', HTML tags, numbering, or quoted restatements of the prompt.\n"
            "If you output anything before 'DLC:', the answer is invalid.\n"
            "Output format:\n"
            "DLC: <one natural and complete detailed localized caption>\n"
            "Do not output bullets, markdown, analysis preambles, or [SEG]."
        )

    @staticmethod
    def generalized_jsd_token_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0):
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([
                    student_log_probs + torch.log1p(-beta_t),
                    teacher_log_probs + torch.log(beta_t),
                ]),
                dim=0,
            )
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student
        return jsd.sum(dim=-1), teacher_entropy

    @staticmethod
    def _sequence_cross_entropy_from_logits(logits, completion_ids):
        targets = completion_ids.reshape(-1).to(logits.device)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets, reduction="mean")

    @staticmethod
    def _token_log_probs_from_logits(logits, completion_ids):
        targets = completion_ids.to(logits.device)
        token_log_probs = F.log_softmax(logits, dim=-1).gather(dim=-1, index=targets.unsqueeze(-1))
        return token_log_probs.squeeze(-1)

    @staticmethod
    def _materialize_autograd_input(value):
        if not isinstance(value, torch.Tensor):
            return value
        # Inference tensors are incompatible with parts of autograd bookkeeping,
        # so convert them to regular tensors before mixing them into training loss.
        return value.clone() if value.is_inference() else value

    def _route_from_iou(self, iou):
        return classify_teacher_route(
            iou=iou,
            low_threshold=self.iou_low_threshold,
            high_threshold=self.iou_high_threshold,
        )

    @staticmethod
    def _build_training_teacher_fields(*, route, iou):
        return {
            "teacher_route": route,
            "caption_to_mask_seg_correct": bool(float(iou) >= 0.5),
        }

    def generate_teacher_structured_diagnosis(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        pipeline_result,
        retry_reason="",
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "target_summary": pipeline_result.target_summary,
                "distractor_summary": pipeline_result.distractor_summary,
                "shared_evidence": pipeline_result.shared_evidence,
                "target_only_evidence": pipeline_result.target_only_evidence,
                "distractor_only_evidence": pipeline_result.distractor_only_evidence,
                "difference_focus": pipeline_result.difference_focus,
                "target_localization_hint": pipeline_result.target_localization_hint,
                "distractor_localization_hint": pipeline_result.distractor_localization_hint,
                "likely_drift_reason": pipeline_result.likely_drift_reason,
                "diagnosis_retry_reason": retry_reason,
            }
        )
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_structured_diagnosis",
        )
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        pipeline_result = self._parse_teacher_structured_diagnosis(raw_prediction, pipeline_result)
        pipeline_result.problem_valid, pipeline_result.problem_failure_reason = self.validate_teacher_problem_identification(
            pipeline_result
        )
        pipeline_result.direction_valid, pipeline_result.direction_failure_reason = self.validate_teacher_correction_direction(
            pipeline_result
        )
        pipeline_result.reason_valid, pipeline_result.reason_failure_reason = self.validate_teacher_reason_explanation(
            pipeline_result
        )
        pipeline_result.diagnosis_valid, pipeline_result.diagnosis_failure_reason = self.validate_teacher_structured_diagnosis(
            pipeline_result
        )
        return pipeline_result

    def _build_teacher_dlc_candidate_record(
        self,
        *,
        image,
        gt_mask,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask_for_prompt,
        ref_mask,
        teacher_fields,
        pipeline_result,
        candidate_style_hint="",
        generation_kwargs=None,
    ):
        del gt_mask
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "target_summary": pipeline_result.target_summary,
                "distractor_summary": pipeline_result.distractor_summary,
                "target_only_evidence": pipeline_result.target_only_evidence,
                "distractor_only_evidence": pipeline_result.distractor_only_evidence,
                "difference_focus": pipeline_result.difference_focus,
                "caption_problem": pipeline_result.caption_problem,
                "correction_direction": pipeline_result.correction_direction,
                "reason": pipeline_result.reason,
                "target_anchor": pipeline_result.target_anchor,
                "distractor_anchor": pipeline_result.distractor_anchor,
                "candidate_style_hint": candidate_style_hint,
            }
        )
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask_for_prompt,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_dlc_candidate",
        )
        generation_kwargs = dict(generation_kwargs or {})
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
            **generation_kwargs,
        )
        detailed_caption_raw = self._extract_labeled_teacher_text(raw_prediction, "DLC")
        if not detailed_caption_raw:
            detailed_caption_raw = raw_prediction
        caption = self._clean_teacher_dlc_caption_text(detailed_caption_raw)
        candidate_result = self._build_empty_teacher_regenerate_pipeline_result()
        candidate_result.target_summary = pipeline_result.target_summary
        candidate_result.distractor_summary = pipeline_result.distractor_summary
        candidate_result.target_only_evidence = pipeline_result.target_only_evidence
        candidate_result.distractor_only_evidence = pipeline_result.distractor_only_evidence
        candidate_result = self._materialize_teacher_caption_result(candidate_result, caption, "structured_candidate")
        failure_reason = self._validate_teacher_dlc(candidate_result)
        reconstruct = self.reconstruct_mask(
            image=image,
            caption=candidate_result.detailed_caption,
            description_status=candidate_result.detailed_status,
            gt_mask=gt_mask_for_prompt,
        )
        pred_mask = None if reconstruct is None else reconstruct.pred_mask
        reconstruct_iou = self._compute_iou(gt_mask_for_prompt, pred_mask) if pred_mask is not None else 0.0
        candidate = {
            "raw_prediction": raw_prediction,
            "caption": candidate_result.detailed_caption,
            "status": candidate_result.detailed_status,
            "failure_reason": failure_reason,
            "completion_ids": candidate_result.detailed_completion_ids,
            "reconstruct_status": None if reconstruct is None else reconstruct.status,
            "reconstruct_iou": float(reconstruct_iou),
            "student_iou": float(iou),
            "target_only_evidence": pipeline_result.target_only_evidence,
            "distractor_only_evidence": pipeline_result.distractor_only_evidence,
            "style_hint": candidate_style_hint,
        }
        candidate["score"] = self._score_teacher_dlc_candidate(candidate)
        return candidate

    def _apply_teacher_dlc_candidate(self, pipeline_result, candidate, source):
        pipeline_result.dlc_raw = candidate.get("raw_prediction", "")
        pipeline_result = self._materialize_teacher_caption_result(
            pipeline_result,
            candidate.get("caption", ""),
            source,
        )
        pipeline_result.detailed_failure_reason = candidate.get("failure_reason", "")
        return pipeline_result

    def _evaluate_teacher_verification_branch(
        self,
        *,
        image,
        gt_mask,
        ref_mask,
        student_question,
        description_status,
        reconstruction,
        iou,
        teacher_prompt_masks,
        teacher_fields,
        pipeline_result,
    ):
        pipeline_result = self.generate_teacher_verification_caption(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            student_question=student_question,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            pipeline_result=pipeline_result,
        )
        verification_iou = 0.0
        if pipeline_result.verification_status == "ok" and not pipeline_result.verification_failure_reason:
            verification_reconstruction = self.reconstruct_mask(
                image=image,
                caption=pipeline_result.verification_caption,
                description_status=pipeline_result.verification_status,
                gt_mask=gt_mask,
            )
            verification_pred_mask = None if verification_reconstruction is None else verification_reconstruction.pred_mask
            if verification_pred_mask is not None:
                verification_iou = self._compute_iou(gt_mask, verification_pred_mask)
        pipeline_result.verification_iou = float(verification_iou)
        return pipeline_result

    def _generate_teacher_repair_caption(
        self,
        *,
        image,
        gt_mask,
        ref_mask,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        teacher_prompt_masks,
        teacher_fields,
        pipeline_result,
        repair_reason,
        rejected_dlc_summary,
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "target_summary": pipeline_result.target_summary,
                "distractor_summary": pipeline_result.distractor_summary,
                "target_only_evidence": pipeline_result.target_only_evidence,
                "distractor_only_evidence": pipeline_result.distractor_only_evidence,
                "caption_problem": pipeline_result.caption_problem,
                "correction_direction": pipeline_result.correction_direction,
                "reason": pipeline_result.reason,
                "target_anchor": pipeline_result.target_anchor,
                "distractor_anchor": pipeline_result.distractor_anchor,
                "dlc_failure_code": repair_reason,
                "rejected_dlc_summary": rejected_dlc_summary,
            }
        )
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_dlc_repair",
        )
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        detailed_caption_raw = self._extract_labeled_teacher_text(raw_prediction, "DLC")
        if not detailed_caption_raw:
            detailed_caption_raw = raw_prediction
        pipeline_result.repair_raw = raw_prediction
        pipeline_result = self._materialize_teacher_caption_result(
            pipeline_result,
            detailed_caption_raw,
            "repair_dlc",
        )
        pipeline_result.detailed_failure_reason = self._validate_teacher_dlc(pipeline_result)
        return pipeline_result

    def generate_teacher_fault_report(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        relation_context,
    ):
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_fault_report",
        )
        teacher_fields["teacher_fault_report_prompt"] = prompt
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        result = self._parse_teacher_fault_report(raw_prediction)
        return self._backfill_teacher_fault_report(
            result,
            relation_context=relation_context,
            student_caption=student_caption,
        )

    def generate_teacher_repair_plan(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        pipeline_result,
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "primary_failure_type": pipeline_result.primary_failure_type,
                "secondary_failure_types": ", ".join(pipeline_result.secondary_failure_types) or "none",
                "failure_confidence": pipeline_result.failure_confidence,
                "target_summary": pipeline_result.target_summary,
                "ref_summary": pipeline_result.ref_summary,
                "missing_evidence": pipeline_result.missing_evidence,
                "distractor_evidence": pipeline_result.distractor_evidence,
                "bad_phrases_in_student": pipeline_result.bad_phrases_in_student,
                "missing_phrases_needed": pipeline_result.missing_phrases_needed,
                "keepable_phrases": pipeline_result.keepable_phrases,
                "evidence_for_failure": pipeline_result.evidence_for_failure,
            }
        )
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_repair_plan",
        )
        teacher_fields["teacher_repair_plan_prompt"] = prompt
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        return self._parse_teacher_repair_plan(raw_prediction, pipeline_result)

    def generate_teacher_problem_identification(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        pipeline_result,
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "target_summary": pipeline_result.target_summary,
                "distractor_summary": pipeline_result.distractor_summary,
                "shared_evidence": pipeline_result.shared_evidence,
                "target_only_evidence": pipeline_result.target_only_evidence,
                "distractor_only_evidence": pipeline_result.distractor_only_evidence,
                "target_localization_hint": pipeline_result.target_localization_hint,
                "distractor_localization_hint": pipeline_result.distractor_localization_hint,
                "likely_drift_reason": pipeline_result.likely_drift_reason,
            }
        )
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_problem_identification",
        )
        teacher_fields["teacher_problem_prompt"] = prompt
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        pipeline_result.problem_raw = raw_prediction
        pipeline_result.caption_problem = self._normalize_teacher_field_text(raw_prediction)
        return pipeline_result

    def generate_teacher_correction_direction(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        pipeline_result,
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "target_summary": pipeline_result.target_summary,
                "distractor_summary": pipeline_result.distractor_summary,
                "shared_evidence": pipeline_result.shared_evidence,
                "target_only_evidence": pipeline_result.target_only_evidence,
                "distractor_only_evidence": pipeline_result.distractor_only_evidence,
                "difference_focus": pipeline_result.difference_focus,
                "caption_problem": pipeline_result.caption_problem,
            }
        )
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_correction_direction",
        )
        teacher_fields["teacher_direction_prompt"] = prompt
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        pipeline_result.direction_raw = raw_prediction
        pipeline_result.correction_direction = self._normalize_teacher_field_text(raw_prediction)
        return pipeline_result

    def generate_teacher_reason_explanation(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        pipeline_result,
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "target_summary": pipeline_result.target_summary,
                "distractor_summary": pipeline_result.distractor_summary,
                "shared_evidence": pipeline_result.shared_evidence,
                "target_only_evidence": pipeline_result.target_only_evidence,
                "distractor_only_evidence": pipeline_result.distractor_only_evidence,
                "target_localization_hint": pipeline_result.target_localization_hint,
                "distractor_localization_hint": pipeline_result.distractor_localization_hint,
                "likely_drift_reason": pipeline_result.likely_drift_reason,
                "caption_problem": pipeline_result.caption_problem,
                "correction_direction": pipeline_result.correction_direction,
            }
        )
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_reason_explanation",
        )
        teacher_fields["teacher_reason_prompt"] = prompt
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        pipeline_result.reason_raw = raw_prediction
        pipeline_result.reason = self._normalize_teacher_field_text(raw_prediction)
        pipeline_result.reason_is_coarse = self._teacher_reason_is_coarse(pipeline_result.reason)
        return pipeline_result

    def generate_teacher_regenerate_single_stage(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        pipeline_result,
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "target_summary": pipeline_result.target_summary,
                "distractor_summary": pipeline_result.distractor_summary,
                "shared_evidence": pipeline_result.shared_evidence,
                "target_only_evidence": pipeline_result.target_only_evidence,
                "distractor_only_evidence": pipeline_result.distractor_only_evidence,
                "difference_focus": pipeline_result.difference_focus,
                "target_localization_hint": pipeline_result.target_localization_hint,
                "distractor_localization_hint": pipeline_result.distractor_localization_hint,
                "likely_drift_reason": pipeline_result.likely_drift_reason,
            }
        )
        prompt = self.build_teacher_regenerate_single_prompt(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            teacher_fields=teacher_fields,
        )
        teacher_fields["teacher_single_stage_prompt"] = prompt
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        pipeline_result.single_stage_raw = "" if raw_prediction is None else str(raw_prediction)
        pipeline_result.dlc_raw = pipeline_result.single_stage_raw
        pipeline_result.verification_raw = pipeline_result.single_stage_raw
        pipeline_result.problem_raw = pipeline_result.single_stage_raw
        pipeline_result.direction_raw = pipeline_result.single_stage_raw
        pipeline_result.reason_raw = pipeline_result.single_stage_raw
        pipeline_result.caption_problem = self._normalize_teacher_field_text(
            self._extract_labeled_teacher_text(raw_prediction, "CAPTION_PROBLEM")
        )
        pipeline_result.correction_direction = self._normalize_teacher_field_text(
            self._extract_labeled_teacher_text(raw_prediction, "CORRECTION_DIRECTION")
        )
        pipeline_result.reason = self._normalize_teacher_field_text(
            self._extract_labeled_teacher_text(raw_prediction, "REASON")
        )
        pipeline_result.problem_valid = bool(pipeline_result.caption_problem)
        pipeline_result.direction_valid = bool(pipeline_result.correction_direction)
        pipeline_result.reason_valid = bool(pipeline_result.reason)
        pipeline_result.diagnosis_valid = bool(
            pipeline_result.problem_valid or pipeline_result.direction_valid or pipeline_result.reason_valid
        )
        pipeline_result.reason_is_coarse = bool(
            pipeline_result.reason and self._teacher_reason_is_coarse(pipeline_result.reason)
        )

        detailed_caption_raw = self._extract_labeled_teacher_text(raw_prediction, "DLC")
        if not detailed_caption_raw:
            fallback_caption = self._clean_caption_text(raw_prediction)
            fallback_status = self._infer_description_status(fallback_caption)
            if fallback_status == "ok":
                detailed_caption_raw = fallback_caption
        detailed_caption = self._clean_teacher_dlc_caption_text(detailed_caption_raw)
        detailed_caption = re.sub(r"^(?:dlc\s*:\s*)+", "", detailed_caption, flags=re.IGNORECASE).strip()
        detailed_caption = re.sub(r"^(?:the task is|task is|region1 is)\s+", "", detailed_caption, flags=re.IGNORECASE)
        detailed_caption = re.sub(r"^(?:horse-\d+|t\d+)\s*", "", detailed_caption, flags=re.IGNORECASE)
        detailed_completion_ids = self._encode_completion_from_caption(detailed_caption)
        detailed_caption, detailed_completion_ids, detailed_was_truncated = self._truncate_caption_completion(
            detailed_caption,
            detailed_completion_ids,
            max_tokens=self.description_max_new_tokens,
        )
        detailed_status = self._infer_description_status(detailed_caption)
        if detailed_status == "ok" and not self._is_caption_content_sufficient(detailed_caption):
            detailed_status = "truncated_caption"
        if detailed_status != "seg_style_answer" and detailed_was_truncated:
            detailed_status = "truncated_caption"
        pipeline_result.detailed_caption = detailed_caption
        pipeline_result.detailed_completion_ids = detailed_completion_ids
        pipeline_result.detailed_status = detailed_status
        pipeline_result.verification_caption = ""
        pipeline_result.verification_status = "empty"
        return pipeline_result

    def generate_teacher_light_diagnosis(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        pipeline_result,
    ):
        # Legacy compatibility helper. Training no longer uses this path directly.
        return self.generate_teacher_reason_explanation(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            pipeline_result=pipeline_result,
        )
    def generate_teacher_dlc_from_diagnosis(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        pipeline_result,
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "target_summary": pipeline_result.target_summary,
                "distractor_summary": pipeline_result.distractor_summary,
                "target_only_evidence": pipeline_result.target_only_evidence,
                "distractor_only_evidence": pipeline_result.distractor_only_evidence,
                "target_localization_hint": pipeline_result.target_localization_hint,
                "caption_problem": pipeline_result.caption_problem,
                "correction_direction": pipeline_result.correction_direction,
                "reason": pipeline_result.reason,
            }
        )
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_dlc_from_diagnosis",
        )
        teacher_fields["teacher_dlc_prompt"] = prompt
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        pipeline_result.dlc_raw = raw_prediction
        detailed_caption = self._clean_teacher_dlc_caption_text(self._extract_labeled_teacher_text(raw_prediction, "DLC"))
        detailed_completion_ids = self._encode_completion_from_caption(detailed_caption)
        detailed_caption, detailed_completion_ids, detailed_was_truncated = self._truncate_caption_completion(
            detailed_caption,
            detailed_completion_ids,
            max_tokens=self.description_max_new_tokens,
        )
        detailed_status = self._infer_description_status(detailed_caption)
        if detailed_status == "ok" and not self._is_caption_content_sufficient(detailed_caption):
            detailed_status = "truncated_caption"
        if detailed_status != "seg_style_answer" and detailed_was_truncated:
            detailed_status = "truncated_caption"
        pipeline_result.detailed_caption = detailed_caption
        pipeline_result.detailed_completion_ids = detailed_completion_ids
        pipeline_result.detailed_status = detailed_status
        failure_reason = self._validate_teacher_dlc(pipeline_result)
        if failure_reason:
            pipeline_result.detailed_failure_reason = failure_reason
        return pipeline_result

    def generate_teacher_dlc_from_plan(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        pipeline_result,
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "primary_failure_type": pipeline_result.primary_failure_type,
                "secondary_failure_types": ", ".join(pipeline_result.secondary_failure_types) or "none",
                "missing_evidence": pipeline_result.missing_evidence,
                "distractor_evidence": pipeline_result.distractor_evidence,
                "bad_phrases_in_student": pipeline_result.bad_phrases_in_student,
                "missing_phrases_needed": pipeline_result.missing_phrases_needed,
                "keepable_phrases": pipeline_result.keepable_phrases,
                "remove_phrases": pipeline_result.remove_phrases,
                "add_phrases": pipeline_result.add_phrases,
                "emphasize_phrases": pipeline_result.emphasize_phrases,
                "deemphasize_phrases": pipeline_result.deemphasize_phrases,
                "localize_with": pipeline_result.localize_with,
                "avoid_phrases": pipeline_result.avoid_phrases,
                "rewrite_strategy": pipeline_result.rewrite_strategy,
            }
        )
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_dlc",
        )
        teacher_fields["teacher_dlc_prompt"] = prompt
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        pipeline_result.dlc_raw = raw_prediction
        detailed_caption = self._clean_teacher_dlc_caption_text(self._extract_labeled_teacher_text(raw_prediction, "DLC"))
        detailed_completion_ids = self._encode_completion_from_caption(detailed_caption)
        detailed_caption, detailed_completion_ids, detailed_was_truncated = self._truncate_caption_completion(
            detailed_caption,
            detailed_completion_ids,
            max_tokens=self.description_max_new_tokens,
        )
        detailed_status = self._infer_description_status(detailed_caption)
        if detailed_status == "ok" and not self._is_caption_content_sufficient(detailed_caption):
            detailed_status = "truncated_caption"
        if detailed_status != "seg_style_answer" and detailed_was_truncated:
            detailed_status = "truncated_caption"
        pipeline_result.detailed_caption = detailed_caption
        pipeline_result.detailed_completion_ids = detailed_completion_ids
        pipeline_result.detailed_status = detailed_status
        failure_reason = self._validate_teacher_dlc(pipeline_result)
        if failure_reason:
            pipeline_result.detailed_failure_reason = failure_reason
        return pipeline_result

    def generate_teacher_verification_caption(
        self,
        *,
        image,
        teacher_prompt_masks,
        student_question,
        description_status,
        reconstruction,
        iou,
        gt_mask,
        ref_mask,
        teacher_fields,
        pipeline_result,
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "detailed_caption": pipeline_result.detailed_caption,
                "target_only_evidence": pipeline_result.target_only_evidence,
                "distractor_only_evidence": pipeline_result.distractor_only_evidence,
                "target_localization_hint": pipeline_result.target_localization_hint,
                "correction_direction": pipeline_result.correction_direction,
            }
        )
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=pipeline_result.detailed_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="teacher_verification_caption_from_diagnosis",
        )
        teacher_fields["teacher_verification_prompt"] = prompt
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
        )
        pipeline_result.verification_raw = raw_prediction
        verification_caption = self._clean_teacher_dlc_caption_text(
            self._extract_labeled_teacher_text(raw_prediction, "VERIFICATION_CAPTION")
        )
        verification_status = self._infer_description_status(verification_caption)
        if verification_status == "ok" and (
            not self._is_caption_content_sufficient(verification_caption)
            or self._is_overly_generic_caption(verification_caption)
        ):
            verification_status = "truncated_caption"
        pipeline_result.verification_caption = verification_caption
        pipeline_result.verification_status = verification_status
        failure_reason = self._validate_teacher_verification_caption(pipeline_result)
        if failure_reason:
            pipeline_result.verification_failure_reason = failure_reason
        return pipeline_result

    def run_teacher_regenerate_pipeline(
        self,
        *,
        image,
        gt_mask,
        ref_mask,
        student_question,
        student_caption,
        description_status,
        reconstruction,
        iou,
        teacher_fields,
        caption_mode_failure=False,
    ):
        teacher_prompt_masks = self._build_teacher_prompt_masks(gt_mask, ref_mask)
        difference_context = build_teacher_regenerate_difference_context(
            model=self,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            image=image,
            student_question=student_question,
            region_model=self.require_teacher_model("Teacher regenerate difference summarization"),
        )
        pipeline_result = self._build_empty_teacher_regenerate_pipeline_result()
        pipeline_result.teacher_pipeline_mode = "structured_2p5"
        pipeline_result.target_summary = difference_context["target_summary"]
        pipeline_result.distractor_summary = difference_context["distractor_summary"]
        pipeline_result.shared_evidence = difference_context["shared_evidence"]
        pipeline_result.target_only_evidence = difference_context["target_only_evidence"]
        pipeline_result.distractor_only_evidence = difference_context["distractor_only_evidence"]
        pipeline_result.difference_focus = difference_context.get("difference_focus", "")
        pipeline_result.target_localization_hint = difference_context["target_localization_hint"]
        pipeline_result.distractor_localization_hint = difference_context["distractor_localization_hint"]
        pipeline_result.likely_drift_reason = difference_context["likely_drift_reason"]
        pipeline_result.difference_context_nontrivial = bool(
            self._teacher_field_is_effective(pipeline_result.target_only_evidence)
            or self._teacher_field_is_effective(pipeline_result.distractor_only_evidence)
        )
        pipeline_result.stop_stage = "difference_context"
        pipeline_result = self.generate_teacher_structured_diagnosis(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            pipeline_result=pipeline_result,
            retry_reason="",
        )
        if not pipeline_result.diagnosis_valid:
            pipeline_result.teacher_diagnosis_retry_count = 1
            retry_reason = pipeline_result.diagnosis_failure_reason
            pipeline_result = self.generate_teacher_structured_diagnosis(
                image=image,
                teacher_prompt_masks=teacher_prompt_masks,
                student_question=student_question,
                student_caption=student_caption,
                description_status=description_status,
                reconstruction=reconstruction,
                iou=iou,
                gt_mask=gt_mask,
                ref_mask=ref_mask,
                teacher_fields=teacher_fields,
                pipeline_result=pipeline_result,
                retry_reason=retry_reason,
            )
        pipeline_result.stop_stage = "structured_diagnosis"

        selected_candidate = None
        candidate_scores = []
        candidate_signal_is_actionable = self._teacher_has_actionable_diagnosis_signal(pipeline_result)
        can_try_candidates = bool(candidate_signal_is_actionable or self._teacher_has_minimal_diagnosis_signal(pipeline_result))
        if can_try_candidates:
            candidate_specs = (
                {"candidate_style_hint": "Use the strongest target-only cue as the main anchor.", "do_sample": False},
                {
                    "candidate_style_hint": "Prefer a compact DLC-bench style sentence with one clear local cue.",
                    "do_sample": True,
                    "temperature": 0.35,
                    "top_p": 0.9,
                },
                {
                    "candidate_style_hint": "Keep the target cue explicit while avoiding distractor-compatible wording.",
                    "do_sample": True,
                    "temperature": 0.55,
                    "top_p": 0.92,
                },
            )
            candidates = []
            for spec in candidate_specs:
                candidate = self._build_teacher_dlc_candidate_record(
                    image=image,
                    gt_mask=gt_mask,
                    teacher_prompt_masks=teacher_prompt_masks,
                    student_question=student_question,
                    student_caption=student_caption,
                    description_status=description_status,
                    reconstruction=reconstruction,
                    iou=iou,
                    gt_mask_for_prompt=gt_mask,
                    ref_mask=ref_mask,
                    teacher_fields=teacher_fields,
                    pipeline_result=pipeline_result,
                    candidate_style_hint=spec["candidate_style_hint"],
                    generation_kwargs={
                        key: value
                        for key, value in spec.items()
                        if key in {"do_sample", "temperature", "top_p"}
                    },
                )
                candidates.append(candidate)
                candidate_scores.append(
                    {
                        "caption": candidate["caption"],
                        "status": candidate["status"],
                        "failure_reason": candidate["failure_reason"],
                        "reconstruct_iou": candidate["reconstruct_iou"],
                        "score": candidate["score"],
                    }
                )
            pipeline_result.teacher_dlc_candidate_count = len(candidates)
            valid_candidates = [candidate for candidate in candidates if self._teacher_candidate_is_usable(candidate)]
            pipeline_result.teacher_dlc_valid_candidate_count = len(valid_candidates)
            ranked_candidates = sorted(
                valid_candidates,
                key=lambda item: (item["score"], item["reconstruct_iou"]),
                reverse=True,
            )
            if ranked_candidates:
                selected_candidate = ranked_candidates[0]
                pipeline_result.teacher_dlc_selected_by = (
                    "candidate_score" if candidate_signal_is_actionable else "candidate_score_degraded"
                )
                pipeline_result = self._apply_teacher_dlc_candidate(
                    pipeline_result,
                    selected_candidate,
                    "structured_candidate" if candidate_signal_is_actionable else "degraded_structured_candidate",
                )

        pipeline_result.teacher_dlc_candidate_scores = tuple(candidate_scores)
        if selected_candidate is None:
            pipeline_result.detailed_failure_reason = "teacher_dlc_invalid:no_candidate_selected"
        pipeline_result.stop_stage = "dlc_candidates"

        if selected_candidate is not None:
            pipeline_result = self._evaluate_teacher_verification_branch(
                image=image,
                gt_mask=gt_mask,
                ref_mask=ref_mask,
                student_question=student_question,
                description_status=description_status,
                reconstruction=reconstruction,
                iou=iou,
                teacher_prompt_masks=teacher_prompt_masks,
                teacher_fields=teacher_fields,
                pipeline_result=pipeline_result,
            )

            best_caption = pipeline_result.detailed_caption
            best_status = pipeline_result.detailed_status
            best_failure_reason = pipeline_result.detailed_failure_reason
            best_iou = float(selected_candidate.get("reconstruct_iou", 0.0) or 0.0)
            if (
                pipeline_result.verification_status == "ok"
                and not pipeline_result.verification_failure_reason
                and float(pipeline_result.verification_iou or 0.0) > best_iou
            ):
                best_caption = pipeline_result.verification_caption
                best_status = pipeline_result.verification_status
                best_failure_reason = ""
                best_iou = float(pipeline_result.verification_iou or 0.0)
                pipeline_result.teacher_verification_used = True
                pipeline_result.teacher_dlc_selected_by = "verification_iou"
                pipeline_result = self._materialize_teacher_caption_result(
                    pipeline_result,
                    best_caption,
                    "verification_caption",
                )
            else:
                pipeline_result.teacher_verification_used = False

            pipeline_result.detailed_status = best_status
            pipeline_result.detailed_failure_reason = best_failure_reason
            teacher_iou_plain = best_iou
        else:
            teacher_iou_plain = 0.0

        teacher_reconstruct_ok = bool(
            pipeline_result.detailed_status == "ok"
            and float(teacher_iou_plain) > 0.0
            and not pipeline_result.detailed_failure_reason
        )
        pipeline_result.verification_iou = float(teacher_iou_plain)
        pipeline_result.gate_passed = (
            bool(teacher_reconstruct_ok)
            if caption_mode_failure
            else (
                self._teacher_regenerate_gate_passed(iou, teacher_iou_plain)
                if teacher_reconstruct_ok
                else False
            )
        )

        if can_try_candidates and not pipeline_result.gate_passed:
            rejected_dlc_summary = "; ".join(
                f"score={item['score']:.3f},iou={item['reconstruct_iou']:.3f},status={item['status']},failure={item['failure_reason']},caption={item['caption']!r}"
                for item in candidate_scores[:3]
            )
            repair_reason = (
                pipeline_result.detailed_failure_reason
                or pipeline_result.verification_failure_reason
                or ("teacher_gate_failed:iou_not_improved_enough" if teacher_reconstruct_ok else "teacher_gate_failed:reconstruct_failed")
            )
            pipeline_result = self._generate_teacher_repair_caption(
                image=image,
                gt_mask=gt_mask,
                ref_mask=ref_mask,
                student_question=student_question,
                student_caption=student_caption,
                description_status=description_status,
                reconstruction=reconstruction,
                iou=iou,
                teacher_prompt_masks=teacher_prompt_masks,
                teacher_fields=teacher_fields,
                pipeline_result=pipeline_result,
                repair_reason=repair_reason,
                rejected_dlc_summary=rejected_dlc_summary,
            )
            pipeline_result.stop_stage = "repair"
            repair_reconstruction = self.reconstruct_mask(
                image=image,
                caption=pipeline_result.detailed_caption,
                description_status=pipeline_result.detailed_status,
                gt_mask=gt_mask,
            )
            repair_pred_mask = None if repair_reconstruction is None else repair_reconstruction.pred_mask
            teacher_iou_plain = self._compute_iou(gt_mask, repair_pred_mask) if repair_pred_mask is not None else 0.0
            pipeline_result.verification_iou = float(teacher_iou_plain)
            teacher_reconstruct_ok = bool(
                repair_reconstruction is not None
                and repair_reconstruction.status == "ok"
                and repair_pred_mask is not None
                and not pipeline_result.detailed_failure_reason
            )
            pipeline_result.gate_passed = (
                bool(teacher_reconstruct_ok)
                if caption_mode_failure
                else (
                    self._teacher_regenerate_gate_passed(iou, teacher_iou_plain)
                    if teacher_reconstruct_ok
                    else False
                )
            )
            if pipeline_result.gate_passed:
                pipeline_result.teacher_dlc_selected_by = "repair_caption"

        if not pipeline_result.gate_passed:
            pipeline_result.teacher_fallback_used = True
            pipeline_result.teacher_fallback_reason = (
                pipeline_result.diagnosis_failure_reason
                or pipeline_result.detailed_failure_reason
                or pipeline_result.verification_failure_reason
                or "teacher_gate_failed"
            )
            fallback_result = self.generate_teacher_regenerate_single_stage(
                image=image,
                teacher_prompt_masks=teacher_prompt_masks,
                student_question=student_question,
                student_caption=student_caption,
                description_status=description_status,
                reconstruction=reconstruction,
                iou=iou,
                gt_mask=gt_mask,
                ref_mask=ref_mask,
                teacher_fields=teacher_fields,
                pipeline_result=pipeline_result,
            )
            fallback_result.teacher_pipeline_mode = "structured_2p5_with_fallback"
            fallback_result.teacher_fallback_used = True
            fallback_result.teacher_fallback_reason = pipeline_result.teacher_fallback_reason
            fallback_result.stop_stage = "single_stage_fallback"
            if self._teacher_field_is_effective(fallback_result.single_stage_raw, invalid_markers=("",)):
                fallback_reconstruction = self.reconstruct_mask(
                    image=image,
                    caption=fallback_result.detailed_caption,
                    description_status=fallback_result.detailed_status,
                    gt_mask=gt_mask,
                )
                fallback_pred_mask = None if fallback_reconstruction is None else fallback_reconstruction.pred_mask
                fallback_iou = self._compute_iou(gt_mask, fallback_pred_mask) if fallback_pred_mask is not None else 0.0
                fallback_result.verification_iou = float(fallback_iou)
                fallback_reconstruct_ok = bool(
                    fallback_reconstruction is not None
                    and fallback_reconstruction.status == "ok"
                    and fallback_pred_mask is not None
                )
                fallback_result.gate_passed = (
                    bool(fallback_reconstruct_ok)
                    if caption_mode_failure
                    else (
                        self._teacher_regenerate_gate_passed(iou, fallback_iou)
                        if fallback_reconstruct_ok
                        else False
                    )
                )
                fallback_result.stop_stage = "passed" if fallback_result.gate_passed else "gate"
                if fallback_result.gate_passed:
                    fallback_result.teacher_selected_caption_source = "single_stage_fallback"
                    return fallback_result
            pipeline_result = fallback_result

        pipeline_result.stop_stage = "passed" if pipeline_result.gate_passed else "gate"
        if not teacher_reconstruct_ok and not pipeline_result.gate_passed:
            pipeline_result.diagnosis_failure_reason = "teacher_gate_failed:reconstruct_failed"
        elif not pipeline_result.gate_passed:
            pipeline_result.diagnosis_failure_reason = "teacher_gate_failed:iou_not_improved_enough"
        return pipeline_result

    @staticmethod
    def _route_prompt_tag(route):
        if route == TEACHER_REGENERATE_ROUTE:
            return "[TEACHER_REGENERATE_ROUTE]"
        if route == ON_POLICY_DISTILL_ROUTE:
            return "[ON_POLICY_DISTILL_ROUTE]"
        if route == GRPO_POSITIVE_ROUTE:
            return "[GRPO_ROUTE]"
        if route:
            return f"[{str(route).upper()}]"
        return ""

    def estimate_opsd_route_for_sample_with_model(
        self,
        *,
        description_model,
        reconstruct_model,
        image,
        prompt_masks,
        student_question,
        gt_mask,
        sample_key=None,
        debug: bool = False,
    ):
        gt_mask_np = self._to_numpy_mask(gt_mask)
        if int(gt_mask_np.sum()) == 0:
            return {
                "sample_key": sample_key,
                "route": "skip",
                "iou": 0.0,
                "description_status": "empty_gt_mask",
                "reconstruct_status": "skipped_empty_gt_mask",
                "description": None,
                "reconstruction": None,
                "pred_mask": None,
            }

        description = self.generate_description_with_model(
            description_model,
            image=image,
            mask_prompts=prompt_masks,
            student_question=student_question,
            apply_mask_focus=True,
        )
        if description.status != "ok":
            reconstruction = ReconstructionResult(
                pred_mask=None,
                question=None,
                raw_prediction="",
                prediction_masks_count=0,
                seg_token_count=0,
                status="skipped_invalid_description",
            )
        else:
            best_result = None
            best_iou = -1.0
            spatial_hint = self._coarse_spatial_hint(gt_mask_np)
            for reconstruct_question_base in self._resolve_reconstruct_questions(description.clean_caption):
                reconstruct_question_variants = [reconstruct_question_base]
                if spatial_hint:
                    reconstruct_question_variants.append(
                        self._append_spatial_hint_to_question(reconstruct_question_base, spatial_hint)
                    )
                for reconstruct_question in reconstruct_question_variants:
                    predict_dict = self._predict_forward_eval(
                        reconstruct_model,
                        image=image,
                        text=reconstruct_question,
                        past_text="",
                        mask_prompts=None,
                        tokenizer=self.tokenizer,
                    )
                    raw_prediction = predict_dict.get("prediction", "")
                    prediction_masks = predict_dict.get("prediction_masks")
                    prediction_masks_count = 0 if prediction_masks is None else len(prediction_masks)
                    if not prediction_masks:
                        result = ReconstructionResult(
                            pred_mask=None,
                            question=reconstruct_question,
                            raw_prediction=raw_prediction,
                            prediction_masks_count=prediction_masks_count,
                            seg_token_count=int(predict_dict.get("seg_token_count", 0) or 0),
                            status="empty_prediction_masks",
                        )
                        candidate_iou = -1.0
                    else:
                        first_mask = prediction_masks[0]
                        if isinstance(first_mask, torch.Tensor):
                            first_mask = first_mask.detach().cpu().numpy()
                        first_mask = np.asarray(first_mask)
                        if first_mask.ndim == 3 and first_mask.shape[0] == 1:
                            first_mask = first_mask[0]
                        pred_mask = self._to_numpy_mask(first_mask)
                        result = ReconstructionResult(
                            pred_mask=pred_mask,
                            question=reconstruct_question,
                            raw_prediction=raw_prediction,
                            prediction_masks_count=prediction_masks_count,
                            seg_token_count=int(predict_dict.get("seg_token_count", 0) or 0),
                            status="ok" if pred_mask.sum() > 0 else "zero_area_mask",
                        )
                        candidate_iou = self._compute_iou(gt_mask_np, pred_mask)
                    if candidate_iou > best_iou:
                        best_result = result
                        best_iou = candidate_iou
            reconstruction = best_result

        reconstruct_status = (
            "missing_reconstruction_result" if reconstruction is None else reconstruction.status
        )
        pred_mask = None if reconstruction is None else reconstruction.pred_mask
        route = "skip"
        iou = 0.0
        if pred_mask is not None:
            iou = self._compute_iou(gt_mask_np, pred_mask)
            route = self._route_from_iou(iou)
        if debug:
            self._debug_sample(
                sample_key=sample_key,
                route=route,
                student_question=student_question,
                raw_prediction=description.raw_prediction,
                caption=description.clean_caption,
                description_status=description.status,
                reconstruct_question=None if reconstruction is None else reconstruction.question,
                raw_reconstruct_prediction="" if reconstruction is None else reconstruction.raw_prediction,
                reconstruct_status=reconstruct_status,
                seg_token_count=0 if reconstruction is None else reconstruction.seg_token_count,
                prediction_masks_count=0 if reconstruction is None else reconstruction.prediction_masks_count,
                pred_mask=pred_mask,
                gt_mask=gt_mask_np,
                iou=iou,
                empty_gt_mask=False,
            )
        return {
            "sample_key": sample_key,
            "route": route,
            "iou": float(iou),
            "description_status": description.status,
            "reconstruct_status": reconstruct_status,
            "description": description,
            "reconstruction": reconstruction,
            "pred_mask": pred_mask,
        }

    def estimate_opsd_route_for_sample(
        self,
        *,
        image,
        prompt_masks,
        student_question,
        gt_mask,
        sample_key=None,
        debug: bool = False,
    ):
        return self.estimate_opsd_route_for_sample_with_model(
            description_model=self.student_model,
            reconstruct_model=self.student_model,
            image=image,
            prompt_masks=prompt_masks,
            student_question=student_question,
            gt_mask=gt_mask,
            sample_key=sample_key,
            debug=debug,
        )

    @staticmethod
    def _resolve_batch_route(routes):
        route_list = [route for route in routes if route not in {None, "", "skip"}]
        if not route_list:
            return None
        unique_routes = sorted(set(route_list))
        if len(unique_routes) != 1:
            raise RuntimeError(f"Mixed OPSD routes in one batch: {unique_routes}")
        return unique_routes[0]

    def _resolve_loss_family(self, batch_route, route_from_manifest=None, online_route=None):
        if batch_route not in {None, "", "skip"}:
            return str(batch_route)
        manifest_route = None if route_from_manifest in {None, "", "skip"} else str(route_from_manifest)
        online_route = None if online_route in {None, "", "skip"} else str(online_route)
        if self.use_online_route_for_loss:
            route_priority = {
                TEACHER_REGENERATE_ROUTE: 0,
                ON_POLICY_DISTILL_ROUTE: 1,
                GRPO_POSITIVE_ROUTE: 2,
            }
            if manifest_route is None:
                return online_route or TEACHER_REGENERATE_ROUTE
            if online_route is None:
                return manifest_route
            manifest_priority = route_priority.get(manifest_route, 99)
            online_priority = route_priority.get(online_route, 99)
            return online_route if online_priority < manifest_priority else manifest_route
        if manifest_route is not None:
            return manifest_route
        if online_route is not None:
            return online_route
        return TEACHER_REGENERATE_ROUTE

    def _build_dummy_completion_ids(self):
        completion_ids = self._encode_completion_from_caption("the object")
        if completion_ids.shape[1] > 0:
            return completion_ids
        fallback_token_id = self.tokenizer.eos_token_id
        if fallback_token_id is None:
            fallback_token_id = self.tokenizer.pad_token_id
        if fallback_token_id is None:
            fallback_token_id = 0
        return torch.tensor([[int(fallback_token_id)]], dtype=torch.long, device=self.device)

    def _build_teacher_prompt_masks(self, gt_mask_np, ref_mask_np):
        return np.stack(
            [
                self._to_numpy_mask(gt_mask_np).astype(np.float32),
                self._to_numpy_mask(ref_mask_np).astype(np.float32),
            ],
            axis=0,
        )

    @staticmethod
    def _zero_ref_mask_like(mask):
        return np.zeros_like(mask, dtype=np.uint8)

    @staticmethod
    def _merge_dummy_reasons(records):
        reasons = []
        for record in records:
            reason = record.get("dummy_reason")
            if reason:
                reasons.append(str(reason))
        return sorted(set(reasons))

    def _attempt_teacher_regenerate_analysis(
        self,
        *,
        image,
        gt_mask_np,
        ref_mask_np,
        student_question,
        description,
        reconstruction,
        iou,
        allow_teacher_ce,
    ):
        result = {
            "teacher_regenerate": None,
            "teacher_reconstruct_ok": None,
            "teacher_gate_passed": None,
            "teacher_iou_plain": None,
            "teacher_completion_len": None,
            "caption_mode_failure": False,
            "teacher_difference_context_nontrivial": False,
            "teacher_diagnosis_valid": False,
            "teacher_verification_caption_status": "empty",
            "teacher_verification_caption": "",
            "teacher_dlc": "",
            "teacher_verification_iou": 0.0,
            "teacher_dlc_valid": False,
            "teacher_target_summary": "",
            "teacher_distractor_summary": "",
            "teacher_shared_evidence": "",
            "teacher_target_only_evidence": "",
            "teacher_distractor_only_evidence": "",
            "teacher_difference_focus": "",
            "teacher_likely_drift_reason": "",
            "teacher_caption_problem": "",
            "teacher_correction_direction": "",
            "teacher_reason": "",
            "teacher_single_stage_raw": "",
            "teacher_structured_diagnosis_raw": "",
            "teacher_repair_raw": "",
            "teacher_problem_raw": "",
            "teacher_direction_raw": "",
            "teacher_reason_raw": "",
            "teacher_problem_valid": False,
            "teacher_direction_valid": False,
            "teacher_reason_valid": False,
            "teacher_reason_is_coarse": False,
            "teacher_problem_failure_reason": "",
            "teacher_direction_failure_reason": "",
            "teacher_reason_failure_reason": "",
            "teacher_diagnosis_failure_reason": "",
            "teacher_pipeline_stop_stage": "difference_context",
            "teacher_pipeline_failure_reason": "",
            "teacher_pipeline_mode": "single_stage",
            "teacher_diagnosis_retry_count": 0,
            "teacher_dlc_candidate_count": 0,
            "teacher_dlc_valid_candidate_count": 0,
            "teacher_dlc_selected_by": "",
            "teacher_verification_used": False,
            "teacher_fallback_used": False,
            "teacher_fallback_reason": "",
            "teacher_selected_caption_source": "",
            "teacher_dlc_candidate_scores": (),
        }
        if not allow_teacher_ce:
            return result
        caption_mode_failure = bool(description.raw_failure_mode == "seg_style_answer" or description.status == "seg_style_answer")
        result["caption_mode_failure"] = caption_mode_failure

        teacher_fields = self._build_training_teacher_fields(
            route=TEACHER_REGENERATE_ROUTE,
            iou=iou,
        )
        teacher_regenerate = self.run_teacher_regenerate_pipeline(
            image=image,
            gt_mask=gt_mask_np,
            ref_mask=ref_mask_np,
            student_question=student_question,
            student_caption=description.clean_caption,
            description_status=description.status,
            reconstruction=reconstruction,
            iou=iou,
            teacher_fields=teacher_fields,
            caption_mode_failure=caption_mode_failure,
        )
        result["teacher_regenerate"] = teacher_regenerate
        result["teacher_completion_len"] = int(teacher_regenerate.detailed_completion_ids.shape[1])
        result["teacher_difference_context_nontrivial"] = bool(teacher_regenerate.difference_context_nontrivial)
        result["teacher_diagnosis_valid"] = bool(teacher_regenerate.diagnosis_valid)
        result["teacher_verification_caption_status"] = str(teacher_regenerate.verification_status)
        result["teacher_verification_caption"] = teacher_regenerate.verification_caption
        result["teacher_dlc"] = teacher_regenerate.detailed_caption
        result["teacher_dlc_valid"] = bool(
            teacher_regenerate.detailed_status == "ok" and not teacher_regenerate.detailed_failure_reason
        )
        result["teacher_target_summary"] = teacher_regenerate.target_summary
        result["teacher_distractor_summary"] = teacher_regenerate.distractor_summary
        result["teacher_shared_evidence"] = teacher_regenerate.shared_evidence
        result["teacher_target_only_evidence"] = teacher_regenerate.target_only_evidence
        result["teacher_distractor_only_evidence"] = teacher_regenerate.distractor_only_evidence
        result["teacher_difference_focus"] = teacher_regenerate.difference_focus
        result["teacher_likely_drift_reason"] = teacher_regenerate.likely_drift_reason
        result["teacher_caption_problem"] = teacher_regenerate.caption_problem
        result["teacher_correction_direction"] = teacher_regenerate.correction_direction
        result["teacher_reason"] = teacher_regenerate.reason
        result["teacher_single_stage_raw"] = teacher_regenerate.single_stage_raw
        result["teacher_structured_diagnosis_raw"] = teacher_regenerate.structured_diagnosis_raw
        result["teacher_repair_raw"] = teacher_regenerate.repair_raw
        result["teacher_problem_raw"] = teacher_regenerate.problem_raw
        result["teacher_direction_raw"] = teacher_regenerate.direction_raw
        result["teacher_reason_raw"] = teacher_regenerate.reason_raw
        result["teacher_problem_valid"] = bool(teacher_regenerate.problem_valid)
        result["teacher_direction_valid"] = bool(teacher_regenerate.direction_valid)
        result["teacher_reason_valid"] = bool(teacher_regenerate.reason_valid)
        result["teacher_reason_is_coarse"] = bool(teacher_regenerate.reason_is_coarse)
        result["teacher_problem_failure_reason"] = str(teacher_regenerate.problem_failure_reason)
        result["teacher_direction_failure_reason"] = str(teacher_regenerate.direction_failure_reason)
        result["teacher_reason_failure_reason"] = str(teacher_regenerate.reason_failure_reason)
        result["teacher_diagnosis_failure_reason"] = str(teacher_regenerate.diagnosis_failure_reason)
        result["teacher_pipeline_stop_stage"] = teacher_regenerate.stop_stage
        result["teacher_pipeline_mode"] = str(teacher_regenerate.teacher_pipeline_mode)
        result["teacher_diagnosis_retry_count"] = int(teacher_regenerate.teacher_diagnosis_retry_count)
        result["teacher_dlc_candidate_count"] = int(teacher_regenerate.teacher_dlc_candidate_count)
        result["teacher_dlc_valid_candidate_count"] = int(teacher_regenerate.teacher_dlc_valid_candidate_count)
        result["teacher_dlc_selected_by"] = str(teacher_regenerate.teacher_dlc_selected_by)
        result["teacher_verification_used"] = bool(teacher_regenerate.teacher_verification_used)
        result["teacher_fallback_used"] = bool(teacher_regenerate.teacher_fallback_used)
        result["teacher_fallback_reason"] = str(teacher_regenerate.teacher_fallback_reason)
        result["teacher_selected_caption_source"] = str(teacher_regenerate.teacher_selected_caption_source)
        result["teacher_dlc_candidate_scores"] = tuple(teacher_regenerate.teacher_dlc_candidate_scores)
        result["teacher_pipeline_failure_reason"] = (
            teacher_regenerate.difference_context_failure_reason
            or teacher_regenerate.diagnosis_failure_reason
            or teacher_regenerate.detailed_failure_reason
            or teacher_regenerate.verification_failure_reason
            or ("" if teacher_regenerate.gate_passed else "teacher_gate_failed")
        )
        if teacher_regenerate.detailed_completion_ids.shape[1] == 0:
            result["teacher_reconstruct_ok"] = False
            result["teacher_gate_passed"] = False
            result["teacher_iou_plain"] = 0.0
            return result

        teacher_iou_plain = float(teacher_regenerate.verification_iou or 0.0)
        result["teacher_verification_iou"] = float(teacher_iou_plain)
        teacher_reconstruct_ok = bool(teacher_regenerate.detailed_status == "ok" and teacher_iou_plain > 0.0)
        teacher_gate_passed = bool(teacher_regenerate.gate_passed)
        result["teacher_reconstruct_ok"] = bool(teacher_reconstruct_ok)
        result["teacher_gate_passed"] = bool(teacher_gate_passed)
        result["teacher_iou_plain"] = float(teacher_iou_plain)
        return result

    def _empty_loss_vector(self):
        return torch.empty(0, device=self.device, dtype=next(self.student_model.parameters()).dtype)

    def _zero_scalar(self, *, requires_grad=False, dtype=None):
        if dtype is None:
            dtype = next(self.student_model.parameters()).dtype
        return torch.zeros((), device=self.device, dtype=dtype, requires_grad=requires_grad)

    def _placeholder_loss_vector(self, count, *, reason):
        count = int(count)
        if count <= 0:
            return self._empty_loss_vector()
        zero = self._zero_scalar(requires_grad=True)
        if self._should_debug_print():
            print(
                "[Sa2VA_OPSD_V2_DDP_DEBUG] "
                f"route_placeholder_loss reason={reason} count={count}"
            )
        return zero.expand(count)

    def compute_regenerate_alignment_loss(
        self,
        image,
        prompt_masks,
        student_question,
        completion_ids,
        loss_weight=1.0,
    ):
        if completion_ids.shape[1] == 0:
            return None
        student_logits = self._forward_sequence_with_model(
            self.student_model,
            image,
            prompt_masks,
            student_question,
            completion_ids,
            apply_mask_focus=True,
        )
        sample_loss = self._sequence_cross_entropy_from_logits(student_logits, completion_ids)
        return sample_loss * float(loss_weight)

    def compute_regenerate_alignment_losses_batch(self, batch_items):
        sample_losses = []
        dummy_reasons = []
        for item in batch_items:
            completion_ids = item.get("completion_ids")
            if completion_ids is None or completion_ids.shape[1] == 0:
                completion_ids = self._build_dummy_completion_ids()
            loss_weight = float(item.get("loss_weight", 1.0))
            if loss_weight == 0.0 and item.get("dummy_reason"):
                dummy_reasons.append(str(item["dummy_reason"]))
            sample_loss = self.compute_regenerate_alignment_loss(
                image=item["image"],
                prompt_masks=item["prompt_masks"],
                student_question=item["student_question"],
                completion_ids=completion_ids,
                loss_weight=loss_weight,
            )
            if sample_loss is not None:
                sample_losses.append(sample_loss)
        if not sample_losses:
            return self._placeholder_loss_vector(
                len(batch_items),
                reason=f"regen-empty-sample-losses dummy_reasons={','.join(dummy_reasons) or 'unknown'}",
            )
        return torch.stack(sample_losses)

    def compute_onpolicy_distill_loss(
        self,
        image,
        prompt_masks,
        student_question,
        teacher_prompt,
        completion_ids,
        teacher_prompt_masks=None,
        iou=0.0,
        loss_weight=1.0,
    ):
        if completion_ids.shape[1] == 0:
            return None
        student_logits = self._forward_sequence_with_model(
            self.student_model,
            image,
            prompt_masks,
            student_question,
            completion_ids,
            apply_mask_focus=True,
        )
        teacher_model = self.require_teacher_model("On-policy distillation")
        with torch.no_grad():
            teacher_logits = self._forward_sequence_with_model(
                teacher_model,
                image,
                teacher_prompt_masks if teacher_prompt_masks is not None else prompt_masks,
                teacher_prompt,
                completion_ids,
                apply_mask_focus=False,
            )
        jsd_tokens, teacher_entropy = self.generalized_jsd_token_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            beta=self.jsd_beta,
            temperature=self.teacher_temperature,
        )
        token_weights = torch.exp(-self.entropy_weight_beta * teacher_entropy)
        token_weights = token_weights / token_weights.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        sample_weight = self.mid_iou_alpha * max(1.0 - float(iou), 0.0)
        return (jsd_tokens * token_weights).mean() * sample_weight * float(loss_weight)

    def compute_onpolicy_distill_losses_batch(self, batch_items):
        sample_losses = []
        dummy_reasons = []
        for item in batch_items:
            completion_ids = item.get("completion_ids")
            if completion_ids is None or completion_ids.shape[1] == 0:
                completion_ids = self._build_dummy_completion_ids()
            loss_weight = float(item.get("loss_weight", 1.0))
            if loss_weight == 0.0 and item.get("dummy_reason"):
                dummy_reasons.append(str(item["dummy_reason"]))
            sample_loss = self.compute_onpolicy_distill_loss(
                image=item["image"],
                prompt_masks=item["prompt_masks"],
                student_question=item["student_question"],
                teacher_prompt=item["teacher_prompt"],
                completion_ids=completion_ids,
                teacher_prompt_masks=item.get("teacher_prompt_masks", item["prompt_masks"]),
                iou=float(item.get("iou", 0.0)),
                loss_weight=loss_weight,
            )
            if sample_loss is not None:
                sample_losses.append(sample_loss)
        if not sample_losses:
            return self._placeholder_loss_vector(
                len(batch_items),
                reason=f"onpolicy-empty-sample-losses dummy_reasons={','.join(dummy_reasons) or 'unknown'}",
            )
        return torch.stack(sample_losses)

    def _sample_grpo_descriptions(self, *, model, image, prompt_masks, student_question):
        target_rollout_count = int(self.grpo_group_size)
        if target_rollout_count <= 0:
            return []
        descriptions = []
        for _ in range(target_rollout_count):
            descriptions.append(
                self.generate_description_with_model(
                    model,
                    image=image,
                    mask_prompts=prompt_masks,
                    student_question=student_question,
                    apply_mask_focus=True,
                )
            )
        return descriptions

    @staticmethod
    def _mask_area_ratio(mask):
        mask = np.asarray(mask)
        if mask.ndim != 2:
            return 0.0
        return float((mask > 0).sum()) / float(max(mask.shape[0] * mask.shape[1], 1))

    @staticmethod
    def _mask_center(mask):
        mask = np.asarray(mask)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        h, w = mask.shape
        return (float(xs.mean()) / max(w, 1), float(ys.mean()) / max(h, 1))

    @staticmethod
    def _mask_center_distance(mask_a, mask_b):
        center_a = Sa2VAOPSDModelV2._mask_center(mask_a)
        center_b = Sa2VAOPSDModelV2._mask_center(mask_b)
        if center_a is None or center_b is None:
            return 1.0
        return float(((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5)

    def _prepare_mask_like_gt(self, gt_mask, candidate_mask):
        gt_mask = self._to_numpy_mask(gt_mask)
        candidate_mask = self._to_numpy_mask(candidate_mask)
        if gt_mask.shape != candidate_mask.shape:
            candidate_mask_t = torch.from_numpy(candidate_mask[None, None].astype(np.float32))
            candidate_mask_t = F.interpolate(candidate_mask_t, size=gt_mask.shape, mode="nearest")[0, 0]
            candidate_mask = (candidate_mask_t.numpy() > 0).astype(np.uint8)
        return candidate_mask

    def _score_confuser_candidate(self, gt_mask, candidate_mask):
        overlap_iou = float(self._compute_iou(gt_mask, candidate_mask))
        center_distance = self._mask_center_distance(gt_mask, candidate_mask)
        return (
            self.grpo_confuser_overlap_weight * overlap_iou
            - self.grpo_confuser_nearby_center_weight * center_distance
        )

    def _select_confuser_masks(self, *, gt_mask, candidate_masks):
        if candidate_masks is None:
            candidate_masks = []
        candidate_count = len(candidate_masks)
        gt_mask = self._to_numpy_mask(gt_mask)
        scored_candidates = []
        for candidate_mask in candidate_masks:
            prepared_mask = self._prepare_mask_like_gt(gt_mask, candidate_mask)
            if int(prepared_mask.sum()) == 0:
                continue
            area_ratio = self._mask_area_ratio(prepared_mask)
            if area_ratio < self.grpo_confuser_min_area_ratio or area_ratio > self.grpo_confuser_max_area_ratio:
                continue
            overlap_iou = float(self._compute_iou(gt_mask, prepared_mask))
            if overlap_iou >= self.grpo_confuser_duplicate_iou_threshold:
                continue
            scored_candidates.append(
                (self._score_confuser_candidate(gt_mask, prepared_mask), prepared_mask)
            )
        if len(scored_candidates) < self.grpo_confuser_min_candidates:
            return None, {
                "confuser_candidate_count": int(candidate_count),
                "selected_confuser_count": 0,
                "scored_confuser_count": int(len(scored_candidates)),
                "grpo_skip_reason": "missing_confuser_masks",
            }
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        selected_masks = []
        for _, prepared_mask in scored_candidates:
            is_duplicate = any(
                float(self._compute_iou(existing_mask, prepared_mask)) >= self.grpo_confuser_duplicate_iou_threshold
                for existing_mask in selected_masks
            )
            if is_duplicate:
                continue
            selected_masks.append(prepared_mask)
            if len(selected_masks) >= self.grpo_confuser_num_negatives:
                break
        if len(selected_masks) < self.grpo_confuser_num_negatives:
            return None, {
                "confuser_candidate_count": int(candidate_count),
                "selected_confuser_count": int(len(selected_masks)),
                "scored_confuser_count": int(len(scored_candidates)),
                "grpo_skip_reason": "missing_confuser_masks",
            }
        return selected_masks, {
            "confuser_candidate_count": int(candidate_count),
            "selected_confuser_count": int(len(selected_masks)),
            "scored_confuser_count": int(len(scored_candidates)),
            "grpo_skip_reason": None,
        }

    def _build_confuser_mcq_prompt(self, caption):
        option_text = ", ".join(self._grpo_option_letters)
        return (
            "<image>\n"
            f"There are {self.grpo_confuser_num_options} candidate regions in the picture, corresponding to options {option_text}.\n"
            "Read the caption and choose the single region option that best matches it.\n"
            f"Caption: {caption}\n"
            f"Answer with one uppercase letter only: {option_text}."
        )

    def _score_caption_against_mask_options(
        self,
        *,
        model,
        image,
        option_masks,
        caption,
        correct_option_idx,
    ):
        mcq_prompt = self._build_confuser_mcq_prompt(caption)
        answer_completion_ids = [
            self._encode_completion_from_caption(option_text)
            for option_text in self._grpo_option_letters
        ]
        samples = [
            {
                "image": image,
                "prompt_masks": option_masks,
                "prompt_text": mcq_prompt,
                "completion_ids": completion_ids,
                "apply_mask_focus": False,
            }
            for completion_ids in answer_completion_ids
        ]
        with self._temporary_eval_model(model):
            with torch.inference_mode():
                answer_logits = self._forward_sequence_multi_sample_with_model(
                    model,
                    samples,
                )["logits"]
        first_step_logits = answer_logits[:, 0, :]
        answer_token_ids = torch.tensor(self._grpo_option_token_ids, device=first_step_logits.device, dtype=torch.long)
        answer_scores = first_step_logits.gather(dim=-1, index=answer_token_ids.unsqueeze(-1)).squeeze(-1)
        option_probs = torch.softmax(answer_scores, dim=0)
        predicted_option_idx = int(option_probs.argmax().item())
        correct_option_prob = float(option_probs[correct_option_idx].item())
        selected_correct = predicted_option_idx == int(correct_option_idx)
        confuser_iou_weights = torch.zeros_like(option_probs)
        for option_idx, candidate_mask in enumerate(option_masks):
            if option_idx == int(correct_option_idx):
                continue
            confuser_iou_weights[option_idx] = float(self._compute_iou(gt_mask=option_masks[correct_option_idx], pred_mask=candidate_mask))
        confuser_penalty = float((option_probs * confuser_iou_weights).sum().item())
        reward_raw = correct_option_prob - confuser_penalty
        reward = float(max(min(reward_raw, 1.0), -1.0))
        return ConfuserSelectionResult(
            option_probs=option_probs.detach(),
            predicted_option_idx=predicted_option_idx,
            correct_option_idx=int(correct_option_idx),
            reward=reward,
            reward_raw=float(reward_raw),
            selected_correct=selected_correct,
            correct_option_prob=correct_option_prob,
            confuser_iou_weights=confuser_iou_weights.detach(),
            confuser_penalty=confuser_penalty,
        )

    def compute_grpo_loss(
        self,
        *,
        image,
        prompt_masks,
        student_question,
        gt_mask,
        confuser_candidate_masks=None,
        force_dummy=False,
        dummy_reason=None,
        dummy_completion_ids=None,
    ):
        old_policy_model = self.require_old_policy_model("GRPO confuser")
        self._log_realtime_memory("grpo_start")

        def _dummy_grpo_result(skip_reason):
            completion_ids = dummy_completion_ids if dummy_completion_ids is not None else self._build_dummy_completion_ids()
            with self._temporary_eval_model(old_policy_model):
                with torch.inference_mode():
                    old_policy_logits = self._forward_sequence_with_model(
                        old_policy_model,
                        image,
                        prompt_masks,
                        student_question,
                        completion_ids,
                        apply_mask_focus=True,
                    )
                    old_token_log_probs = self._token_log_probs_from_logits(
                        old_policy_logits,
                        completion_ids,
                    )
            old_token_log_probs = self._materialize_autograd_input(old_token_log_probs.detach())
            student_logits = self._forward_sequence_batch_with_model(
                self.student_model,
                image,
                prompt_masks,
                student_question,
                completion_ids,
                apply_mask_focus=True,
            )
            current_token_log_probs = self._token_log_probs_from_logits(
                student_logits,
                completion_ids,
            )
            ratio = torch.exp(current_token_log_probs - old_token_log_probs)
            token_weights = torch.ones_like(ratio, dtype=ratio.dtype)
            sample_losses = -(ratio * 0.0 * token_weights).sum(dim=-1) / token_weights.sum(dim=-1).clamp_min(1.0)
            self._log_realtime_memory("grpo_dummy_result", extra=f"skip_reason={skip_reason}")
            return sample_losses.mean(), {
                "reward_sum": 0.0,
                "reward_count": 0,
                "reward_raw_sum": 0.0,
                "gt_prob_sum": 0.0,
                "confuser_penalty_sum": 0.0,
                "reward_raw_count": 0,
                "rollout_mcq_confidences": [],
                "rollout_rewards": [],
                "rollout_mcq_correct": [],
                "mcq_correct_count": 0,
                "mcq_total_count": 0,
                "mcq_correct_conf_sum": 0.0,
                "skip_reason": str(skip_reason),
                "confuser_candidate_count": 0,
                "selected_confuser_count": 0,
                "scored_confuser_count": 0,
            }

        rollout_entries = []
        rollout_mcq_confidences = []
        rollout_rewards = []
        rollout_mcq_correct = []
        reward_raw_sum = 0.0
        gt_prob_sum = 0.0
        confuser_penalty_sum = 0.0
        if force_dummy:
            return _dummy_grpo_result(dummy_reason or "forced_dummy")
        confuser_masks, confuser_meta = self._select_confuser_masks(
            gt_mask=gt_mask,
            candidate_masks=confuser_candidate_masks,
        )
        self._log_realtime_memory(
            "grpo_after_confuser_select",
            extra=(
                f"candidate_count={confuser_meta.get('confuser_candidate_count')} "
                f"selected_count={confuser_meta.get('selected_confuser_count')} "
                f"scored_count={confuser_meta.get('scored_confuser_count')}"
            ),
        )
        if confuser_masks is None:
            sample_loss, grpo_meta = _dummy_grpo_result("missing_confuser_masks")
            grpo_meta.update(confuser_meta)
            grpo_meta["zero_reward_variance"] = False
            return sample_loss, grpo_meta
        descriptions = self._sample_grpo_descriptions(
            model=old_policy_model,
            image=image,
            prompt_masks=prompt_masks,
            student_question=student_question,
        )
        self._log_realtime_memory("grpo_after_rollout_sample", extra=f"rollout_count={len(descriptions)}")
        for rollout_idx, description in enumerate(descriptions):
            completion_ids = description.completion_ids
            if completion_ids.shape[1] == 0:
                continue
            option_masks = [self._to_numpy_mask(gt_mask), *[self._to_numpy_mask(mask) for mask in confuser_masks]]
            random.shuffle(option_masks)
            correct_option_idx = next(
                idx for idx, candidate_mask in enumerate(option_masks)
                if float(self._compute_iou(gt_mask, candidate_mask)) >= self.grpo_confuser_duplicate_iou_threshold
            )
            selection = self._score_caption_against_mask_options(
                model=old_policy_model,
                image=image,
                option_masks=np.stack(option_masks, axis=0).astype(np.float32),
                caption=description.clean_caption,
                correct_option_idx=correct_option_idx,
            )
            reward_value = selection.reward
            with self._temporary_eval_model(old_policy_model):
                with torch.inference_mode():
                    old_policy_logits = self._forward_sequence_with_model(
                        old_policy_model,
                        image,
                        prompt_masks,
                        student_question,
                        completion_ids,
                        apply_mask_focus=True,
                    )
                    old_token_log_probs = self._token_log_probs_from_logits(
                        old_policy_logits,
                        completion_ids,
                    )
            old_token_log_probs = self._materialize_autograd_input(old_token_log_probs.detach())
            rollout_mcq_confidences.append(selection.correct_option_prob)
            rollout_rewards.append(reward_value)
            rollout_mcq_correct.append(int(selection.selected_correct))
            reward_raw_sum += float(selection.reward_raw)
            gt_prob_sum += float(selection.correct_option_prob)
            confuser_penalty_sum += float(selection.confuser_penalty)
            rollout_entries.append(
                {
                    "completion_ids": completion_ids,
                    "old_token_log_probs": old_token_log_probs,
                    "reward_value": reward_value,
                    "correct_option_prob": selection.correct_option_prob,
                    "selected_correct": bool(selection.selected_correct),
                }
            )
            self._log_realtime_memory(
                "grpo_after_rollout_eval",
                extra=(
                    f"rollout_idx={rollout_idx} completion_len={int(completion_ids.shape[1])} "
                    f"reward={reward_value:.4f} reward_raw={float(selection.reward_raw):.4f} "
                    f"gt_prob={float(selection.correct_option_prob):.4f} "
                    f"confuser_penalty={float(selection.confuser_penalty):.4f} "
                    f"selected_correct={int(selection.selected_correct)}"
                ),
            )

        if not rollout_entries:
            sample_loss, grpo_meta = _dummy_grpo_result("empty_rollout_entries")
            grpo_meta.update(confuser_meta)
            grpo_meta["grpo_skip_reason"] = "empty_rollout_entries"
            grpo_meta["zero_reward_variance"] = False
            return sample_loss, grpo_meta

        reward_tensor = torch.tensor(
            [entry["reward_value"] for entry in rollout_entries],
            device=self.device,
            dtype=rollout_entries[0]["old_token_log_probs"].dtype,
        )
        reward_span = float((reward_tensor.max() - reward_tensor.min()).item())
        zero_reward_variance = reward_span < self.grpo_advantage_eps
        if zero_reward_variance:
            advantages = torch.zeros_like(reward_tensor)
        else:
            reward_std = reward_tensor.std(unbiased=False).clamp_min(self.grpo_advantage_eps)
            advantages = (reward_tensor - reward_tensor.mean()) / reward_std
        self._log_realtime_memory(
            "grpo_after_advantages",
            extra=f"rollout_entries={len(rollout_entries)} zero_reward_variance={int(zero_reward_variance)}",
        )

        completion_pad_id = self.tokenizer.pad_token_id
        if completion_pad_id is None:
            completion_pad_id = self.tokenizer.eos_token_id
        if completion_pad_id is None:
            completion_pad_id = 0
        completion_batch, completion_mask = self._pad_tensor_rows(
            [entry["completion_ids"] for entry in rollout_entries],
            pad_value=completion_pad_id,
            dtype=torch.long,
            device=self.device,
        )
        old_token_log_probs_batch, _ = self._pad_tensor_rows(
            [entry["old_token_log_probs"] for entry in rollout_entries],
            pad_value=0.0,
            dtype=rollout_entries[0]["old_token_log_probs"].dtype,
            device=self.device,
        )
        student_logits = self._forward_sequence_batch_with_model(
            self.student_model,
            image,
            prompt_masks,
            student_question,
            completion_batch,
            apply_mask_focus=True,
        )
        self._log_realtime_memory(
            "grpo_after_current_forward",
            extra=(
                f"completion_batch_shape={tuple(completion_batch.shape)} "
                f"logits_shape={tuple(student_logits.shape)}"
            ),
        )
        current_token_log_probs_batch = self._token_log_probs_from_logits(
            student_logits,
            completion_batch,
        )
        log_ratio = (current_token_log_probs_batch - old_token_log_probs_batch).clamp(min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped_ratio = ratio.clamp(1.0 - self.grpo_clip_eps, 1.0 + self.grpo_clip_eps)
        advantage_batch = advantages.unsqueeze(1)
        surrogate = torch.min(ratio * advantage_batch, clipped_ratio * advantage_batch)
        token_weights = completion_mask.to(dtype=surrogate.dtype)
        sample_losses = -(surrogate * token_weights).sum(dim=-1) / token_weights.sum(dim=-1).clamp_min(1.0)
        if not torch.isfinite(sample_losses).all():
            finite_mask = torch.isfinite(sample_losses)
            if self._should_debug_print():
                bad_count = int((~finite_mask).sum().item())
                total_count = int(sample_losses.numel())
                print(
                    "[Sa2VA_OPSD_V2_WARN] "
                    f"Dropping non-finite grpo sample losses: {bad_count}/{total_count} invalid."
                )
            sample_losses = sample_losses[finite_mask]
        if sample_losses.numel() == 0:
            sample_loss, grpo_meta = _dummy_grpo_result("non_finite_grpo_losses")
            grpo_meta.update(confuser_meta)
            grpo_meta["grpo_skip_reason"] = "non_finite_grpo_losses"
            grpo_meta["zero_reward_variance"] = bool(zero_reward_variance)
            return sample_loss, grpo_meta
        self._log_realtime_memory(
            "grpo_after_loss",
            extra=f"sample_loss_count={int(sample_losses.numel())} mean_loss={float(sample_losses.mean().detach().item()):.6f}",
        )

        return sample_losses.mean(), {
            "reward_sum": float(reward_tensor.sum().item()),
            "reward_count": len(rollout_entries),
            "reward_raw_sum": float(reward_raw_sum),
            "gt_prob_sum": float(gt_prob_sum),
            "confuser_penalty_sum": float(confuser_penalty_sum),
            "reward_raw_count": int(len(rollout_entries)),
            "rollout_mcq_confidences": rollout_mcq_confidences,
            "rollout_rewards": rollout_rewards,
            "rollout_mcq_correct": rollout_mcq_correct,
            "mcq_correct_count": int(sum(rollout_mcq_correct)),
            "mcq_total_count": int(len(rollout_mcq_correct)),
            "mcq_correct_conf_sum": float(
                sum(
                    entry["correct_option_prob"]
                    for entry in rollout_entries
                    if entry["selected_correct"]
                )
            ),
            "confuser_candidate_count": int(confuser_meta.get("confuser_candidate_count", 0)),
            "selected_confuser_count": int(confuser_meta.get("selected_confuser_count", 0)),
            "scored_confuser_count": int(confuser_meta.get("scored_confuser_count", 0)),
            "grpo_skip_reason": "zero_reward_variance" if zero_reward_variance else None,
            "zero_reward_variance": bool(zero_reward_variance),
        }

    def compute_grpo_losses_batch(self, batch_items):
        sample_losses = []
        reward_sum = 0.0
        reward_count = 0
        reward_raw_sum = 0.0
        gt_prob_sum = 0.0
        confuser_penalty_sum = 0.0
        reward_raw_count = 0
        rollout_mcq_confidences = []
        rollout_rewards = []
        rollout_mcq_correct = []
        mcq_correct_count = 0
        mcq_total_count = 0
        mcq_correct_conf_sum = 0.0
        skip_reasons = []
        local_modes = []
        zero_reward_variance_count = 0
        missing_confuser_count = 0
        nonzero_reward_count = 0
        for item in batch_items:
            sample_loss, grpo_meta = self.compute_grpo_loss(
                image=item["image"],
                prompt_masks=item["prompt_masks"],
                student_question=item["student_question"],
                gt_mask=item["gt_mask"],
                confuser_candidate_masks=item.get("confuser_candidate_masks"),
                force_dummy=bool(item.get("is_dummy", False)),
                dummy_reason=item.get("dummy_reason"),
                dummy_completion_ids=item.get("completion_ids"),
            )
            debug_record = item.get("debug_record")
            if isinstance(debug_record, dict):
                debug_record["grpo_skip_reason"] = grpo_meta.get("grpo_skip_reason", grpo_meta.get("skip_reason"))
                debug_record["confuser_candidate_count"] = grpo_meta.get("confuser_candidate_count")
                debug_record["selected_confuser_count"] = grpo_meta.get("selected_confuser_count")
                debug_record["scored_confuser_count"] = grpo_meta.get("scored_confuser_count")
            grpo_mode = str(grpo_meta.get("grpo_skip_reason") or "real")
            if grpo_mode == "zero_reward_variance":
                grpo_mode = "real"
            local_modes.append(grpo_mode)
            reward_sum += grpo_meta["reward_sum"]
            reward_count += grpo_meta["reward_count"]
            reward_raw_sum += float(grpo_meta.get("reward_raw_sum", 0.0))
            gt_prob_sum += float(grpo_meta.get("gt_prob_sum", 0.0))
            confuser_penalty_sum += float(grpo_meta.get("confuser_penalty_sum", 0.0))
            reward_raw_count += int(grpo_meta.get("reward_raw_count", 0))
            zero_reward_variance_count += int(bool(grpo_meta.get("zero_reward_variance", False)))
            missing_confuser_count += int((grpo_meta.get("grpo_skip_reason") or grpo_meta.get("skip_reason")) == "missing_confuser_masks")
            nonzero_reward_count += int(sum(1 for reward in grpo_meta.get("rollout_rewards", []) if float(reward) > 0.0))
            rollout_mcq_confidences.extend(grpo_meta.get("rollout_mcq_confidences", []))
            rollout_rewards.extend(grpo_meta.get("rollout_rewards", []))
            rollout_mcq_correct.extend(grpo_meta.get("rollout_mcq_correct", []))
            mcq_correct_count += int(grpo_meta.get("mcq_correct_count", 0))
            mcq_total_count += int(grpo_meta.get("mcq_total_count", 0))
            mcq_correct_conf_sum += float(grpo_meta.get("mcq_correct_conf_sum", 0.0))
            if sample_loss is not None:
                sample_losses.append(sample_loss * float(item.get("loss_weight", 1.0)))
            else:
                skip_reason = grpo_meta.get("skip_reason")
                if skip_reason:
                    skip_reasons.append(str(skip_reason))
        ddp_fallback_reasons = None
        if self._dist_is_initialized():
            payload = {
                "rank": self._dist_rank(),
                "modes": tuple(local_modes),
            }
            gathered_payloads = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_payloads, payload)
            gathered_mode_signatures = {tuple(item.get("modes", ())) for item in gathered_payloads}
            if len(gathered_mode_signatures) > 1:
                ddp_fallback_reasons = [
                    f"rank{item.get('rank')}:{','.join(item.get('modes', ())) or 'empty'}"
                    for item in gathered_payloads
                ]
        if ddp_fallback_reasons is not None:
            if self._should_debug_print():
                print(
                    "[Sa2VA_OPSD_V2_DDP_DEBUG] "
                    "grpo_mode_mismatch forcing_dummy "
                    f"reasons={ddp_fallback_reasons}",
                    flush=True,
                )
            sample_losses = []
            reward_sum = 0.0
            reward_count = 0
            reward_raw_sum = 0.0
            gt_prob_sum = 0.0
            confuser_penalty_sum = 0.0
            reward_raw_count = 0
            zero_reward_variance_count = 0
            missing_confuser_count = 0
            nonzero_reward_count = 0
            rollout_mcq_confidences = []
            rollout_rewards = []
            rollout_mcq_correct = []
            mcq_correct_count = 0
            mcq_total_count = 0
            mcq_correct_conf_sum = 0.0
            skip_reasons = [f"ddp_grpo_mode_mismatch:{reason}" for reason in ddp_fallback_reasons]
            for item in batch_items:
                sample_loss, grpo_meta = self.compute_grpo_loss(
                    image=item["image"],
                    prompt_masks=item["prompt_masks"],
                    student_question=item["student_question"],
                    gt_mask=item["gt_mask"],
                    confuser_candidate_masks=item.get("confuser_candidate_masks"),
                    force_dummy=True,
                    dummy_reason="ddp_grpo_mode_mismatch",
                    dummy_completion_ids=item.get("completion_ids"),
                )
                debug_record = item.get("debug_record")
                if isinstance(debug_record, dict):
                    debug_record["grpo_skip_reason"] = "ddp_grpo_mode_mismatch"
                reward_sum += grpo_meta["reward_sum"]
                reward_count += grpo_meta["reward_count"]
                reward_raw_sum += float(grpo_meta.get("reward_raw_sum", 0.0))
                gt_prob_sum += float(grpo_meta.get("gt_prob_sum", 0.0))
                confuser_penalty_sum += float(grpo_meta.get("confuser_penalty_sum", 0.0))
                reward_raw_count += int(grpo_meta.get("reward_raw_count", 0))
                zero_reward_variance_count += int(bool(grpo_meta.get("zero_reward_variance", False)))
                missing_confuser_count += int((grpo_meta.get("grpo_skip_reason") or grpo_meta.get("skip_reason")) == "missing_confuser_masks")
                nonzero_reward_count += int(sum(1 for reward in grpo_meta.get("rollout_rewards", []) if float(reward) > 0.0))
                rollout_mcq_confidences.extend(grpo_meta.get("rollout_mcq_confidences", []))
                rollout_rewards.extend(grpo_meta.get("rollout_rewards", []))
                rollout_mcq_correct.extend(grpo_meta.get("rollout_mcq_correct", []))
                mcq_correct_count += int(grpo_meta.get("mcq_correct_count", 0))
                mcq_total_count += int(grpo_meta.get("mcq_total_count", 0))
                mcq_correct_conf_sum += float(grpo_meta.get("mcq_correct_conf_sum", 0.0))
                if sample_loss is not None:
                    sample_losses.append(sample_loss * float(item.get("loss_weight", 1.0)))
        if not sample_losses:
            return self._placeholder_loss_vector(
                len(batch_items),
                reason=f"grpo-empty-sample-losses skip_reasons={','.join(skip_reasons) or 'unknown'}",
            ), {
                "reward_sum": reward_sum,
                "reward_count": reward_count,
                "reward_raw_sum": reward_raw_sum,
                "gt_prob_sum": gt_prob_sum,
                "confuser_penalty_sum": confuser_penalty_sum,
                "reward_raw_count": reward_raw_count,
                "rollout_mcq_confidences": rollout_mcq_confidences,
                "rollout_rewards": rollout_rewards,
                "rollout_mcq_correct": rollout_mcq_correct,
                "mcq_correct_count": mcq_correct_count,
                "mcq_total_count": mcq_total_count,
                "mcq_correct_conf_sum": mcq_correct_conf_sum,
                "skip_reasons": skip_reasons,
                "zero_reward_variance_count": zero_reward_variance_count,
                "missing_confuser_count": missing_confuser_count,
                "nonzero_reward_count": nonzero_reward_count,
            }
        return torch.stack(sample_losses), {
            "reward_sum": reward_sum,
            "reward_count": reward_count,
            "reward_raw_sum": reward_raw_sum,
            "gt_prob_sum": gt_prob_sum,
            "confuser_penalty_sum": confuser_penalty_sum,
            "reward_raw_count": reward_raw_count,
            "rollout_mcq_confidences": rollout_mcq_confidences,
            "rollout_rewards": rollout_rewards,
            "rollout_mcq_correct": rollout_mcq_correct,
            "mcq_correct_count": mcq_correct_count,
            "mcq_total_count": mcq_total_count,
            "mcq_correct_conf_sum": mcq_correct_conf_sum,
            "skip_reasons": skip_reasons,
            "zero_reward_variance_count": zero_reward_variance_count,
            "missing_confuser_count": missing_confuser_count,
            "nonzero_reward_count": nonzero_reward_count,
        }

    def forward(self, data, data_samples=None, mode="loss"):
        del data_samples, mode
        images = data["images"]
        prompt_masks_batch = data["prompt_masks"]
        student_questions = data["student_questions"]
        gt_masks = data["gt_masks"]
        confuser_candidate_masks_batch = data.get("confuser_candidate_masks") or [None] * len(images)
        sample_keys = data.get("sample_keys") or data.get("npz_paths") or [None] * len(images)
        routes = data.get("routes") or [None] * len(images)
        batch_route = self._resolve_batch_route(routes)

        zero = self._zero_scalar(dtype=next(self.student_model.parameters()).dtype)
        total_loss = None
        total_iou = 0.0
        routed_count = 0
        optimized_count = 0
        nonempty_gt_count = 0
        nonempty_caption_count = 0
        caption_token_sum = 0.0
        description_ok_count = 0
        description_empty_count = 0
        description_truncated_count = 0
        description_seg_style_count = 0
        description_raw_seg_style_count = 0
        reconstruct_ok_count = 0
        reconstruct_failed_count = 0
        reconstruct_skip_count = 0
        empty_gt_mask_count = 0
        seg_correct_count = 0
        teacher_regenerate_count = 0
        on_policy_distill_count = 0
        grpo_positive_count = 0
        regen_loss_count = 0
        onpolicy_loss_count = 0
        grpo_loss_count = 0
        total_regen_ce = None
        total_onpolicy_jsd = None
        total_grpo = None
        grpo_reward_sum = 0.0
        grpo_reward_count = 0
        grpo_reward_raw_sum = 0.0
        grpo_reward_raw_count = 0
        grpo_gt_prob_sum = 0.0
        grpo_confuser_penalty_sum = 0.0
        grpo_rollout_mcq_confidences = []
        grpo_rollout_rewards = []
        grpo_rollout_mcq_correct = []
        grpo_mcq_correct_count = 0
        grpo_mcq_total_count = 0
        grpo_mcq_correct_conf_sum = 0.0
        recovery_caption_count = 0
        invalid_caption_penalty_count = 0
        hard_reconstruct_failure_count = 0
        teacher_regenerate_ce_applied_count = 0
        teacher_regenerate_suppressed_count = 0
        teacher_regenerate_verified_count = 0
        teacher_regenerate_rejected_count = 0
        teacher_regenerate_verified_iou_sum = 0.0
        teacher_reconstruct_ok_count = 0
        teacher_positive_gain_count = 0
        teacher_iou_gain_sum = 0.0
        teacher_regenerate_analysis_count = 0
        teacher_difference_context_nontrivial_count = 0
        teacher_regenerate_verification_valid_count = 0
        teacher_regenerate_verification_iou_sum = 0.0
        teacher_problem_valid_count = 0
        teacher_direction_valid_count = 0
        teacher_reason_valid_count = 0
        teacher_reason_coarse_count = 0
        teacher_diagnosis_valid_count = 0
        teacher_dlc_valid_count = 0
        caption_mode_failure_count = 0
        onpolicy_blocked_by_seg_style_count = 0
        grpo_blocked_by_seg_style_count = 0
        teacher_recovery_seg_style_count = 0
        teacher_recovery_seg_style_success_count = 0
        recovery_ce_applied_count = 0
        recovery_suppressed_count = 0
        reconstruct_invalid_caption_skip_count = 0
        reconstruct_empty_prediction_masks_count = 0
        scene_spill_caption_count = 0
        low_density_long_caption_count = 0
        detail_sufficient_caption_count = 0
        generic_caption_count = 0
        repetitive_caption_count = 0
        grpo_zero_reward_variance_count = 0
        grpo_nonzero_reward_count = 0
        grpo_missing_confuser_count = 0
        last_sample_key = None
        last_caption = ""
        last_teacher_prompt = ""
        last_route = batch_route or ""
        rank_debug_records = []

        regen_entries = []
        onpolicy_entries = []
        grpo_entries = []

        for image, prompt_masks, student_question, gt_mask, confuser_candidate_masks, sample_key, route_from_manifest in zip(
            images, prompt_masks_batch, student_questions, gt_masks, confuser_candidate_masks_batch, sample_keys, routes
        ):
            gt_mask_np = self._to_numpy_mask(gt_mask)
            empty_gt_mask = int(gt_mask_np.sum()) == 0
            if empty_gt_mask:
                empty_gt_mask_count += 1
                self._debug_sample(
                    sample_key=sample_key,
                    route="empty_gt_skip",
                    student_question=student_question,
                    raw_prediction="",
                    caption="",
                    description_status="empty_gt_mask",
                    reconstruct_question=None,
                    raw_reconstruct_prediction="",
                    reconstruct_status="skipped_invalid_description",
                    seg_token_count=0,
                    prediction_masks_count=0,
                    pred_mask=None,
                    gt_mask=gt_mask_np,
                    iou=0.0,
                    empty_gt_mask=True,
                )
                continue

            nonempty_gt_count += 1

            description = self.generate_description(image=image, mask_prompts=prompt_masks, student_question=student_question)
            student_caption_trainable = self._is_caption_trainable_for_student_losses(description)
            caption_token_count = self._caption_token_count(description.clean_caption)
            if caption_token_count > 0:
                nonempty_caption_count += 1
                caption_token_sum += caption_token_count
            if self._is_caption_content_sufficient(description.clean_caption):
                detail_sufficient_caption_count += 1
            if self._is_overly_generic_caption(description.clean_caption):
                generic_caption_count += 1
            if self._has_repetitive_caption_pattern(description.clean_caption):
                repetitive_caption_count += 1
            if self._caption_scene_spill_hit_count(description.clean_caption) >= 2:
                scene_spill_caption_count += 1
            if self._is_low_density_long_caption(description.clean_caption):
                low_density_long_caption_count += 1
            if description.status == "ok":
                description_ok_count += 1
            elif description.status == "empty":
                description_empty_count += 1
                invalid_caption_penalty_count += 1
            elif description.status == "truncated_caption":
                description_truncated_count += 1
                invalid_caption_penalty_count += 1
            elif description.status == "seg_style_answer":
                description_seg_style_count += 1
                invalid_caption_penalty_count += 1
            if description.raw_failure_mode == "seg_style_answer":
                description_raw_seg_style_count += 1
            if self._is_caption_mode_failure_status(description.status):
                caption_mode_failure_count += 1

            reconstruction = None
            if description.status == "ok":
                reconstruction = self.reconstruct_mask(
                    image=image,
                    caption=description.clean_caption,
                    description_status=description.status,
                    spatial_hint=self._coarse_spatial_hint(gt_mask_np),
                    gt_mask=gt_mask_np,
                )
            elif self.enable_invalid_caption_recovery:
                reconstruction = self._invalid_reconstruction_placeholder("skipped_invalid_description")
            reconstruct_status = "missing_reconstruction_result" if reconstruction is None else reconstruction.status
            reconstruct_question = None if reconstruction is None else reconstruction.question
            raw_reconstruct_prediction = "" if reconstruction is None else reconstruction.raw_prediction
            prediction_masks_count = 0 if reconstruction is None else reconstruction.prediction_masks_count
            seg_token_count = 0 if reconstruction is None else reconstruction.seg_token_count
            pred_mask = None if reconstruction is None else reconstruction.pred_mask
            if pred_mask is None:
                if reconstruct_status == "skipped_invalid_description":
                    reconstruct_invalid_caption_skip_count += 1
                elif reconstruct_status == "empty_prediction_masks":
                    reconstruct_empty_prediction_masks_count += 1
                reconstruct_skip_count += 1
                iou = 0.0
                ref_mask_np = self._zero_ref_mask_like(gt_mask_np)
            else:
                iou = self._compute_iou(gt_mask_np, pred_mask)
                ref_mask_np = self._to_numpy_mask(pred_mask)
                routed_count += 1
                if iou >= 0.5:
                    seg_correct_count += 1
            online_route = self._route_from_iou(iou)
            loss_family = self._resolve_loss_family(batch_route, route_from_manifest=route_from_manifest, online_route=online_route)
            if reconstruct_status == "ok":
                reconstruct_ok_count += 1
            elif reconstruct_status not in {"ok", "skipped_invalid_description"}:
                reconstruct_failed_count += 1

            self._debug_sample(
                sample_key=sample_key,
                route=f"{loss_family} (online={online_route})" if loss_family != online_route else loss_family,
                student_question=student_question,
                raw_prediction=description.raw_prediction,
                caption=description.clean_caption,
                description_status=description.status,
                raw_caption_failure_mode=description.raw_failure_mode,
                clean_description_status=description.clean_status,
                student_caption_trainable=student_caption_trainable,
                reconstruct_question=reconstruct_question,
                raw_reconstruct_prediction=raw_reconstruct_prediction,
                reconstruct_status=reconstruct_status,
                seg_token_count=seg_token_count,
                prediction_masks_count=prediction_masks_count,
                pred_mask=pred_mask,
                gt_mask=gt_mask_np,
                iou=iou,
                empty_gt_mask=False,
            )

            is_recovery_case = description.status != "ok" or pred_mask is None
            if is_recovery_case:
                recovery_caption_count += 1
            max_regen_count = max(int(np.ceil(self.max_teacher_regenerate_fraction * max(nonempty_gt_count, 1))), 1)
            max_recovery_count = max(int(np.ceil(self.max_recovery_fraction * max(nonempty_gt_count, 1))), 1)
            allow_teacher_ce = teacher_regenerate_ce_applied_count < max_regen_count
            if is_recovery_case:
                allow_teacher_ce = allow_teacher_ce and (recovery_ce_applied_count < max_recovery_count)
            effective_reconstruction = reconstruction or self._invalid_reconstruction_placeholder("skipped_invalid_description")
            teacher_analysis = self._attempt_teacher_regenerate_analysis(
                image=image,
                gt_mask_np=gt_mask_np,
                ref_mask_np=ref_mask_np,
                student_question=student_question,
                description=description,
                reconstruction=effective_reconstruction,
                iou=iou,
                allow_teacher_ce=allow_teacher_ce,
            )
            teacher_reconstruct_ok = teacher_analysis["teacher_reconstruct_ok"]
            teacher_gate_passed = teacher_analysis["teacher_gate_passed"]
            teacher_iou_plain = teacher_analysis["teacher_iou_plain"]
            teacher_completion_len = teacher_analysis["teacher_completion_len"]
            teacher_regenerate = teacher_analysis["teacher_regenerate"]
            teacher_difference_context_nontrivial = bool(
                teacher_analysis.get("teacher_difference_context_nontrivial", False)
            )
            teacher_diagnosis_valid = bool(teacher_analysis.get("teacher_diagnosis_valid", False))
            teacher_verification_caption_status = str(teacher_analysis.get("teacher_verification_caption_status", "empty"))
            teacher_verification_caption = str(teacher_analysis.get("teacher_verification_caption", ""))
            teacher_dlc = str(teacher_analysis.get("teacher_dlc", ""))
            teacher_verification_iou = float(teacher_analysis.get("teacher_verification_iou", 0.0))
            teacher_dlc_valid = bool(teacher_analysis.get("teacher_dlc_valid", False))
            teacher_target_summary = str(teacher_analysis.get("teacher_target_summary", ""))
            teacher_distractor_summary = str(teacher_analysis.get("teacher_distractor_summary", ""))
            teacher_target_only_evidence = str(teacher_analysis.get("teacher_target_only_evidence", ""))
            teacher_distractor_only_evidence = str(teacher_analysis.get("teacher_distractor_only_evidence", ""))
            teacher_difference_focus = str(teacher_analysis.get("teacher_difference_focus", ""))
            teacher_likely_drift_reason = str(teacher_analysis.get("teacher_likely_drift_reason", ""))
            teacher_caption_problem = str(teacher_analysis.get("teacher_caption_problem", ""))
            teacher_correction_direction = str(teacher_analysis.get("teacher_correction_direction", ""))
            teacher_reason = str(teacher_analysis.get("teacher_reason", ""))
            teacher_single_stage_raw = str(teacher_analysis.get("teacher_single_stage_raw", ""))
            teacher_problem_raw = str(teacher_analysis.get("teacher_problem_raw", ""))
            teacher_direction_raw = str(teacher_analysis.get("teacher_direction_raw", ""))
            teacher_reason_raw = str(teacher_analysis.get("teacher_reason_raw", ""))
            teacher_problem_valid = bool(teacher_analysis.get("teacher_problem_valid", False))
            teacher_direction_valid = bool(teacher_analysis.get("teacher_direction_valid", False))
            teacher_reason_valid = bool(teacher_analysis.get("teacher_reason_valid", False))
            teacher_reason_is_coarse = bool(teacher_analysis.get("teacher_reason_is_coarse", False))
            teacher_pipeline_stop_stage = str(teacher_analysis.get("teacher_pipeline_stop_stage", "difference_context"))
            teacher_pipeline_failure_reason = str(teacher_analysis.get("teacher_pipeline_failure_reason", ""))
            teacher_pipeline_mode = str(teacher_analysis.get("teacher_pipeline_mode", "single_stage"))
            teacher_diagnosis_retry_count = int(teacher_analysis.get("teacher_diagnosis_retry_count", 0))
            teacher_dlc_candidate_count = int(teacher_analysis.get("teacher_dlc_candidate_count", 0))
            teacher_dlc_valid_candidate_count = int(teacher_analysis.get("teacher_dlc_valid_candidate_count", 0))
            teacher_dlc_selected_by = str(teacher_analysis.get("teacher_dlc_selected_by", ""))
            teacher_verification_used = bool(teacher_analysis.get("teacher_verification_used", False))
            teacher_fallback_used = bool(teacher_analysis.get("teacher_fallback_used", False))
            teacher_fallback_reason = str(teacher_analysis.get("teacher_fallback_reason", ""))
            teacher_selected_caption_source = str(teacher_analysis.get("teacher_selected_caption_source", ""))
            teacher_dlc_candidate_scores = tuple(teacher_analysis.get("teacher_dlc_candidate_scores", ()))
            if allow_teacher_ce:
                teacher_regenerate_analysis_count += 1
            if teacher_difference_context_nontrivial:
                teacher_difference_context_nontrivial_count += 1
            if teacher_problem_valid:
                teacher_problem_valid_count += 1
            if teacher_direction_valid:
                teacher_direction_valid_count += 1
            if teacher_reason_valid:
                teacher_reason_valid_count += 1
            if teacher_diagnosis_valid:
                teacher_diagnosis_valid_count += 1
            if teacher_dlc_valid:
                teacher_dlc_valid_count += 1
            if teacher_verification_caption_status == "ok":
                teacher_regenerate_verification_valid_count += 1
            teacher_regenerate_verification_iou_sum += teacher_verification_iou
            if teacher_reason_is_coarse:
                teacher_reason_coarse_count += 1
            caption_mode_failure = bool(teacher_analysis.get("caption_mode_failure", False))
            if caption_mode_failure:
                teacher_recovery_seg_style_count += 1
            teacher_iou_gain = self._teacher_regenerate_iou_improvement(iou, teacher_iou_plain)
            if teacher_reconstruct_ok:
                teacher_reconstruct_ok_count += 1
                if caption_mode_failure:
                    teacher_recovery_seg_style_success_count += 1
            if teacher_iou_gain > 0.0:
                teacher_positive_gain_count += 1
                teacher_iou_gain_sum += float(teacher_iou_gain)
            if allow_teacher_ce:
                if teacher_reconstruct_ok and teacher_gate_passed:
                    teacher_regenerate_ce_applied_count += 1
                    teacher_regenerate_verified_count += 1
                    teacher_regenerate_verified_iou_sum += float(teacher_iou_plain or 0.0)
                    if is_recovery_case:
                        recovery_ce_applied_count += 1
                else:
                    teacher_regenerate_suppressed_count += 1
                    teacher_regenerate_rejected_count += 1
                    hard_reconstruct_failure_count += 1
                    if is_recovery_case:
                        recovery_suppressed_count += 1
            elif is_recovery_case:
                teacher_regenerate_suppressed_count += 1
                hard_reconstruct_failure_count += 1
                recovery_suppressed_count += 1

            teacher_prompt = ""
            dummy_reason = None
            is_dummy = False
            sample_debug_record = {
                "sample_key": sample_key,
                "manifest_route": route_from_manifest,
                "loss_family": loss_family,
                "online_route": online_route,
                "reconstruct_status": reconstruct_status,
                "iou": float(iou),
                "allow_teacher_ce": bool(allow_teacher_ce),
                "teacher_reconstruct_ok": teacher_reconstruct_ok,
                "teacher_gate_passed": teacher_gate_passed,
                "teacher_iou_plain": teacher_iou_plain,
                "teacher_completion_len": teacher_completion_len,
                "teacher_difference_context_nontrivial": teacher_difference_context_nontrivial,
                "teacher_diagnosis_valid": teacher_diagnosis_valid,
                "teacher_verification_caption_status": teacher_verification_caption_status,
                "teacher_verification_caption": teacher_verification_caption,
                "teacher_verification_iou": teacher_verification_iou,
                "teacher_dlc": teacher_dlc,
                "target_summary": teacher_target_summary,
                "distractor_summary": teacher_distractor_summary,
                "shared_evidence": str(teacher_analysis.get("teacher_shared_evidence", "")),
                "target_only_evidence": teacher_target_only_evidence,
                "distractor_only_evidence": teacher_distractor_only_evidence,
                "difference_focus": str(teacher_analysis.get("teacher_difference_focus", "")),
                "likely_drift_reason": teacher_likely_drift_reason,
                "caption_problem": teacher_caption_problem,
                "correction_direction": teacher_correction_direction,
                "reason": teacher_reason,
                "single_stage_raw": teacher_single_stage_raw,
                "structured_diagnosis_raw": str(teacher_analysis.get("teacher_structured_diagnosis_raw", "")),
                "repair_raw": str(teacher_analysis.get("teacher_repair_raw", "")),
                "problem_raw": teacher_problem_raw,
                "direction_raw": teacher_direction_raw,
                "reason_raw": teacher_reason_raw,
                "teacher_problem_valid": teacher_problem_valid,
                "teacher_direction_valid": teacher_direction_valid,
                "teacher_reason_valid": teacher_reason_valid,
                "teacher_reason_is_coarse": teacher_reason_is_coarse,
                "teacher_problem_failure_reason": str(teacher_analysis.get("teacher_problem_failure_reason", "")),
                "teacher_direction_failure_reason": str(teacher_analysis.get("teacher_direction_failure_reason", "")),
                "teacher_reason_failure_reason": str(teacher_analysis.get("teacher_reason_failure_reason", "")),
                "teacher_diagnosis_failure_reason": str(teacher_analysis.get("teacher_diagnosis_failure_reason", "")),
                "teacher_pipeline_mode": teacher_pipeline_mode,
                "teacher_diagnosis_retry_count": teacher_diagnosis_retry_count,
                "teacher_dlc_candidate_count": teacher_dlc_candidate_count,
                "teacher_dlc_valid_candidate_count": teacher_dlc_valid_candidate_count,
                "teacher_dlc_selected_by": teacher_dlc_selected_by,
                "teacher_verification_used": teacher_verification_used,
                "teacher_fallback_used": teacher_fallback_used,
                "teacher_fallback_reason": teacher_fallback_reason,
                "teacher_selected_caption_source": teacher_selected_caption_source,
                "teacher_dlc_candidate_scores": teacher_dlc_candidate_scores,
                "teacher_pipeline_stop_stage": teacher_pipeline_stop_stage,
                "teacher_pipeline_failure_reason": teacher_pipeline_failure_reason,
                "raw_caption_failure_mode": description.raw_failure_mode,
                "clean_description_status": description.clean_status,
                "student_caption_trainable": bool(student_caption_trainable),
                "is_dummy": False,
                "dummy_reason": None,
                "entry_added": False,
                "loss_branch": loss_family,
                "grpo_skip_reason": None,
                "confuser_candidate_count": None,
                "selected_confuser_count": None,
                "scored_confuser_count": None,
            }
            if loss_family == TEACHER_REGENERATE_ROUTE:
                teacher_regenerate_count += 1
                teacher_prompt = self._route_prompt_tag(loss_family)
                regen_completion = None if teacher_regenerate is None else teacher_regenerate.detailed_completion_ids
                if regen_completion is None or regen_completion.shape[1] == 0:
                    dummy_reason = "teacher_empty_completion" if teacher_regenerate is not None else "teacher_not_available"
                elif not (teacher_reconstruct_ok and teacher_gate_passed):
                    dummy_reason = "teacher_gate_failed"
                if dummy_reason is None:
                    regen_entries.append(
                        {
                            "image": image,
                            "prompt_masks": prompt_masks,
                            "student_question": student_question,
                            "completion_ids": regen_completion,
                            "is_dummy": False,
                            "loss_weight": 1.0,
                            "dummy_reason": None,
                        }
                    )
                    sample_debug_record["entry_added"] = True
                else:
                    is_dummy = True
                sample_debug_record["is_dummy"] = bool(dummy_reason is not None)
                sample_debug_record["dummy_reason"] = dummy_reason
                last_caption = (
                    teacher_regenerate.detailed_caption
                    if teacher_regenerate is not None and teacher_regenerate.detailed_caption
                    else description.clean_caption
                )
            elif loss_family == ON_POLICY_DISTILL_ROUTE:
                on_policy_distill_count += 1
                teacher_fields = self._build_training_teacher_fields(
                    route=loss_family,
                    iou=iou,
                )
                teacher_prompt = self.build_teacher_privileged_prompt_v3(
                    student_question=student_question,
                    student_caption=description.clean_caption,
                    description_status=description.status,
                    reconstruction=effective_reconstruction,
                    iou=iou,
                    gt_mask=gt_mask_np,
                    ref_mask=ref_mask_np,
                    teacher_fields=teacher_fields,
                )
                teacher_prompt_masks = self._build_teacher_prompt_masks(gt_mask_np, ref_mask_np)
                onpolicy_completion = description.completion_ids
                if onpolicy_completion.shape[1] == 0:
                    dummy_reason = "empty_completion"
                elif not student_caption_trainable:
                    dummy_reason = f"invalid_caption:{description.status}"
                    if description.raw_failure_mode == "seg_style_answer" or description.status == "seg_style_answer":
                        onpolicy_blocked_by_seg_style_count += 1
                elif pred_mask is None:
                    dummy_reason = f"missing_pred_mask:{reconstruct_status}"
                if dummy_reason is None:
                    onpolicy_entries.append(
                        {
                            "image": image,
                            "prompt_masks": prompt_masks,
                            "student_question": student_question,
                            "teacher_prompt": teacher_prompt,
                            "completion_ids": onpolicy_completion,
                            "teacher_prompt_masks": teacher_prompt_masks,
                            "iou": iou,
                            "is_dummy": False,
                            "loss_weight": 1.0,
                            "dummy_reason": None,
                        }
                    )
                    sample_debug_record["entry_added"] = True
                else:
                    is_dummy = True
                sample_debug_record["is_dummy"] = bool(dummy_reason is not None)
                sample_debug_record["dummy_reason"] = dummy_reason
                last_caption = description.clean_caption
            else:
                grpo_positive_count += 1
                if not student_caption_trainable:
                    dummy_reason = f"invalid_caption:{description.status}"
                    if description.raw_failure_mode == "seg_style_answer" or description.status == "seg_style_answer":
                        grpo_blocked_by_seg_style_count += 1
                elif pred_mask is None:
                    dummy_reason = f"missing_pred_mask:{reconstruct_status}"
                if dummy_reason is None:
                    grpo_entries.append(
                        {
                            "image": image,
                            "prompt_masks": prompt_masks,
                            "student_question": student_question,
                            "gt_mask": gt_mask_np,
                            "confuser_candidate_masks": confuser_candidate_masks,
                            "completion_ids": None,
                            "is_dummy": False,
                            "loss_weight": 1.0,
                            "dummy_reason": None,
                            "debug_record": sample_debug_record,
                        }
                    )
                    sample_debug_record["entry_added"] = True
                else:
                    is_dummy = True
                sample_debug_record["is_dummy"] = bool(dummy_reason is not None)
                sample_debug_record["dummy_reason"] = dummy_reason
                teacher_prompt = self._route_prompt_tag(loss_family)
                last_caption = description.clean_caption

            rank_debug_records.append(sample_debug_record)
            last_sample_key = sample_key
            last_route = loss_family
            last_teacher_prompt = teacher_prompt
            total_iou += iou

        if regen_entries:
            regen_losses = self.compute_regenerate_alignment_losses_batch(regen_entries)
            if regen_losses.numel() > 0:
                regen_loss_count += int(regen_losses.shape[0])
                regen_loss_sum = regen_losses.sum()
                total_regen_ce = regen_loss_sum if total_regen_ce is None else total_regen_ce + regen_loss_sum
                total_loss = regen_loss_sum if total_loss is None else total_loss + regen_loss_sum
                optimized_count += int(regen_losses.shape[0])

        if onpolicy_entries:
            onpolicy_losses = self.compute_onpolicy_distill_losses_batch(onpolicy_entries)
            if onpolicy_losses.numel() > 0:
                onpolicy_loss_count += int(onpolicy_losses.shape[0])
                onpolicy_loss_sum = onpolicy_losses.sum()
                total_onpolicy_jsd = onpolicy_loss_sum if total_onpolicy_jsd is None else total_onpolicy_jsd + onpolicy_loss_sum
                total_loss = onpolicy_loss_sum if total_loss is None else total_loss + onpolicy_loss_sum
                optimized_count += int(onpolicy_losses.shape[0])

        if grpo_entries:
            grpo_losses, grpo_meta = self.compute_grpo_losses_batch(grpo_entries)
            grpo_reward_sum += grpo_meta["reward_sum"]
            grpo_reward_count += grpo_meta["reward_count"]
            grpo_reward_raw_sum += float(grpo_meta.get("reward_raw_sum", 0.0))
            grpo_reward_raw_count += int(grpo_meta.get("reward_raw_count", 0))
            grpo_gt_prob_sum += float(grpo_meta.get("gt_prob_sum", 0.0))
            grpo_confuser_penalty_sum += float(grpo_meta.get("confuser_penalty_sum", 0.0))
            grpo_zero_reward_variance_count += int(grpo_meta.get("zero_reward_variance_count", 0))
            grpo_nonzero_reward_count += int(grpo_meta.get("nonzero_reward_count", 0))
            grpo_missing_confuser_count += int(grpo_meta.get("missing_confuser_count", 0))
            grpo_rollout_mcq_confidences.extend(grpo_meta.get("rollout_mcq_confidences", []))
            grpo_rollout_rewards.extend(grpo_meta.get("rollout_rewards", []))
            grpo_rollout_mcq_correct.extend(grpo_meta.get("rollout_mcq_correct", []))
            grpo_mcq_correct_count += int(grpo_meta.get("mcq_correct_count", 0))
            grpo_mcq_total_count += int(grpo_meta.get("mcq_total_count", 0))
            grpo_mcq_correct_conf_sum += float(grpo_meta.get("mcq_correct_conf_sum", 0.0))
            if grpo_losses.numel() > 0:
                grpo_loss_count += int(grpo_losses.shape[0])
                grpo_loss_sum = grpo_losses.sum()
                total_grpo = grpo_loss_sum if total_grpo is None else total_grpo + grpo_loss_sum
                total_loss = grpo_loss_sum if total_loss is None else total_loss + grpo_loss_sum
                optimized_count += int(grpo_losses.shape[0])

        entry_count = len(regen_entries) + len(onpolicy_entries) + len(grpo_entries)
        dummy_entry_count = sum(int(item.get("is_dummy", False)) for item in regen_entries)
        dummy_entry_count += sum(int(item.get("is_dummy", False)) for item in onpolicy_entries)
        dummy_entry_count += sum(int(item.get("is_dummy", False)) for item in grpo_entries)
        real_entry_count = entry_count - dummy_entry_count
        dummy_reasons = self._merge_dummy_reasons(regen_entries + onpolicy_entries + grpo_entries)

        self._log_ddp_route_alignment_debug(
            {
                "rank": self._dist_rank(),
                "batch_route": batch_route,
                "loss_family": self._resolve_loss_family(batch_route),
                "entry_count": entry_count,
                "real_entry_count": real_entry_count,
                "dummy_entry_count": dummy_entry_count,
                "dummy_reasons": dummy_reasons,
                "regen_entry_count": len(regen_entries),
                "onpolicy_entry_count": len(onpolicy_entries),
                "grpo_entry_count": len(grpo_entries),
                "regen_loss_count": regen_loss_count,
                "onpolicy_loss_count": onpolicy_loss_count,
                "grpo_loss_count": grpo_loss_count,
                "optimized_count": optimized_count,
                "records": rank_debug_records,
            }
        )

        self._cumulative_valid_count += routed_count
        self._cumulative_loss_count += optimized_count
        self._cumulative_nonempty_gt_count += nonempty_gt_count
        self._cumulative_nonempty_caption_count += nonempty_caption_count
        self._cumulative_caption_token_sum += caption_token_sum
        self._cumulative_description_ok_count += description_ok_count
        self._cumulative_description_empty_count += description_empty_count
        self._cumulative_description_truncated_count += description_truncated_count
        self._cumulative_description_seg_style_count += description_seg_style_count
        self._cumulative_reconstruct_ok_count += reconstruct_ok_count
        self._cumulative_reconstruct_failed_count += reconstruct_failed_count
        self._cumulative_reconstruct_skip_count += reconstruct_skip_count
        self._cumulative_reconstruct_invalid_caption_skip_count += reconstruct_invalid_caption_skip_count
        self._cumulative_reconstruct_empty_prediction_masks_count += reconstruct_empty_prediction_masks_count
        self._cumulative_empty_gt_mask_count += empty_gt_mask_count
        self._cumulative_seg_correct_count += seg_correct_count
        self._cumulative_teacher_regenerate_count += teacher_regenerate_count
        self._cumulative_on_policy_distill_count += on_policy_distill_count
        self._cumulative_grpo_positive_count += grpo_positive_count
        self._cumulative_regen_loss_count += regen_loss_count
        self._cumulative_onpolicy_loss_count += onpolicy_loss_count
        self._cumulative_grpo_loss_count += grpo_loss_count
        self._cumulative_iou_sum += total_iou
        self._cumulative_grpo_reward_sum += grpo_reward_sum
        self._cumulative_grpo_reward_count += grpo_reward_count
        self._cumulative_grpo_mcq_correct_count += grpo_mcq_correct_count
        self._cumulative_grpo_mcq_count += grpo_mcq_total_count
        self._cumulative_grpo_mcq_correct_conf_sum += grpo_mcq_correct_conf_sum
        self._cumulative_recovery_caption_count += recovery_caption_count
        self._cumulative_invalid_caption_penalty_count += invalid_caption_penalty_count
        self._cumulative_hard_reconstruct_failure_count += hard_reconstruct_failure_count
        self._cumulative_teacher_regenerate_ce_applied_count += teacher_regenerate_ce_applied_count
        self._cumulative_teacher_regenerate_suppressed_count += teacher_regenerate_suppressed_count
        self._cumulative_teacher_regenerate_verified_count += teacher_regenerate_verified_count
        self._cumulative_teacher_regenerate_rejected_count += teacher_regenerate_rejected_count
        self._cumulative_teacher_regenerate_verified_iou_sum += teacher_regenerate_verified_iou_sum
        self._cumulative_teacher_regenerate_analysis_count += teacher_regenerate_analysis_count
        self._cumulative_teacher_difference_context_nontrivial_count += (
            teacher_difference_context_nontrivial_count
        )
        self._cumulative_teacher_problem_valid_count += teacher_problem_valid_count
        self._cumulative_teacher_direction_valid_count += teacher_direction_valid_count
        self._cumulative_teacher_reason_valid_count += teacher_reason_valid_count
        self._cumulative_teacher_reason_coarse_count += teacher_reason_coarse_count
        self._cumulative_teacher_regenerate_verification_valid_count += teacher_regenerate_verification_valid_count
        self._cumulative_teacher_regenerate_verification_iou_sum += teacher_regenerate_verification_iou_sum
        self._cumulative_teacher_diagnosis_valid_count += teacher_diagnosis_valid_count
        self._cumulative_teacher_dlc_valid_count += teacher_dlc_valid_count
        self._cumulative_recovery_ce_applied_count += recovery_ce_applied_count
        self._cumulative_recovery_suppressed_count += recovery_suppressed_count
        self._cumulative_scene_spill_caption_count += scene_spill_caption_count
        self._cumulative_low_density_long_caption_count += low_density_long_caption_count
        self._cumulative_detail_sufficient_caption_count += detail_sufficient_caption_count
        self._cumulative_generic_caption_count += generic_caption_count
        self._cumulative_repetitive_caption_count += repetitive_caption_count
        if total_loss is not None:
            self._cumulative_total_loss_sum += float(total_loss.detach().item())
        if total_regen_ce is not None:
            self._cumulative_regen_ce_sum += float(total_regen_ce.detach().item())
        if total_onpolicy_jsd is not None:
            self._cumulative_onpolicy_jsd_sum += float(total_onpolicy_jsd.detach().item())
        if total_grpo is not None:
            self._cumulative_grpo_sum += float(total_grpo.detach().item())

        window_payload = {
            "valid_count": routed_count,
            "loss_count": optimized_count,
            "nonempty_gt_count": nonempty_gt_count,
            "nonempty_caption_count": nonempty_caption_count,
            "description_ok_count": description_ok_count,
            "description_empty_count": description_empty_count,
            "description_truncated_count": description_truncated_count,
            "description_seg_style_count": description_seg_style_count,
            "description_raw_seg_style_count": description_raw_seg_style_count,
            "reconstruct_ok_count": reconstruct_ok_count,
            "reconstruct_failed_count": reconstruct_failed_count,
            "reconstruct_skip_count": reconstruct_skip_count,
            "reconstruct_invalid_caption_skip_count": reconstruct_invalid_caption_skip_count,
            "reconstruct_empty_prediction_masks_count": reconstruct_empty_prediction_masks_count,
            "empty_gt_mask_count": empty_gt_mask_count,
            "seg_correct_count": seg_correct_count,
            "teacher_regenerate_count": teacher_regenerate_count,
            "on_policy_distill_count": on_policy_distill_count,
            "grpo_positive_count": grpo_positive_count,
            "regen_loss_count": regen_loss_count,
            "onpolicy_loss_count": onpolicy_loss_count,
            "grpo_loss_count": grpo_loss_count,
            "grpo_reward_count": grpo_reward_count,
            "grpo_reward_raw_count": grpo_reward_raw_count,
            "grpo_mcq_correct_count": grpo_mcq_correct_count,
            "grpo_mcq_count": grpo_mcq_total_count,
            "recovery_caption_count": recovery_caption_count,
            "invalid_caption_penalty_count": invalid_caption_penalty_count,
            "hard_reconstruct_failure_count": hard_reconstruct_failure_count,
            "teacher_regenerate_ce_applied_count": teacher_regenerate_ce_applied_count,
            "teacher_regenerate_suppressed_count": teacher_regenerate_suppressed_count,
            "teacher_regenerate_verified_count": teacher_regenerate_verified_count,
            "teacher_regenerate_rejected_count": teacher_regenerate_rejected_count,
            "teacher_reconstruct_ok_count": teacher_reconstruct_ok_count,
            "teacher_positive_gain_count": teacher_positive_gain_count,
            "teacher_regenerate_analysis_count": teacher_regenerate_analysis_count,
            "teacher_difference_context_nontrivial_count": teacher_difference_context_nontrivial_count,
            "teacher_problem_valid_count": teacher_problem_valid_count,
            "teacher_direction_valid_count": teacher_direction_valid_count,
            "teacher_reason_valid_count": teacher_reason_valid_count,
            "teacher_reason_coarse_count": teacher_reason_coarse_count,
            "teacher_regenerate_verification_valid_count": teacher_regenerate_verification_valid_count,
            "teacher_diagnosis_valid_count": teacher_diagnosis_valid_count,
            "teacher_dlc_valid_count": teacher_dlc_valid_count,
            "caption_mode_failure_count": caption_mode_failure_count,
            "onpolicy_blocked_by_seg_style_count": onpolicy_blocked_by_seg_style_count,
            "grpo_blocked_by_seg_style_count": grpo_blocked_by_seg_style_count,
            "teacher_recovery_seg_style_count": teacher_recovery_seg_style_count,
            "teacher_recovery_seg_style_success_count": teacher_recovery_seg_style_success_count,
            "grpo_zero_reward_variance_count": grpo_zero_reward_variance_count,
            "grpo_nonzero_reward_count": grpo_nonzero_reward_count,
            "grpo_missing_confuser_count": grpo_missing_confuser_count,
            "recovery_ce_applied_count": recovery_ce_applied_count,
            "recovery_suppressed_count": recovery_suppressed_count,
            "scene_spill_caption_count": scene_spill_caption_count,
            "low_density_long_caption_count": low_density_long_caption_count,
            "detail_sufficient_caption_count": detail_sufficient_caption_count,
            "generic_caption_count": generic_caption_count,
            "repetitive_caption_count": repetitive_caption_count,
            "iou_sum": total_iou,
            "caption_token_sum": caption_token_sum,
            "total_loss_sum": 0.0 if total_loss is None else float(total_loss.detach().item()),
            "regen_ce_sum": 0.0 if total_regen_ce is None else float(total_regen_ce.detach().item()),
            "onpolicy_jsd_sum": 0.0 if total_onpolicy_jsd is None else float(total_onpolicy_jsd.detach().item()),
            "grpo_sum": 0.0 if total_grpo is None else float(total_grpo.detach().item()),
            "grpo_reward_sum": grpo_reward_sum,
            "grpo_reward_raw_sum": grpo_reward_raw_sum,
            "grpo_gt_prob_sum": grpo_gt_prob_sum,
            "grpo_confuser_penalty_sum": grpo_confuser_penalty_sum,
            "grpo_mcq_correct_conf_sum": grpo_mcq_correct_conf_sum,
            "teacher_regenerate_verified_iou_sum": teacher_regenerate_verified_iou_sum,
            "teacher_regenerate_verification_iou_sum": teacher_regenerate_verification_iou_sum,
            "teacher_iou_gain_sum": teacher_iou_gain_sum,
        }
        self._append_window_metrics(window_payload)
        window_totals = self._aggregate_window_metrics()
        window_valid_count = max(window_totals["valid_count"], 1)
        window_loss_count = max(window_totals["loss_count"], 1)
        window_nonempty_gt_count = max(window_totals["nonempty_gt_count"], 1)
        window_nonempty_caption_count = max(window_totals["nonempty_caption_count"], 1)
        window_route_count = max(
            window_totals["teacher_regenerate_count"]
            + window_totals["on_policy_distill_count"]
            + window_totals["grpo_positive_count"],
            1,
        )
        window_verifier_iou = window_totals["iou_sum"] / window_valid_count
        window_seg_correct_rate = window_totals["seg_correct_count"] / window_valid_count
        window_all_sample_seg_success_rate = window_totals["reconstruct_ok_count"] / window_nonempty_gt_count
        window_all_sample_seg_correct_rate = window_totals["seg_correct_count"] / window_nonempty_gt_count
        window_avg_caption_tokens = window_totals["caption_token_sum"] / window_nonempty_caption_count
        window_teacher_regenerate_rate = window_totals["teacher_regenerate_count"] / window_route_count
        window_on_policy_distill_rate = window_totals["on_policy_distill_count"] / window_route_count
        window_grpo_positive_rate = window_totals["grpo_positive_count"] / window_route_count
        window_grpo_reward_mean = window_totals["grpo_reward_sum"] / max(window_totals["grpo_reward_count"], 1)
        window_grpo_reward_raw_mean = window_totals["grpo_reward_raw_sum"] / max(window_totals["grpo_reward_raw_count"], 1)
        window_grpo_gt_prob_mean = window_totals["grpo_gt_prob_sum"] / max(window_totals["grpo_reward_raw_count"], 1)
        window_grpo_confuser_penalty_mean = (
            window_totals["grpo_confuser_penalty_sum"] / max(window_totals["grpo_reward_raw_count"], 1)
        )
        window_grpo_mcq_acc = window_totals["grpo_mcq_correct_count"] / max(window_totals["grpo_mcq_count"], 1)
        window_grpo_mcq_correct_conf_mean = (
            window_totals["grpo_mcq_correct_conf_sum"] / max(window_totals["grpo_mcq_correct_count"], 1)
        )
        window_caption_invalid_rate = window_totals["invalid_caption_penalty_count"] / window_nonempty_gt_count
        window_caption_empty_rate = window_totals["description_empty_count"] / window_nonempty_gt_count
        window_caption_truncated_rate = window_totals["description_truncated_count"] / window_nonempty_gt_count
        window_caption_seg_style_rate = window_totals["description_seg_style_count"] / window_nonempty_gt_count
        window_caption_seg_style_rate_raw = window_totals["description_raw_seg_style_count"] / window_nonempty_gt_count
        window_caption_mode_failure_rate = window_totals["caption_mode_failure_count"] / window_nonempty_gt_count
        window_reconstruct_invalid_caption_skip_rate = (
            window_totals["reconstruct_invalid_caption_skip_count"] / window_nonempty_gt_count
        )
        window_reconstruct_empty_prediction_masks_rate = (
            window_totals["reconstruct_empty_prediction_masks_count"] / window_nonempty_gt_count
        )
        window_detail_sufficient_caption_rate = window_totals["detail_sufficient_caption_count"] / window_nonempty_gt_count
        window_scene_spill_caption_rate = window_totals["scene_spill_caption_count"] / window_nonempty_gt_count
        window_teacher_regenerate_gate_pass_rate = (
            window_totals["teacher_regenerate_verified_count"]
            / max(window_totals["teacher_regenerate_verified_count"] + window_totals["teacher_regenerate_rejected_count"], 1)
        )
        window_teacher_regenerate_verified_iou_mean = (
            window_totals["teacher_regenerate_verified_iou_sum"] / max(window_totals["teacher_regenerate_verified_count"], 1)
        )
        window_teacher_difference_context_nontrivial_rate = (
            window_totals["teacher_difference_context_nontrivial_count"]
            / max(window_totals["teacher_regenerate_analysis_count"], 1)
        )
        window_teacher_problem_valid_rate = (
            window_totals["teacher_problem_valid_count"]
            / max(window_totals["teacher_regenerate_analysis_count"], 1)
        )
        window_teacher_direction_valid_rate = (
            window_totals["teacher_direction_valid_count"]
            / max(window_totals["teacher_regenerate_analysis_count"], 1)
        )
        window_teacher_reason_valid_rate = (
            window_totals["teacher_reason_valid_count"]
            / max(window_totals["teacher_regenerate_analysis_count"], 1)
        )
        window_teacher_reason_coarse_rate = (
            window_totals["teacher_reason_coarse_count"]
            / max(window_totals["teacher_regenerate_analysis_count"], 1)
        )
        window_teacher_regenerate_verification_caption_valid_rate = (
            window_totals["teacher_regenerate_verification_valid_count"]
            / max(window_totals["teacher_regenerate_analysis_count"], 1)
        )
        window_teacher_regenerate_verification_iou_mean = (
            window_totals["teacher_regenerate_verification_iou_sum"]
            / max(window_totals["teacher_regenerate_analysis_count"], 1)
        )
        window_teacher_diagnosis_valid_rate = (
            window_totals["teacher_diagnosis_valid_count"] / max(window_totals["teacher_regenerate_analysis_count"], 1)
        )
        window_teacher_dlc_valid_rate = (
            window_totals["teacher_dlc_valid_count"] / max(window_totals["teacher_regenerate_analysis_count"], 1)
        )
        window_teacher_verification_gate_pass_rate = window_teacher_regenerate_gate_pass_rate
        window_loss_opsd_total = window_totals["total_loss_sum"] / window_loss_count
        window_regen_ce = window_totals["regen_ce_sum"] / max(window_totals["regen_loss_count"], 1)
        window_onpolicy_jsd = window_totals["onpolicy_jsd_sum"] / max(window_totals["onpolicy_loss_count"], 1)
        window_grpo = window_totals["grpo_sum"] / max(window_totals["grpo_loss_count"], 1)
        window_teacher_positive_gain_rate = (
            window_totals["teacher_positive_gain_count"] / max(window_totals["teacher_reconstruct_ok_count"], 1)
        )
        window_teacher_iou_gain_mean = (
            window_totals["teacher_iou_gain_sum"] / max(window_totals["teacher_positive_gain_count"], 1)
        )
        window_grpo_zero_reward_variance_rate = (
            window_totals["grpo_zero_reward_variance_count"] / max(window_totals["grpo_loss_count"], 1)
        )
        window_grpo_nonzero_reward_rate = (
            window_totals["grpo_nonzero_reward_count"] / max(window_totals["grpo_reward_count"], 1)
        )
        window_grpo_missing_confuser_rate = (
            window_totals["grpo_missing_confuser_count"] / max(window_totals["grpo_positive_count"], 1)
        )

        cumulative_valid_count = max(self._cumulative_valid_count, 1)
        cumulative_loss_count = max(self._cumulative_loss_count, 1)
        cumulative_nonempty_gt_count = max(self._cumulative_nonempty_gt_count, 1)
        cumulative_nonempty_caption_count = max(self._cumulative_nonempty_caption_count, 1)
        cumulative_verifier_iou = self._cumulative_iou_sum / cumulative_valid_count
        cumulative_seg_correct_rate = self._cumulative_seg_correct_count / cumulative_valid_count
        cumulative_all_sample_reconstruct_attempt_rate = self._cumulative_valid_count / cumulative_nonempty_gt_count
        cumulative_all_sample_seg_success_rate = self._cumulative_reconstruct_ok_count / cumulative_nonempty_gt_count
        cumulative_all_sample_seg_correct_rate = self._cumulative_seg_correct_count / cumulative_nonempty_gt_count
        cumulative_valid_caption_cond_seg_correct_rate = (
            self._cumulative_seg_correct_count / max(self._cumulative_description_ok_count, 1)
        )
        cumulative_loss_opsd_total = self._cumulative_total_loss_sum / cumulative_loss_count
        cumulative_nonempty_caption_rate = (
            self._cumulative_nonempty_caption_count / cumulative_nonempty_gt_count
        )
        cumulative_valid_caption_rate = self._cumulative_description_ok_count / cumulative_nonempty_gt_count
        cumulative_avg_caption_tokens = (
            self._cumulative_caption_token_sum / cumulative_nonempty_caption_count
        )
        cumulative_route_count = max(
            self._cumulative_teacher_regenerate_count
            + self._cumulative_on_policy_distill_count
            + self._cumulative_grpo_positive_count,
            1,
        )
        cumulative_teacher_regenerate_rate = self._cumulative_teacher_regenerate_count / cumulative_route_count
        cumulative_on_policy_distill_rate = self._cumulative_on_policy_distill_count / cumulative_route_count
        cumulative_grpo_positive_rate = self._cumulative_grpo_positive_count / cumulative_route_count
        cumulative_regen_ce = self._cumulative_regen_ce_sum / max(self._cumulative_regen_loss_count, 1)
        cumulative_onpolicy_jsd = self._cumulative_onpolicy_jsd_sum / max(self._cumulative_onpolicy_loss_count, 1)
        cumulative_grpo = self._cumulative_grpo_sum / max(self._cumulative_grpo_loss_count, 1)
        cumulative_grpo_reward_mean = self._cumulative_grpo_reward_sum / max(self._cumulative_grpo_reward_count, 1)
        cumulative_grpo_mcq_acc = (
            self._cumulative_grpo_mcq_correct_count / max(self._cumulative_grpo_mcq_count, 1)
        )
        cumulative_grpo_mcq_correct_conf_mean = (
            self._cumulative_grpo_mcq_correct_conf_sum / max(self._cumulative_grpo_mcq_correct_count, 1)
        )
        cumulative_recovery_caption_rate = self._cumulative_recovery_caption_count / cumulative_nonempty_gt_count
        cumulative_invalid_caption_penalty_rate = self._cumulative_invalid_caption_penalty_count / cumulative_nonempty_gt_count
        cumulative_caption_empty_rate = self._cumulative_description_empty_count / cumulative_nonempty_gt_count
        cumulative_caption_truncated_rate = self._cumulative_description_truncated_count / cumulative_nonempty_gt_count
        cumulative_caption_seg_style_rate = self._cumulative_description_seg_style_count / cumulative_nonempty_gt_count
        cumulative_reconstruct_invalid_caption_skip_rate = (
            self._cumulative_reconstruct_invalid_caption_skip_count / cumulative_nonempty_gt_count
        )
        cumulative_reconstruct_empty_prediction_masks_rate = (
            self._cumulative_reconstruct_empty_prediction_masks_count / cumulative_nonempty_gt_count
        )
        cumulative_detail_sufficient_caption_rate = self._cumulative_detail_sufficient_caption_count / cumulative_nonempty_gt_count
        cumulative_generic_caption_rate = self._cumulative_generic_caption_count / cumulative_nonempty_gt_count
        cumulative_repetitive_caption_rate = self._cumulative_repetitive_caption_count / cumulative_nonempty_gt_count
        cumulative_scene_spill_caption_rate = self._cumulative_scene_spill_caption_count / cumulative_nonempty_gt_count
        cumulative_low_density_long_caption_rate = self._cumulative_low_density_long_caption_count / cumulative_nonempty_gt_count
        cumulative_teacher_regenerate_gate_pass_rate = (
            self._cumulative_teacher_regenerate_verified_count
            / max(
                self._cumulative_teacher_regenerate_verified_count
                + self._cumulative_teacher_regenerate_rejected_count,
                1,
            )
        )
        cumulative_teacher_regenerate_verified_iou_mean = (
            self._cumulative_teacher_regenerate_verified_iou_sum
            / max(self._cumulative_teacher_regenerate_verified_count, 1)
        )
        cumulative_teacher_difference_context_nontrivial_rate = (
            self._cumulative_teacher_difference_context_nontrivial_count
            / max(self._cumulative_teacher_regenerate_analysis_count, 1)
        )
        cumulative_teacher_problem_valid_rate = (
            self._cumulative_teacher_problem_valid_count
            / max(self._cumulative_teacher_regenerate_analysis_count, 1)
        )
        cumulative_teacher_direction_valid_rate = (
            self._cumulative_teacher_direction_valid_count
            / max(self._cumulative_teacher_regenerate_analysis_count, 1)
        )
        cumulative_teacher_reason_valid_rate = (
            self._cumulative_teacher_reason_valid_count
            / max(self._cumulative_teacher_regenerate_analysis_count, 1)
        )
        cumulative_teacher_reason_coarse_rate = (
            self._cumulative_teacher_reason_coarse_count
            / max(self._cumulative_teacher_regenerate_analysis_count, 1)
        )
        cumulative_teacher_regenerate_verification_caption_valid_rate = (
            self._cumulative_teacher_regenerate_verification_valid_count
            / max(self._cumulative_teacher_regenerate_analysis_count, 1)
        )
        cumulative_teacher_regenerate_verification_iou_mean = (
            self._cumulative_teacher_regenerate_verification_iou_sum
            / max(self._cumulative_teacher_regenerate_analysis_count, 1)
        )
        cumulative_teacher_diagnosis_valid_rate = (
            self._cumulative_teacher_diagnosis_valid_count
            / max(self._cumulative_teacher_regenerate_analysis_count, 1)
        )
        cumulative_teacher_dlc_valid_rate = (
            self._cumulative_teacher_dlc_valid_count / max(self._cumulative_teacher_regenerate_analysis_count, 1)
        )
        cumulative_teacher_verification_caption_valid_rate = (
            self._cumulative_teacher_regenerate_verification_valid_count
            / max(self._cumulative_teacher_regenerate_analysis_count, 1)
        )
        cumulative_teacher_verification_gate_pass_rate = cumulative_teacher_regenerate_gate_pass_rate

        if optimized_count == 0:
            self._log_pre_return_debug(
                batch_route=batch_route,
                last_route=last_route,
                optimized_count=optimized_count,
                regen_loss_count=regen_loss_count,
                onpolicy_loss_count=onpolicy_loss_count,
                grpo_loss_count=grpo_loss_count,
                total_loss=total_loss,
                total_regen_ce=total_regen_ce,
                total_onpolicy_jsd=total_onpolicy_jsd,
                total_grpo=total_grpo,
                rank_debug_records=rank_debug_records,
            )
            metrics = {
                "loss_opsd_total": self._zero_scalar(requires_grad=True, dtype=zero.dtype),
                "opsd_regen_ce": zero,
                "opsd_onpolicy_jsd": zero,
                "opsd_grpo": zero,
                "grpo_reward_mean": zero,
                "grpo_reward_raw_mean": zero,
                "grpo_gt_prob_mean": zero,
                "grpo_confuser_penalty_mean": zero,
                "grpo_mcq_acc": zero,
                "grpo_mcq_correct_conf_mean": zero,
                "grpo_group_size": self._metric_tensor(float(self.grpo_group_size), zero.dtype),
                "verifier_iou": self._metric_tensor(window_verifier_iou, zero.dtype),
                "seg_correct_rate": self._metric_tensor(window_seg_correct_rate, zero.dtype),
                "all_sample_seg_success_rate": self._metric_tensor(window_all_sample_seg_success_rate, zero.dtype),
                "all_sample_seg_correct_rate": self._metric_tensor(window_all_sample_seg_correct_rate, zero.dtype),
                "avg_caption_tokens": self._metric_tensor(window_avg_caption_tokens, zero.dtype),
                "teacher_regenerate_rate": self._metric_tensor(window_teacher_regenerate_rate, zero.dtype),
                "on_policy_distill_rate": self._metric_tensor(window_on_policy_distill_rate, zero.dtype),
                "grpo_positive_rate": self._metric_tensor(window_grpo_positive_rate, zero.dtype),
                "caption_invalid_rate": self._metric_tensor(window_caption_invalid_rate, zero.dtype),
                "caption_empty_rate": self._metric_tensor(window_caption_empty_rate, zero.dtype),
                "caption_truncated_rate": self._metric_tensor(window_caption_truncated_rate, zero.dtype),
                "caption_seg_style_rate": self._metric_tensor(window_caption_seg_style_rate, zero.dtype),
                "caption_seg_style_rate_raw": self._metric_tensor(window_caption_seg_style_rate_raw, zero.dtype),
                "caption_mode_failure_rate": self._metric_tensor(window_caption_mode_failure_rate, zero.dtype),
                "reconstruct_invalid_caption_skip_rate": self._metric_tensor(
                    window_reconstruct_invalid_caption_skip_rate, zero.dtype
                ),
                "reconstruct_empty_prediction_masks_rate": self._metric_tensor(
                    window_reconstruct_empty_prediction_masks_rate, zero.dtype
                ),
                "detail_sufficient_caption_rate": self._metric_tensor(window_detail_sufficient_caption_rate, zero.dtype),
                "scene_spill_caption_rate": self._metric_tensor(window_scene_spill_caption_rate, zero.dtype),
                "teacher_regenerate_ce_applied_count": self._metric_tensor(window_totals["teacher_regenerate_ce_applied_count"], zero.dtype),
                "teacher_regenerate_suppressed_count": self._metric_tensor(window_totals["teacher_regenerate_suppressed_count"], zero.dtype),
                "teacher_regenerate_verified_count": self._metric_tensor(window_totals["teacher_regenerate_verified_count"], zero.dtype),
                "teacher_regenerate_rejected_count": self._metric_tensor(window_totals["teacher_regenerate_rejected_count"], zero.dtype),
                "teacher_regenerate_gate_pass_rate": self._metric_tensor(
                    window_teacher_regenerate_gate_pass_rate, zero.dtype
                ),
                "teacher_regenerate_verified_iou_mean": self._metric_tensor(
                    window_teacher_regenerate_verified_iou_mean, zero.dtype
                ),
                "teacher_difference_context_nontrivial_rate": self._metric_tensor(
                    window_teacher_difference_context_nontrivial_rate, zero.dtype
                ),
                "teacher_problem_valid_rate": self._metric_tensor(
                    window_teacher_problem_valid_rate, zero.dtype
                ),
                "teacher_direction_valid_rate": self._metric_tensor(
                    window_teacher_direction_valid_rate, zero.dtype
                ),
                "teacher_reason_valid_rate": self._metric_tensor(
                    window_teacher_reason_valid_rate, zero.dtype
                ),
                "teacher_reason_coarse_rate": self._metric_tensor(
                    window_teacher_reason_coarse_rate, zero.dtype
                ),
                "teacher_regenerate_verification_caption_valid_rate": self._metric_tensor(
                    window_teacher_regenerate_verification_caption_valid_rate, zero.dtype
                ),
                "teacher_regenerate_verification_iou_mean": self._metric_tensor(
                    window_teacher_regenerate_verification_iou_mean, zero.dtype
                ),
                "teacher_diagnosis_valid_rate": self._metric_tensor(
                    window_teacher_diagnosis_valid_rate, zero.dtype
                ),
                "teacher_dlc_valid_rate": self._metric_tensor(window_teacher_dlc_valid_rate, zero.dtype),
                "teacher_verification_caption_valid_rate": self._metric_tensor(
                    window_teacher_regenerate_verification_caption_valid_rate, zero.dtype
                ),
                "teacher_verification_gate_pass_rate": self._metric_tensor(
                    window_teacher_verification_gate_pass_rate, zero.dtype
                ),
                "teacher_regenerate_dlc_ce_applied_count": self._metric_tensor(
                    window_totals["teacher_regenerate_ce_applied_count"], zero.dtype
                ),
                "onpolicy_blocked_by_seg_style_count": self._metric_tensor(
                    window_totals["onpolicy_blocked_by_seg_style_count"], zero.dtype
                ),
                "grpo_blocked_by_seg_style_count": self._metric_tensor(
                    window_totals["grpo_blocked_by_seg_style_count"], zero.dtype
                ),
                "teacher_recovery_seg_style_count": self._metric_tensor(
                    window_totals["teacher_recovery_seg_style_count"], zero.dtype
                ),
                "teacher_recovery_seg_style_success_count": self._metric_tensor(
                    window_totals["teacher_recovery_seg_style_success_count"], zero.dtype
                ),
                "teacher_positive_gain_rate": self._metric_tensor(window_teacher_positive_gain_rate, zero.dtype),
                "teacher_iou_gain_mean": self._metric_tensor(window_teacher_iou_gain_mean, zero.dtype),
                "grpo_zero_reward_variance_rate": self._metric_tensor(window_grpo_zero_reward_variance_rate, zero.dtype),
                "grpo_nonzero_reward_rate": self._metric_tensor(window_grpo_nonzero_reward_rate, zero.dtype),
                "grpo_missing_confuser_rate": self._metric_tensor(window_grpo_missing_confuser_rate, zero.dtype),
            }
            return metrics

        avg_total_loss = total_loss / optimized_count
        avg_regen_ce = zero if total_regen_ce is None else total_regen_ce / max(regen_loss_count, 1)
        avg_onpolicy_jsd = zero if total_onpolicy_jsd is None else total_onpolicy_jsd / max(onpolicy_loss_count, 1)
        avg_grpo = zero if total_grpo is None else total_grpo / max(grpo_loss_count, 1)
        batch_route_count = max(
            teacher_regenerate_count + on_policy_distill_count + grpo_positive_count,
            1,
        )
        batch_teacher_regenerate_rate = teacher_regenerate_count / batch_route_count
        batch_on_policy_distill_rate = on_policy_distill_count / batch_route_count
        batch_grpo_positive_rate = grpo_positive_count / batch_route_count
        batch_grpo_reward_mean = grpo_reward_sum / max(grpo_reward_count, 1)
        batch_teacher_regenerate_gate_pass_rate = (
            teacher_regenerate_verified_count
            / max(teacher_regenerate_verified_count + teacher_regenerate_rejected_count, 1)
        )
        batch_teacher_regenerate_verified_iou_mean = (
            teacher_regenerate_verified_iou_sum / max(teacher_regenerate_verified_count, 1)
        )
        batch_grpo_mcq_acc = grpo_mcq_correct_count / max(grpo_mcq_total_count, 1)
        batch_grpo_mcq_correct_conf_mean = grpo_mcq_correct_conf_sum / max(grpo_mcq_correct_count, 1)
        grpo_rollout_conf_text = self._format_float_list(grpo_rollout_mcq_confidences)
        grpo_rollout_rewards_text = self._format_float_list(grpo_rollout_rewards)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(
                    f"[Sa2VA_OPSD_V2] last_sample_key={last_sample_key!r} last_route={last_route} "
                    f"batch_avg_iou={total_iou / max(routed_count, 1):.4f} "
                    f"batch_seg_correct_rate={seg_correct_count / max(routed_count, 1):.4f} "
                    f"batch_regen_rate={batch_teacher_regenerate_rate:.4f} "
                    f"batch_onpolicy_rate={batch_on_policy_distill_rate:.4f} "
                    f"batch_grpo_rate={batch_grpo_positive_rate:.4f} "
                    f"teacher_regen_verified={teacher_regenerate_verified_count} "
                    f"teacher_regen_rejected={teacher_regenerate_rejected_count} "
                    f"teacher_regen_gate_pass_rate={batch_teacher_regenerate_gate_pass_rate:.4f} "
                    f"teacher_regen_verified_iou_mean={batch_teacher_regenerate_verified_iou_mean:.4f} "
                    f"teacher_regenerate_ce_applied={teacher_regenerate_ce_applied_count} "
                    f"teacher_regenerate_suppressed={teacher_regenerate_suppressed_count} "
                    f"window_teacher_difference_context_nontrivial_rate={window_teacher_difference_context_nontrivial_rate:.4f} "
                    f"window_teacher_problem_valid_rate={window_teacher_problem_valid_rate:.4f} "
                    f"window_teacher_direction_valid_rate={window_teacher_direction_valid_rate:.4f} "
                    f"window_teacher_reason_valid_rate={window_teacher_reason_valid_rate:.4f} "
                    f"window_teacher_reason_coarse_rate={window_teacher_reason_coarse_rate:.4f} "
                    f"window_teacher_regenerate_verification_caption_valid_rate={window_teacher_regenerate_verification_caption_valid_rate:.4f} "
                    f"window_teacher_regenerate_verification_iou_mean={window_teacher_regenerate_verification_iou_mean:.4f} "
                    f"window_teacher_diagnosis_valid_rate={window_teacher_diagnosis_valid_rate:.4f} "
                    f"window_teacher_dlc_valid_rate={window_teacher_dlc_valid_rate:.4f} "
                    f"window_teacher_verification_gate_pass_rate={window_teacher_verification_gate_pass_rate:.4f} "
                    f"teacher_verification_caption={teacher_verification_caption!r} "
                    f"teacher_dlc={teacher_dlc!r} "
                    f"teacher_caption_problem={teacher_caption_problem!r} "
                    f"teacher_correction_direction={teacher_correction_direction!r} "
                    f"teacher_reason={teacher_reason!r} "
                    f"teacher_single_stage_raw={teacher_single_stage_raw!r} "
                    f"teacher_difference_focus={teacher_difference_focus!r} "
                    f"teacher_problem_raw={teacher_problem_raw!r} "
                    f"teacher_direction_raw={teacher_direction_raw!r} "
                    f"teacher_reason_raw={teacher_reason_raw!r} "
                    f"teacher_problem_valid={teacher_problem_valid} "
                    f"teacher_direction_valid={teacher_direction_valid} "
                    f"teacher_reason_valid={teacher_reason_valid} "
                    f"window_avg_caption_tokens={window_avg_caption_tokens:.2f} "
                    f"window_caption_seg_style_rate_raw={window_caption_seg_style_rate_raw:.4f} "
                    f"window_caption_mode_failure_rate={window_caption_mode_failure_rate:.4f} "
                    f"window_onpolicy_blocked_by_seg_style_count={window_totals['onpolicy_blocked_by_seg_style_count']} "
                    f"window_grpo_blocked_by_seg_style_count={window_totals['grpo_blocked_by_seg_style_count']} "
                    f"window_teacher_recovery_seg_style_count={window_totals['teacher_recovery_seg_style_count']} "
                    f"window_teacher_recovery_seg_style_success_count={window_totals['teacher_recovery_seg_style_success_count']} "
                    f"cumulative_teacher_difference_context_nontrivial_rate={cumulative_teacher_difference_context_nontrivial_rate:.4f} "
                    f"cumulative_teacher_problem_valid_rate={cumulative_teacher_problem_valid_rate:.4f} "
                    f"cumulative_teacher_direction_valid_rate={cumulative_teacher_direction_valid_rate:.4f} "
                    f"cumulative_teacher_reason_valid_rate={cumulative_teacher_reason_valid_rate:.4f} "
                    f"cumulative_teacher_reason_coarse_rate={cumulative_teacher_reason_coarse_rate:.4f} "
                    f"cumulative_teacher_regenerate_verification_caption_valid_rate={cumulative_teacher_regenerate_verification_caption_valid_rate:.4f} "
                    f"cumulative_teacher_regenerate_verification_iou_mean={cumulative_teacher_regenerate_verification_iou_mean:.4f} "
                    f"cumulative_teacher_diagnosis_valid_rate={cumulative_teacher_diagnosis_valid_rate:.4f} "
                    f"cumulative_teacher_dlc_valid_rate={cumulative_teacher_dlc_valid_rate:.4f} "
                    f"cumulative_teacher_verification_caption_valid_rate={cumulative_teacher_verification_caption_valid_rate:.4f} "
                    f"cumulative_teacher_verification_gate_pass_rate={cumulative_teacher_verification_gate_pass_rate:.4f} "
                    f"cumulative_teacher_regenerate_dlc_ce_applied_count={self._cumulative_teacher_regenerate_ce_applied_count} "
                    f"window_teacher_positive_gain_rate={window_teacher_positive_gain_rate:.4f} "
                    f"window_teacher_iou_gain_mean={window_teacher_iou_gain_mean:.4f} "
                    f"window_grpo_zero_reward_variance_rate={window_grpo_zero_reward_variance_rate:.4f} "
                    f"window_grpo_nonzero_reward_rate={window_grpo_nonzero_reward_rate:.4f} "
                    f"window_grpo_missing_confuser_rate={window_grpo_missing_confuser_rate:.4f} "
                    f"window_grpo_reward_raw_mean={window_grpo_reward_raw_mean:.4f} "
                    f"window_grpo_gt_prob_mean={window_grpo_gt_prob_mean:.4f} "
                    f"window_grpo_confuser_penalty_mean={window_grpo_confuser_penalty_mean:.4f} "
                    f"grpo_mcq_acc={batch_grpo_mcq_acc:.4f} "
                    f"grpo_mcq_correct_conf_mean={batch_grpo_mcq_correct_conf_mean:.4f} "
                    f"grpo_rollout_confidences={grpo_rollout_conf_text} "
                    f"grpo_rollout_rewards={grpo_rollout_rewards_text} "
                    f"{self._format_cuda_memory_stats()}"
                )
        else:
            print(
                f"[Sa2VA_OPSD_V2] last_sample_key={last_sample_key!r} last_route={last_route} "
                f"batch_avg_iou={total_iou / max(routed_count, 1):.4f} "
                f"batch_seg_correct_rate={seg_correct_count / max(routed_count, 1):.4f} "
                f"batch_regen_rate={batch_teacher_regenerate_rate:.4f} "
                f"batch_onpolicy_rate={batch_on_policy_distill_rate:.4f} "
                f"batch_grpo_rate={batch_grpo_positive_rate:.4f} "
                f"teacher_regen_verified={teacher_regenerate_verified_count} "
                f"teacher_regen_rejected={teacher_regenerate_rejected_count} "
                f"teacher_regen_gate_pass_rate={batch_teacher_regenerate_gate_pass_rate:.4f} "
                f"teacher_regen_verified_iou_mean={batch_teacher_regenerate_verified_iou_mean:.4f} "
                f"teacher_regenerate_ce_applied={teacher_regenerate_ce_applied_count} "
                f"teacher_regenerate_suppressed={teacher_regenerate_suppressed_count} "
                f"window_teacher_difference_context_nontrivial_rate={window_teacher_difference_context_nontrivial_rate:.4f} "
                f"window_teacher_problem_valid_rate={window_teacher_problem_valid_rate:.4f} "
                f"window_teacher_direction_valid_rate={window_teacher_direction_valid_rate:.4f} "
                f"window_teacher_reason_valid_rate={window_teacher_reason_valid_rate:.4f} "
                f"window_teacher_reason_coarse_rate={window_teacher_reason_coarse_rate:.4f} "
                f"window_teacher_regenerate_verification_caption_valid_rate={window_teacher_regenerate_verification_caption_valid_rate:.4f} "
                    f"window_teacher_regenerate_verification_iou_mean={window_teacher_regenerate_verification_iou_mean:.4f} "
                    f"window_teacher_diagnosis_valid_rate={window_teacher_diagnosis_valid_rate:.4f} "
                    f"window_teacher_dlc_valid_rate={window_teacher_dlc_valid_rate:.4f} "
                    f"window_teacher_verification_gate_pass_rate={window_teacher_verification_gate_pass_rate:.4f} "
                    f"teacher_verification_caption={teacher_verification_caption!r} "
                    f"teacher_dlc={teacher_dlc!r} "
                    f"teacher_caption_problem={teacher_caption_problem!r} "
                    f"teacher_correction_direction={teacher_correction_direction!r} "
                    f"teacher_reason={teacher_reason!r} "
                    f"teacher_single_stage_raw={teacher_single_stage_raw!r} "
                    f"teacher_difference_focus={teacher_difference_focus!r} "
                    f"teacher_problem_raw={teacher_problem_raw!r} "
                    f"teacher_direction_raw={teacher_direction_raw!r} "
                    f"teacher_reason_raw={teacher_reason_raw!r} "
                    f"teacher_problem_valid={teacher_problem_valid} "
                    f"teacher_direction_valid={teacher_direction_valid} "
                    f"teacher_reason_valid={teacher_reason_valid} "
                    f"window_avg_caption_tokens={window_avg_caption_tokens:.2f} "
                f"window_caption_seg_style_rate_raw={window_caption_seg_style_rate_raw:.4f} "
                f"window_caption_mode_failure_rate={window_caption_mode_failure_rate:.4f} "
                f"window_onpolicy_blocked_by_seg_style_count={window_totals['onpolicy_blocked_by_seg_style_count']} "
                f"window_grpo_blocked_by_seg_style_count={window_totals['grpo_blocked_by_seg_style_count']} "
                f"window_teacher_recovery_seg_style_count={window_totals['teacher_recovery_seg_style_count']} "
                f"window_teacher_recovery_seg_style_success_count={window_totals['teacher_recovery_seg_style_success_count']} "
                f"cumulative_teacher_difference_context_nontrivial_rate={cumulative_teacher_difference_context_nontrivial_rate:.4f} "
                f"cumulative_teacher_problem_valid_rate={cumulative_teacher_problem_valid_rate:.4f} "
                f"cumulative_teacher_direction_valid_rate={cumulative_teacher_direction_valid_rate:.4f} "
                f"cumulative_teacher_reason_valid_rate={cumulative_teacher_reason_valid_rate:.4f} "
                f"cumulative_teacher_reason_coarse_rate={cumulative_teacher_reason_coarse_rate:.4f} "
                f"cumulative_teacher_regenerate_verification_caption_valid_rate={cumulative_teacher_regenerate_verification_caption_valid_rate:.4f} "
                f"cumulative_teacher_regenerate_verification_iou_mean={cumulative_teacher_regenerate_verification_iou_mean:.4f} "
                f"cumulative_teacher_diagnosis_valid_rate={cumulative_teacher_diagnosis_valid_rate:.4f} "
                f"cumulative_teacher_dlc_valid_rate={cumulative_teacher_dlc_valid_rate:.4f} "
                f"cumulative_teacher_verification_caption_valid_rate={cumulative_teacher_verification_caption_valid_rate:.4f} "
                f"cumulative_teacher_verification_gate_pass_rate={cumulative_teacher_verification_gate_pass_rate:.4f} "
                f"cumulative_teacher_regenerate_dlc_ce_applied_count={self._cumulative_teacher_regenerate_ce_applied_count} "
                f"window_teacher_positive_gain_rate={window_teacher_positive_gain_rate:.4f} "
                f"window_teacher_iou_gain_mean={window_teacher_iou_gain_mean:.4f} "
                f"window_grpo_zero_reward_variance_rate={window_grpo_zero_reward_variance_rate:.4f} "
                f"window_grpo_nonzero_reward_rate={window_grpo_nonzero_reward_rate:.4f} "
                f"window_grpo_missing_confuser_rate={window_grpo_missing_confuser_rate:.4f} "
                f"window_grpo_reward_raw_mean={window_grpo_reward_raw_mean:.4f} "
                f"window_grpo_gt_prob_mean={window_grpo_gt_prob_mean:.4f} "
                f"window_grpo_confuser_penalty_mean={window_grpo_confuser_penalty_mean:.4f} "
                f"grpo_mcq_acc={batch_grpo_mcq_acc:.4f} "
                f"grpo_mcq_correct_conf_mean={batch_grpo_mcq_correct_conf_mean:.4f} "
                f"grpo_rollout_confidences={grpo_rollout_conf_text} "
                f"grpo_rollout_rewards={grpo_rollout_rewards_text} "
                f"{self._format_cuda_memory_stats()}"
            )
        self._log_pre_return_debug(
            batch_route=batch_route,
            last_route=last_route,
            optimized_count=optimized_count,
            regen_loss_count=regen_loss_count,
            onpolicy_loss_count=onpolicy_loss_count,
            grpo_loss_count=grpo_loss_count,
            total_loss=total_loss,
            total_regen_ce=total_regen_ce,
            total_onpolicy_jsd=total_onpolicy_jsd,
            total_grpo=total_grpo,
            rank_debug_records=rank_debug_records,
        )
        metrics = {
            "loss_opsd_total": avg_total_loss,
            "opsd_regen_ce": avg_regen_ce.detach(),
            "opsd_onpolicy_jsd": avg_onpolicy_jsd.detach(),
            "opsd_grpo": avg_grpo.detach(),
            "grpo_reward_mean": self._metric_tensor(window_grpo_reward_mean, avg_total_loss.dtype),
            "grpo_reward_raw_mean": self._metric_tensor(window_grpo_reward_raw_mean, avg_total_loss.dtype),
            "grpo_gt_prob_mean": self._metric_tensor(window_grpo_gt_prob_mean, avg_total_loss.dtype),
            "grpo_confuser_penalty_mean": self._metric_tensor(window_grpo_confuser_penalty_mean, avg_total_loss.dtype),
            "grpo_mcq_acc": self._metric_tensor(window_grpo_mcq_acc, avg_total_loss.dtype),
            "grpo_mcq_correct_conf_mean": self._metric_tensor(
                window_grpo_mcq_correct_conf_mean, avg_total_loss.dtype
            ),
            "grpo_group_size": self._metric_tensor(float(self.grpo_group_size), avg_total_loss.dtype),
            "verifier_iou": self._metric_tensor(window_verifier_iou, avg_total_loss.dtype),
            "seg_correct_rate": self._metric_tensor(window_seg_correct_rate, avg_total_loss.dtype),
            "all_sample_seg_success_rate": self._metric_tensor(
                window_all_sample_seg_success_rate, avg_total_loss.dtype
            ),
            "all_sample_seg_correct_rate": self._metric_tensor(
                window_all_sample_seg_correct_rate, avg_total_loss.dtype
            ),
            "avg_caption_tokens": self._metric_tensor(window_avg_caption_tokens, avg_total_loss.dtype),
            "teacher_regenerate_rate": self._metric_tensor(window_teacher_regenerate_rate, avg_total_loss.dtype),
            "on_policy_distill_rate": self._metric_tensor(window_on_policy_distill_rate, avg_total_loss.dtype),
            "grpo_positive_rate": self._metric_tensor(window_grpo_positive_rate, avg_total_loss.dtype),
            "caption_invalid_rate": self._metric_tensor(window_caption_invalid_rate, avg_total_loss.dtype),
            "caption_empty_rate": self._metric_tensor(window_caption_empty_rate, avg_total_loss.dtype),
            "caption_truncated_rate": self._metric_tensor(window_caption_truncated_rate, avg_total_loss.dtype),
            "caption_seg_style_rate": self._metric_tensor(window_caption_seg_style_rate, avg_total_loss.dtype),
            "caption_seg_style_rate_raw": self._metric_tensor(window_caption_seg_style_rate_raw, avg_total_loss.dtype),
            "caption_mode_failure_rate": self._metric_tensor(window_caption_mode_failure_rate, avg_total_loss.dtype),
            "reconstruct_invalid_caption_skip_rate": self._metric_tensor(
                window_reconstruct_invalid_caption_skip_rate, avg_total_loss.dtype
            ),
            "reconstruct_empty_prediction_masks_rate": self._metric_tensor(
                window_reconstruct_empty_prediction_masks_rate, avg_total_loss.dtype
            ),
            "detail_sufficient_caption_rate": self._metric_tensor(window_detail_sufficient_caption_rate, avg_total_loss.dtype),
            "scene_spill_caption_rate": self._metric_tensor(window_scene_spill_caption_rate, avg_total_loss.dtype),
            "teacher_regenerate_ce_applied_count": self._metric_tensor(window_totals["teacher_regenerate_ce_applied_count"], avg_total_loss.dtype),
            "teacher_regenerate_suppressed_count": self._metric_tensor(window_totals["teacher_regenerate_suppressed_count"], avg_total_loss.dtype),
            "teacher_regenerate_verified_count": self._metric_tensor(window_totals["teacher_regenerate_verified_count"], avg_total_loss.dtype),
            "teacher_regenerate_rejected_count": self._metric_tensor(window_totals["teacher_regenerate_rejected_count"], avg_total_loss.dtype),
            "teacher_regenerate_gate_pass_rate": self._metric_tensor(window_teacher_regenerate_gate_pass_rate, avg_total_loss.dtype),
            "teacher_regenerate_verified_iou_mean": self._metric_tensor(
                window_teacher_regenerate_verified_iou_mean, avg_total_loss.dtype
            ),
            "teacher_difference_context_nontrivial_rate": self._metric_tensor(
                window_teacher_difference_context_nontrivial_rate, avg_total_loss.dtype
            ),
            "teacher_problem_valid_rate": self._metric_tensor(
                window_teacher_problem_valid_rate, avg_total_loss.dtype
            ),
            "teacher_direction_valid_rate": self._metric_tensor(
                window_teacher_direction_valid_rate, avg_total_loss.dtype
            ),
            "teacher_reason_valid_rate": self._metric_tensor(
                window_teacher_reason_valid_rate, avg_total_loss.dtype
            ),
            "teacher_reason_coarse_rate": self._metric_tensor(
                window_teacher_reason_coarse_rate, avg_total_loss.dtype
            ),
            "teacher_regenerate_verification_caption_valid_rate": self._metric_tensor(
                window_teacher_regenerate_verification_caption_valid_rate, avg_total_loss.dtype
            ),
            "teacher_regenerate_verification_iou_mean": self._metric_tensor(
                window_teacher_regenerate_verification_iou_mean, avg_total_loss.dtype
            ),
            "teacher_diagnosis_valid_rate": self._metric_tensor(
                window_teacher_diagnosis_valid_rate, avg_total_loss.dtype
            ),
            "teacher_dlc_valid_rate": self._metric_tensor(window_teacher_dlc_valid_rate, avg_total_loss.dtype),
            "teacher_verification_caption_valid_rate": self._metric_tensor(
                window_teacher_regenerate_verification_caption_valid_rate, avg_total_loss.dtype
            ),
            "teacher_verification_gate_pass_rate": self._metric_tensor(
                window_teacher_verification_gate_pass_rate, avg_total_loss.dtype
            ),
            "teacher_regenerate_dlc_ce_applied_count": self._metric_tensor(
                window_totals["teacher_regenerate_ce_applied_count"], avg_total_loss.dtype
            ),
            "onpolicy_blocked_by_seg_style_count": self._metric_tensor(
                window_totals["onpolicy_blocked_by_seg_style_count"], avg_total_loss.dtype
            ),
            "grpo_blocked_by_seg_style_count": self._metric_tensor(
                window_totals["grpo_blocked_by_seg_style_count"], avg_total_loss.dtype
            ),
            "teacher_recovery_seg_style_count": self._metric_tensor(
                window_totals["teacher_recovery_seg_style_count"], avg_total_loss.dtype
            ),
            "teacher_recovery_seg_style_success_count": self._metric_tensor(
                window_totals["teacher_recovery_seg_style_success_count"], avg_total_loss.dtype
            ),
            "teacher_positive_gain_rate": self._metric_tensor(window_teacher_positive_gain_rate, avg_total_loss.dtype),
            "teacher_iou_gain_mean": self._metric_tensor(window_teacher_iou_gain_mean, avg_total_loss.dtype),
            "grpo_zero_reward_variance_rate": self._metric_tensor(window_grpo_zero_reward_variance_rate, avg_total_loss.dtype),
            "grpo_nonzero_reward_rate": self._metric_tensor(window_grpo_nonzero_reward_rate, avg_total_loss.dtype),
            "grpo_missing_confuser_rate": self._metric_tensor(window_grpo_missing_confuser_rate, avg_total_loss.dtype),
        }
        return metrics
