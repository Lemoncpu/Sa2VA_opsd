import copy
import inspect
import random
import re
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModel
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from projects.sa2va.datasets.common import SEG_QUESTIONS
from projects.sa2va.evaluation.teacher_diagnosis_common import (
    GRPO_POSITIVE_ROUTE,
    ON_POLICY_DISTILL_ROUTE,
    TEACHER_REGENERATE_ROUTE,
    build_mask_relation_context,
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
    selected_correct: bool
    correct_option_prob: float


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
        grpo_confuser_duplicate_iou_threshold=0.95,
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
        self._cumulative_recovery_ce_applied_count = 0
        self._cumulative_recovery_suppressed_count = 0
        self._cumulative_scene_spill_caption_count = 0
        self._cumulative_low_density_long_caption_count = 0
        self._cumulative_detail_sufficient_caption_count = 0
        self._cumulative_generic_caption_count = 0
        self._cumulative_repetitive_caption_count = 0

        self.teacher_summary_template = teacher_summary_template or (
            "You are optimizing the following task: given a gtmask, generate a caption that describes it. "
            "You are now given the original input, the student question, and privileged verification information. "
            "Use these privileged signals to improve the caption generation.\n"
            "Original student prompt: {student_question}\n"
            "Student caption: {student_caption}\n"
            "Verifier caption used for reconstruction: {verifier_caption}\n"
            "Reconstruction question: {reconstruct_question}\n"
            "Description generation status: {description_status}\n"
            "Reconstruction status: {reconstruct_status}\n"
            "caption_to_mask_seg_correct: {caption_to_mask_seg_correct}\n"
            "IoU between gtmask (region1) and refmask (region2): {iou:.4f}\n"
            "Reconstruction produced a valid mask: {has_mask}\n"
            "region1 = gtmask summary: {gtmask_summary}\n"
            "region2 = refmask summary: {refmask_summary}\n"
            "Compare region1 and region2, then infer how the caption should be revised so the reconstruction moves from region2 toward region1.\n"
            "Use that strategy to better model the student's caption tokens."
        )
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
        self._ensure_generation_ready(model)
        return model

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

    def _ensure_generation_ready(self, model):
        self._ensure_runtime_state(model)
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

    def require_teacher_model(self, context="this operation"):
        if self.has_teacher_model():
            return self.teacher_model
        raise RuntimeError(
            f"{context} requires enable_teacher=True. "
            "Teacher-free evaluation does not construct teacher_model."
        )

    def train(self, mode=True):
        super().train(mode)
        if self.has_teacher_model():
            self.teacher_model.eval()
        return self

    def to(self, *args, **kwargs):
        target_dtype = kwargs.get("dtype")
        if target_dtype is None and len(args) == 1 and isinstance(args[0], torch.dtype):
            target_dtype = args[0]
        if target_dtype is not None:
            self.student_model.to(dtype=target_dtype)
            if self.has_teacher_model():
                self.teacher_model.to(dtype=target_dtype)
        self.student_model.to(self.device)
        if self.has_teacher_model():
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
        return self

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
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        has_teacher_state = any(k.startswith("teacher_model.") for k in state_dict)

        if not self.has_teacher_model():
            filtered_state = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith(("teacher_model.",))
            }
            return super().load_state_dict(filtered_state, strict=strict)

        if has_teacher_state:
            return super().load_state_dict(state_dict, strict=strict)

        filtered_state = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith(("teacher_model.",))
        }
        result = super().load_state_dict(filtered_state, strict=False)
        self._sync_teacher()
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

    @staticmethod
    def _teacher_regenerate_gate_passed(student_iou, teacher_iou):
        return (float(teacher_iou) - float(student_iou)) > 0.5

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

    def _debug_sample(
        self,
        *,
        sample_key,
        route,
        student_question,
        raw_prediction,
        caption,
        description_status,
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
                if "processor" in signature.parameters and "processor" not in kwargs:
                    kwargs["processor"] = self.processor
                return model.predict_forward(**kwargs)

    def _clone_generation_config(self, model, overrides=None):
        base_config = getattr(model, "gen_config", None)
        if base_config is None:
            base_config = getattr(getattr(model, "language_model", None), "generation_config", None)
        generation_config = copy.deepcopy(base_config) if base_config is not None else GenerationConfig()
        tokenizer = self.tokenizer
        if getattr(generation_config, "bos_token_id", None) is None and tokenizer is not None:
            generation_config.bos_token_id = tokenizer.bos_token_id
        if getattr(generation_config, "eos_token_id", None) is None and tokenizer is not None:
            generation_config.eos_token_id = tokenizer.eos_token_id
        if getattr(generation_config, "pad_token_id", None) is None and tokenizer is not None:
            generation_config.pad_token_id = (
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        for key, value in (overrides or {}).items():
            setattr(generation_config, key, value)
        return generation_config

    @staticmethod
    def _extract_completion_ids(sequences, prompt_ids):
        if sequences.ndim != 2 or prompt_ids.ndim != 2:
            return sequences
        prompt_len = prompt_ids.shape[1]
        if sequences.shape[1] > prompt_len and torch.equal(
            sequences[:, :prompt_len].to(prompt_ids.device),
            prompt_ids,
        ):
            return sequences[:, prompt_len:]
        return sequences

    def _generate_caption_with_model(
        self,
        model,
        *,
        image,
        prompt_masks,
        prompt_text,
        apply_mask_focus=True,
        generation_overrides=None,
    ):
        self._ensure_generation_ready(model)
        with self._temporary_eval_model(model):
            with torch.inference_mode():
                formatted_prompt_masks = self._to_teacher_prompt_masks(prompt_masks)
                prompt_image = self._build_mask_focused_image(image, prompt_masks) if apply_mask_focus and prompt_masks is not None else image
                mm_inputs = self._build_forward_inputs(model, prompt_image, formatted_prompt_masks, prompt_text)
                inputs_embeds = self._compose_inputs_embeds(model, mm_inputs)
                generation_config = self._clone_generation_config(model, generation_overrides)
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=mm_inputs["attention_mask"],
                    generation_config=generation_config,
                    bos_token_id=self.tokenizer.bos_token_id,
                    stopping_criteria=getattr(model, "stop_criteria", None),
                    output_hidden_states=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
        generated_ids = self._extract_completion_ids(outputs.sequences, mm_inputs["input_ids"])
        raw_prediction = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False).strip()
        clean_caption = self._clean_caption_text(raw_prediction)
        status = self._infer_description_status(clean_caption)
        if status == "ok" and not self._is_caption_content_sufficient(clean_caption):
            status = "truncated_caption"
        completion_ids = self._encode_completion_from_caption(clean_caption, model=model)
        return DescriptionResult(
            raw_prediction=raw_prediction,
            clean_caption=clean_caption,
            completion_ids=completion_ids,
            status=status,
        )

    def predict_text_with_masks(
        self,
        model,
        *,
        image,
        text,
        mask_prompts=None,
        apply_mask_focus=False,
        generation_overrides=None,
    ):
        self._ensure_generation_ready(model)
        prompt_image = image
        prompt_masks_for_forward = None
        if mask_prompts is not None:
            prompt_masks_for_forward = self._to_teacher_prompt_masks(mask_prompts)
            if apply_mask_focus:
                prompt_image = self._build_mask_focused_image(image, mask_prompts)
        with self._temporary_eval_model(model):
            with torch.inference_mode():
                mm_inputs = self._build_forward_inputs(model, prompt_image, prompt_masks_for_forward, text)
                inputs_embeds = self._compose_inputs_embeds(model, mm_inputs)
                generation_config = self._clone_generation_config(model, generation_overrides)
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=mm_inputs["attention_mask"],
                    generation_config=generation_config,
                    bos_token_id=self.tokenizer.bos_token_id,
                    stopping_criteria=getattr(model, "stop_criteria", None),
                    output_hidden_states=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
        prediction = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False).strip()
        return {"prediction": prediction}

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
        status = self._infer_description_status(clean_caption)
        if status == "ok" and not self._is_caption_content_sufficient(clean_caption):
            status = "truncated_caption"
        return DescriptionResult(
            raw_prediction=raw_prediction,
            clean_caption=clean_caption,
            completion_ids=predicted_ids,
            status=status,
        )

    def generate_description_with_model(
        self,
        model,
        *,
        image,
        mask_prompts,
        student_question,
        apply_mask_focus=True,
        generation_overrides=None,
    ):
        if generation_overrides is not None:
            return self._generate_caption_with_model(
                model,
                image=image,
                prompt_masks=mask_prompts,
                prompt_text=student_question,
                apply_mask_focus=apply_mask_focus,
                generation_overrides=generation_overrides,
            )
        formatted_mask_prompts = self._format_mask_prompts_for_predict_forward(mask_prompts)
        prompt_image = self._build_mask_focused_image(image, mask_prompts) if apply_mask_focus else image
        predict_dict = self._predict_forward_eval(
            model,
            image=prompt_image,
            text=student_question,
            past_text="",
            mask_prompts=formatted_mask_prompts,
            tokenizer=self.tokenizer,
        )
        raw_prediction = predict_dict.get("prediction", "")
        clean_caption = self._clean_caption_text(raw_prediction)
        status = self._infer_description_status(clean_caption)
        if status == "ok" and not self._is_caption_content_sufficient(clean_caption):
            status = "truncated_caption"
        return DescriptionResult(
            raw_prediction=raw_prediction,
            clean_caption=clean_caption,
            completion_ids=self._encode_completion_from_caption(clean_caption, model=model),
            status=status,
        )

    def generate_description(self, image, mask_prompts, student_question):
        return self.generate_description_with_model(
            self.student_model,
            image=image,
            mask_prompts=mask_prompts,
            student_question=student_question,
            apply_mask_focus=True,
        )

    def generate_teacher_caption_with_privileged_prompt(
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
    ):
        teacher_prompt_masks = np.stack(
            [
                self._to_numpy_mask(gt_mask).astype(np.float32),
                self._to_numpy_mask(ref_mask).astype(np.float32),
            ],
            axis=0,
        )
        teacher_prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode="regenerate_caption",
        )
        teacher_fields["teacher_regenerate_prompt"] = teacher_prompt
        teacher_model = self.require_teacher_model("Teacher privileged regeneration")
        return self.generate_description_with_model(
            teacher_model,
            image=image,
            mask_prompts=teacher_prompt_masks,
            student_question=teacher_prompt,
            apply_mask_focus=True,
            generation_overrides={
                "max_new_tokens": self.low_iou_regen_max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 4,
            },
        )

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
        prompt = (
            "<image>\n"
            "You are a teacher supervising a caption-to-mask model. The student first writes a caption for the target, and that caption is then used to reconstruct a segmentation mask. "
            "You are given privileged access to the target mask (region1 = gtmask) and the reconstructed mask (region2 = refmask). "
            "Your job is to analyze, at pixel and region level, why the current student caption produces the current reconstructed mask, and then provide the correct supervision for this route.\n"
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
        if generation_mode == "regenerate_caption":
            prompt = (
                "<image>\n"
                "You are a teacher supervising a failed caption-to-mask reconstruction.\n"
                "Region1 is the gtmask and region2 is the mask reconstructed from the student's caption.\n"
                f"Student prompt: {clean_question}\n"
                f"Failed student caption: {student_caption}\n"
                f"Description status: {description_status}\n"
                f"Reconstruction status: {reconstruction.status}\n"
                f"Current IoU between region1 and region2: {iou:.4f}\n"
                f"Shared overlap summary: {relation_context['overlap_summary']}\n"
                f"Region1-only summary (missing target pixels): {relation_context['gt_only_summary']}\n"
                f"Region2-only summary (distractor pixels wrongly predicted): {relation_context['ref_only_summary']}\n"
                "Before writing the new caption, reason in this order internally:\n"
                "1. Analyze what region1 actually contains.\n"
                "2. Analyze what region2 actually contains.\n"
                "3. Compare the two masks and identify missing target evidence and extra distractor evidence.\n"
                "4. Diagnose why the student's caption leads to region2 instead of region1, and which phrases are wrong, missing, too generic, or misleading.\n"
                "5. Regenerate a better caption that would move reconstruction from region2 toward region1.\n"
                f"{route_guidance}\n"
                "Output requirements:\n"
                "- Return exactly one natural and complete sentence describing region1.\n"
                "- Focus on visible appearance, attributes, parts, markings, clothing, pose, and only the minimum local context needed to localize the target.\n"
                "- Prefer concrete visible details over generic statements.\n"
                "- Fix the specific mistakes that caused region2 to differ from region1.\n"
                "- Do not explain.\n"
                "- Do not output labels.\n"
                "- Do not mention region1 or region2.\n"
                "- Do not describe anything that is not visible.\n"
                "- Do not copy the failed student caption if it still matches region2."
            )
        return prompt

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

    def _empty_loss_vector(self):
        return torch.empty(0, device=self.device, dtype=next(self.student_model.parameters()).dtype)

    def compute_regenerate_alignment_loss(
        self,
        image,
        prompt_masks,
        student_question,
        completion_ids,
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
        return self._sequence_cross_entropy_from_logits(student_logits, completion_ids)

    def compute_regenerate_alignment_losses_batch(self, batch_items):
        sample_losses = []
        for item in batch_items:
            completion_ids = item["completion_ids"]
            if completion_ids.shape[1] == 0:
                continue
            sample_loss = self.compute_regenerate_alignment_loss(
                image=item["image"],
                prompt_masks=item["prompt_masks"],
                student_question=item["student_question"],
                completion_ids=completion_ids,
            )
            if sample_loss is not None:
                sample_losses.append(sample_loss)
        if not sample_losses:
            return self._empty_loss_vector()
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
        return (jsd_tokens * token_weights).mean() * sample_weight

    def compute_onpolicy_distill_losses_batch(self, batch_items):
        sample_losses = []
        for item in batch_items:
            completion_ids = item["completion_ids"]
            if completion_ids.shape[1] == 0:
                continue
            sample_loss = self.compute_onpolicy_distill_loss(
                image=item["image"],
                prompt_masks=item["prompt_masks"],
                student_question=item["student_question"],
                teacher_prompt=item["teacher_prompt"],
                completion_ids=completion_ids,
                teacher_prompt_masks=item.get("teacher_prompt_masks", item["prompt_masks"]),
                iou=float(item.get("iou", 0.0)),
            )
            if sample_loss is not None:
                sample_losses.append(sample_loss)
        if not sample_losses:
            return self._empty_loss_vector()
        return torch.stack(sample_losses)

    def _sample_grpo_descriptions(self, *, image, prompt_masks, student_question):
        target_rollout_count = int(self.grpo_group_size)
        if target_rollout_count <= 0:
            return []
        generation_overrides = {
            "max_new_tokens": self.grpo_sample_max_new_tokens,
            "do_sample": True,
            "num_beams": 1,
            "temperature": self.grpo_sample_temperature,
            "top_p": self.grpo_sample_top_p,
        }
        descriptions = []
        for _ in range(target_rollout_count):
            descriptions.append(
                self.generate_description_with_model(
                    self.student_model,
                    image=image,
                    mask_prompts=prompt_masks,
                    student_question=student_question,
                    apply_mask_focus=True,
                    generation_overrides=generation_overrides,
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
            return None
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
            return None
        return selected_masks

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
        with self._temporary_eval_model(self.student_model):
            with torch.inference_mode():
                answer_logits = self._forward_sequence_multi_sample_with_model(
                    self.student_model,
                    samples,
                )["logits"]
        first_step_logits = answer_logits[:, 0, :]
        answer_token_ids = torch.tensor(self._grpo_option_token_ids, device=first_step_logits.device, dtype=torch.long)
        answer_scores = first_step_logits.gather(dim=-1, index=answer_token_ids.unsqueeze(-1)).squeeze(-1)
        option_probs = torch.softmax(answer_scores, dim=0)
        predicted_option_idx = int(option_probs.argmax().item())
        correct_option_prob = float(option_probs[correct_option_idx].item())
        selected_correct = predicted_option_idx == int(correct_option_idx)
        reward = correct_option_prob if selected_correct else 0.0
        if not selected_correct and not self.grpo_confuser_zero_reward_on_wrong:
            reward = correct_option_prob
        return ConfuserSelectionResult(
            option_probs=option_probs.detach(),
            predicted_option_idx=predicted_option_idx,
            correct_option_idx=int(correct_option_idx),
            reward=float(reward),
            selected_correct=selected_correct,
            correct_option_prob=correct_option_prob,
        )

    def compute_grpo_loss(
        self,
        *,
        image,
        prompt_masks,
        student_question,
        gt_mask,
        confuser_candidate_masks=None,
    ):
        rollout_entries = []
        rollout_mcq_confidences = []
        rollout_rewards = []
        rollout_mcq_correct = []
        confuser_masks = self._select_confuser_masks(
            gt_mask=gt_mask,
            candidate_masks=confuser_candidate_masks,
        )
        if confuser_masks is None:
            return None, {
                "reward_sum": 0.0,
                "reward_count": 0,
                "rollout_mcq_confidences": [],
                "rollout_rewards": [],
                "rollout_mcq_correct": [],
                "mcq_correct_count": 0,
                "mcq_total_count": 0,
                "mcq_correct_conf_sum": 0.0,
            }
        descriptions = self._sample_grpo_descriptions(
            image=image,
            prompt_masks=prompt_masks,
            student_question=student_question,
        )
        for description in descriptions:
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
                image=image,
                option_masks=np.stack(option_masks, axis=0).astype(np.float32),
                caption=description.clean_caption,
                correct_option_idx=correct_option_idx,
            )
            reward_value = selection.reward
            with self._temporary_eval_model(self.student_model):
                with torch.inference_mode():
                    old_policy_logits = self._forward_sequence_with_model(
                        self.student_model,
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
            rollout_entries.append(
                {
                    "completion_ids": completion_ids,
                    "old_token_log_probs": old_token_log_probs,
                    "reward_value": reward_value,
                    "correct_option_prob": selection.correct_option_prob,
                    "selected_correct": bool(selection.selected_correct),
                }
            )

        if not rollout_entries:
            return None, {
                "reward_sum": 0.0,
                "reward_count": 0,
                "rollout_mcq_confidences": [],
                "rollout_rewards": [],
                "rollout_mcq_correct": [],
                "mcq_correct_count": 0,
                "mcq_total_count": 0,
                "mcq_correct_conf_sum": 0.0,
            }

        reward_tensor = torch.tensor(
            [entry["reward_value"] for entry in rollout_entries],
            device=self.device,
            dtype=rollout_entries[0]["old_token_log_probs"].dtype,
        )
        reward_std = reward_tensor.std(unbiased=False).clamp_min(self.grpo_advantage_eps)
        advantages = (reward_tensor - reward_tensor.mean()) / reward_std

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
        current_token_log_probs = self._token_log_probs_from_logits(
            student_logits,
            completion_batch,
        )
        ratio = torch.exp(current_token_log_probs - old_token_log_probs_batch)
        clipped_ratio = ratio.clamp(1.0 - self.grpo_clip_eps, 1.0 + self.grpo_clip_eps)
        advantage_batch = advantages.unsqueeze(1)
        surrogate = torch.min(ratio * advantage_batch, clipped_ratio * advantage_batch)
        token_weights = completion_mask.to(dtype=surrogate.dtype)
        sample_losses = -(surrogate * token_weights).sum(dim=-1) / token_weights.sum(dim=-1).clamp_min(1.0)

        return sample_losses.mean(), {
            "reward_sum": float(reward_tensor.sum().item()),
            "reward_count": len(rollout_entries),
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
        }

    def compute_grpo_losses_batch(self, batch_items):
        sample_losses = []
        reward_sum = 0.0
        reward_count = 0
        rollout_mcq_confidences = []
        rollout_rewards = []
        rollout_mcq_correct = []
        mcq_correct_count = 0
        mcq_total_count = 0
        mcq_correct_conf_sum = 0.0
        for item in batch_items:
            sample_loss, grpo_meta = self.compute_grpo_loss(
                image=item["image"],
                prompt_masks=item["prompt_masks"],
                student_question=item["student_question"],
                gt_mask=item["gt_mask"],
                confuser_candidate_masks=item.get("confuser_candidate_masks"),
            )
            reward_sum += grpo_meta["reward_sum"]
            reward_count += grpo_meta["reward_count"]
            rollout_mcq_confidences.extend(grpo_meta.get("rollout_mcq_confidences", []))
            rollout_rewards.extend(grpo_meta.get("rollout_rewards", []))
            rollout_mcq_correct.extend(grpo_meta.get("rollout_mcq_correct", []))
            mcq_correct_count += int(grpo_meta.get("mcq_correct_count", 0))
            mcq_total_count += int(grpo_meta.get("mcq_total_count", 0))
            mcq_correct_conf_sum += float(grpo_meta.get("mcq_correct_conf_sum", 0.0))
            if sample_loss is not None:
                sample_losses.append(sample_loss)
        if not sample_losses:
            return self._empty_loss_vector(), {
                "reward_sum": reward_sum,
                "reward_count": reward_count,
                "rollout_mcq_confidences": rollout_mcq_confidences,
                "rollout_rewards": rollout_rewards,
                "rollout_mcq_correct": rollout_mcq_correct,
                "mcq_correct_count": mcq_correct_count,
                "mcq_total_count": mcq_total_count,
                "mcq_correct_conf_sum": mcq_correct_conf_sum,
            }
        return torch.stack(sample_losses), {
            "reward_sum": reward_sum,
            "reward_count": reward_count,
            "rollout_mcq_confidences": rollout_mcq_confidences,
            "rollout_rewards": rollout_rewards,
            "rollout_mcq_correct": rollout_mcq_correct,
            "mcq_correct_count": mcq_correct_count,
            "mcq_total_count": mcq_total_count,
            "mcq_correct_conf_sum": mcq_correct_conf_sum,
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

        zero = next(self.student_model.parameters()).sum() * 0.0
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
        recovery_ce_applied_count = 0
        recovery_suppressed_count = 0
        reconstruct_invalid_caption_skip_count = 0
        reconstruct_empty_prediction_masks_count = 0
        scene_spill_caption_count = 0
        low_density_long_caption_count = 0
        detail_sufficient_caption_count = 0
        generic_caption_count = 0
        repetitive_caption_count = 0
        last_sample_key = None
        last_caption = ""
        last_teacher_prompt = ""
        last_route = ""

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
                last_sample_key = sample_key
                last_route = TEACHER_REGENERATE_ROUTE
                teacher_regenerate_count += 1
                is_recovery_case = description.status != "ok"
                if is_recovery_case:
                    recovery_caption_count += 1
                max_regen_count = max(int(np.ceil(self.max_teacher_regenerate_fraction * max(nonempty_gt_count, 1))), 1)
                max_recovery_count = max(int(np.ceil(self.max_recovery_fraction * max(nonempty_gt_count, 1))), 1)
                allow_teacher_ce = teacher_regenerate_ce_applied_count < max_regen_count
                if is_recovery_case:
                    allow_teacher_ce = allow_teacher_ce and (recovery_ce_applied_count < max_recovery_count)
                if allow_teacher_ce:
                    zero_ref_mask = np.zeros_like(gt_mask_np, dtype=np.uint8)
                    teacher_fields = self._build_training_teacher_fields(
                        route=TEACHER_REGENERATE_ROUTE,
                        iou=0.0,
                    )
                    recovery_reconstruction = reconstruction or self._invalid_reconstruction_placeholder("skipped_invalid_description")
                    teacher_regenerate = self.generate_teacher_caption_with_privileged_prompt(
                        image=image,
                        gt_mask=gt_mask_np,
                        ref_mask=zero_ref_mask,
                        student_question=student_question,
                        student_caption=description.clean_caption,
                        description_status=description.status,
                        reconstruction=recovery_reconstruction,
                        iou=0.0,
                        teacher_fields=teacher_fields,
                    )
                    teacher_reconstruction = self.reconstruct_mask(
                        image=image,
                        caption=teacher_regenerate.clean_caption,
                        description_status=teacher_regenerate.status,
                        gt_mask=gt_mask_np,
                    )
                    teacher_pred_mask = None if teacher_reconstruction is None else teacher_reconstruction.pred_mask
                    teacher_iou_plain = (
                        self._compute_iou(gt_mask_np, teacher_pred_mask) if teacher_pred_mask is not None else 0.0
                    )
                    teacher_reconstruct_ok = (
                        teacher_reconstruction is not None
                        and teacher_reconstruction.status == "ok"
                        and teacher_pred_mask is not None
                    )
                    if teacher_reconstruct_ok and self._teacher_regenerate_gate_passed(0.0, teacher_iou_plain):
                        regen_entries.append(
                            {
                                "image": image,
                                "prompt_masks": prompt_masks,
                                "student_question": student_question,
                                "completion_ids": teacher_regenerate.completion_ids,
                            }
                        )
                        teacher_regenerate_ce_applied_count += 1
                        teacher_regenerate_verified_count += 1
                        teacher_regenerate_verified_iou_sum += teacher_iou_plain
                        if is_recovery_case:
                            recovery_ce_applied_count += 1
                        last_caption = teacher_regenerate.clean_caption or description.clean_caption
                    else:
                        teacher_regenerate_suppressed_count += 1
                        teacher_regenerate_rejected_count += 1
                        hard_reconstruct_failure_count += 1
                        if is_recovery_case:
                            recovery_suppressed_count += 1
                        last_caption = description.clean_caption
                else:
                    teacher_regenerate_suppressed_count += 1
                    hard_reconstruct_failure_count += 1
                    if is_recovery_case:
                        recovery_suppressed_count += 1
                    last_caption = description.clean_caption
                last_teacher_prompt = self._route_prompt_tag(TEACHER_REGENERATE_ROUTE)
                self._debug_sample(
                    sample_key=sample_key,
                    route=TEACHER_REGENERATE_ROUTE,
                    student_question=student_question,
                    raw_prediction=description.raw_prediction,
                    caption=description.clean_caption,
                    description_status=description.status,
                    reconstruct_question=reconstruct_question,
                    raw_reconstruct_prediction=raw_reconstruct_prediction,
                    reconstruct_status=reconstruct_status,
                    seg_token_count=seg_token_count,
                    prediction_masks_count=prediction_masks_count,
                    pred_mask=None,
                    gt_mask=gt_mask_np,
                    iou=0.0,
                    empty_gt_mask=False,
                )
                continue

            iou = self._compute_iou(gt_mask_np, pred_mask)
            ref_mask_np = self._to_numpy_mask(pred_mask)
            online_route = self._route_from_iou(iou)
            if self.use_online_route_for_loss:
                route = online_route
            else:
                route = batch_route or route_from_manifest or online_route
            routed_count += 1
            if iou >= 0.5:
                seg_correct_count += 1
            if reconstruct_status == "ok":
                reconstruct_ok_count += 1
            elif reconstruct_status not in {"ok", "skipped_invalid_description"}:
                reconstruct_failed_count += 1

            self._debug_sample(
                sample_key=sample_key,
                route=f"{route} (online={online_route})" if route != online_route else route,
                student_question=student_question,
                raw_prediction=description.raw_prediction,
                caption=description.clean_caption,
                description_status=description.status,
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

            teacher_prompt = ""
            if route == TEACHER_REGENERATE_ROUTE:
                teacher_regenerate_count += 1
                teacher_prompt = self._route_prompt_tag(route)
                max_regen_count = max(int(np.ceil(self.max_teacher_regenerate_fraction * max(nonempty_gt_count, 1))), 1)
                if teacher_regenerate_ce_applied_count < max_regen_count:
                    teacher_fields = self._build_training_teacher_fields(
                        route=route,
                        iou=iou,
                    )
                    teacher_regenerate = self.generate_teacher_caption_with_privileged_prompt(
                        image=image,
                        gt_mask=gt_mask_np,
                        ref_mask=ref_mask_np,
                        student_question=student_question,
                        student_caption=description.clean_caption,
                        description_status=description.status,
                        reconstruction=reconstruction,
                        iou=iou,
                        teacher_fields=teacher_fields,
                    )
                    teacher_reconstruction = self.reconstruct_mask(
                        image=image,
                        caption=teacher_regenerate.clean_caption,
                        description_status=teacher_regenerate.status,
                        gt_mask=gt_mask_np,
                    )
                    teacher_pred_mask = None if teacher_reconstruction is None else teacher_reconstruction.pred_mask
                    teacher_iou_plain = (
                        self._compute_iou(gt_mask_np, teacher_pred_mask) if teacher_pred_mask is not None else 0.0
                    )
                    teacher_reconstruct_ok = (
                        teacher_reconstruction is not None
                        and teacher_reconstruction.status == "ok"
                        and teacher_pred_mask is not None
                    )
                    if teacher_reconstruct_ok and self._teacher_regenerate_gate_passed(iou, teacher_iou_plain):
                        regen_entries.append(
                            {
                                "image": image,
                                "prompt_masks": prompt_masks,
                                "student_question": student_question,
                                "completion_ids": teacher_regenerate.completion_ids,
                            }
                        )
                        teacher_regenerate_ce_applied_count += 1
                        teacher_regenerate_verified_count += 1
                        teacher_regenerate_verified_iou_sum += teacher_iou_plain
                        last_caption = teacher_regenerate.clean_caption or description.clean_caption
                    else:
                        teacher_regenerate_suppressed_count += 1
                        teacher_regenerate_rejected_count += 1
                        hard_reconstruct_failure_count += 1
                        last_caption = description.clean_caption
                else:
                    teacher_regenerate_suppressed_count += 1
                    hard_reconstruct_failure_count += 1
                    last_caption = description.clean_caption
            elif route == ON_POLICY_DISTILL_ROUTE:
                on_policy_distill_count += 1
                teacher_fields = self._build_training_teacher_fields(
                    route=route,
                    iou=iou,
                )
                teacher_prompt = self.build_teacher_privileged_prompt_v3(
                    student_question=student_question,
                    student_caption=description.clean_caption,
                    description_status=description.status,
                    reconstruction=reconstruction,
                    iou=iou,
                    gt_mask=gt_mask_np,
                    ref_mask=ref_mask_np,
                    teacher_fields=teacher_fields,
                )
                teacher_prompt_masks = np.stack(
                    [
                        gt_mask_np.astype(np.float32),
                        ref_mask_np.astype(np.float32),
                    ],
                    axis=0,
                )
                onpolicy_entries.append(
                    {
                        "image": image,
                        "prompt_masks": prompt_masks,
                        "student_question": student_question,
                        "teacher_prompt": teacher_prompt,
                        "completion_ids": description.completion_ids,
                        "teacher_prompt_masks": teacher_prompt_masks,
                        "iou": iou,
                    }
                )
                last_caption = description.clean_caption
            else:
                grpo_positive_count += 1
                grpo_entries.append(
                    {
                        "image": image,
                        "prompt_masks": prompt_masks,
                        "student_question": student_question,
                        "gt_mask": gt_mask_np,
                        "confuser_candidate_masks": confuser_candidate_masks,
                    }
                )
                teacher_prompt = self._route_prompt_tag(route)
                last_caption = description.clean_caption

            last_sample_key = sample_key
            last_route = route
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

        if optimized_count == 0:
            metrics = {
                "loss_opsd_total": zero,
                "opsd_regen_ce": zero,
                "opsd_onpolicy_jsd": zero,
                "opsd_grpo": zero,
                "grpo_reward_mean": zero,
                "grpo_mcq_acc": zero,
                "grpo_mcq_correct_conf_mean": zero,
                "grpo_group_size": self._metric_tensor(float(self.grpo_group_size), zero.dtype),
                "verifier_iou": self._metric_tensor(cumulative_verifier_iou, zero.dtype),
                "seg_correct_rate": self._metric_tensor(cumulative_seg_correct_rate, zero.dtype),
                "all_sample_seg_success_rate": self._metric_tensor(cumulative_all_sample_seg_success_rate, zero.dtype),
                "all_sample_seg_correct_rate": self._metric_tensor(cumulative_all_sample_seg_correct_rate, zero.dtype),
                "avg_caption_tokens": self._metric_tensor(cumulative_avg_caption_tokens, zero.dtype),
                "teacher_regenerate_rate": self._metric_tensor(cumulative_teacher_regenerate_rate, zero.dtype),
                "on_policy_distill_rate": self._metric_tensor(cumulative_on_policy_distill_rate, zero.dtype),
                "grpo_positive_rate": self._metric_tensor(cumulative_grpo_positive_rate, zero.dtype),
                "caption_invalid_rate": self._metric_tensor(cumulative_invalid_caption_penalty_rate, zero.dtype),
                "caption_empty_rate": self._metric_tensor(cumulative_caption_empty_rate, zero.dtype),
                "caption_truncated_rate": self._metric_tensor(cumulative_caption_truncated_rate, zero.dtype),
                "caption_seg_style_rate": self._metric_tensor(cumulative_caption_seg_style_rate, zero.dtype),
                "reconstruct_invalid_caption_skip_rate": self._metric_tensor(
                    cumulative_reconstruct_invalid_caption_skip_rate, zero.dtype
                ),
                "reconstruct_empty_prediction_masks_rate": self._metric_tensor(
                    cumulative_reconstruct_empty_prediction_masks_rate, zero.dtype
                ),
                "detail_sufficient_caption_rate": self._metric_tensor(cumulative_detail_sufficient_caption_rate, zero.dtype),
                "scene_spill_caption_rate": self._metric_tensor(cumulative_scene_spill_caption_rate, zero.dtype),
                "teacher_regenerate_ce_applied_count": self._metric_tensor(self._cumulative_teacher_regenerate_ce_applied_count, zero.dtype),
                "teacher_regenerate_suppressed_count": self._metric_tensor(self._cumulative_teacher_regenerate_suppressed_count, zero.dtype),
                "teacher_regenerate_verified_count": self._metric_tensor(self._cumulative_teacher_regenerate_verified_count, zero.dtype),
                "teacher_regenerate_rejected_count": self._metric_tensor(self._cumulative_teacher_regenerate_rejected_count, zero.dtype),
                "teacher_regenerate_gate_pass_rate": self._metric_tensor(
                    cumulative_teacher_regenerate_gate_pass_rate, zero.dtype
                ),
                "teacher_regenerate_verified_iou_mean": self._metric_tensor(
                    cumulative_teacher_regenerate_verified_iou_mean, zero.dtype
                ),
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
                    f"cum_avg_caption_tokens={cumulative_avg_caption_tokens:.2f} "
                    f"grpo_mcq_acc={batch_grpo_mcq_acc:.4f} "
                    f"grpo_mcq_correct_conf_mean={batch_grpo_mcq_correct_conf_mean:.4f} "
                    f"grpo_rollout_confidences={grpo_rollout_conf_text} "
                    f"grpo_rollout_rewards={grpo_rollout_rewards_text}"
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
                f"cum_avg_caption_tokens={cumulative_avg_caption_tokens:.2f} "
                f"grpo_mcq_acc={batch_grpo_mcq_acc:.4f} "
                f"grpo_mcq_correct_conf_mean={batch_grpo_mcq_correct_conf_mean:.4f} "
                f"grpo_rollout_confidences={grpo_rollout_conf_text} "
                f"grpo_rollout_rewards={grpo_rollout_rewards_text}"
            )
        metrics = {
            "loss_opsd_total": avg_total_loss,
            "opsd_regen_ce": avg_regen_ce.detach(),
            "opsd_onpolicy_jsd": avg_onpolicy_jsd.detach(),
            "opsd_grpo": avg_grpo.detach(),
            "grpo_reward_mean": self._metric_tensor(batch_grpo_reward_mean, avg_total_loss.dtype),
            "grpo_mcq_acc": self._metric_tensor(cumulative_grpo_mcq_acc, avg_total_loss.dtype),
            "grpo_mcq_correct_conf_mean": self._metric_tensor(
                cumulative_grpo_mcq_correct_conf_mean, avg_total_loss.dtype
            ),
            "grpo_group_size": self._metric_tensor(float(self.grpo_group_size), avg_total_loss.dtype),
            "verifier_iou": self._metric_tensor(cumulative_verifier_iou, avg_total_loss.dtype),
            "seg_correct_rate": self._metric_tensor(cumulative_seg_correct_rate, avg_total_loss.dtype),
            "all_sample_seg_success_rate": self._metric_tensor(
                cumulative_all_sample_seg_success_rate, avg_total_loss.dtype
            ),
            "all_sample_seg_correct_rate": self._metric_tensor(
                cumulative_all_sample_seg_correct_rate, avg_total_loss.dtype
            ),
            "avg_caption_tokens": self._metric_tensor(cumulative_avg_caption_tokens, avg_total_loss.dtype),
            "teacher_regenerate_rate": self._metric_tensor(cumulative_teacher_regenerate_rate, avg_total_loss.dtype),
            "on_policy_distill_rate": self._metric_tensor(cumulative_on_policy_distill_rate, avg_total_loss.dtype),
            "grpo_positive_rate": self._metric_tensor(cumulative_grpo_positive_rate, avg_total_loss.dtype),
            "caption_invalid_rate": self._metric_tensor(cumulative_invalid_caption_penalty_rate, avg_total_loss.dtype),
            "caption_empty_rate": self._metric_tensor(cumulative_caption_empty_rate, avg_total_loss.dtype),
            "caption_truncated_rate": self._metric_tensor(cumulative_caption_truncated_rate, avg_total_loss.dtype),
            "caption_seg_style_rate": self._metric_tensor(cumulative_caption_seg_style_rate, avg_total_loss.dtype),
            "reconstruct_invalid_caption_skip_rate": self._metric_tensor(
                cumulative_reconstruct_invalid_caption_skip_rate, avg_total_loss.dtype
            ),
            "reconstruct_empty_prediction_masks_rate": self._metric_tensor(
                cumulative_reconstruct_empty_prediction_masks_rate, avg_total_loss.dtype
            ),
            "detail_sufficient_caption_rate": self._metric_tensor(cumulative_detail_sufficient_caption_rate, avg_total_loss.dtype),
            "scene_spill_caption_rate": self._metric_tensor(cumulative_scene_spill_caption_rate, avg_total_loss.dtype),
            "teacher_regenerate_ce_applied_count": self._metric_tensor(self._cumulative_teacher_regenerate_ce_applied_count, avg_total_loss.dtype),
            "teacher_regenerate_suppressed_count": self._metric_tensor(self._cumulative_teacher_regenerate_suppressed_count, avg_total_loss.dtype),
            "teacher_regenerate_verified_count": self._metric_tensor(self._cumulative_teacher_regenerate_verified_count, avg_total_loss.dtype),
            "teacher_regenerate_rejected_count": self._metric_tensor(self._cumulative_teacher_regenerate_rejected_count, avg_total_loss.dtype),
            "teacher_regenerate_gate_pass_rate": self._metric_tensor(cumulative_teacher_regenerate_gate_pass_rate, avg_total_loss.dtype),
            "teacher_regenerate_verified_iou_mean": self._metric_tensor(
                cumulative_teacher_regenerate_verified_iou_mean, avg_total_loss.dtype
            ),
        }
        return metrics
