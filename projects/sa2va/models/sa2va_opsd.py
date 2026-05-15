import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModel
from transformers import AutoModel, AutoTokenizer, GenerationConfig


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]


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
    status: str


class Sa2VAOPSDModel(BaseModel):
    """Single-GPU OPSD wrapper around the public Sa2VA chat model."""

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
        student_generation_config=None,
        teacher_summary_template=None,
        reconstruct_question_template=None,
        device="cuda:0",
        use_flash_attn=True,
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
        self.teacher_model_path = None if not self.enable_teacher else (teacher_model_path or model_path)
        self.tokenizer_path = tokenizer_path or model_path
        self.teacher_temperature = teacher_temperature
        self.jsd_beta = jsd_beta
        self.privileged_iou_precision = privileged_iou_precision
        self.device = torch.device(device)
        self.use_flash_attn = use_flash_attn
        self.debug_print_limit = 3
        self._debug_print_count = 0

        self.teacher_summary_template = teacher_summary_template or (
            "You are given privileged verification feedback.\n"
            "Original task: {student_question}\n"
            "Student caption: {student_caption}\n"
            "Description generation status: {description_status}\n"
            "Reconstructed mask IoU with the reference mask: {iou:.4f}\n"
            "Reconstruction produced a valid mask: {has_mask}\n"
            "Use this privileged verification information to better model the student's caption tokens."
        )
        self.reconstruct_question_template = reconstruct_question_template or (
            "<image>\nPlease segment the region that matches this description: {caption}"
        )

        self._validate_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False,
        )
        self.model_dtype = self._resolve_torch_dtype(torch_dtype)
        self.student_model = self._load_model(self.model_path)
        self.teacher_model = None
        if self.enable_teacher:
            self.teacher_model = self._load_model(self.teacher_model_path)
            self._sync_teacher()

        generation_kwargs = {
            "max_new_tokens": 96,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if student_generation_config is not None:
            generation_kwargs.update(student_generation_config)
            self.student_generation_config = GenerationConfig(**generation_kwargs)
            self._apply_generation_overrides(self.student_model, generation_kwargs)
        else:
            self.student_generation_config = None

    def _validate_device(self):
        if self.device.type != "cuda":
            raise ValueError(f"Sa2VAOPSDModel only supports CUDA. Got {self.device}.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Sa2VA_OPSD training.")
        if torch.cuda.device_count() <= (self.device.index or 0):
            raise RuntimeError(
                f"Configured device {self.device} is not visible. "
                f"Visible device count={torch.cuda.device_count()}."
            )

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
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.model_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=self.use_flash_attn,
        )
        model.to(self.device)
        self._ensure_runtime_state(model)
        self._ensure_generation_ready(model)
        return model

    @staticmethod
    def _ensure_runtime_state(model):
        if not hasattr(model, "_count"):
            model._count = 0

    def _ensure_generation_ready(self, model):
        self._ensure_runtime_state(model)
        if not getattr(model, "init_prediction_config", False) or not hasattr(model, "stop_criteria"):
            model.preparing_for_generation(tokenizer=self.tokenizer)
        self._ensure_runtime_state(model)
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

    @staticmethod
    def _apply_generation_overrides(model, generation_kwargs):
        if not hasattr(model, "gen_config"):
            return
        for key, value in generation_kwargs.items():
            setattr(model.gen_config, key, value)

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

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        teacher_prefix = "teacher_model."
        return {k: v for k, v in state.items() if not k.startswith(teacher_prefix)}

    def load_state_dict(self, state_dict, strict=True):
        filtered_state = {
            k: v for k, v in state_dict.items() if not k.startswith("teacher_model.")
        }
        result = super().load_state_dict(filtered_state, strict=strict)
        if self.has_teacher_model():
            self._sync_teacher()
        return result

    @staticmethod
    def _to_numpy_mask(mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        return (np.asarray(mask) > 0).astype(np.uint8)

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

    def _format_mask_prompts_for_predict_forward(self, prompt_masks):
        stacked_masks = np.stack(
            [np.asarray(item, dtype=np.float32) for item in prompt_masks],
            axis=0,
        )
        return [stacked_masks]

    def _build_forward_inputs(self, model, image, prompt_masks, question_text):
        self._ensure_generation_ready(model)
        ori_image_size = image.size

        g_image = np.array(image)
        g_image = model.extra_image_processor.apply_image(g_image)
        g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(model.torch_dtype)
        g_pixel_values = torch.stack(
            [model.grounding_encoder.preprocess_image(g_image)]
        ).to(device=self.device, dtype=model.torch_dtype)

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

        mask_prompts, vp_token_str = self._create_region_prompt(model, prompt_masks)
        vp_overall_mask = torch.tensor([False] * (len(images) - 1) + [True], device=self.device)

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
            "g_pixel_values": g_pixel_values,
            "input_ids": ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt_masks": mask_prompts,
            "vp_overall_mask": vp_overall_mask,
            "ori_image_size": ori_image_size,
            "student_prompt": full_human_prompt,
        }

    def _build_forward_inputs_no_vp(self, model, image, text):
        self._ensure_generation_ready(model)
        ori_image_size = image.size

        g_image = np.array(image)
        g_image = model.extra_image_processor.apply_image(g_image)
        g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(model.torch_dtype)
        g_pixel_values = torch.stack(
            [model.grounding_encoder.preprocess_image(g_image)]
        ).to(device=self.device, dtype=model.torch_dtype)

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
        num_image_tokens = pixel_values.shape[0] * model.patch_token
        image_token_str = (
            f"{model.IMG_START_TOKEN}"
            f"{model.IMG_CONTEXT_TOKEN * num_image_tokens}"
            f"{model.IMG_END_TOKEN}\n"
        )
        input_text = text.replace("<image>", image_token_str)
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
            "g_pixel_values": g_pixel_values,
            "input_ids": ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt_masks": None,
            "vp_overall_mask": None,
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
            selected = input_ids.reshape(batch_size * seq_len) == model.img_context_token_id
            expected_tokens = int(selected.sum().item())
            flat_input_embeds[selected] = vit_embeds.reshape(-1, hidden_dim)[:expected_tokens]
            return flat_input_embeds.reshape(batch_size, seq_len, hidden_dim)

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
        actual_tokens = vp_embeds.shape[0]
        if actual_tokens < expected_tokens:
            raise RuntimeError(
                f"VP embed count mismatch for {type(model).__name__}: "
                f"expected {expected_tokens}, got {actual_tokens}."
            )
        flat_input_embeds[selected] = vp_embeds[:expected_tokens]
        return flat_input_embeds.reshape(batch_size, seq_len, hidden_dim)

    def _generate_with_model(self, model, mm_inputs, generation_config, output_hidden_states=False):
        inputs_embeds = self._compose_inputs_embeds(model, mm_inputs)
        return model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=mm_inputs["attention_mask"],
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            return_dict_in_generate=True,
            stopping_criteria=model.stop_criteria,
        )

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

    @staticmethod
    def _clean_caption_text(caption):
        caption = "" if caption is None else str(caption)
        caption = caption.replace("<|im_end|>", "")
        caption = caption.replace("<|end|>", "")
        caption = caption.replace("<|endoftext|>", "")
        caption = re.sub(r"<\|[^>\n]*\|>", " ", caption)
        caption = re.sub(r"<\|.*$", "", caption)
        caption = re.sub(r"<[^>\n]*$", "", caption)
        caption = re.sub(r"\s+", " ", caption).strip()
        for prefix in ("Sure, ", "Sure. ", "Certainly, "):
            if caption.startswith(prefix):
                caption = caption[len(prefix) :].strip()
        return caption

    @staticmethod
    def _infer_description_status(caption):
        if caption is None:
            return "decode_error"
        normalized = caption.strip()
        if not normalized:
            return "empty"
        lowered = re.sub(r"[\s\.\!\?]+", " ", normalized.lower()).strip()
        seg_style_patterns = {
            "[seg]",
            "it is [seg]",
            "the segmentation result is [seg]",
            "segmentation result is [seg]",
        }
        if lowered in seg_style_patterns:
            return "seg_style_answer"
        return "ok"

    def _encode_completion_from_caption(self, caption):
        encoded = self.tokenizer(
            caption,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return encoded

    @staticmethod
    def _mask_bbox(mask):
        if mask is None:
            return None
        mask = np.asarray(mask)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def _should_debug_print(self):
        if self._debug_print_count >= self.debug_print_limit:
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _debug_sample(
        self,
        *,
        student_question,
        raw_prediction,
        caption,
        description_status,
        reconstruct_question,
        raw_reconstruct_prediction,
        reconstruct_status,
        prediction_masks_count,
        pred_mask,
        gt_mask,
        iou,
    ):
        if not self._should_debug_print():
            return
        pred_sum = None if pred_mask is None else int(np.asarray(pred_mask).sum())
        gt_sum = None if gt_mask is None else int(np.asarray(gt_mask).sum())
        print(
            "[Sa2VA_OPSD_DEBUG] "
            f"student_question={student_question!r}\n"
            f"[Sa2VA_OPSD_DEBUG] raw_prediction={raw_prediction!r}\n"
            f"[Sa2VA_OPSD_DEBUG] clean_caption={caption!r}\n"
            f"[Sa2VA_OPSD_DEBUG] description_status={description_status}\n"
            f"[Sa2VA_OPSD_DEBUG] reconstruct_question={reconstruct_question!r}\n"
            f"[Sa2VA_OPSD_DEBUG] raw_reconstruct_prediction={raw_reconstruct_prediction!r}\n"
            f"[Sa2VA_OPSD_DEBUG] reconstruct_status={reconstruct_status}\n"
            f"[Sa2VA_OPSD_DEBUG] prediction_masks_count={prediction_masks_count}\n"
            f"[Sa2VA_OPSD_DEBUG] pred_mask_sum={pred_sum} gt_mask_sum={gt_sum}\n"
            f"[Sa2VA_OPSD_DEBUG] pred_mask_shape={None if pred_mask is None else tuple(np.asarray(pred_mask).shape)} "
            f"gt_mask_shape={None if gt_mask is None else tuple(np.asarray(gt_mask).shape)}\n"
            f"[Sa2VA_OPSD_DEBUG] pred_bbox={self._mask_bbox(pred_mask)} gt_bbox={self._mask_bbox(gt_mask)}\n"
            f"[Sa2VA_OPSD_DEBUG] iou={iou:.4f} "
            f"description_failed={description_status != 'ok'} "
            f"reconstruction_failed={reconstruct_status not in {'ok', 'skipped_invalid_description'}}"
        )
        self._debug_print_count += 1

    def _predict_forward_eval(self, model, **kwargs):
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return model.predict_forward(**kwargs)
        finally:
            if was_training:
                model.train()

    def _forward_sequence_with_model(self, model, image, prompt_masks, prompt_text, completion_ids):
        mm_inputs = self._build_forward_inputs(model, image, prompt_masks, prompt_text)
        prompt_len = mm_inputs["input_ids"].shape[1]
        full_ids = torch.cat([mm_inputs["input_ids"], completion_ids.to(self.device)], dim=1)
        full_attention_mask = torch.ones_like(full_ids, dtype=torch.bool)
        full_position_ids = torch.arange(full_ids.shape[1], device=self.device).unsqueeze(0)
        inputs_embeds = self._compose_inputs_embeds(model, mm_inputs, input_ids=full_ids)
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            position_ids=full_position_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits[:, prompt_len - 1 : -1, :]

    def generate_description(self, image, prompt_masks, student_question):
        formatted_prompt_masks = self._format_mask_prompts_for_predict_forward(prompt_masks)
        predict_dict = self._predict_forward_eval(
            self.student_model,
            image=image,
            text=student_question,
            past_text="",
            mask_prompts=formatted_prompt_masks,
            tokenizer=self.tokenizer,
        )
        raw_prediction = predict_dict.get("prediction", "")
        clean_caption = self._clean_caption_text(raw_prediction)
        status = self._infer_description_status(clean_caption)
        completion_ids = self._encode_completion_from_caption(clean_caption)
        return DescriptionResult(
            raw_prediction=raw_prediction,
            clean_caption=clean_caption,
            completion_ids=completion_ids,
            status=status,
        )

    def reconstruct_mask_from_description(self, image, caption, description_status):
        if description_status != "ok":
            return ReconstructionResult(
                pred_mask=None,
                question=None,
                raw_prediction="",
                prediction_masks_count=0,
                status="skipped_invalid_description",
            )

        reconstruct_question = self.reconstruct_question_template.format(caption=caption)
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
        if not prediction_masks:
            return ReconstructionResult(
                pred_mask=None,
                question=reconstruct_question,
                raw_prediction=raw_prediction,
                prediction_masks_count=prediction_masks_count,
                status="empty_prediction_masks",
            )

        first_mask = prediction_masks[0]
        if isinstance(first_mask, torch.Tensor):
            first_mask = first_mask.detach().cpu().numpy()
        first_mask = np.asarray(first_mask)
        if first_mask.ndim == 3 and first_mask.shape[0] == 1:
            first_mask = first_mask[0]

        pred_mask = self._to_numpy_mask(first_mask)
        status = "ok" if pred_mask.sum() > 0 else "empty_prediction_masks"
        return ReconstructionResult(
            pred_mask=pred_mask,
            question=reconstruct_question,
            raw_prediction=raw_prediction,
            prediction_masks_count=prediction_masks_count,
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

    def build_teacher_privileged_prompt(self, student_question, student_caption, iou, has_mask, description_status):
        clean_question = self._strip_image_placeholder(student_question)
        summary = self.teacher_summary_template.format(
            student_question=clean_question,
            student_caption=student_caption,
            description_status=description_status,
            iou=round(iou, self.privileged_iou_precision),
            has_mask="yes" if has_mask else "no",
        )
        return f"<image>\n{clean_question}\n\n{summary}"

    @staticmethod
    def generalized_jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0):
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_log_probs + torch.log1p(-beta_t),
                        teacher_log_probs + torch.log(beta_t),
                    ]
                ),
                dim=0,
            )
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

        return jsd.sum(dim=-1).mean()

    def compute_opsd_loss(self, image, prompt_masks, student_question, teacher_prompt, completion_ids):
        student_logits = self._forward_sequence_with_model(
            self.student_model,
            image,
            prompt_masks,
            student_question,
            completion_ids,
        )
        teacher_model = self.require_teacher_model("OPSD distillation")
        with torch.no_grad():
            teacher_logits = self._forward_sequence_with_model(
                teacher_model,
                image,
                prompt_masks,
                teacher_prompt,
                completion_ids,
            )
        return self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            beta=self.jsd_beta,
            temperature=self.teacher_temperature,
        )

    def forward(self, data, data_samples=None, mode="loss"):
        del data_samples, mode

        images = data["images"]
        prompt_masks_batch = data["prompt_masks"]
        student_questions = data["student_questions"]
        gt_masks = data["gt_masks"]

        total_jsd = None
        total_iou = 0.0
        valid_count = 0
        description_ok_count = 0
        description_empty_count = 0
        description_seg_style_count = 0
        reconstruct_ok_count = 0
        reconstruct_failed_count = 0
        last_caption = ""
        last_teacher_prompt = ""

        for image, prompt_masks, student_question, gt_mask in zip(
            images, prompt_masks_batch, student_questions, gt_masks
        ):
            description = self.generate_description(
                image=image,
                prompt_masks=prompt_masks,
                student_question=student_question,
            )
            if description.status == "ok":
                description_ok_count += 1
            elif description.status == "empty":
                description_empty_count += 1
            elif description.status == "seg_style_answer":
                description_seg_style_count += 1

            reconstruction = self.reconstruct_mask_from_description(
                image=image,
                caption=description.clean_caption,
                description_status=description.status,
            )
            iou = self._compute_iou(gt_mask, reconstruction.pred_mask)
            if reconstruction.status == "ok":
                reconstruct_ok_count += 1
            elif reconstruction.status not in {"ok", "skipped_invalid_description"}:
                reconstruct_failed_count += 1

            self._debug_sample(
                student_question=student_question,
                raw_prediction=description.raw_prediction,
                caption=description.clean_caption,
                description_status=description.status,
                reconstruct_question=reconstruction.question,
                raw_reconstruct_prediction=reconstruction.raw_prediction,
                reconstruct_status=reconstruction.status,
                prediction_masks_count=reconstruction.prediction_masks_count,
                pred_mask=reconstruction.pred_mask,
                gt_mask=self._to_numpy_mask(gt_mask),
                iou=iou,
            )

            teacher_prompt = self.build_teacher_privileged_prompt(
                student_question=student_question,
                student_caption=description.clean_caption,
                iou=iou,
                has_mask=reconstruction.pred_mask is not None,
                description_status=description.status,
            )
            last_caption = description.clean_caption
            last_teacher_prompt = teacher_prompt

            if description.status != "ok":
                continue

            jsd_loss = self.compute_opsd_loss(
                image=image,
                prompt_masks=prompt_masks,
                student_question=student_question,
                teacher_prompt=teacher_prompt,
                completion_ids=description.completion_ids,
            )
            total_jsd = jsd_loss if total_jsd is None else total_jsd + jsd_loss
            total_iou += iou
            valid_count += 1

        if valid_count == 0:
            zero = next(self.student_model.parameters()).sum() * 0.0
            return {
                "loss_opsd_jsd": zero,
                "verifier_iou": zero,
                "description_ok_count": zero,
                "description_empty_count": torch.tensor(float(description_empty_count), device=self.device, dtype=zero.dtype),
                "description_seg_style_count": torch.tensor(float(description_seg_style_count), device=self.device, dtype=zero.dtype),
                "reconstruct_ok_count": torch.tensor(float(reconstruct_ok_count), device=self.device, dtype=zero.dtype),
                "reconstruct_failed_count": torch.tensor(float(reconstruct_failed_count), device=self.device, dtype=zero.dtype),
            }

        avg_jsd = total_jsd / valid_count
        avg_iou = torch.tensor(total_iou / valid_count, device=self.device, dtype=avg_jsd.dtype)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(
                    f"[Sa2VA_OPSD_2B] caption='{last_caption}' iou={total_iou / valid_count:.4f}\n"
                    f"[Sa2VA_OPSD_2B] teacher_prompt={last_teacher_prompt}"
                )
        else:
            print(
                f"[Sa2VA_OPSD_2B] caption='{last_caption}' iou={total_iou / valid_count:.4f}\n"
                f"[Sa2VA_OPSD_2B] teacher_prompt={last_teacher_prompt}"
            )
        return {
            "loss_opsd_jsd": avg_jsd,
            "verifier_iou": avg_iou,
            "description_ok_count": torch.tensor(float(description_ok_count), device=self.device, dtype=avg_jsd.dtype),
            "description_empty_count": torch.tensor(float(description_empty_count), device=self.device, dtype=avg_jsd.dtype),
            "description_seg_style_count": torch.tensor(float(description_seg_style_count), device=self.device, dtype=avg_jsd.dtype),
            "reconstruct_ok_count": torch.tensor(float(reconstruct_ok_count), device=self.device, dtype=avg_jsd.dtype),
            "reconstruct_failed_count": torch.tensor(float(reconstruct_failed_count), device=self.device, dtype=avg_jsd.dtype),
        }
