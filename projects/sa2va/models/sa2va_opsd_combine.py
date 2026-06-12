import random
import re
from dataclasses import dataclass

import numpy as np
import torch

from projects.sa2va.evaluation.teacher_diagnosis_common import (
    GRPO_POSITIVE_ROUTE,
    ON_POLICY_DISTILL_ROUTE,
    TEACHER_REGENERATE_ROUTE,
    build_mask_relation_context,
)
from projects.sa2va.models.sa2va_opsd_v2 import (
    ConfuserSelectionResult,
    DescriptionResult,
)
from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3


@dataclass
class RouteDecision:
    route: str
    manifest_score_primary: float = 0.0
    manifest_score_secondary: float = 0.0
    dlc_choose_one_correct: bool = False
    dlc_choose_one_confidence: float = 0.0
    referring_iou: float = 0.0
    failure_reason: str = ""


@dataclass
class StudentModeOutputs:
    dlc_result: DescriptionResult | None = None
    referring_result: DescriptionResult | None = None
    dlc_raw_prediction: str = ""
    referring_raw_prediction: str = ""
    combine_raw_prediction: str = ""


@dataclass
class TeacherCaptionResult:
    prompt: str
    result: DescriptionResult


class Sa2VAOPSDCombineModel(Sa2VAOPSDModelV3):
    def __init__(
        self,
        *args,
        train_mode="dlc",
        combine_choose_one_low_conf_threshold=0.2,
        dlc_onpolicy_conf_threshold=0.5,
        referring_iou_sft_threshold=0.5,
        referring_iou_grpo_threshold=0.85,
        combine_grpo_choose_one_conf_threshold=0.5,
        combine_grpo_iou_threshold=0.85,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        train_mode = str(train_mode).strip().lower()
        if train_mode not in {"dlc", "referring", "combine"}:
            raise ValueError(f"Unsupported train_mode={train_mode!r}.")
        if not (0.0 <= float(combine_choose_one_low_conf_threshold) <= float(dlc_onpolicy_conf_threshold) <= 1.0):
            raise ValueError("Expected 0 <= combine_choose_one_low_conf_threshold <= dlc_onpolicy_conf_threshold <= 1.")
        if not (0.0 <= float(referring_iou_sft_threshold) <= float(referring_iou_grpo_threshold) <= 1.0):
            raise ValueError("Expected 0 <= referring_iou_sft_threshold <= referring_iou_grpo_threshold <= 1.")
        if not (0.0 <= float(combine_grpo_iou_threshold) <= 1.0):
            raise ValueError("Expected 0 <= combine_grpo_iou_threshold <= 1.")
        self.train_mode = train_mode
        self.combine_choose_one_low_conf_threshold = float(combine_choose_one_low_conf_threshold)
        self.dlc_onpolicy_conf_threshold = float(dlc_onpolicy_conf_threshold)
        self.referring_iou_sft_threshold = float(referring_iou_sft_threshold)
        self.referring_iou_grpo_threshold = float(referring_iou_grpo_threshold)
        self.combine_grpo_choose_one_conf_threshold = float(combine_grpo_choose_one_conf_threshold)
        self.combine_grpo_iou_threshold = float(combine_grpo_iou_threshold)

    def build_dlc_student_prompt(self, student_question):
        clean_question = self._strip_image_placeholder(student_question)
        return (
            "<image>\n"
            "Describe only the masked target as one detailed localized caption.\n"
            "Focus on visible, distinguishing, local details that separate the target from nearby similar regions.\n"
            "Do not mention segmentation, masks, labels, or [SEG].\n"
            f"Task: {clean_question}"
        )

    def build_referring_student_prompt(self, student_question):
        clean_question = self._strip_image_placeholder(student_question)
        return (
            "<image>\n"
            "Write one short referring expression for the masked target.\n"
            "The expression should help a segmentation model locate the target precisely.\n"
            "Keep it short, concrete, and visually grounded. Do not mention masks or [SEG].\n"
            f"Task: {clean_question}"
        )

    def build_combine_student_prompt(self, student_question):
        clean_question = self._strip_image_placeholder(student_question)
        return (
            "<image>\n"
            "Generate two captions for the same masked target.\n"
            "The first is a detailed localized caption. The second is a short referring expression.\n"
            "Output exactly two lines in this format and nothing else:\n"
            "DLC: ...\n"
            "REFERRING: ...\n"
            "Do not mention masks, segmentation, or [SEG].\n"
            f"Task: {clean_question}"
        )

    def _generate_caption_with_prompt(
        self,
        model,
        *,
        image,
        mask_prompts,
        prompt_text,
        apply_mask_focus=True,
        max_new_tokens=None,
    ):
        formatted_mask_prompts = self._format_mask_prompts_for_predict_forward(mask_prompts)
        prompt_image = self._build_mask_focused_image(image, mask_prompts) if apply_mask_focus else image
        predict_dict = self._predict_forward_eval(
            model,
            image=prompt_image,
            text=prompt_text,
            past_text="",
            mask_prompts=formatted_mask_prompts,
            tokenizer=self.tokenizer,
            max_new_tokens=self.description_max_new_tokens if max_new_tokens is None else int(max_new_tokens),
            do_sample=False,
            repetition_penalty=self.description_repetition_penalty,
            no_repeat_ngram_size=self.description_no_repeat_ngram_size,
            bad_words_ids=self._caption_bad_words_ids,
        )
        raw_prediction = "" if predict_dict is None else str(predict_dict.get("prediction", ""))
        clean_caption = self._clean_caption_text(raw_prediction)
        completion_ids = self._encode_completion_from_caption(clean_caption, model=model)
        return self._finalize_description_result(
            raw_prediction=raw_prediction,
            clean_caption=clean_caption,
            completion_ids=completion_ids,
        )

    @staticmethod
    def _extract_labeled_text(raw_text, label):
        pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*(?=^\s*[A-Z_]+\s*:|\Z)"
        match = re.search(pattern, raw_text or "", flags=re.DOTALL | re.MULTILINE)
        return "" if match is None else match.group(1).strip()

    def _finalize_labeled_caption(self, *, raw_prediction, labeled_text):
        clean_caption = self._clean_caption_text(labeled_text)
        completion_ids = self._encode_completion_from_caption(clean_caption)
        return self._finalize_description_result(
            raw_prediction=raw_prediction,
            clean_caption=clean_caption,
            completion_ids=completion_ids,
        )

    def generate_student_mode_outputs(self, *, image, mask_prompts, student_question):
        if self.train_mode == "dlc":
            prompt = self.build_dlc_student_prompt(student_question)
            dlc_result = self._generate_caption_with_prompt(
                self.student_model,
                image=image,
                mask_prompts=mask_prompts,
                prompt_text=prompt,
                apply_mask_focus=True,
            )
            return StudentModeOutputs(
                dlc_result=dlc_result,
                dlc_raw_prediction=dlc_result.raw_prediction,
            )
        if self.train_mode == "referring":
            prompt = self.build_referring_student_prompt(student_question)
            referring_result = self._generate_caption_with_prompt(
                self.student_model,
                image=image,
                mask_prompts=mask_prompts,
                prompt_text=prompt,
                apply_mask_focus=True,
            )
            return StudentModeOutputs(
                referring_result=referring_result,
                referring_raw_prediction=referring_result.raw_prediction,
            )
        prompt = self.build_combine_student_prompt(student_question)
        formatted_mask_prompts = self._format_mask_prompts_for_predict_forward(mask_prompts)
        prompt_image = self._build_mask_focused_image(image, mask_prompts)
        predict_dict = self._predict_forward_eval(
            self.student_model,
            image=prompt_image,
            text=prompt,
            past_text="",
            mask_prompts=formatted_mask_prompts,
            tokenizer=self.tokenizer,
            max_new_tokens=self.description_max_new_tokens * 2,
            do_sample=False,
            repetition_penalty=self.description_repetition_penalty,
            no_repeat_ngram_size=self.description_no_repeat_ngram_size,
            bad_words_ids=self._caption_bad_words_ids,
        )
        raw_prediction = "" if predict_dict is None else str(predict_dict.get("prediction", ""))
        dlc_text = self._extract_labeled_text(raw_prediction, "DLC")
        referring_text = self._extract_labeled_text(raw_prediction, "REFERRING")
        dlc_result = self._finalize_labeled_caption(raw_prediction=raw_prediction, labeled_text=dlc_text)
        referring_result = self._finalize_labeled_caption(raw_prediction=raw_prediction, labeled_text=referring_text)
        return StudentModeOutputs(
            dlc_result=dlc_result,
            referring_result=referring_result,
            dlc_raw_prediction=dlc_result.raw_prediction,
            referring_raw_prediction=referring_result.raw_prediction,
            combine_raw_prediction=raw_prediction,
        )

    def _sample_grpo_captions_with_prompt(self, *, model, image, prompt_masks, prompt_text):
        target_rollout_count = int(self.grpo_group_size)
        if target_rollout_count <= 0:
            return []
        descriptions = []
        for _ in range(target_rollout_count):
            descriptions.append(
                self._generate_caption_with_prompt(
                    model,
                    image=image,
                    mask_prompts=prompt_masks,
                    prompt_text=prompt_text,
                    apply_mask_focus=True,
                )
            )
        return descriptions

    def run_dlc_choose_one(self, *, image, caption, gt_mask, confuser_candidate_masks):
        if not caption:
            return {
                "selection": None,
                "wrong_confuser_mask": None,
                "failure_reason": "empty_caption",
            }
        confuser_masks, confuser_meta = self._select_confuser_masks(
            gt_mask=gt_mask,
            candidate_masks=confuser_candidate_masks,
        )
        if confuser_masks is None:
            return {
                "selection": None,
                "wrong_confuser_mask": None,
                "failure_reason": confuser_meta.get("grpo_skip_reason") or "missing_confuser_masks",
            }
        option_masks = [self._to_numpy_mask(gt_mask), *[self._to_numpy_mask(mask) for mask in confuser_masks]]
        random.shuffle(option_masks)
        correct_option_idx = next(
            idx for idx, candidate_mask in enumerate(option_masks)
            if float(self._compute_iou(gt_mask, candidate_mask)) >= self.grpo_confuser_duplicate_iou_threshold
        )
        selection = self._score_caption_against_mask_options(
            model=self.student_model,
            image=image,
            option_masks=np.stack(option_masks, axis=0).astype(np.float32),
            caption=caption,
            correct_option_idx=correct_option_idx,
        )
        wrong_confuser_mask = None
        if not selection.selected_correct:
            wrong_confuser_mask = self._to_numpy_mask(option_masks[selection.predicted_option_idx])
        return {
            "selection": selection,
            "wrong_confuser_mask": wrong_confuser_mask,
            "failure_reason": "",
        }

    def determine_route_decision(
        self,
        *,
        dlc_result,
        referring_result,
        dlc_selection_payload,
        referring_iou,
    ):
        if self.train_mode == "dlc":
            selection = dlc_selection_payload.get("selection")
            failure_reason = dlc_selection_payload.get("failure_reason", "")
            if selection is None:
                return RouteDecision(route=TEACHER_REGENERATE_ROUTE, failure_reason=failure_reason)
            confidence = float(selection.correct_option_prob)
            if not selection.selected_correct:
                route = TEACHER_REGENERATE_ROUTE
            elif confidence < self.dlc_onpolicy_conf_threshold:
                route = ON_POLICY_DISTILL_ROUTE
            else:
                route = GRPO_POSITIVE_ROUTE
            return RouteDecision(
                route=route,
                manifest_score_primary=confidence,
                dlc_choose_one_correct=bool(selection.selected_correct),
                dlc_choose_one_confidence=confidence,
                failure_reason=failure_reason,
            )
        if self.train_mode == "referring":
            iou = float(referring_iou)
            if iou < self.referring_iou_sft_threshold:
                route = TEACHER_REGENERATE_ROUTE
            elif iou < self.referring_iou_grpo_threshold:
                route = ON_POLICY_DISTILL_ROUTE
            else:
                route = GRPO_POSITIVE_ROUTE
            return RouteDecision(
                route=route,
                manifest_score_primary=iou,
                referring_iou=iou,
            )
        selection = dlc_selection_payload.get("selection")
        failure_reason = dlc_selection_payload.get("failure_reason", "")
        confidence = 0.0 if selection is None else float(selection.correct_option_prob)
        selected_correct = False if selection is None else bool(selection.selected_correct)
        iou = float(referring_iou)
        if (not selected_correct) or confidence < self.combine_choose_one_low_conf_threshold:
            route = TEACHER_REGENERATE_ROUTE
        elif confidence < self.combine_grpo_choose_one_conf_threshold or iou < self.combine_grpo_iou_threshold:
            route = ON_POLICY_DISTILL_ROUTE
        else:
            route = GRPO_POSITIVE_ROUTE
        return RouteDecision(
            route=route,
            manifest_score_primary=confidence,
            manifest_score_secondary=iou,
            dlc_choose_one_correct=selected_correct,
            dlc_choose_one_confidence=confidence,
            referring_iou=iou,
            failure_reason=failure_reason,
        )

    def build_teacher_privileged_prompt_dlc(
        self,
        *,
        student_question,
        student_caption,
        gt_mask,
        wrong_confuser_mask,
        route,
    ):
        clean_question = self._strip_image_placeholder(student_question)
        relation_context = build_mask_relation_context(
            model=self,
            gt_mask=gt_mask,
            ref_mask=wrong_confuser_mask if wrong_confuser_mask is not None else self._zero_ref_mask_like(gt_mask),
        )
        return (
            "<image>\n"
            "You are supervising a mask-to-detailed-caption task.\n"
            "region1 is the gtmask. region2 is the wrong confuser region selected by the student caption.\n"
            f"Teacher route: {route}\n"
            f"Student prompt: {clean_question}\n"
            f"Student caption: {student_caption}\n"
            f"Target summary (region1): {relation_context['gt_summary']}\n"
            f"Wrong confuser summary (region2): {relation_context['ref_summary']}\n"
            f"Shared overlap summary: {relation_context['overlap_summary']}\n"
            f"Target-only summary: {relation_context['gt_only_summary']}\n"
            f"Confuser-only summary: {relation_context['ref_only_summary']}\n"
            "Write one corrected detailed localized caption for region1 only.\n"
            "Use visible, distinguishing, local details that separate region1 from region2.\n"
            "Do not mention masks, segmentation, labels, or [SEG]."
        )

    def build_teacher_privileged_prompt_referring(
        self,
        *,
        student_question,
        student_caption,
        gt_mask,
        ref_mask,
        route,
    ):
        clean_question = self._strip_image_placeholder(student_question)
        relation_context = build_mask_relation_context(
            model=self,
            gt_mask=gt_mask,
            ref_mask=ref_mask if ref_mask is not None else self._zero_ref_mask_like(gt_mask),
        )
        return (
            "<image>\n"
            "You are supervising a caption-to-mask reconstruction task.\n"
            "region1 is the gtmask. region2 is the reconstructed refmask from the student's short referring caption.\n"
            f"Teacher route: {route}\n"
            f"Student prompt: {clean_question}\n"
            f"Student caption: {student_caption}\n"
            f"Target summary (region1): {relation_context['gt_summary']}\n"
            f"Reconstructed summary (region2): {relation_context['ref_summary']}\n"
            f"Shared overlap summary: {relation_context['overlap_summary']}\n"
            f"Target-only summary: {relation_context['gt_only_summary']}\n"
            f"Ref-only summary: {relation_context['ref_only_summary']}\n"
            "Write one short referring expression for region1 that is easier to reconstruct correctly.\n"
            "Keep it short, concrete, visually grounded, and target-specific.\n"
            "Do not mention masks, segmentation, labels, or [SEG]."
        )

    def build_teacher_privileged_prompt_combine_dlc(self, **kwargs):
        return self.build_teacher_privileged_prompt_dlc(**kwargs)

    def build_teacher_privileged_prompt_combine_referring(self, **kwargs):
        return self.build_teacher_privileged_prompt_referring(**kwargs)

    def _generate_teacher_caption_result(
        self,
        *,
        prompt,
        image,
        teacher_prompt_masks,
        max_new_tokens=None,
    ):
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        clean_caption = self._clean_caption_text(raw_prediction)
        completion_ids = self._encode_completion_from_caption(clean_caption)
        result = self._finalize_description_result(
            raw_prediction=raw_prediction,
            clean_caption=clean_caption,
            completion_ids=completion_ids,
        )
        return TeacherCaptionResult(prompt=prompt, result=result)

    def _route_prompt_for_mode(
        self,
        *,
        student_question,
        student_caption,
        gt_mask,
        ref_mask=None,
        wrong_confuser_mask=None,
        route,
        branch,
    ):
        if branch == "dlc":
            return self.build_teacher_privileged_prompt_dlc(
                student_question=student_question,
                student_caption=student_caption,
                gt_mask=gt_mask,
                wrong_confuser_mask=wrong_confuser_mask,
                route=route,
            )
        return self.build_teacher_privileged_prompt_referring(
            student_question=student_question,
            student_caption=student_caption,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            route=route,
        )

    def compute_referring_grpo_loss(
        self,
        *,
        image,
        prompt_masks,
        student_question,
        gt_mask,
        force_dummy=False,
        dummy_reason=None,
        dummy_completion_ids=None,
    ):
        old_policy_model = self.require_old_policy_model("GRPO referring")

        def _dummy_result(skip_reason):
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
                    old_token_log_probs = self._token_log_probs_from_logits(old_policy_logits, completion_ids)
            old_token_log_probs = self._materialize_autograd_input(old_token_log_probs.detach())
            student_logits = self._forward_sequence_batch_with_model(
                self.student_model,
                image,
                prompt_masks,
                student_question,
                completion_ids,
                apply_mask_focus=True,
            )
            current_token_log_probs = self._token_log_probs_from_logits(student_logits, completion_ids)
            ratio = torch.exp(current_token_log_probs - old_token_log_probs)
            sample_losses = -(ratio * 0.0).mean(dim=-1)
            return sample_losses.mean(), {
                "reward_sum": 0.0,
                "reward_count": 0,
                "reward_raw_sum": 0.0,
                "reward_raw_count": 0,
                "skip_reason": str(skip_reason),
            }

        if force_dummy:
            return _dummy_result(dummy_reason or "forced_dummy")
        descriptions = self._sample_grpo_captions_with_prompt(
            model=old_policy_model,
            image=image,
            prompt_masks=prompt_masks,
            prompt_text=student_question,
        )
        rollout_entries = []
        for description in descriptions:
            if description.completion_ids.shape[1] == 0 or not self._is_caption_trainable_for_student_losses(description):
                continue
            reconstruction = self.reconstruct_mask(
                image=image,
                caption=description.clean_caption,
                description_status=description.status,
                gt_mask=gt_mask,
            )
            reward_value = 0.0 if reconstruction.pred_mask is None else float(self._compute_iou(gt_mask, reconstruction.pred_mask))
            with self._temporary_eval_model(old_policy_model):
                with torch.inference_mode():
                    old_policy_logits = self._forward_sequence_with_model(
                        old_policy_model,
                        image,
                        prompt_masks,
                        student_question,
                        description.completion_ids,
                        apply_mask_focus=True,
                    )
                    old_token_log_probs = self._token_log_probs_from_logits(old_policy_logits, description.completion_ids)
            rollout_entries.append(
                {
                    "completion_ids": description.completion_ids,
                    "old_token_log_probs": self._materialize_autograd_input(old_token_log_probs.detach()),
                    "reward_value": reward_value,
                }
            )
        if not rollout_entries:
            return _dummy_result("empty_rollout_entries")
        reward_tensor = torch.tensor(
            [entry["reward_value"] for entry in rollout_entries],
            device=self.device,
            dtype=rollout_entries[0]["old_token_log_probs"].dtype,
        )
        reward_span = float((reward_tensor.max() - reward_tensor.min()).item())
        if reward_span < self.grpo_advantage_eps:
            advantages = torch.zeros_like(reward_tensor)
        else:
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
        current_token_log_probs_batch = self._token_log_probs_from_logits(student_logits, completion_batch)
        log_ratio = (current_token_log_probs_batch - old_token_log_probs_batch).clamp(min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped_ratio = ratio.clamp(1.0 - self.grpo_clip_eps, 1.0 + self.grpo_clip_eps)
        advantage_batch = advantages.unsqueeze(1)
        surrogate = torch.min(ratio * advantage_batch, clipped_ratio * advantage_batch)
        token_weights = completion_mask.to(dtype=surrogate.dtype)
        sample_losses = -(surrogate * token_weights).sum(dim=-1) / token_weights.sum(dim=-1).clamp_min(1.0)
        if sample_losses.numel() == 0:
            return _dummy_result("non_finite_grpo_losses")
        return sample_losses.mean(), {
            "reward_sum": float(reward_tensor.sum().item()),
            "reward_count": len(rollout_entries),
            "reward_raw_sum": float(reward_tensor.sum().item()),
            "reward_raw_count": int(len(rollout_entries)),
            "skip_reason": None,
        }

    def compute_referring_grpo_losses_batch(self, batch_items):
        sample_losses = []
        reward_sum = 0.0
        reward_count = 0
        reward_raw_sum = 0.0
        reward_raw_count = 0
        dummy_reasons = []
        for item in batch_items:
            sample_loss, meta = self.compute_referring_grpo_loss(
                image=item["image"],
                prompt_masks=item["prompt_masks"],
                student_question=item["student_question"],
                gt_mask=item["gt_mask"],
                force_dummy=bool(item.get("is_dummy", False)),
                dummy_reason=item.get("dummy_reason"),
                dummy_completion_ids=item.get("completion_ids"),
            )
            reward_sum += float(meta.get("reward_sum", 0.0))
            reward_count += int(meta.get("reward_count", 0))
            reward_raw_sum += float(meta.get("reward_raw_sum", 0.0))
            reward_raw_count += int(meta.get("reward_raw_count", 0))
            if meta.get("skip_reason"):
                dummy_reasons.append(str(meta["skip_reason"]))
            if sample_loss is not None:
                sample_losses.append(sample_loss * float(item.get("loss_weight", 1.0)))
        if not sample_losses:
            return self._placeholder_loss_vector(
                len(batch_items),
                reason=f"ref-grpo-empty-sample-losses dummy_reasons={','.join(dummy_reasons) or 'unknown'}",
            ), {
                "reward_sum": reward_sum,
                "reward_count": reward_count,
                "reward_raw_sum": reward_raw_sum,
                "reward_raw_count": reward_raw_count,
            }
        return torch.stack(sample_losses), {
            "reward_sum": reward_sum,
            "reward_count": reward_count,
            "reward_raw_sum": reward_raw_sum,
            "reward_raw_count": reward_raw_count,
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
        route_teacher_count = 0
        route_onpolicy_count = 0
        route_grpo_count = 0
        dlc_choose_one_correct_count = 0
        dlc_choose_one_confidence_sum = 0.0
        referring_iou_sum = 0.0
        referring_seg_correct_count = 0
        combine_dual_caption_valid_count = 0
        combine_dual_supervision_applied_count = 0
        regen_entries = []
        onpolicy_entries = []
        grpo_entries = []
        ref_grpo_entries = []
        rank_debug_records = []
        last_debug = {}

        for image, prompt_masks, student_question, gt_mask, confuser_candidate_masks, sample_key, route_from_manifest in zip(
            images, prompt_masks_batch, student_questions, gt_masks, confuser_candidate_masks_batch, sample_keys, routes
        ):
            gt_mask_np = self._to_numpy_mask(gt_mask)
            if int(gt_mask_np.sum()) == 0:
                continue

            outputs = self.generate_student_mode_outputs(
                image=image,
                mask_prompts=prompt_masks,
                student_question=student_question,
            )
            dlc_result = outputs.dlc_result
            referring_result = outputs.referring_result
            dlc_selection_payload = {"selection": None, "wrong_confuser_mask": None, "failure_reason": ""}
            referring_iou = 0.0
            ref_mask_np = self._zero_ref_mask_like(gt_mask_np)
            if dlc_result is not None and self._is_caption_trainable_for_student_losses(dlc_result):
                dlc_selection_payload = self.run_dlc_choose_one(
                    image=image,
                    caption=dlc_result.clean_caption,
                    gt_mask=gt_mask_np,
                    confuser_candidate_masks=confuser_candidate_masks,
                )
            if referring_result is not None and self._is_caption_trainable_for_student_losses(referring_result):
                reconstruction = self.reconstruct_mask(
                    image=image,
                    caption=referring_result.clean_caption,
                    description_status=referring_result.status,
                    gt_mask=gt_mask_np,
                )
                if reconstruction.pred_mask is not None:
                    ref_mask_np = self._to_numpy_mask(reconstruction.pred_mask)
                    referring_iou = float(self._compute_iou(gt_mask_np, ref_mask_np))
            route_decision = self.determine_route_decision(
                dlc_result=dlc_result,
                referring_result=referring_result,
                dlc_selection_payload=dlc_selection_payload,
                referring_iou=referring_iou,
            )
            online_route = route_decision.route
            loss_family = self._resolve_loss_family(
                batch_route,
                route_from_manifest=route_from_manifest,
                online_route=online_route,
            )
            routed_count += 1
            total_iou += float(referring_iou)
            if route_decision.dlc_choose_one_correct:
                dlc_choose_one_correct_count += 1
            dlc_choose_one_confidence_sum += float(route_decision.dlc_choose_one_confidence)
            referring_iou_sum += float(referring_iou)
            if referring_iou >= self.referring_iou_sft_threshold:
                referring_seg_correct_count += 1
            if self.train_mode == "combine":
                dual_valid = bool(
                    dlc_result is not None and self._is_caption_trainable_for_student_losses(dlc_result)
                    and referring_result is not None and self._is_caption_trainable_for_student_losses(referring_result)
                )
                if dual_valid:
                    combine_dual_caption_valid_count += 1

            wrong_confuser_mask = dlc_selection_payload.get("wrong_confuser_mask")
            if loss_family == TEACHER_REGENERATE_ROUTE:
                route_teacher_count += 1
                if self.train_mode in {"dlc", "combine"} and dlc_result is not None:
                    teacher_prompt = self._route_prompt_for_mode(
                        student_question=student_question,
                        student_caption=dlc_result.clean_caption,
                        gt_mask=gt_mask_np,
                        wrong_confuser_mask=wrong_confuser_mask,
                        route=loss_family,
                        branch="dlc",
                    )
                    teacher_masks = self._build_teacher_prompt_masks(
                        gt_mask_np,
                        wrong_confuser_mask if wrong_confuser_mask is not None else self._zero_ref_mask_like(gt_mask_np),
                    )
                    teacher_dlc = self._generate_teacher_caption_result(
                        prompt=teacher_prompt,
                        image=image,
                        teacher_prompt_masks=teacher_masks,
                    )
                    if self._is_caption_trainable_for_student_losses(teacher_dlc.result):
                        regen_entries.append(
                            {
                                "image": image,
                                "prompt_masks": prompt_masks,
                                "student_question": self.build_dlc_student_prompt(student_question),
                                "completion_ids": teacher_dlc.result.completion_ids,
                                "loss_weight": 1.0,
                            }
                        )
                        if self.train_mode == "combine":
                            combine_dual_supervision_applied_count += 1
                if self.train_mode in {"referring", "combine"} and referring_result is not None:
                    teacher_prompt = self._route_prompt_for_mode(
                        student_question=student_question,
                        student_caption=referring_result.clean_caption,
                        gt_mask=gt_mask_np,
                        ref_mask=ref_mask_np,
                        route=loss_family,
                        branch="referring",
                    )
                    teacher_masks = self._build_teacher_prompt_masks(gt_mask_np, ref_mask_np)
                    teacher_ref = self._generate_teacher_caption_result(
                        prompt=teacher_prompt,
                        image=image,
                        teacher_prompt_masks=teacher_masks,
                    )
                    if self._is_caption_trainable_for_student_losses(teacher_ref.result):
                        regen_entries.append(
                            {
                                "image": image,
                                "prompt_masks": prompt_masks,
                                "student_question": self.build_referring_student_prompt(student_question),
                                "completion_ids": teacher_ref.result.completion_ids,
                                "loss_weight": 1.0,
                            }
                        )
                        if self.train_mode == "combine":
                            combine_dual_supervision_applied_count += 1
            elif loss_family == ON_POLICY_DISTILL_ROUTE:
                route_onpolicy_count += 1
                if self.train_mode in {"dlc", "combine"} and dlc_result is not None and self._is_caption_trainable_for_student_losses(dlc_result):
                    teacher_prompt = self._route_prompt_for_mode(
                        student_question=student_question,
                        student_caption=dlc_result.clean_caption,
                        gt_mask=gt_mask_np,
                        wrong_confuser_mask=wrong_confuser_mask,
                        route=loss_family,
                        branch="dlc",
                    )
                    teacher_masks = self._build_teacher_prompt_masks(
                        gt_mask_np,
                        wrong_confuser_mask if wrong_confuser_mask is not None else self._zero_ref_mask_like(gt_mask_np),
                    )
                    onpolicy_entries.append(
                        {
                            "image": image,
                            "prompt_masks": prompt_masks,
                            "student_question": self.build_dlc_student_prompt(student_question),
                            "teacher_prompt": teacher_prompt,
                            "completion_ids": dlc_result.completion_ids,
                            "teacher_prompt_masks": teacher_masks,
                            "iou": max(1.0 - route_decision.dlc_choose_one_confidence, 0.0),
                        }
                    )
                if self.train_mode in {"referring", "combine"} and referring_result is not None and self._is_caption_trainable_for_student_losses(referring_result):
                    teacher_prompt = self._route_prompt_for_mode(
                        student_question=student_question,
                        student_caption=referring_result.clean_caption,
                        gt_mask=gt_mask_np,
                        ref_mask=ref_mask_np,
                        route=loss_family,
                        branch="referring",
                    )
                    teacher_masks = self._build_teacher_prompt_masks(gt_mask_np, ref_mask_np)
                    onpolicy_entries.append(
                        {
                            "image": image,
                            "prompt_masks": prompt_masks,
                            "student_question": self.build_referring_student_prompt(student_question),
                            "teacher_prompt": teacher_prompt,
                            "completion_ids": referring_result.completion_ids,
                            "teacher_prompt_masks": teacher_masks,
                            "iou": float(referring_iou),
                        }
                    )
            else:
                route_grpo_count += 1
                if self.train_mode == "dlc":
                    grpo_entries.append(
                        {
                            "image": image,
                            "prompt_masks": prompt_masks,
                            "student_question": self.build_dlc_student_prompt(student_question),
                            "gt_mask": gt_mask_np,
                            "confuser_candidate_masks": confuser_candidate_masks,
                        }
                    )
                elif self.train_mode == "referring":
                    ref_grpo_entries.append(
                        {
                            "image": image,
                            "prompt_masks": prompt_masks,
                            "student_question": self.build_referring_student_prompt(student_question),
                            "gt_mask": gt_mask_np,
                        }
                    )
                else:
                    grpo_entries.append(
                        {
                            "image": image,
                            "prompt_masks": prompt_masks,
                            "student_question": self.build_dlc_student_prompt(student_question),
                            "gt_mask": gt_mask_np,
                            "confuser_candidate_masks": confuser_candidate_masks,
                        }
                    )
                    if referring_result is not None and self._is_caption_trainable_for_student_losses(referring_result):
                        teacher_prompt = self._route_prompt_for_mode(
                            student_question=student_question,
                            student_caption=referring_result.clean_caption,
                            gt_mask=gt_mask_np,
                            ref_mask=ref_mask_np,
                            route=loss_family,
                            branch="referring",
                        )
                        teacher_masks = self._build_teacher_prompt_masks(gt_mask_np, ref_mask_np)
                        onpolicy_entries.append(
                            {
                                "image": image,
                                "prompt_masks": prompt_masks,
                                "student_question": self.build_referring_student_prompt(student_question),
                                "teacher_prompt": teacher_prompt,
                                "completion_ids": referring_result.completion_ids,
                                "teacher_prompt_masks": teacher_masks,
                                "iou": float(referring_iou),
                            }
                        )

            sample_debug_record = {
                "sample_key": sample_key,
                "loss_family": loss_family,
                "online_route": online_route,
                "train_mode": self.train_mode,
                "dlc_caption": "" if dlc_result is None else dlc_result.clean_caption,
                "referring_caption": "" if referring_result is None else referring_result.clean_caption,
                "dlc_choose_one_correct": bool(route_decision.dlc_choose_one_correct),
                "dlc_choose_one_confidence": float(route_decision.dlc_choose_one_confidence),
                "referring_iou": float(referring_iou),
                "wrong_confuser_available": bool(wrong_confuser_mask is not None),
            }
            rank_debug_records.append(sample_debug_record)
            last_debug = sample_debug_record

        if regen_entries:
            regen_losses = self.compute_regenerate_alignment_losses_batch(regen_entries)
            if regen_losses.numel() > 0:
                regen_loss_sum = regen_losses.sum()
                total_loss = regen_loss_sum if total_loss is None else total_loss + regen_loss_sum
        if onpolicy_entries:
            onpolicy_losses = self.compute_onpolicy_distill_losses_batch(onpolicy_entries)
            if onpolicy_losses.numel() > 0:
                onpolicy_loss_sum = onpolicy_losses.sum()
                total_loss = onpolicy_loss_sum if total_loss is None else total_loss + onpolicy_loss_sum
        grpo_reward_sum = 0.0
        grpo_reward_count = 0
        if grpo_entries:
            grpo_losses, grpo_meta = self.compute_grpo_losses_batch(grpo_entries)
            grpo_reward_sum += float(grpo_meta.get("reward_sum", 0.0))
            grpo_reward_count += int(grpo_meta.get("reward_count", 0))
            if grpo_losses.numel() > 0:
                grpo_loss_sum = grpo_losses.sum()
                total_loss = grpo_loss_sum if total_loss is None else total_loss + grpo_loss_sum
        if ref_grpo_entries:
            ref_grpo_losses, ref_grpo_meta = self.compute_referring_grpo_losses_batch(ref_grpo_entries)
            grpo_reward_sum += float(ref_grpo_meta.get("reward_sum", 0.0))
            grpo_reward_count += int(ref_grpo_meta.get("reward_count", 0))
            if ref_grpo_losses.numel() > 0:
                ref_grpo_loss_sum = ref_grpo_losses.sum()
                total_loss = ref_grpo_loss_sum if total_loss is None else total_loss + ref_grpo_loss_sum
        if total_loss is None:
            total_loss = zero

        self._log_ddp_route_alignment_debug(
            {
                "rank": self._dist_rank(),
                "batch_route": batch_route,
                "loss_family": self._resolve_loss_family(batch_route),
                "entry_count": len(regen_entries) + len(onpolicy_entries) + len(grpo_entries) + len(ref_grpo_entries),
                "regen_entry_count": len(regen_entries),
                "onpolicy_entry_count": len(onpolicy_entries),
                "grpo_entry_count": len(grpo_entries) + len(ref_grpo_entries),
                "records": rank_debug_records,
            }
        )

        if self._should_debug_print() and last_debug:
            print(
                "[Sa2VA_OPSD_COMBINE] "
                f"train_mode={last_debug.get('train_mode')} route={last_debug.get('loss_family')} "
                f"dlc_caption={last_debug.get('dlc_caption')!r} "
                f"referring_caption={last_debug.get('referring_caption')!r} "
                f"dlc_choose_one_correct={int(last_debug.get('dlc_choose_one_correct', False))} "
                f"dlc_choose_one_confidence={float(last_debug.get('dlc_choose_one_confidence', 0.0)):.4f} "
                f"referring_iou={float(last_debug.get('referring_iou', 0.0)):.4f} "
                f"wrong_confuser_available={int(last_debug.get('wrong_confuser_available', False))}",
                flush=True,
            )

        route_count = max(routed_count, 1)
        avg_total_loss = total_loss.detach() if isinstance(total_loss, torch.Tensor) else zero
        metrics = {
            "loss_opsd_total": total_loss,
            "train_mode_id": self._metric_tensor({"dlc": 0.0, "referring": 1.0, "combine": 2.0}[self.train_mode], avg_total_loss.dtype),
            "route_teacher_regenerate_rate": self._metric_tensor(route_teacher_count / route_count, avg_total_loss.dtype),
            "route_on_policy_distill_rate": self._metric_tensor(route_onpolicy_count / route_count, avg_total_loss.dtype),
            "route_grpo_positive_rate": self._metric_tensor(route_grpo_count / route_count, avg_total_loss.dtype),
            "dlc_choose_one_correct_rate": self._metric_tensor(dlc_choose_one_correct_count / route_count, avg_total_loss.dtype),
            "dlc_choose_one_confidence_mean": self._metric_tensor(dlc_choose_one_confidence_sum / route_count, avg_total_loss.dtype),
            "referring_iou_mean": self._metric_tensor(referring_iou_sum / route_count, avg_total_loss.dtype),
            "referring_seg_correct_rate": self._metric_tensor(referring_seg_correct_count / route_count, avg_total_loss.dtype),
            "combine_dual_caption_valid_rate": self._metric_tensor(combine_dual_caption_valid_count / route_count, avg_total_loss.dtype),
            "combine_dual_supervision_applied_rate": self._metric_tensor(combine_dual_supervision_applied_count / route_count, avg_total_loss.dtype),
            "grpo_reward_mean": self._metric_tensor(grpo_reward_sum / max(grpo_reward_count, 1), avg_total_loss.dtype),
            "avg_referring_iou": self._metric_tensor(total_iou / route_count, avg_total_loss.dtype),
        }
        return metrics
