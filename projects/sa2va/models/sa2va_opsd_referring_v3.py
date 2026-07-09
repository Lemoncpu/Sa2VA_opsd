import re

from projects.sa2va.datasets.common import DEFAULT_MASK_TO_REFERRING_QUESTION
from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3


class Sa2VAOPSDReferringModelV3(Sa2VAOPSDModelV3):
    """Referring-expression variant that reuses the DLC OPSD training stack."""

    @staticmethod
    def _referring_prompt_text(question_text):
        clean_question = Sa2VAOPSDModelV3._strip_image_placeholder(question_text or "")
        if not clean_question:
            return DEFAULT_MASK_TO_REFERRING_QUESTION
        if "detailed, localized caption" in clean_question.lower():
            return DEFAULT_MASK_TO_REFERRING_QUESTION
        return f"<image>{clean_question}" if not clean_question.startswith("<image>") else clean_question

    def _normalize_referring_expression(self, caption):
        caption = self._clean_caption_text(caption)
        caption = re.sub(
            r"^(?:the\s+)?(?:target|region|masked region|highlighted region)\b(?:\s*[:,.-]\s*|\s+)",
            "",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(r"\bregion\s*1\b", "", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\bregion1\b", "", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\s+", " ", caption).strip(" ,.")
        caption = self._canonicalize_referring_expression(caption)
        return re.sub(r"\s+", " ", caption).strip(" ,.")

    def _rewrite_caption_prompt_for_referring(self, prompt):
        replacements = (
            ("one natural and complete detailed localized caption", "one short, concrete, visually grounded referring expression"),
            ("one corrected detailed localized caption", "one corrected short referring expression"),
            ("corrected detailed localized caption", "corrected short referring expression"),
            ("detailed localized caption", "short referring expression"),
            ("DLC-bench style caption", "short referring expression"),
            ("Write a shorter verification caption for reconstruction gating.", "Write a shorter verifier-friendly referring expression for reconstruction gating."),
            ("Write a shorter verifier-friendly caption derived from the DLC and gtmask.", "Write a shorter verifier-friendly referring expression derived from the current referring expression and gtmask."),
            ("The verification caption must be shorter than the DLC.", "The verification caption must be shorter than the current referring expression."),
            ("It must be shorter than the DLC.", "It must be shorter than the current referring expression."),
        )
        rewritten = prompt
        for source, target in replacements:
            rewritten = rewritten.replace(source, target)
        return rewritten

    def _finalize_description_result(self, *, raw_prediction, clean_caption, completion_ids):
        normalized_caption = self._normalize_referring_expression(clean_caption)
        return super()._finalize_description_result(
            raw_prediction=raw_prediction,
            clean_caption=normalized_caption,
            completion_ids=completion_ids,
        )

    def _clean_teacher_dlc_caption_text(self, caption):
        normalized = super()._clean_teacher_dlc_caption_text(caption)
        return self._normalize_referring_expression(normalized)

    def _is_caption_content_sufficient(self, caption):
        token_count = self._caption_token_count(caption)
        if token_count >= 2 and self._is_np_like_caption(caption):
            return not self._is_overly_generic_caption(caption)
        if token_count < max(self.min_caption_tokens, 2):
            return False
        return super()._is_caption_content_sufficient(caption)

    def _teacher_candidate_length_score(self, caption):
        token_count = self._caption_token_count(caption)
        if 4 <= token_count <= 12:
            return 0.5
        if 2 <= token_count <= 16:
            return 0.2
        return -0.2

    def generate_description_with_model(
        self,
        model,
        *,
        image,
        mask_prompts,
        student_question,
        apply_mask_focus=True,
    ):
        return super().generate_description_with_model(
            model,
            image=image,
            mask_prompts=mask_prompts,
            student_question=self._referring_prompt_text(student_question),
            apply_mask_focus=apply_mask_focus,
        )

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
        prompt = super().build_teacher_privileged_prompt_v3(
            student_question=self._referring_prompt_text(student_question),
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode=generation_mode,
        )
        return self._rewrite_caption_prompt_for_referring(prompt)

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
        prompt = super().build_teacher_regenerate_single_prompt(
            student_question=self._referring_prompt_text(student_question),
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            teacher_fields=teacher_fields,
        )
        return self._rewrite_caption_prompt_for_referring(prompt)

