import re

import numpy as np

from projects.sa2va.datasets.common import DEFAULT_MASK_TO_REFERRING_QUESTION
from projects.sa2va.evaluation.teacher_diagnosis_common import (
    build_mask_relation_context,
    build_teacher_regenerate_difference_context,
)
from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3


class Sa2VAOPSDReferringModelV3(Sa2VAOPSDModelV3):
    """Referring-expression variant that reuses the DLC OPSD training stack."""

    _REFERRING_DIRECTION_WORDS = {
        "left", "right", "top", "bottom", "middle", "center", "front", "back", "upper", "lower"
    }
    _REFERRING_ORDINAL_WORDS = {
        "first", "second", "third", "fourth", "fifth", "1st", "2nd", "3rd", "4th", "5th"
    }
    _REFERRING_RELATION_WORDS = {
        "behind", "beside", "near", "under", "over", "above", "below", "between", "with"
    }
    _REFERRING_BODYPART_WORDS = {
        "arm", "head", "hand", "leg", "hair", "face", "tail", "wing", "foot", "feet"
    }
    _REFERRING_HARD_LOSS_WEIGHT = 1.35

    @staticmethod
    def _referring_failure_type_set():
        return {
            "too_generic",
            "wrong_attribute",
            "wrong_part_focus",
            "wrong_spatial_anchor",
            "distractor_leak",
            "scene_spill",
            "mixed_target",
            "underspecified_local_detail",
            "empty_or_malformed",
            "unknown",
        }

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
        if 2 <= token_count <= 4:
            return 0.8
        if 5 <= token_count <= 6:
            return 0.45
        if token_count == 1:
            return 0.15
        if 7 <= token_count <= 8:
            return 0.05
        return -0.35

    @classmethod
    def _tokenize_referring_expression(cls, caption):
        return re.findall(r"[a-z0-9']+", (caption or "").lower())

    @classmethod
    def _is_hard_referring_expression(cls, caption):
        tokens = set(cls._tokenize_referring_expression(caption))
        if not tokens:
            return False
        caption_lower = (caption or "").lower()
        has_direction = bool(tokens & cls._REFERRING_DIRECTION_WORDS)
        has_ordinal = bool(tokens & cls._REFERRING_ORDINAL_WORDS)
        has_relation = bool(tokens & cls._REFERRING_RELATION_WORDS) or any(
            phrase in caption_lower for phrase in ("next to", "in front of", "on top of", "from the left", "from the right")
        )
        has_bodypart = bool(tokens & cls._REFERRING_BODYPART_WORDS)
        hard_signal_count = sum((has_direction, has_ordinal, has_relation, has_bodypart))
        return hard_signal_count >= 2 or (len(tokens) >= 7 and hard_signal_count >= 1)

    def _training_loss_weight_for_sample(
        self,
        *,
        loss_family,
        student_caption="",
        teacher_caption="",
    ):
        del loss_family
        if self._is_hard_referring_expression(teacher_caption) or self._is_hard_referring_expression(student_caption):
            return self._REFERRING_HARD_LOSS_WEIGHT
        return 1.0

    def _referring_teacher_regenerate_gate_passed(self, student_caption, student_iou, teacher_iou):
        if self._teacher_regenerate_gate_passed(student_iou, teacher_iou):
            return True
        if not self._is_hard_referring_expression(student_caption):
            return False
        student_iou = float(student_iou)
        teacher_iou = float(teacher_iou)
        iou_gain = teacher_iou - student_iou
        return teacher_iou >= 0.55 and iou_gain >= 0.08

    @staticmethod
    def _referring_fault_report_labels():
        return (
            "GTMASK_DESC",
            "REFMASK_DESC",
            "MISSING_EVIDENCE",
            "DISTRACTOR_EVIDENCE",
            "PRIMARY_FAILURE_TYPE",
            "BAD_PHRASES_IN_STUDENT",
            "MISSING_PHRASES_NEEDED",
            "KEEPABLE_PHRASES",
            "REFERRING",
        )

    def _build_referring_teacher_prompt_masks(self, gt_mask_np, ref_mask_np):
        gt_mask = self._to_numpy_mask(gt_mask_np).astype(np.uint8)
        ref_mask = self._to_numpy_mask(ref_mask_np).astype(np.uint8)
        overlap_mask = np.logical_and(gt_mask > 0, ref_mask > 0).astype(np.uint8)
        gt_only_mask = np.logical_and(gt_mask > 0, ref_mask == 0).astype(np.uint8)
        ref_only_mask = np.logical_and(ref_mask > 0, gt_mask == 0).astype(np.uint8)
        return np.stack(
            [
                gt_mask.astype(np.float32),
                ref_mask.astype(np.float32),
                overlap_mask.astype(np.float32),
                gt_only_mask.astype(np.float32),
                ref_only_mask.astype(np.float32),
            ],
            axis=0,
        )

    def _humanize_referring_failure_type(self, failure_type):
        failure_type = self._normalize_teacher_field_text(failure_type).lower()
        mapping = {
            "too_generic": "The student caption is too generic for stable reconstruction.",
            "wrong_attribute": "The student caption anchors on a wrong or misleading visual attribute.",
            "wrong_part_focus": "The student caption focuses on the wrong local part of the target.",
            "wrong_spatial_anchor": "The student caption uses a weak or wrong spatial anchor.",
            "distractor_leak": "The student caption still fits the distractor-side region.",
            "scene_spill": "The student caption spills into broader scene context instead of the target.",
            "mixed_target": "The student caption mixes target evidence with another nearby region.",
            "underspecified_local_detail": "The student caption misses the local detail needed to isolate the target.",
            "empty_or_malformed": "The student caption is empty or malformed for reconstruction.",
            "unknown": "The student caption does not give a stable target-specific anchor.",
        }
        return mapping.get(failure_type, mapping["unknown"])

    def _build_referring_programmatic_direction(self, result):
        actions = []
        missing_needed = result.missing_phrases_needed if self._teacher_field_is_effective(
            result.missing_phrases_needed,
            invalid_markers=("",),
        ) else result.target_only_evidence
        if self._teacher_field_is_effective(missing_needed):
            actions.append(f"add {missing_needed}")
        if self._teacher_field_is_effective(result.distractor_only_evidence):
            actions.append(f"avoid wording that still fits {result.distractor_only_evidence}")
        if self._teacher_field_is_effective(result.keepable_phrases):
            actions.append(f"keep only the reliable part {result.keepable_phrases}")
        if not actions:
            actions.append("rewrite with one concrete target-specific visual cue")
        return " and ".join(actions).strip()

    def _build_referring_programmatic_reason(self, result):
        missing_text = (
            result.target_only_evidence
            if self._teacher_field_is_effective(result.target_only_evidence)
            else "the missing target-side cue"
        )
        distractor_text = (
            result.distractor_only_evidence
            if self._teacher_field_is_effective(result.distractor_only_evidence)
            else "the distractor-side region"
        )
        return (
            f"Reconstruction currently misses {missing_text} and is still compatible with "
            f"{distractor_text}, so the caption needs a tighter target-only anchor."
        )

    def _backfill_referring_fault_report(
        self,
        result,
        *,
        relation_context,
        student_caption,
    ):
        if result.primary_failure_type not in self._referring_failure_type_set():
            result.primary_failure_type = "unknown"
        if not self._teacher_field_is_effective(result.target_summary, invalid_markers=("",)):
            result.target_summary = self._normalize_teacher_field_text(relation_context.get("gt_summary", ""))
        if not self._teacher_field_is_effective(result.distractor_summary, invalid_markers=("",)):
            result.distractor_summary = self._normalize_teacher_field_text(relation_context.get("ref_summary", ""))
        if not self._teacher_field_is_effective(result.target_only_evidence):
            result.target_only_evidence = self._normalize_teacher_field_text(relation_context.get("gt_only_summary", ""))
        if not self._teacher_field_is_effective(result.distractor_only_evidence):
            result.distractor_only_evidence = self._normalize_teacher_field_text(relation_context.get("ref_only_summary", ""))
        trimmed_caption = self._clean_caption_text(student_caption)
        if not self._teacher_field_is_effective(result.bad_phrases_in_student):
            result.bad_phrases_in_student = trimmed_caption if trimmed_caption else "unknown"
        if not self._teacher_field_is_effective(result.missing_phrases_needed):
            result.missing_phrases_needed = (
                result.target_only_evidence if self._teacher_field_is_effective(result.target_only_evidence) else "none"
            )
        if not self._teacher_field_is_effective(result.keepable_phrases):
            result.keepable_phrases = "none"
        if not self._teacher_field_is_effective(result.caption_problem, invalid_markers=("",)):
            base_problem = self._humanize_referring_failure_type(result.primary_failure_type)
            if self._teacher_field_is_effective(result.bad_phrases_in_student):
                base_problem += f" Likely misleading phrase: {result.bad_phrases_in_student}."
            result.caption_problem = base_problem
        if not self._teacher_field_is_effective(result.correction_direction, invalid_markers=("",)):
            result.correction_direction = self._build_referring_programmatic_direction(result)
        if not self._teacher_field_is_effective(result.reason, invalid_markers=("",)):
            result.reason = self._build_referring_programmatic_reason(result)
        if not self._teacher_field_is_effective(result.target_anchor, invalid_markers=("",)):
            result.target_anchor = self._normalize_teacher_field_text(result.missing_phrases_needed or result.target_only_evidence)
        if not self._teacher_field_is_effective(result.distractor_anchor, invalid_markers=("",)):
            result.distractor_anchor = self._normalize_teacher_field_text(result.distractor_only_evidence)
        result.problem_valid = bool(self._teacher_field_is_effective(result.caption_problem, invalid_markers=("",)))
        result.direction_valid = bool(self._teacher_field_is_effective(result.correction_direction, invalid_markers=("",)))
        result.reason_valid = bool(self._teacher_field_is_effective(result.reason, invalid_markers=("",)))
        return result

    def _parse_referring_fault_report(self, raw_prediction, base_result=None):
        result = base_result or self._build_empty_teacher_regenerate_pipeline_result()
        text = "" if raw_prediction is None else str(raw_prediction)
        result.structured_diagnosis_raw = text
        result.diagnosis_raw = text
        sections = self._parse_teacher_labeled_sections(text, self._referring_fault_report_labels())
        result.target_summary = self._normalize_teacher_field_text(sections.get("GTMASK_DESC", ""))
        result.distractor_summary = self._normalize_teacher_field_text(sections.get("REFMASK_DESC", ""))
        result.target_only_evidence = self._normalize_teacher_field_text(sections.get("MISSING_EVIDENCE", ""))
        result.distractor_only_evidence = self._normalize_teacher_field_text(sections.get("DISTRACTOR_EVIDENCE", ""))
        result.primary_failure_type = self._normalize_teacher_field_text(
            sections.get("PRIMARY_FAILURE_TYPE", "")
        ).lower()
        result.bad_phrases_in_student = self._clean_teacher_phrase_list_text(
            sections.get("BAD_PHRASES_IN_STUDENT", "")
        )
        result.missing_phrases_needed = self._clean_teacher_phrase_list_text(
            sections.get("MISSING_PHRASES_NEEDED", "")
        )
        result.keepable_phrases = self._clean_teacher_phrase_list_text(
            sections.get("KEEPABLE_PHRASES", "")
        )
        referring_text = self._normalize_teacher_field_text(sections.get("REFERRING", ""))
        result.caption_problem = self._humanize_referring_failure_type(result.primary_failure_type)
        result.correction_direction = ""
        result.reason = ""
        result.target_anchor = self._normalize_teacher_field_text(result.missing_phrases_needed or result.target_only_evidence)
        result.distractor_anchor = self._normalize_teacher_field_text(result.distractor_only_evidence)
        if referring_text:
            result = self._materialize_teacher_caption_result(
                result,
                referring_text,
                "structured_referring_direct",
            )
        return result

    def validate_referring_fault_report(self, result):
        if result.primary_failure_type not in self._referring_failure_type_set():
            return False, "referring_fault_report_invalid:bad_primary_type"
        if not self._teacher_field_is_effective(result.target_summary, invalid_markers=("",)):
            return False, "referring_fault_report_invalid:missing_target_summary"
        if not self._teacher_field_is_effective(result.distractor_summary, invalid_markers=("",)):
            return False, "referring_fault_report_invalid:missing_ref_summary"
        if not (
            self._teacher_field_is_effective(result.target_only_evidence)
            or self._teacher_field_is_effective(result.distractor_only_evidence)
        ):
            return False, "referring_fault_report_invalid:no_difference_evidence"
        if not any(
            self._teacher_field_is_effective(value, invalid_markers=("",))
            for value in (
                result.bad_phrases_in_student,
                result.missing_phrases_needed,
                result.keepable_phrases,
            )
        ):
            return False, "referring_fault_report_invalid:no_phrase_level_signal"
        result.diagnosis_valid = True
        return True, ""

    def _evaluate_referring_candidate(
        self,
        *,
        image,
        gt_mask,
        raw_prediction,
        caption_text,
        pipeline_result,
        student_iou,
        source,
        style_hint="",
    ):
        caption = self._clean_teacher_dlc_caption_text(caption_text)
        candidate_result = self._build_empty_teacher_regenerate_pipeline_result()
        candidate_result.target_summary = pipeline_result.target_summary
        candidate_result.distractor_summary = pipeline_result.distractor_summary
        candidate_result.target_only_evidence = pipeline_result.target_only_evidence
        candidate_result.distractor_only_evidence = pipeline_result.distractor_only_evidence
        candidate_result = self._materialize_teacher_caption_result(candidate_result, caption, source)
        failure_reason = self._validate_teacher_dlc(candidate_result)
        reconstruct = self.reconstruct_mask(
            image=image,
            caption=candidate_result.detailed_caption,
            description_status=candidate_result.detailed_status,
            gt_mask=gt_mask,
        )
        pred_mask = None if reconstruct is None else reconstruct.pred_mask
        reconstruct_iou = self._compute_iou(gt_mask, pred_mask) if pred_mask is not None else 0.0
        candidate = {
            "raw_prediction": raw_prediction,
            "caption": candidate_result.detailed_caption,
            "status": candidate_result.detailed_status,
            "failure_reason": failure_reason,
            "completion_ids": candidate_result.detailed_completion_ids,
            "reconstruct_status": None if reconstruct is None else reconstruct.status,
            "reconstruct_iou": float(reconstruct_iou),
            "student_iou": float(student_iou),
            "target_only_evidence": pipeline_result.target_only_evidence,
            "distractor_only_evidence": pipeline_result.distractor_only_evidence,
            "style_hint": style_hint,
        }
        candidate["score"] = self._score_teacher_dlc_candidate(candidate)
        return candidate

    def _generate_referring_candidate_record(
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
        candidate_style_hint,
        generation_kwargs=None,
        repair_mode=False,
    ):
        teacher_fields = dict(teacher_fields)
        teacher_fields.update(
            {
                "target_summary": pipeline_result.target_summary,
                "distractor_summary": pipeline_result.distractor_summary,
                "target_only_evidence": pipeline_result.target_only_evidence,
                "distractor_only_evidence": pipeline_result.distractor_only_evidence,
                "primary_failure_type": getattr(pipeline_result, "primary_failure_type", ""),
                "bad_phrases_in_student": getattr(pipeline_result, "bad_phrases_in_student", ""),
                "missing_phrases_needed": getattr(pipeline_result, "missing_phrases_needed", ""),
                "keepable_phrases": getattr(pipeline_result, "keepable_phrases", ""),
                "caption_problem": pipeline_result.caption_problem,
                "correction_direction": pipeline_result.correction_direction,
                "reason": pipeline_result.reason,
                "candidate_style_hint": candidate_style_hint,
            }
        )
        generation_mode = "referring_repair" if repair_mode else "referring_rewrite_candidate"
        prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=teacher_fields,
            generation_mode=generation_mode,
        )
        generation_kwargs = dict(generation_kwargs or {})
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
            **generation_kwargs,
        )
        referring_text = self._extract_labeled_teacher_text(raw_prediction, "REFERRING")
        if not referring_text:
            referring_text = raw_prediction
        return self._evaluate_referring_candidate(
            image=image,
            gt_mask=gt_mask,
            raw_prediction=raw_prediction,
            caption_text=referring_text,
            pipeline_result=pipeline_result,
            student_iou=iou,
            source="referring_candidate_repair" if repair_mode else "referring_candidate",
            style_hint=candidate_style_hint,
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
        if generation_mode == "referring_fault_report_rewrite":
            clean_question = self._strip_image_placeholder(self._referring_prompt_text(student_question))
            relation_context = build_mask_relation_context(
                model=self,
                gt_mask=gt_mask,
                ref_mask=ref_mask if ref_mask is not None else self._zero_ref_mask_like(gt_mask),
            )
            gt_only_summary = relation_context["gt_only_summary"]
            ref_only_summary = relation_context["ref_only_summary"]
            student_caption = self._normalize_teacher_field_text(student_caption)
            return (
                "<image>\n"
                "You are supervising a short referring expression reconstruction task.\n"
                "Compare the marked regions visually before rewriting.\n"
                "region1 = gtmask = the true target.\n"
                "region2 = refmask = the mask reconstructed from the student's current referring expression.\n"
                "region3 = shared overlap between gtmask and refmask.\n"
                "region4 = gtmask minus refmask = target pixels the current referring expression still misses.\n"
                "region5 = refmask minus gtmask = distractor pixels the current referring expression wrongly includes.\n"
                f"Student prompt: {clean_question}\n"
                f"Student referring expression: {student_caption}\n"
                f"Description status: {description_status}\n"
                f"Reconstruction status: {reconstruction.status}\n"
                f"IoU between region1 and region2: {iou:.4f}\n"
                f"Target summary from mask stats: {relation_context['gt_summary']}\n"
                f"Ref summary from mask stats: {relation_context['ref_summary']}\n"
                f"Shared overlap summary from mask stats: {relation_context['overlap_summary']}\n"
                f"Missing target-side summary from mask stats: {gt_only_summary}\n"
                f"Distractor-side leak summary from mask stats: {ref_only_summary}\n"
                f"Difference focus: {teacher_fields.get('difference_focus', '')}\n"
                f"Likely drift reason: {teacher_fields.get('likely_drift_reason', '')}\n"
                "Output exactly these 9 lines and nothing else:\n"
                "GTMASK_DESC:\n"
                "REFMASK_DESC:\n"
                "MISSING_EVIDENCE:\n"
                "DISTRACTOR_EVIDENCE:\n"
                "PRIMARY_FAILURE_TYPE:\n"
                "BAD_PHRASES_IN_STUDENT:\n"
                "MISSING_PHRASES_NEEDED:\n"
                "KEEPABLE_PHRASES:\n"
                "REFERRING:\n"
                "Rules:\n"
                "- PRIMARY_FAILURE_TYPE must be exactly one of: too_generic, wrong_attribute, wrong_part_focus, wrong_spatial_anchor, distractor_leak, scene_spill, mixed_target, underspecified_local_detail, empty_or_malformed, unknown.\n"
                "- GTMASK_DESC and REFMASK_DESC must describe only visible content, not masks or labels.\n"
                "- MISSING_EVIDENCE must name what region1 contains that region2 still misses.\n"
                "- DISTRACTOR_EVIDENCE must name what region2 wrongly includes.\n"
                "- BAD_PHRASES_IN_STUDENT must be a short comma-separated phrase list or unknown.\n"
                "- MISSING_PHRASES_NEEDED must be a short comma-separated phrase list or none.\n"
                "- KEEPABLE_PHRASES must be a short comma-separated phrase list or none.\n"
                "- REFERRING must be one short target-specific referring expression for region1 only, ideally 2 to 6 words as a compact noun phrase.\n"
                "- REFERRING should avoid full-sentence style unless a tiny relational phrase is absolutely necessary.\n"
                "- REFERRING must not start with 'the target', 'the region', 'region1', or any explanation template.\n"
                "- Do not output markdown, bullets, JSON, [SEG], or any labels beyond the 9 required field names."
            )
        if generation_mode in {"referring_rewrite_candidate", "referring_repair"}:
            clean_question = self._strip_image_placeholder(self._referring_prompt_text(student_question))
            student_caption = self._normalize_teacher_field_text(student_caption)
            instruction = "Write one better short referring expression for region1." if generation_mode == "referring_rewrite_candidate" else "Repair the failed short referring expression and write one better expression for region1."
            return (
                "<image>\n"
                "You are supervising a short referring expression reconstruction task.\n"
                "region1 is the true target, region2 is the reconstructed distractor-leaning region, region3 is the shared overlap, region4 is the missing target-only area, and region5 is the leaked distractor-only area.\n"
                f"Student prompt: {clean_question}\n"
                f"Student referring expression: {student_caption}\n"
                f"Target summary: {teacher_fields.get('target_summary', '')}\n"
                f"Reconstructed summary: {teacher_fields.get('distractor_summary', '')}\n"
                f"Missing target-side evidence: {teacher_fields.get('target_only_evidence', '')}\n"
                f"Distractor-side leak evidence: {teacher_fields.get('distractor_only_evidence', '')}\n"
                f"PRIMARY_FAILURE_TYPE: {teacher_fields.get('primary_failure_type', '')}\n"
                f"BAD_PHRASES_IN_STUDENT: {teacher_fields.get('bad_phrases_in_student', '')}\n"
                f"MISSING_PHRASES_NEEDED: {teacher_fields.get('missing_phrases_needed', '')}\n"
                f"KEEPABLE_PHRASES: {teacher_fields.get('keepable_phrases', '')}\n"
                f"CAPTION_PROBLEM: {teacher_fields.get('caption_problem', '')}\n"
                f"CORRECTION_DIRECTION: {teacher_fields.get('correction_direction', '')}\n"
                f"REASON: {teacher_fields.get('reason', '')}\n"
                f"Style hint: {teacher_fields.get('candidate_style_hint', '')}\n"
                f"{instruction}\n"
                "Output exactly one line:\n"
                "REFERRING: <one short target-specific referring expression>\n"
                "Rules:\n"
                "- Keep it short, concrete, visually grounded, and close to a RefCOCO-style noun phrase.\n"
                "- Prefer 2 to 6 words unless one extra relational phrase is necessary.\n"
                "- Prefer the target-only cue over scene-level context.\n"
                "- Avoid wording that still fits the distractor-side leak.\n"
                "- Do not start with 'the target', 'the region', 'region1', or an explanation template.\n"
                "- Do not output analysis, markdown, bullets, [SEG], or extra labels."
            )
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
        teacher_prompt_masks = self._build_referring_teacher_prompt_masks(gt_mask, ref_mask)
        fallback_prompt_masks = self._build_teacher_prompt_masks(gt_mask, ref_mask)
        difference_context = build_teacher_regenerate_difference_context(
            model=self,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            image=image,
            student_question=student_question,
            region_model=self.require_teacher_model("Teacher regenerate difference summarization"),
        )
        relation_context = build_mask_relation_context(model=self, gt_mask=gt_mask, ref_mask=ref_mask)
        pipeline_result = self._build_empty_teacher_regenerate_pipeline_result()
        pipeline_result.teacher_pipeline_mode = "referring_structured_fault_report"
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

        diagnosis_teacher_fields = dict(teacher_fields)
        diagnosis_teacher_fields.update(
            {
                "difference_focus": pipeline_result.difference_focus,
                "likely_drift_reason": pipeline_result.likely_drift_reason,
            }
        )
        diagnosis_prompt = self.build_teacher_privileged_prompt_v3(
            student_question=student_question,
            student_caption=student_caption,
            description_status=description_status,
            reconstruction=reconstruction,
            iou=iou,
            gt_mask=gt_mask,
            ref_mask=ref_mask,
            teacher_fields=diagnosis_teacher_fields,
            generation_mode="referring_fault_report_rewrite",
        )
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=diagnosis_prompt,
        )
        pipeline_result = self._parse_referring_fault_report(raw_prediction, base_result=pipeline_result)
        pipeline_result = self._backfill_referring_fault_report(
            pipeline_result,
            relation_context=relation_context,
            student_caption=student_caption,
        )
        pipeline_result.diagnosis_valid, pipeline_result.diagnosis_failure_reason = self.validate_referring_fault_report(
            pipeline_result
        )
        if not pipeline_result.diagnosis_valid:
            pipeline_result.teacher_diagnosis_retry_count = 1
            retry_fields = dict(diagnosis_teacher_fields)
            retry_fields["candidate_style_hint"] = (
                "Retry with shorter fields and keep every line concrete. "
                "Make REFERRING a compact target-only expression."
            )
            retry_prompt = self.build_teacher_privileged_prompt_v3(
                student_question=student_question,
                student_caption=student_caption,
                description_status=description_status,
                reconstruction=reconstruction,
                iou=iou,
                gt_mask=gt_mask,
                ref_mask=ref_mask,
                teacher_fields=retry_fields,
                generation_mode="referring_fault_report_rewrite",
            )
            retry_raw = self._predict_teacher_privileged_text(
                image=image,
                teacher_prompt_masks=teacher_prompt_masks,
                teacher_prompt=retry_prompt,
            )
            pipeline_result = self._parse_referring_fault_report(retry_raw, base_result=pipeline_result)
            pipeline_result = self._backfill_referring_fault_report(
                pipeline_result,
                relation_context=relation_context,
                student_caption=student_caption,
            )
            pipeline_result.diagnosis_valid, pipeline_result.diagnosis_failure_reason = self.validate_referring_fault_report(
                pipeline_result
            )
        pipeline_result.stop_stage = "structured_diagnosis"

        candidates = []
        candidate_scores = []
        if self._teacher_field_is_effective(pipeline_result.detailed_caption, invalid_markers=("",)):
            direct_candidate = self._evaluate_referring_candidate(
                image=image,
                gt_mask=gt_mask,
                raw_prediction=pipeline_result.structured_diagnosis_raw,
                caption_text=pipeline_result.detailed_caption,
                pipeline_result=pipeline_result,
                student_iou=iou,
                source="structured_referring_direct",
                style_hint="direct_rewrite_from_fault_report",
            )
            candidates.append(direct_candidate)
            candidate_scores.append(
                {
                    "caption": direct_candidate["caption"],
                    "status": direct_candidate["status"],
                    "failure_reason": direct_candidate["failure_reason"],
                    "reconstruct_iou": direct_candidate["reconstruct_iou"],
                    "score": direct_candidate["score"],
                }
            )

        if pipeline_result.diagnosis_valid or self._teacher_has_minimal_diagnosis_signal(pipeline_result):
            candidate_specs = (
                {
                    "candidate_style_hint": "Prefer the strongest missing target-side cue as the main anchor.",
                    "do_sample": False,
                },
                {
                    "candidate_style_hint": "Keep the expression very short but preserve one local distinguishing cue.",
                    "do_sample": True,
                    "temperature": 0.35,
                    "top_p": 0.9,
                },
                {
                    "candidate_style_hint": "Avoid distractor-compatible wording and keep only the most target-specific phrase.",
                    "do_sample": True,
                    "temperature": 0.55,
                    "top_p": 0.92,
                },
            )
            for spec in candidate_specs:
                candidate = self._generate_referring_candidate_record(
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
        pipeline_result.teacher_dlc_candidate_scores = tuple(candidate_scores)
        selected_candidate = None
        if valid_candidates:
            selected_candidate = sorted(
                valid_candidates,
                key=lambda item: (item["score"], item["reconstruct_iou"]),
                reverse=True,
            )[0]
            pipeline_result.teacher_dlc_selected_by = "referring_candidate_score"
            pipeline_result = self._apply_teacher_dlc_candidate(
                pipeline_result,
                selected_candidate,
                "structured_referring_candidate",
            )
        else:
            pipeline_result.detailed_failure_reason = "teacher_dlc_invalid:no_candidate_selected"
        pipeline_result.stop_stage = "referring_candidates"

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
            best_iou = float(selected_candidate.get("reconstruct_iou", 0.0) or 0.0)
            if (
                pipeline_result.verification_status == "ok"
                and not pipeline_result.verification_failure_reason
                and float(pipeline_result.verification_iou or 0.0) > best_iou
            ):
                pipeline_result.teacher_verification_used = True
                pipeline_result.teacher_dlc_selected_by = "verification_iou"
                pipeline_result = self._materialize_teacher_caption_result(
                    pipeline_result,
                    pipeline_result.verification_caption,
                    "verification_caption",
                )
                pipeline_result.detailed_failure_reason = ""
                best_iou = float(pipeline_result.verification_iou or 0.0)
            else:
                pipeline_result.teacher_verification_used = False
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
                self._referring_teacher_regenerate_gate_passed(student_caption, iou, teacher_iou_plain)
                if teacher_reconstruct_ok
                else False
            )
        )

        if (pipeline_result.diagnosis_valid or self._teacher_has_minimal_diagnosis_signal(pipeline_result)) and not pipeline_result.gate_passed:
            repair_candidate = self._generate_referring_candidate_record(
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
                candidate_style_hint="Use only one target-only cue and remove every distractor-compatible phrase.",
                generation_kwargs={"do_sample": False},
                repair_mode=True,
            )
            repair_iou = float(repair_candidate.get("reconstruct_iou", 0.0) or 0.0)
            if self._teacher_candidate_is_usable(repair_candidate) and repair_iou > teacher_iou_plain:
                pipeline_result = self._apply_teacher_dlc_candidate(
                    pipeline_result,
                    repair_candidate,
                    "referring_repair_candidate",
                )
                pipeline_result.teacher_dlc_selected_by = "repair_candidate"
                pipeline_result.verification_iou = repair_iou
                teacher_reconstruct_ok = bool(
                    pipeline_result.detailed_status == "ok" and not pipeline_result.detailed_failure_reason and repair_iou > 0.0
                )
                pipeline_result.gate_passed = (
                    bool(teacher_reconstruct_ok)
                    if caption_mode_failure
                    else (
                        self._referring_teacher_regenerate_gate_passed(student_caption, iou, repair_iou)
                        if teacher_reconstruct_ok
                        else False
                    )
                )
            pipeline_result.stop_stage = "repair"

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
                teacher_prompt_masks=fallback_prompt_masks,
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
            fallback_result.teacher_pipeline_mode = "referring_structured_fault_report_with_fallback"
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
                        self._referring_teacher_regenerate_gate_passed(student_caption, iou, fallback_iou)
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
