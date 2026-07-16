import re

import numpy as np
import torch
import torch.nn.functional as F

from projects.sa2va.datasets.referring_prompts import DEFAULT_MASK_TO_REFERRING_QUESTION
from projects.sa2va.evaluation.teacher_diagnosis_common import (
    build_mask_relation_context,
)
from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3


class Sa2VAOPSDReferringModelV3(Sa2VAOPSDModelV3):
    """Referring-expression variant that reuses the DLC OPSD training stack."""

    _REFERRING_DIRECTION_WORDS = {
        "left", "right", "top", "bottom", "middle", "center", "front", "back", "upper", "lower",
        "leftmost", "rightmost", "topmost", "bottommost",
    }
    _REFERRING_ORDINAL_WORDS = {
        "first", "second", "third", "fourth", "fifth", "1st", "2nd", "3rd", "4th", "5th"
    }
    _REFERRING_RELATION_WORDS = {
        "behind", "beside", "near", "under", "over", "above", "below", "between", "with",
        "next", "holding", "wearing", "carrying", "by"
    }
    _REFERRING_BODYPART_WORDS = {
        "arm", "head", "hand", "leg", "hair", "face", "tail", "wing", "foot", "feet"
    }
    _REFERRING_COLOR_WORDS = {
        "red", "blue", "green", "yellow", "black", "white", "brown", "gray", "grey", "orange", "pink", "purple"
    }
    _REFERRING_TEMPLATE_PREFIXES = (
        "the target",
        "the region",
        "region1",
        "this is",
        "there is",
        "it is",
    )
    _REFERRING_STYLE_FORBIDDEN_PHRASES = (
        "appears to",
        "seems to",
        "in the image",
        "in this image",
        "visible in",
    )
    _REFERRING_GENERIC_CATEGORY_WORDS = {
        "person", "people", "man", "woman", "boy", "girl", "dog", "cat", "chair", "table", "car", "bus", "bike",
    }
    _REFERRING_CUE_META_PATTERNS = (
        r"area_ratio\s*=",
        r"\bbbox\s*=",
        r"\bcenter\s*=",
        r"</?p>",
        r"\bregion\d+\b",
        r"\bregion\s+\d+\b",
        r"\[[0-9.,\s-]+\]",
    )
    _REFERRING_CUE_STOPWORDS = {
        "a", "an", "the", "this", "that", "these", "those", "target", "region",
        "cue", "keep", "drop", "none", "unknown",
    }
    _REFERRING_TRAILING_FRAGMENT_WORDS = {
        "a", "an", "the", "with", "without", "in", "on", "at", "of", "for", "from",
        "to", "by", "near", "next", "behind", "under", "over", "above", "below",
        "between", "and", "or",
    }

    def __init__(
        self,
        *args,
        referring_hard_loss_weight_base=1.2,
        referring_hard_loss_weight_compound=1.4,
        referring_hard_loss_weight_max=1.55,
        enable_referring_direct_mask_loss=True,
        referring_direct_mask_loss_weight=0.7,
        referring_teacher_direct_mask_loss_weight=0.35,
        referring_confuser_separation_loss_weight=0.15,
        referring_direct_mask_loss_min_iou_gate=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._short_referring_log_mode = True
        self.referring_hard_loss_weight_base = float(referring_hard_loss_weight_base)
        self.referring_hard_loss_weight_compound = float(referring_hard_loss_weight_compound)
        self.referring_hard_loss_weight_max = float(referring_hard_loss_weight_max)
        self.enable_referring_direct_mask_loss = bool(enable_referring_direct_mask_loss)
        self.referring_direct_mask_loss_weight = float(referring_direct_mask_loss_weight)
        self.referring_teacher_direct_mask_loss_weight = float(referring_teacher_direct_mask_loss_weight)
        self.referring_confuser_separation_loss_weight = float(referring_confuser_separation_loss_weight)
        self.referring_direct_mask_loss_min_iou_gate = float(referring_direct_mask_loss_min_iou_gate)

    @staticmethod
    def _referring_failure_type_set():
        return {
            "wrong_subject",
            "missing_attribute",
            "wrong_attribute",
            "missing_position",
        }

    @staticmethod
    def _referring_prompt_text(question_text):
        clean_question = Sa2VAOPSDModelV3._strip_image_placeholder(question_text or "")
        if not clean_question:
            return DEFAULT_MASK_TO_REFERRING_QUESTION
        if "detailed, localized caption" in clean_question.lower():
            return DEFAULT_MASK_TO_REFERRING_QUESTION
        return f"<image>{clean_question}" if not clean_question.startswith("<image>") else clean_question

    @classmethod
    def _has_referring_template_prefix(cls, caption):
        caption_lower = (caption or "").strip().lower()
        return any(caption_lower.startswith(prefix) for prefix in cls._REFERRING_TEMPLATE_PREFIXES)

    @classmethod
    def _looks_like_full_sentence(cls, caption):
        caption = (caption or "").strip()
        if not caption:
            return False
        token_count = len(cls._tokenize_referring_expression(caption))
        caption_lower = caption.lower()
        return token_count >= 8 or any(phrase in caption_lower for phrase in cls._REFERRING_STYLE_FORBIDDEN_PHRASES)

    @classmethod
    def _is_referring_category_only_expression(cls, caption):
        tokens = cls._tokenize_referring_expression(caption)
        if not tokens:
            return False
        cue_words = (
            cls._REFERRING_DIRECTION_WORDS
            | cls._REFERRING_ORDINAL_WORDS
            | cls._REFERRING_RELATION_WORDS
            | cls._REFERRING_BODYPART_WORDS
            | cls._REFERRING_COLOR_WORDS
        )
        has_cue = bool(set(tokens) & cue_words) or any(
            phrase in (caption or "").lower()
            for phrase in ("next to", "in front of", "on top of", "from the left", "from the right")
        )
        if has_cue:
            return False
        filtered = [token for token in tokens if token not in {"the", "a", "an"}]
        return len(filtered) <= 2 and all(token in cls._REFERRING_GENERIC_CATEGORY_WORDS for token in filtered)

    def _referring_style_is_usable(self, caption):
        token_count = self._caption_token_count(caption)
        if token_count < 2 or token_count > 9:
            return False
        if self._has_referring_template_prefix(caption):
            return False
        if self._looks_like_full_sentence(caption):
            return False
        if self._caption_scene_spill_hit_count(caption) >= 2:
            return False
        if self._is_referring_category_only_expression(caption):
            return False
        return True

    def _referring_style_score(self, caption):
        token_count = self._caption_token_count(caption)
        score = 0.0
        if 2 <= token_count <= 4:
            score += 1.0
        elif 5 <= token_count <= 6:
            score += 0.7
        elif 7 <= token_count <= 9:
            score += 0.2
        else:
            score -= 0.8
        if self._has_referring_template_prefix(caption):
            score -= 0.8
        if self._looks_like_full_sentence(caption):
            score -= 0.6
        if self._caption_scene_spill_hit_count(caption) >= 2:
            score -= 0.5
        if self._is_referring_category_only_expression(caption):
            score -= 0.7
        return score

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

    def _force_short_referring_expression(self, caption):
        caption = self._normalize_referring_expression(caption)
        caption = re.sub(r"^(?:referring|caption|description|dlc)\s*:\s*", "", caption, flags=re.IGNORECASE)
        caption = re.sub(
            r"\bstanding on the (left|right|top|bottom) side of the image\b",
            r"\1",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(
            r"\bon the (left|right|top|bottom) side of the image\b",
            r"\1",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(r"\b(?:in|on|at)\s+the image\b", "", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\bof the image\b", "", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\bthat is\b.*$", "", caption, flags=re.IGNORECASE)
        caption = re.sub(r"[,.].*$", "", caption).strip(" ,.")
        tokens = self._tokenize_referring_expression(caption)
        if len(tokens) > 9:
            caption = " ".join(tokens[:9])
        caption = re.sub(r"\s+", " ", caption).strip(" ,.")
        while True:
            trailing = self._tokenize_referring_expression(caption)
            if not trailing or trailing[-1] not in self._REFERRING_TRAILING_FRAGMENT_WORDS:
                break
            caption = re.sub(rf"\b{re.escape(trailing[-1])}\b[\s,.;:!?-]*$", "", caption, flags=re.IGNORECASE).strip(" ,.")
        return re.sub(r"\s+", " ", caption).strip(" ,.")

    def _referring_teacher_max_new_tokens(self):
        return max(8, min(int(self.description_max_new_tokens), 14))

    def _referring_verification_max_new_tokens(self):
        return max(6, min(int(self.description_max_new_tokens), 10))

    def _build_referring_verification_fallback(self, pipeline_result):
        detailed_tokens = self._caption_token_count(pipeline_result.detailed_caption)
        candidates = [
            getattr(pipeline_result, "keepable_phrases", ""),
            getattr(pipeline_result, "target_only_evidence", ""),
            getattr(pipeline_result, "missing_phrases_needed", ""),
            pipeline_result.detailed_caption,
        ]
        for candidate in candidates:
            candidate = self._force_short_referring_expression(candidate)
            if not candidate:
                continue
            tokens = self._caption_token_count(candidate)
            if detailed_tokens > 0 and tokens >= detailed_tokens:
                words = self._tokenize_referring_expression(candidate)
                if len(words) >= 2:
                    candidate = " ".join(words[: max(2, min(len(words) - 1, 4))])
                    candidate = self._force_short_referring_expression(candidate)
                    tokens = self._caption_token_count(candidate)
            if not candidate or not self._referring_style_is_usable(candidate):
                continue
            if detailed_tokens > 0 and tokens >= detailed_tokens:
                continue
            return candidate
        return ""

    @classmethod
    def _looks_like_meta_cue(cls, text):
        lowered = (text or "").lower()
        if not lowered:
            return True
        return any(re.search(pattern, lowered) for pattern in cls._REFERRING_CUE_META_PATTERNS)

    def _sanitize_referring_cue(self, value, *, allow_none=True):
        raw = self._normalize_teacher_field_text(value)
        if not raw:
            return "none" if allow_none else ""
        raw = re.sub(r"(?i)\b(?:keep_cue|drop_cue|keep cue|drop cue)\s*:\s*", "", raw)
        raw = re.sub(r"</?p>", " ", raw, flags=re.IGNORECASE)
        candidates = []
        for part in re.split(r"[,;\n]", raw):
            item = self._normalize_teacher_field_text(part)
            if not item or self._looks_like_meta_cue(item):
                continue
            item = self._force_short_referring_expression(item)
            tokens = [
                token for token in self._tokenize_referring_expression(item)
                if token not in self._REFERRING_CUE_STOPWORDS
            ]
            if not tokens:
                continue
            if not any(re.search(r"[a-z]", token) for token in tokens):
                continue
            cleaned = " ".join(tokens[:6]).strip()
            if cleaned:
                candidates.append(cleaned)
        if not candidates:
            return "none" if allow_none else ""
        return ", ".join(dict.fromkeys(candidates))

    def _sanitize_referring_prompt_summary(self, value):
        cleaned = self._sanitize_referring_cue(value, allow_none=False)
        return cleaned

    def _fallback_referring_keep_cue(self, student_caption, failure_type):
        caption = self._force_short_referring_expression(student_caption)
        tokens = [
            token for token in self._tokenize_referring_expression(caption)
            if token not in self._REFERRING_CUE_STOPWORDS
        ]
        if failure_type == "missing_position":
            spatial_tokens = [
                token for token in tokens
                if token in (self._REFERRING_DIRECTION_WORDS | self._REFERRING_ORDINAL_WORDS)
            ]
            return " ".join(spatial_tokens[:3]).strip()
        return " ".join(tokens[:4]).strip()

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
        return self._force_short_referring_expression(normalized)

    def _materialize_teacher_caption_result(self, result, caption, source):
        return super()._materialize_teacher_caption_result(
            result,
            self._force_short_referring_expression(caption),
            source,
        )

    def _is_caption_content_sufficient(self, caption):
        if not self._referring_style_is_usable(caption):
            return False
        token_count = self._caption_token_count(caption)
        if token_count >= 2 and self._is_np_like_caption(caption):
            return not self._is_overly_generic_caption(caption)
        if token_count < max(self.min_caption_tokens, 2):
            return False
        return super()._is_caption_content_sufficient(caption)

    def _teacher_candidate_length_score(self, caption):
        token_count = self._caption_token_count(caption)
        if 2 <= token_count <= 4:
            return 1.0
        if 5 <= token_count <= 6:
            return 0.6
        if 7 <= token_count <= 9:
            return 0.15
        if token_count == 1:
            return -0.1
        return -0.7

    @classmethod
    def _tokenize_referring_expression(cls, caption):
        return re.findall(r"[a-z0-9']+", (caption or "").lower())

    @classmethod
    def _is_hard_referring_expression(cls, caption):
        tags = cls._analyze_referring_difficulty(caption)
        return bool(tags["hard_sample"])

    @classmethod
    def _analyze_referring_difficulty(cls, caption):
        tokens = set(cls._tokenize_referring_expression(caption))
        caption_lower = (caption or "").lower()
        has_direction = bool(tokens & cls._REFERRING_DIRECTION_WORDS)
        has_ordinal = bool(tokens & cls._REFERRING_ORDINAL_WORDS)
        has_relation = bool(tokens & cls._REFERRING_RELATION_WORDS) or any(
            phrase in caption_lower for phrase in ("next to", "in front of", "on top of", "from the left", "from the right")
        )
        has_bodypart = bool(tokens & cls._REFERRING_BODYPART_WORDS)
        same_category_like = bool(tokens & cls._REFERRING_GENERIC_CATEGORY_WORDS) and (has_direction or has_ordinal or has_relation)
        compound = sum((has_direction, has_ordinal, has_relation)) >= 2
        hard_sample = compound or has_bodypart or (same_category_like and (has_direction or has_ordinal))
        return {
            "spatial_required": bool(has_direction),
            "ordinal_required": bool(has_ordinal),
            "relation_required": bool(has_relation),
            "bodypart_required": bool(has_bodypart),
            "same_category_multi_instance_like": bool(same_category_like),
            "hard_sample": bool(hard_sample),
            "compound_hard_sample": bool(compound or (has_direction and has_ordinal)),
        }

    def _training_loss_weight_for_sample(
        self,
        *,
        loss_family,
        student_caption="",
        teacher_caption="",
    ):
        del loss_family
        student_tags = self._analyze_referring_difficulty(student_caption)
        teacher_tags = self._analyze_referring_difficulty(teacher_caption)
        active = {
            key: bool(student_tags[key] or teacher_tags[key])
            for key in (
                "spatial_required",
                "ordinal_required",
                "relation_required",
                "bodypart_required",
                "same_category_multi_instance_like",
                "compound_hard_sample",
                "hard_sample",
            )
        }
        cue_count = sum(
            int(active[key])
            for key in ("spatial_required", "ordinal_required", "relation_required", "bodypart_required")
        )
        if active["compound_hard_sample"]:
            return self.referring_hard_loss_weight_max
        if cue_count >= 2 or active["same_category_multi_instance_like"]:
            return self.referring_hard_loss_weight_compound
        if active["hard_sample"]:
            return self.referring_hard_loss_weight_base
        return 1.0

    def _referring_teacher_regenerate_gate_passed(self, student_caption, student_iou, teacher_iou):
        if self._teacher_regenerate_gate_passed(student_iou, teacher_iou):
            return True
        student_iou = float(student_iou)
        teacher_iou = float(teacher_iou)
        iou_gain = teacher_iou - student_iou
        if teacher_iou >= 0.8 and iou_gain >= -0.02:
            return True
        if teacher_iou >= 0.7 and iou_gain >= 0.03:
            return True
        if not self._is_hard_referring_expression(student_caption):
            return False
        return teacher_iou >= 0.55 and iou_gain >= 0.08

    @staticmethod
    def _referring_fault_report_labels():
        return (
            "ERROR_TYPE",
            "KEEP_CUE",
            "DROP_CUE",
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

    def _infer_minimal_referring_failure_type(self, result, student_caption):
        caption_lower = (student_caption or "").lower()
        target_text = " ".join(
            part.lower()
            for part in (
                getattr(result, "keepable_phrases", ""),
            )
            if part
        )
        distractor_text = " ".join(
            part.lower()
            for part in (
                getattr(result, "must_avoid_phrases", ""),
            )
            if part
        )
        has_position = any(token in " ".join((caption_lower, target_text, distractor_text)) for token in (self._REFERRING_DIRECTION_WORDS | self._REFERRING_ORDINAL_WORDS))
        if self._teacher_field_is_effective(result.distractor_only_evidence) and (
            not self._teacher_field_is_effective(getattr(result, "keepable_phrases", ""))
            or any(token in distractor_text for token in self._tokenize_referring_expression(caption_lower))
        ):
            return "wrong_subject"
        if self._teacher_field_is_effective(getattr(result, "must_avoid_phrases", "")) and self._teacher_field_is_effective(getattr(result, "keepable_phrases", "")):
            return "wrong_attribute"
        if has_position:
            return "missing_position"
        return "missing_attribute"

    def _backfill_referring_fault_report(
        self,
        result,
        *,
        relation_context,
        student_caption,
    ):
        if result.primary_failure_type not in self._referring_failure_type_set():
            result.primary_failure_type = self._infer_minimal_referring_failure_type(result, student_caption)
        result.secondary_failure_type = ""
        result.target_summary = self._sanitize_referring_prompt_summary(relation_context.get("gt_summary", ""))
        result.distractor_summary = self._sanitize_referring_prompt_summary(relation_context.get("ref_summary", ""))
        result.target_only_evidence = self._sanitize_referring_prompt_summary(relation_context.get("gt_only_summary", ""))
        result.distractor_only_evidence = self._sanitize_referring_prompt_summary(relation_context.get("ref_only_summary", ""))
        trimmed_caption = self._clean_caption_text(student_caption)
        result.bad_phrases_in_student = trimmed_caption if trimmed_caption else "unknown"
        result.keepable_phrases = self._sanitize_referring_cue(result.keepable_phrases)
        result.must_avoid_phrases = self._sanitize_referring_cue(result.must_avoid_phrases)
        if not self._teacher_field_is_effective(result.keepable_phrases):
            fallback_keep = self._fallback_referring_keep_cue(trimmed_caption, result.primary_failure_type)
            if self._teacher_field_is_effective(result.target_only_evidence):
                result.keepable_phrases = result.target_only_evidence
            elif fallback_keep:
                result.keepable_phrases = fallback_keep
            else:
                result.keepable_phrases = "none"
        if not self._teacher_field_is_effective(getattr(result, "must_avoid_phrases", "")):
            result.must_avoid_phrases = result.distractor_only_evidence if self._teacher_field_is_effective(result.distractor_only_evidence) else "none"
        result.must_avoid_phrases = self._sanitize_referring_cue(result.must_avoid_phrases)
        result.missing_phrases_needed = result.keepable_phrases
        result.caption_problem = ""
        result.correction_direction = ""
        result.reason = ""
        result.target_anchor = ""
        result.distractor_anchor = ""
        result.problem_valid = False
        result.direction_valid = False
        result.reason_valid = False
        result.problem_failure_reason = ""
        result.direction_failure_reason = ""
        result.reason_failure_reason = ""
        return result

    def _parse_referring_fault_report(self, raw_prediction, base_result=None):
        result = base_result or self._build_empty_teacher_regenerate_pipeline_result()
        text = "" if raw_prediction is None else str(raw_prediction)
        result.structured_diagnosis_raw = text
        result.diagnosis_raw = text
        sections = self._parse_teacher_labeled_sections(text, self._referring_fault_report_labels())
        result.primary_failure_type = self._normalize_teacher_field_text(
            sections.get("ERROR_TYPE", "")
        ).lower()
        result.keepable_phrases = self._sanitize_referring_cue(
            sections.get("KEEP_CUE", "")
        )
        result.must_avoid_phrases = self._sanitize_referring_cue(
            sections.get("DROP_CUE", "")
        )
        result.bad_phrases_in_student = result.must_avoid_phrases
        result.missing_phrases_needed = result.keepable_phrases
        referring_text = self._normalize_teacher_field_text(sections.get("REFERRING", ""))
        result.caption_problem = ""
        result.correction_direction = ""
        result.reason = ""
        result.target_anchor = ""
        result.distractor_anchor = ""
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
        keep_cue = self._sanitize_referring_cue(result.keepable_phrases)
        drop_cue = self._sanitize_referring_cue(result.must_avoid_phrases)
        detailed_caption = self._force_short_referring_expression(result.detailed_caption)
        if not any(self._teacher_field_is_effective(value, invalid_markers=("",)) for value in (keep_cue, drop_cue, detailed_caption)):
            return False, "referring_fault_report_invalid:no_phrase_level_signal"
        lower_text = " ".join(part.lower() for part in (keep_cue, drop_cue, detailed_caption, result.target_only_evidence) if part)
        if result.primary_failure_type == "missing_position" and not any(token in lower_text for token in (self._REFERRING_DIRECTION_WORDS | self._REFERRING_ORDINAL_WORDS)):
            return False, "referring_fault_report_invalid:missing_spatial_cue"
        if result.primary_failure_type == "wrong_subject" and not (
            self._teacher_field_is_effective(drop_cue) or self._teacher_field_is_effective(detailed_caption, invalid_markers=("",))
        ):
            return False, "referring_fault_report_invalid:missing_wrong_subject_drop_cue"
        if result.primary_failure_type == "missing_attribute" and not (
            self._teacher_field_is_effective(keep_cue) or self._teacher_field_is_effective(detailed_caption, invalid_markers=("",))
        ):
            return False, "referring_fault_report_invalid:missing_attribute_keep_cue"
        if result.primary_failure_type == "wrong_attribute" and not (
            (self._teacher_field_is_effective(keep_cue) and self._teacher_field_is_effective(drop_cue))
            or self._teacher_field_is_effective(detailed_caption, invalid_markers=("",))
        ):
            return False, "referring_fault_report_invalid:missing_attribute_swap_cue"
        result.keepable_phrases = keep_cue
        result.must_avoid_phrases = drop_cue
        result.detailed_caption = detailed_caption
        result.diagnosis_valid = True
        return True, ""

    def _referring_type_specific_style_hint(self, failure_type, *, repair_mode=False):
        action = "Repair the expression" if repair_mode else "Rewrite the expression"
        mapping = {
            "wrong_subject": f"{action} around the true target subject only and remove the larger nearby subject.",
            "missing_attribute": f"{action} by adding one short target-only attribute and nothing else.",
            "wrong_attribute": f"{action} by dropping the wrong attribute and keeping the correct subject cue.",
            "missing_position": f"{action} with one short left-right-middle or order cue and keep the noun phrase short.",
        }
        return mapping.get(failure_type, f"{action} as one short RefCOCO-style noun phrase with one target-only cue.")

    def _referring_candidate_cue_score(self, caption, pipeline_result):
        caption_lower = (caption or "").lower()
        score = 0.0
        target_text = " ".join(
            item.lower()
            for item in (
                getattr(pipeline_result, "missing_phrases_needed", ""),
                getattr(pipeline_result, "keepable_phrases", ""),
            )
            if self._teacher_field_is_effective(item)
        )
        distractor_text = " ".join(
            item.lower()
            for item in (
                getattr(pipeline_result, "must_avoid_phrases", ""),
            )
            if self._teacher_field_is_effective(item)
        )
        for token in self._split_teacher_field_list(target_text):
            if token and token.lower() in caption_lower:
                score += 0.25
        for token in self._split_teacher_field_list(distractor_text):
            if token and token.lower() in caption_lower:
                score -= 0.3
        return score

    def _referring_candidate_type_constraint_passed(self, caption, pipeline_result):
        failure_type = getattr(pipeline_result, "primary_failure_type", "")
        caption_lower = (caption or "").lower()
        if failure_type == "missing_position":
            return any(token in caption_lower for token in (self._REFERRING_DIRECTION_WORDS | self._REFERRING_ORDINAL_WORDS))
        if failure_type in {"wrong_subject", "missing_attribute", "wrong_attribute"}:
            keep_cue = self._sanitize_referring_cue(getattr(pipeline_result, "keepable_phrases", ""))
            keep_ok = (not self._teacher_field_is_effective(keep_cue)) or any(
                token.lower() in caption_lower
                for token in self._split_teacher_field_list(keep_cue)
            )
            must_avoid = self._sanitize_referring_cue(getattr(pipeline_result, "must_avoid_phrases", ""))
            drop_ok = not any(
                token.lower() in caption_lower
                for token in self._split_teacher_field_list(must_avoid)
            )
            if failure_type == "missing_attribute":
                return bool(keep_ok)
            if failure_type == "wrong_subject" and not self._teacher_field_is_effective(must_avoid):
                return True
            if failure_type == "wrong_attribute" and not (
                self._teacher_field_is_effective(keep_cue) or self._teacher_field_is_effective(must_avoid)
            ):
                return True
            return bool(keep_ok and drop_ok)
        return True

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
            "style_score": self._referring_style_score(candidate_result.detailed_caption),
            "cue_score": self._referring_candidate_cue_score(candidate_result.detailed_caption, pipeline_result),
            "cue_passed": bool(self._referring_candidate_type_constraint_passed(candidate_result.detailed_caption, pipeline_result)),
        }
        candidate["score"] = (
            float(candidate["style_score"])
            + float(candidate["cue_score"])
            + float(candidate["reconstruct_iou"])
            - (0.75 if failure_reason else 0.0)
        )
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
                "gt_summary": pipeline_result.target_summary,
                "ref_summary": pipeline_result.distractor_summary,
                "gt_only_summary": pipeline_result.target_only_evidence,
                "ref_only_summary": pipeline_result.distractor_only_evidence,
                "primary_failure_type": getattr(pipeline_result, "primary_failure_type", ""),
                "keepable_phrases": getattr(pipeline_result, "keepable_phrases", ""),
                "candidate_style_hint": candidate_style_hint,
                "must_avoid_phrases": getattr(pipeline_result, "must_avoid_phrases", ""),
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
            max_new_tokens=self._referring_teacher_max_new_tokens(),
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

    def _teacher_candidate_is_usable(self, candidate):
        if not super()._teacher_candidate_is_usable(candidate):
            return False
        if not self._referring_style_is_usable(candidate.get("caption", "")):
            return False
        return bool(candidate.get("cue_passed", True))

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
            gt_summary = self._sanitize_referring_prompt_summary(relation_context["gt_summary"]) or "none"
            ref_summary = self._sanitize_referring_prompt_summary(relation_context["ref_summary"]) or "none"
            gt_only_summary = self._sanitize_referring_prompt_summary(relation_context["gt_only_summary"]) or "none"
            ref_only_summary = self._sanitize_referring_prompt_summary(relation_context["ref_only_summary"]) or "none"
            student_caption = self._normalize_teacher_field_text(student_caption)
            return (
                "<image>\n"
                "You are supervising a short referring expression reconstruction task.\n"
                "Only care about the main target subject itself. Do not add scene, background, or activity description unless it is part of the target subject phrase.\n"
                "region1 = gtmask = the true target.\n"
                "region2 = refmask = the mask reconstructed from the student's current referring expression.\n"
                f"Student prompt: {clean_question}\n"
                f"Student referring expression: {student_caption}\n"
                f"Description status: {description_status}\n"
                f"Reconstruction status: {reconstruction.status}\n"
                f"IoU between region1 and region2: {iou:.4f}\n"
                f"Target subject cue: {gt_summary}\n"
                f"Reconstructed subject cue: {ref_summary}\n"
                f"Target-only cue: {gt_only_summary}\n"
                f"Distractor-only cue: {ref_only_summary}\n"
                "Output exactly these 4 lines and nothing else:\n"
                "ERROR_TYPE:\n"
                "KEEP_CUE:\n"
                "DROP_CUE:\n"
                "REFERRING:\n"
                "Rules:\n"
                "- ERROR_TYPE must be exactly one of: wrong_subject, missing_attribute, wrong_attribute, missing_position.\n"
                "- KEEP_CUE must be a very short target-side cue to keep or add. Use none only if necessary.\n"
                "- DROP_CUE must be a very short wrong cue to drop. Use none if there is no wrong cue.\n"
                "- REFERRING must be one short target-specific referring expression for region1 only, ideally 2 to 5 words as a compact noun phrase.\n"
                "- Focus only on the target subject, not the broader scene.\n"
                "- REFERRING must end cleanly with a noun phrase and must not end with words like with, in, on, of, or and.\n"
                "- REFERRING must not start with 'the target', 'the region', 'region1', or any explanation template.\n"
                "- Do not output markdown, bullets, JSON, [SEG], or any labels beyond the 4 required field names."
            )
        if generation_mode in {"referring_rewrite_candidate", "referring_repair"}:
            clean_question = self._strip_image_placeholder(self._referring_prompt_text(student_question))
            student_caption = self._normalize_teacher_field_text(student_caption)
            instruction = "Write one better short referring expression for region1." if generation_mode == "referring_rewrite_candidate" else "Repair the failed short referring expression and write one better expression for region1."
            return (
                "<image>\n"
                "You are supervising a short referring expression reconstruction task.\n"
                "Only care about the target subject itself, not the broader scene.\n"
                "region1 is the true target and region2 is the reconstructed distractor-leaning region.\n"
                f"Student prompt: {clean_question}\n"
                f"Student referring expression: {student_caption}\n"
                f"Target summary: {teacher_fields.get('gt_summary', '')}\n"
                f"Reconstructed summary: {teacher_fields.get('ref_summary', '')}\n"
                f"Missing target-side evidence: {teacher_fields.get('gt_only_summary', '')}\n"
                f"Distractor-side leak evidence: {teacher_fields.get('ref_only_summary', '')}\n"
                f"ERROR_TYPE: {teacher_fields.get('primary_failure_type', '')}\n"
                f"KEEP_CUE: {teacher_fields.get('keepable_phrases', '')}\n"
                f"DROP_CUE: {teacher_fields.get('must_avoid_phrases', '')}\n"
                f"Style hint: {teacher_fields.get('candidate_style_hint', '')}\n"
                f"{instruction}\n"
                "Output exactly one line:\n"
                "REFERRING: <one short target-specific referring expression>\n"
                "Rules:\n"
                "- Keep it short, concrete, visually grounded, and close to a RefCOCO-style noun phrase.\n"
                "- Prefer 2 to 5 words, and never exceed 6 words.\n"
                "- Prefer subject-level cue words over scene-level context.\n"
                "- Keep KEEP_CUE if it helps and remove DROP_CUE if it is wrong.\n"
                "- End with a clean noun phrase, not with dangling words like with, in, on, of, or and.\n"
                "- Do not start with 'the target', 'the region', 'region1', or an explanation template.\n"
                "- Do not output analysis, markdown, bullets, [SEG], or extra labels."
            )
        if generation_mode == "referring_verification_caption":
            clean_question = self._strip_image_placeholder(self._referring_prompt_text(student_question))
            student_caption = self._normalize_teacher_field_text(student_caption)
            return (
                "<image>\n"
                "You are writing a very short verifier-friendly referring expression for region1.\n"
                "Only keep the minimum subject cue that still isolates the target.\n"
                f"Student prompt: {clean_question}\n"
                f"Current referring expression: {student_caption}\n"
                f"KEEP_CUE: {teacher_fields.get('keepable_phrases', '')}\n"
                f"TARGET_ONLY_CUE: {teacher_fields.get('target_only_evidence', '')}\n"
                f"DROP_CUE: {teacher_fields.get('must_avoid_phrases', '')}\n"
                "Output exactly one line:\n"
                "VERIFICATION_CAPTION: <one very short referring expression>\n"
                "Rules:\n"
                "- Use 2 to 4 words whenever possible, never exceed 5 words.\n"
                "- It must be shorter than the current referring expression.\n"
                "- Keep one target-only distinguishing cue when possible.\n"
                "- Avoid DROP_CUE and avoid any scene description.\n"
                "- End with a clean noun phrase, not with with, in, on, of, or and.\n"
                "- Do not output explanations, markdown, bullets, [SEG], or extra labels."
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
        prompt = self._rewrite_caption_prompt_for_referring(prompt)
        prompt = prompt.replace(
            "Your answer must start immediately with 'DLC:' on the first line.\n",
            "Your answer must start immediately with 'REFERRING:' on the first line.\n",
        )
        prompt = prompt.replace(
            "DLC: <one natural and complete detailed localized caption>\n",
            "REFERRING: <one short target-specific referring expression>\n",
        )
        prompt = prompt.replace(
            "Write only one corrected detailed localized caption for region1.\n",
            "Write only one corrected short referring expression for region1.\n",
        )
        prompt += (
            "\nExtra rules for the referring expression:\n"
            "- Prefer 2 to 5 words, and never exceed 6 words.\n"
            "- Write a compact noun phrase, not a full sentence.\n"
            "- End cleanly; do not end with with, in, on, of, or and.\n"
            "- Keep only the minimum target-specific cue needed to isolate region1.\n"
        )
        return prompt

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
                "gt_summary": pipeline_result.target_summary,
                "ref_summary": pipeline_result.distractor_summary,
                "gt_only_summary": pipeline_result.target_only_evidence,
                "ref_only_summary": pipeline_result.distractor_only_evidence,
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
            max_new_tokens=self._referring_teacher_max_new_tokens(),
        )
        pipeline_result.single_stage_raw = "" if raw_prediction is None else str(raw_prediction)
        pipeline_result.dlc_raw = pipeline_result.single_stage_raw
        pipeline_result.verification_raw = pipeline_result.single_stage_raw
        pipeline_result.problem_raw = pipeline_result.single_stage_raw
        pipeline_result.direction_raw = pipeline_result.single_stage_raw
        pipeline_result.reason_raw = pipeline_result.single_stage_raw
        pipeline_result.caption_problem = ""
        pipeline_result.correction_direction = ""
        pipeline_result.reason = ""
        pipeline_result.problem_valid = False
        pipeline_result.direction_valid = False
        pipeline_result.reason_valid = False
        pipeline_result.diagnosis_valid = False
        pipeline_result.reason_is_coarse = False

        detailed_caption_raw = self._extract_labeled_teacher_text(raw_prediction, "REFERRING")
        if not detailed_caption_raw:
            detailed_caption_raw = self._extract_labeled_teacher_text(raw_prediction, "DLC")
        if not detailed_caption_raw:
            fallback_caption = self._clean_caption_text(raw_prediction)
            fallback_status = self._infer_description_status(fallback_caption)
            if fallback_status == "ok":
                detailed_caption_raw = fallback_caption
        detailed_caption = self._force_short_referring_expression(detailed_caption_raw)
        detailed_completion_ids = self._encode_completion_from_caption(detailed_caption)
        detailed_caption, detailed_completion_ids, detailed_was_truncated = self._truncate_caption_completion(
            detailed_caption,
            detailed_completion_ids,
            max_tokens=self.description_max_new_tokens,
        )
        detailed_caption = self._force_short_referring_expression(detailed_caption)
        detailed_status = self._infer_description_status(detailed_caption)
        if detailed_status == "ok" and (
            not self._is_caption_content_sufficient(detailed_caption)
            or not self._referring_style_is_usable(detailed_caption)
        ):
            detailed_status = "truncated_caption"
        if detailed_status != "seg_style_answer" and detailed_was_truncated:
            detailed_status = "truncated_caption"
        pipeline_result.detailed_caption = detailed_caption
        pipeline_result.detailed_completion_ids = detailed_completion_ids
        pipeline_result.detailed_status = detailed_status
        pipeline_result.verification_caption = ""
        pipeline_result.verification_status = "empty"
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
                "keepable_phrases": getattr(pipeline_result, "keepable_phrases", ""),
                "must_avoid_phrases": getattr(pipeline_result, "must_avoid_phrases", ""),
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
            generation_mode="referring_verification_caption",
        )
        raw_prediction = self._predict_teacher_privileged_text(
            image=image,
            teacher_prompt_masks=teacher_prompt_masks,
            teacher_prompt=prompt,
            max_new_tokens=self._referring_verification_max_new_tokens(),
        )
        pipeline_result.verification_raw = raw_prediction
        verification_caption = self._clean_teacher_dlc_caption_text(
            self._extract_labeled_teacher_text(raw_prediction, "VERIFICATION_CAPTION")
        )
        verification_caption = self._force_short_referring_expression(verification_caption)
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
            fallback_caption = self._build_referring_verification_fallback(pipeline_result)
            if fallback_caption:
                pipeline_result.verification_caption = fallback_caption
                pipeline_result.verification_status = self._infer_description_status(fallback_caption)
                retry_reason = self._validate_teacher_verification_caption(pipeline_result)
                if not retry_reason:
                    pipeline_result.verification_failure_reason = ""
                    return pipeline_result
                failure_reason = retry_reason
        if failure_reason:
            pipeline_result.verification_failure_reason = failure_reason
        return pipeline_result

    def _metric_tensor_like(self, value, reference):
        return self._metric_tensor(float(value), reference.dtype)

    def _build_grounding_pixel_values(self, image):
        g_image = np.array(image)
        g_image = self.student_model.extra_image_processor.apply_image(g_image)
        g_pixel = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.student_model.torch_dtype)
        return torch.stack([
            self.student_model.grounding_encoder.preprocess_image(g_pixel)
        ]).to(self.student_model.torch_dtype)

    def _encode_completion_from_raw_prediction(self, raw_prediction):
        text = "" if raw_prediction is None else str(raw_prediction).strip()
        if not text:
            return torch.empty((1, 0), dtype=torch.long, device=self.device)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return torch.empty((1, 0), dtype=torch.long, device=self.device)
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _compute_referring_mask_losses_from_caption(
        self,
        *,
        image,
        caption,
        gt_mask_np,
        confuser_candidate_masks=None,
    ):
        reconstruct_question = self._resolve_reconstruct_questions(caption)[0]
        with torch.no_grad():
            reconstruction = self.reconstruct_mask(
                image=image,
                caption=caption,
                description_status="ok",
                gt_mask=gt_mask_np,
            )
        raw_prediction = reconstruction.raw_prediction if reconstruction is not None else ""
        completion_ids = self._encode_completion_from_raw_prediction(raw_prediction)
        if completion_ids.shape[1] == 0:
            return None
        model_outputs = self._forward_sequence_multi_sample_with_model(
            self.student_model,
            [
                {
                    "image": image,
                    "prompt_masks": None,
                    "prompt_text": reconstruct_question,
                    "completion_ids": completion_ids,
                    "apply_mask_focus": False,
                }
            ],
            output_hidden_states=True,
        )
        completion_hidden_states = model_outputs.get("completion_hidden_states")
        if completion_hidden_states is None or completion_hidden_states.shape[1] == 0:
            return None
        seg_mask = completion_ids[0] == self.student_model.seg_token_idx
        if int(seg_mask.sum().item()) == 0:
            return None
        seg_hidden_states = completion_hidden_states[0][seg_mask]
        all_seg_hidden_states = self.student_model.text_hidden_fcs(seg_hidden_states)
        if all_seg_hidden_states.ndim == 1:
            all_seg_hidden_states = all_seg_hidden_states.unsqueeze(0)
        g_pixel_values = self._build_grounding_pixel_values(image)
        sam_states = self.student_model.grounding_encoder.get_sam2_embeddings(g_pixel_values)
        mask_logits = self.student_model.grounding_encoder.language_embd_inference(
            sam_states,
            [all_seg_hidden_states[0].unsqueeze(0)],
        )
        mask_logits = F.interpolate(
            mask_logits,
            size=gt_mask_np.shape,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        gt_mask_t = torch.from_numpy(gt_mask_np.astype(np.float32)).to(device=mask_logits.device, dtype=mask_logits.dtype).unsqueeze(0)
        bce = F.binary_cross_entropy_with_logits(mask_logits, gt_mask_t)
        probs = torch.sigmoid(mask_logits)
        intersection = (probs * gt_mask_t).sum(dim=(-2, -1))
        denom = probs.sum(dim=(-2, -1)) + gt_mask_t.sum(dim=(-2, -1))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0))
        dice = dice.mean()
        confuser_penalty = mask_logits.new_zeros(())
        if confuser_candidate_masks:
            scored_masks = sorted(
                confuser_candidate_masks,
                key=lambda candidate_mask: self._score_confuser_candidate(gt_mask_np, candidate_mask),
                reverse=True,
            )[:3]
            penalty_terms = []
            for candidate_mask in scored_masks:
                prepared = self._prepare_mask_like_gt(gt_mask_np, candidate_mask)
                confuser_t = torch.from_numpy(prepared.astype(np.float32)).to(device=mask_logits.device, dtype=mask_logits.dtype)
                confuser_overlap = (probs[0] * confuser_t).sum() / confuser_t.sum().clamp_min(1.0)
                penalty_terms.append(confuser_overlap)
            if penalty_terms:
                confuser_penalty = torch.stack(penalty_terms).mean()
        return {
            "bce": bce,
            "dice": dice,
            "loss": bce + dice,
            "confuser_penalty": confuser_penalty,
            "reconstruction": reconstruction,
        }

    def _compute_referring_aux_losses(self, data):
        zero = self._zero_scalar(requires_grad=False)
        student_losses = []
        teacher_losses = []
        confuser_penalties = []
        hard_sample_count = 0
        spatial_count = 0
        ordinal_count = 0
        relation_count = 0
        same_category_like_count = 0
        last_failure_type = "none"
        last_secondary_type = "none"
        last_cue_passed = False
        routes = data.get("routes") or []
        for idx, (image, prompt_masks, student_question, gt_mask, confuser_candidate_masks) in enumerate(
            zip(
                data["images"],
                data["prompt_masks"],
                data["student_questions"],
                data["gt_masks"],
                data["confuser_candidate_masks"],
            )
        ):
            gt_mask_np = self._to_numpy_mask(gt_mask)
            with torch.no_grad():
                description = self.generate_description(image=image, mask_prompts=prompt_masks, student_question=student_question)
            difficulty = self._analyze_referring_difficulty(description.clean_caption)
            hard_sample_count += int(difficulty["hard_sample"])
            spatial_count += int(difficulty["spatial_required"])
            ordinal_count += int(difficulty["ordinal_required"])
            relation_count += int(difficulty["relation_required"])
            same_category_like_count += int(difficulty["same_category_multi_instance_like"])
            if description.status == "ok" and self._caption_token_count(description.clean_caption) >= 2:
                student_mask_result = self._compute_referring_mask_losses_from_caption(
                    image=image,
                    caption=description.clean_caption,
                    gt_mask_np=gt_mask_np,
                    confuser_candidate_masks=confuser_candidate_masks,
                )
                if student_mask_result is not None:
                    reconstruction = student_mask_result["reconstruction"]
                    pred_mask = None if reconstruction is None else reconstruction.pred_mask
                    iou = self._compute_iou(gt_mask_np, pred_mask) if pred_mask is not None else 0.0
                    if iou >= self.referring_direct_mask_loss_min_iou_gate:
                        student_losses.append(student_mask_result["loss"])
                        confuser_penalties.append(student_mask_result["confuser_penalty"])
                else:
                    reconstruction = self._invalid_reconstruction_placeholder("referring_direct_mask_unavailable")
            else:
                reconstruction = self._invalid_reconstruction_placeholder("referring_caption_invalid")
            pred_mask = None if reconstruction is None else reconstruction.pred_mask
            iou = self._compute_iou(gt_mask_np, pred_mask) if pred_mask is not None else 0.0
            route_from_manifest = routes[idx] if idx < len(routes) else None
            online_route = self._route_from_iou(iou)
            loss_family = self._resolve_loss_family(None, route_from_manifest=route_from_manifest, online_route=online_route)
            with torch.no_grad():
                teacher_analysis = self._attempt_teacher_regenerate_analysis(
                    image=image,
                    gt_mask_np=gt_mask_np,
                    ref_mask_np=self._zero_ref_mask_like(gt_mask_np) if pred_mask is None else self._to_numpy_mask(pred_mask),
                    student_question=student_question,
                    description=description,
                    reconstruction=reconstruction,
                    iou=iou,
                    allow_teacher_ce=True,
                )
            teacher_regenerate = teacher_analysis.get("teacher_regenerate")
            if teacher_regenerate is not None:
                last_failure_type = getattr(teacher_regenerate, "primary_failure_type", "none") or "none"
                last_secondary_type = getattr(teacher_regenerate, "secondary_failure_type", "none") or "none"
                last_cue_passed = bool(getattr(teacher_regenerate, "cue_passed", False))
            if (
                loss_family == "teacher_regenerate"
                and teacher_analysis.get("teacher_reconstruct_ok")
                and teacher_analysis.get("teacher_gate_passed")
                and teacher_regenerate is not None
                and self._teacher_field_is_effective(teacher_regenerate.detailed_caption, invalid_markers=("",))
            ):
                teacher_mask_result = self._compute_referring_mask_losses_from_caption(
                    image=image,
                    caption=teacher_regenerate.detailed_caption,
                    gt_mask_np=gt_mask_np,
                    confuser_candidate_masks=confuser_candidate_masks,
                )
                if teacher_mask_result is not None:
                    teacher_losses.append(teacher_mask_result["loss"])
        batch_count = max(len(data["images"]), 1)
        student_loss = torch.stack(student_losses).mean() if student_losses else zero
        teacher_loss = torch.stack(teacher_losses).mean() if teacher_losses else zero
        confuser_penalty = torch.stack(confuser_penalties).mean() if confuser_penalties else zero
        return {
            "student_loss": student_loss,
            "teacher_loss": teacher_loss,
            "confuser_penalty": confuser_penalty,
            "hard_sample_rate": hard_sample_count / batch_count,
            "spatial_rate": spatial_count / batch_count,
            "ordinal_rate": ordinal_count / batch_count,
            "relation_rate": relation_count / batch_count,
            "same_category_like_rate": same_category_like_count / batch_count,
            "last_failure_type": last_failure_type,
            "last_secondary_type": last_secondary_type,
            "last_cue_passed": last_cue_passed,
        }

    def forward(self, data, data_samples=None, mode="loss"):
        metrics = super().forward(data, data_samples=data_samples, mode=mode)
        if isinstance(metrics, dict):
            if "teacher_dlc_valid_rate" in metrics:
                metrics["teacher_referring_valid_rate"] = metrics.pop("teacher_dlc_valid_rate")
            if "teacher_regenerate_dlc_ce_applied_count" in metrics:
                metrics["teacher_regenerate_referring_ce_applied_count"] = metrics.pop("teacher_regenerate_dlc_ce_applied_count")
        if (
            mode != "loss"
            or not self.training
            or not self.enable_referring_direct_mask_loss
            or not isinstance(metrics, dict)
            or "loss_opsd_total" not in metrics
        ):
            return metrics
        aux = self._compute_referring_aux_losses(data)
        total_aux = (
            aux["student_loss"] * self.referring_direct_mask_loss_weight
            + aux["teacher_loss"] * self.referring_teacher_direct_mask_loss_weight
            + aux["confuser_penalty"] * self.referring_confuser_separation_loss_weight
        )
        metrics["loss_opsd_total"] = metrics["loss_opsd_total"] + total_aux
        metrics["referring_direct_mask_loss"] = aux["student_loss"].detach()
        metrics["referring_teacher_direct_mask_loss"] = aux["teacher_loss"].detach()
        metrics["referring_confuser_separation_loss"] = aux["confuser_penalty"].detach()
        metrics["referring_hard_sample_rate"] = self._metric_tensor_like(aux["hard_sample_rate"], metrics["loss_opsd_total"])
        metrics["referring_spatial_sample_rate"] = self._metric_tensor_like(aux["spatial_rate"], metrics["loss_opsd_total"])
        metrics["referring_ordinal_sample_rate"] = self._metric_tensor_like(aux["ordinal_rate"], metrics["loss_opsd_total"])
        metrics["referring_relation_sample_rate"] = self._metric_tensor_like(aux["relation_rate"], metrics["loss_opsd_total"])
        metrics["referring_same_category_like_rate"] = self._metric_tensor_like(
            aux["same_category_like_rate"], metrics["loss_opsd_total"]
        )
        print(
            "[Sa2VA_REFERRING_V3] "
            f"teacher_primary_failure_type={aux['last_failure_type']} "
            f"teacher_cue_passed={int(aux['last_cue_passed'])}",
            flush=True,
        )
        return metrics

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
        relation_context = build_mask_relation_context(model=self, gt_mask=gt_mask, ref_mask=ref_mask)
        pipeline_result = self._build_empty_teacher_regenerate_pipeline_result()
        pipeline_result.teacher_pipeline_mode = "referring_minimal_4field"
        pipeline_result.target_summary = self._normalize_teacher_field_text(relation_context.get("gt_summary", ""))
        pipeline_result.distractor_summary = self._normalize_teacher_field_text(relation_context.get("ref_summary", ""))
        pipeline_result.target_only_evidence = self._normalize_teacher_field_text(relation_context.get("gt_only_summary", ""))
        pipeline_result.distractor_only_evidence = self._normalize_teacher_field_text(relation_context.get("ref_only_summary", ""))
        pipeline_result.shared_evidence = ""
        pipeline_result.difference_focus = ""
        pipeline_result.target_localization_hint = ""
        pipeline_result.distractor_localization_hint = ""
        pipeline_result.likely_drift_reason = ""
        pipeline_result.difference_context_nontrivial = False
        pipeline_result.stop_stage = "minimal_relation_cues"
        diagnosis_teacher_fields = dict(teacher_fields)
        diagnosis_teacher_fields.update(
            {
                "gt_summary": pipeline_result.target_summary,
                "ref_summary": pipeline_result.distractor_summary,
                "gt_only_summary": pipeline_result.target_only_evidence,
                "ref_only_summary": pipeline_result.distractor_only_evidence,
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
            primary_hint = self._referring_type_specific_style_hint(
                getattr(pipeline_result, "primary_failure_type", ""),
                repair_mode=False,
            )
            candidate_specs = (
                {
                    "candidate_style_hint": primary_hint,
                    "do_sample": False,
                },
                {
                    "candidate_style_hint": f"{primary_hint} Keep it to 2 to 5 words if possible.",
                    "do_sample": True,
                    "temperature": 0.35,
                    "top_p": 0.9,
                },
                {
                    "candidate_style_hint": f"{primary_hint} Remove every distractor-compatible phrase.",
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
            pipeline_result.teacher_pipeline_mode = "referring_candidate_selected"
            pipeline_result.teacher_dlc_selected_by = "candidate_score"
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
                pipeline_result.teacher_pipeline_mode = "referring_verification_selected"
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
        pipeline_result.cue_passed = bool(
            selected_candidate is not None and bool(selected_candidate.get("cue_passed", False))
        )
        pipeline_result.verification_iou = float(teacher_iou_plain)
        iou_gate_passed = (
            bool(teacher_reconstruct_ok)
            if caption_mode_failure
            else (
                self._referring_teacher_regenerate_gate_passed(student_caption, iou, teacher_iou_plain)
                if teacher_reconstruct_ok
                else False
            )
        )
        pipeline_result.gate_passed = bool(iou_gate_passed and pipeline_result.cue_passed)

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
                candidate_style_hint=self._referring_type_specific_style_hint(
                    getattr(pipeline_result, "primary_failure_type", ""),
                    repair_mode=True,
                ),
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
                pipeline_result.teacher_pipeline_mode = "referring_candidate_selected"
                pipeline_result.teacher_dlc_selected_by = "repair_candidate"
                pipeline_result.verification_iou = repair_iou
                teacher_reconstruct_ok = bool(
                    pipeline_result.detailed_status == "ok" and not pipeline_result.detailed_failure_reason and repair_iou > 0.0
                )
                pipeline_result.cue_passed = bool(repair_candidate.get("cue_passed", False))
                iou_gate_passed = (
                    bool(teacher_reconstruct_ok)
                    if caption_mode_failure
                    else (
                        self._referring_teacher_regenerate_gate_passed(student_caption, iou, repair_iou)
                        if teacher_reconstruct_ok
                        else False
                    )
                )
                pipeline_result.gate_passed = bool(iou_gate_passed and pipeline_result.cue_passed)
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
            fallback_result.teacher_fallback_used = True
            fallback_result.teacher_fallback_reason = pipeline_result.teacher_fallback_reason
            fallback_result.stop_stage = "single_stage_fallback"
            fallback_result.teacher_pipeline_mode = "referring_single_stage_fallback"
            if self._teacher_field_is_effective(fallback_result.single_stage_raw, invalid_markers=("",)):
                fallback_style_ok = (
                    self._referring_style_is_usable(fallback_result.detailed_caption)
                    and self._is_caption_content_sufficient(fallback_result.detailed_caption)
                )
                if not fallback_style_ok:
                    fallback_result.detailed_failure_reason = "teacher_dlc_invalid:referring_style_gate_failed"
                    fallback_result.cue_passed = False
                    fallback_result.gate_passed = False
                else:
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
                    fallback_result.cue_passed = self._referring_candidate_type_constraint_passed(
                        fallback_result.detailed_caption,
                        fallback_result,
                    )
                    fallback_result.gate_passed = bool(fallback_result.gate_passed and fallback_result.cue_passed)
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
