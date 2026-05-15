import re

import numpy as np


FIVE_LABELS = ("GTMASK", "REFMASK", "CAPTION_PROBLEM", "CORRECTION_DIRECTION", "REASON")
THREE_LABELS = ("CAPTION_PROBLEM", "CORRECTION_DIRECTION", "REASON")
CAPTION_TO_MASK_SUCCESS_IOU_THRESHOLD = 0.5
LOW_IOU_TEACHER_REGENERATE_THRESHOLD = 0.3
HIGH_QUALITY_IOU_THRESHOLD = 0.8
TEACHER_REGENERATE_ROUTE = "teacher_regenerate"
ON_POLICY_DISTILL_ROUTE = "on_policy_distill"
GRPO_POSITIVE_ROUTE = "grpo_positive"


def normalize_teacher_output(text):
    text = (text or "").replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("<|end|>", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def parse_labeled_sections(text, labels):
    normalized = normalize_teacher_output(text)
    parsed = {}
    joined_labels = "|".join(map(re.escape, labels))
    for label in labels:
        pattern = rf"(?ms)^\s*{re.escape(label)}:\s*(.*?)(?=^\s*(?:{joined_labels}):|\Z)"
        match = re.search(pattern, normalized)
        parsed[label.lower()] = match.group(1).strip() if match else ""
    return parsed


def labeled_field_count(fields, labels):
    return sum(bool((fields.get(label.lower()) or "").strip()) for label in labels)


def is_degenerate_teacher_output(text):
    normalized = re.sub(r"\s+", " ", normalize_teacher_output(text).lower()).strip()
    if not normalized:
        return True
    if "[seg]" in normalized:
        return True
    explicit_bad_patterns = (
        "region summary for region1",
        "region summary for region2",
        "does not match the observed region",
        "region descriptions do not match the image",
        "do not match the image",
        "adjust the bounding box",
        "correct the description of region1",
        "correct the description of region2",
    )
    if any(pattern in normalized for pattern in explicit_bad_patterns):
        return True
    repeated_phrases = (
        "the caption is too vague",
        "specify the exact",
        "improve segmentation accuracy",
        "provide more specific information",
    )
    if any(normalized.count(phrase) >= 3 for phrase in repeated_phrases):
        return True
    tokens = re.findall(r"[a-z0-9']+", normalized)
    if len(tokens) >= 48:
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        if unique_ratio < 0.4:
            return True
    return False


def score_labeled_output(text, fields, labels):
    score = labeled_field_count(fields, labels) * 100
    normalized = normalize_teacher_output(text)
    if labels and normalized.startswith(f"{labels[0]}:"):
        score += 10
    if not is_degenerate_teacher_output(normalized):
        score += 5
    score -= min(len(normalized) // 400, 5)
    return score


def format_teacher_structured_output(fields):
    return "\n".join(
        [
            f"GTMASK: {fields['gtmask']}",
            f"REFMASK: {fields['refmask']}",
            f"CAPTION_PROBLEM: {fields['caption_problem']}",
            f"CORRECTION_DIRECTION: {fields['correction_direction']}",
            f"REASON: {fields['reason']}",
        ]
    )


def build_mask_relation_context(*, model, gt_mask, ref_mask):
    gt_mask = np.asarray(gt_mask).astype(np.uint8)
    ref_mask = np.asarray(ref_mask).astype(np.uint8)
    gt_only = np.logical_and(gt_mask > 0, ref_mask == 0).astype(np.uint8)
    ref_only = np.logical_and(ref_mask > 0, gt_mask == 0).astype(np.uint8)
    return {
        "gt_summary": model._mask_summary(gt_mask),
        "ref_summary": model._mask_summary(ref_mask),
        "gt_only_summary": "none" if int(gt_only.sum()) == 0 else model._mask_summary(gt_only),
        "ref_only_summary": "none" if int(ref_only.sum()) == 0 else model._mask_summary(ref_only),
    }


def build_region_description_fallback(*, model, mask, role_name, paired_region_text="", iou=None):
    mask = np.asarray(mask).astype(np.uint8)
    area = int(mask.sum())
    if area == 0:
        return f"{role_name} is empty or nearly empty."
    if role_name == "region2" and paired_region_text and iou is not None and iou >= 0.75:
        return f"Nearly the same target as region1: {paired_region_text}."
    return f"{role_name} summary: {model._mask_summary(mask)}."


def _is_region_description_usable(model, description):
    clean_caption = (description.clean_caption or "").strip()
    if not clean_caption:
        return False
    if description.status == "ok":
        return True
    if model._caption_token_count(clean_caption) >= 2 and not model._is_overly_generic_caption(clean_caption):
        return True
    return False


def describe_region_with_retries(*, model, region_model, image, mask, student_question):
    mask = np.asarray(mask).astype(np.float32)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=0)

    candidates = [
        model.generate_description_with_model(
            region_model,
            image=image,
            mask_prompts=mask,
            student_question=student_question,
            apply_mask_focus=True,
        ),
        model.generate_description_with_model(
            region_model,
            image=image,
            mask_prompts=mask,
            student_question=student_question,
            apply_mask_focus=False,
        ),
    ]
    return max(
        candidates,
        key=lambda item: model._description_quality_score(item.raw_prediction, item.clean_caption, item.status),
    )


def caption_to_mask_seg_correct(*, iou):
    return float(iou) >= CAPTION_TO_MASK_SUCCESS_IOU_THRESHOLD


def classify_teacher_route(
    *,
    iou,
    low_threshold=LOW_IOU_TEACHER_REGENERATE_THRESHOLD,
    high_threshold=HIGH_QUALITY_IOU_THRESHOLD,
):
    iou = float(iou)
    if iou <= float(low_threshold):
        return TEACHER_REGENERATE_ROUTE
    if iou >= float(high_threshold):
        return GRPO_POSITIVE_ROUTE
    return ON_POLICY_DISTILL_ROUTE


def _bool_text(value):
    return "true" if bool(value) else "false"


def build_compare_prompt_variants(
    *,
    sample,
    reconstruction,
    iou,
    gt_desc,
    ref_desc,
    gt_mask,
    ref_mask,
    route,
):
    del gt_desc, ref_desc
    student_question = sample.get("student_question", "").replace("<image>", "").strip()
    caption = sample.get("caption", "") or ""
    description_status = sample.get("description_status", "ok")
    relation_context = build_mask_relation_context(
        model=sample["model"],
        gt_mask=gt_mask,
        ref_mask=ref_mask,
    )
    seg_correct = caption_to_mask_seg_correct(iou=iou)
    if route == TEACHER_REGENERATE_ROUTE:
        route_instruction = (
            "The current sample is unusable. Diagnose why the current caption failed and provide evidence that can support a full rewrite for region1."
        )
    elif route == ON_POLICY_DISTILL_ROUTE:
        route_instruction = (
            "The current sample is partially usable. Focus on the missing or misleading target cue while keeping the student's caption semantics on-policy."
        )
    else:
        route_instruction = (
            "The current sample is already high quality, so only make minimal corrections if a real mismatch remains."
        )
    base_context = (
        "You are optimizing the following task: given a gtmask, generate a caption that describes it. "
        "You are now given the original input, the student question, and privileged verification information. "
        "Use these privileged signals to improve the caption generation.\n"
        f"Teacher route: {route}\n"
        f"Student prompt: {student_question}\n"
        f"Student caption: {caption}\n"
        f"Description status: {description_status}\n"
        f"Reconstruction status: {reconstruction.status}\n"
        f"caption_to_mask_seg_correct: {_bool_text(seg_correct)}\n"
        "IoU is the intersection-over-union between gtmask and refmask: intersection / union.\n"
        "If IoU is close to 0, gtmask and refmask are weakly related or completely different.\n"
        "If IoU is close to 1, gtmask and refmask are very similar.\n"
        f"Current IoU: {iou:.4f}\n"
        f"Unique non-overlap area in gtmask: {relation_context['gt_only_summary']}\n"
        f"Unique non-overlap area in refmask: {relation_context['ref_only_summary']}\n"
    )
    return [
        f"""<image>
You are optimizing the following task: given a gtmask, generate a caption that describes it. You are now given the original input, the student question, and privileged verification information. Use these privileged signals to improve the caption generation.
- region1 = gtmask = the original target mask
- The system first generates a caption from region1.
- Then it uses that caption to segment again and gets region2 = refmask.

Your job is to compare gtmask(region1) and refmask(region2), judge why the caption led to region2 instead of region1, and improve the caption for the gtmask description task.
Use the visual difference between region1 and region2.
The observed region descriptions below may help, but if they conflict with the image, trust the image.
Judge what problem the current IoU indicates from the facts below. Do not rely on any pre-labeled failure category.
Use the IoU value, the caption_to_mask_seg_correct flag, and the non-overlap areas to decide the problem.
{route_instruction}

{base_context}
Output exactly 3 lines and nothing else:
CAPTION_PROBLEM:
CORRECTION_DIRECTION:
REASON:
""",
        f"""<image>
You are optimizing the following task: given a gtmask, generate a caption that describes it. You are now given the original input, the student question, and privileged verification information. Use these privileged signals to improve the caption generation.
region1 is the original gtmask.
region2 is the refmask produced after re-segmenting from the generated caption.
Compare gtmask(region1) and refmask(region2), explain why the caption led to region2, and say how to improve the caption.
Judge what problem the current IoU indicates from the facts below. Do not rely on any pre-labeled failure category.
Use the IoU value, the caption_to_mask_seg_correct flag, and the difference between the non-overlap areas to decide how to revise the caption.
{route_instruction}
student caption: {caption}
caption_to_mask_seg_correct: {_bool_text(seg_correct)}
Current IoU: {iou:.4f}
Unique non-overlap area in gtmask: {relation_context['gt_only_summary']}
Unique non-overlap area in refmask: {relation_context['ref_only_summary']}
Return only:
CAPTION_PROBLEM:
CORRECTION_DIRECTION:
REASON:
""",
    ]


def build_compare_fallback(
    *,
    model,
    sample,
    reconstruction,
    iou,
    gt_mask,
    ref_mask,
    gt_desc,
    ref_desc,
    route,
):
    caption = (sample.get("caption") or "").strip()
    token_count = model._caption_token_count(caption)
    generic_caption = model._is_overly_generic_caption(caption)
    seg_correct = caption_to_mask_seg_correct(iou=iou)

    if sample.get("description_status") != "ok":
        problem = "The first-stage caption was empty or malformed, so reconstruction had no stable target."
        correction = "Write a valid sentence that directly describes the masked target with clearly visible details."
        reason = "Without a usable first caption, the reconstructor has no stable target to follow."
    elif route == TEACHER_REGENERATE_ROUTE:
        if reconstruction.status != "ok" or int(np.asarray(ref_mask).sum()) == 0:
            problem = "The current caption does not reconstruct a usable target mask from the gtmask description task."
            correction = "Rewrite the caption from scratch so it directly describes the gtmask target with visible attributes and the minimum local context needed to recover region1."
            reason = "The reconstruction result is missing or unusable, so the current student trajectory should be replaced rather than lightly edited."
        else:
            problem = "The current caption still does not describe the gtmask precisely enough for reliable reconstruction."
            correction = "Rewrite the caption so region1 is described directly, then add one strong visible attribute, relation, or side cue that separates gtmask from nearby distractors."
            reason = "The IoU and non-overlap areas show that the current student trajectory is too far from region1 and needs a stronger reset."
    elif route == ON_POLICY_DISTILL_ROUTE:
        problem = "The current caption is partially useful but still misses target-specific evidence for stable reconstruction."
        correction = "Keep the correct target category when possible, then add one stronger visible attribute, relation, or side cue that separates gtmask from nearby distractors."
        reason = "The IoU is moderate, so the student caption has useful structure but still needs a guided on-policy correction."
    else:
        problem = "The caption already reconstructs a region close to the gtmask and only a small local mismatch may remain."
        correction = "Keep the caption stable and change at most one precise local detail if the current mismatch is real."
        reason = "The IoU is already high, so large caption edits are more likely to hurt than help."

    if generic_caption or token_count <= 2:
        correction += " Avoid single-word or overly generic captions."

    if gt_desc and ref_desc and gt_desc != ref_desc and not seg_correct:
        reason = (
            f"region1 is currently interpreted as '{gt_desc}', while region2 behaves more like '{ref_desc}'. "
            "The correction should push the caption toward region1 and away from region2."
        )

    return {
        "caption_problem": problem,
        "correction_direction": correction,
        "reason": reason,
    }


def _empty_teacher_metrics(prefix, *, route):
    metrics = {
        f"{prefix}_caption": "",
        f"{prefix}_raw_prediction": "",
        f"{prefix}_status": "skipped_inactive_route",
        f"{prefix}_prompt": "",
        f"{prefix}_reconstruct_status": "skipped_inactive_route",
        f"{prefix}_reconstruct_question": None,
        f"{prefix}_reconstruct_raw_prediction": "",
        f"{prefix}_prediction_masks_count": 0,
        f"{prefix}_pred_mask_sum": None,
        f"{prefix}_reconstruct_iou": 0.0,
        f"{prefix}_success_iou_gt_0_5": False,
        f"{prefix}_route": route,
    }
    if prefix == "teacher_predict":
        metrics["teacher_predict_caption_self_iou"] = 0.0
    return metrics


def _build_caption_reconstruction_metrics(
    *,
    prefix,
    model,
    image,
    gt_mask,
    caption_result,
    prompt_text,
    route,
):
    predict_caption = (caption_result.clean_caption or "").strip()
    metrics = {
        f"{prefix}_caption": predict_caption,
        f"{prefix}_raw_prediction": normalize_teacher_output(caption_result.raw_prediction),
        f"{prefix}_status": caption_result.status,
        f"{prefix}_prompt": prompt_text or "",
        f"{prefix}_reconstruct_status": "skipped_no_teacher_caption",
        f"{prefix}_reconstruct_question": None,
        f"{prefix}_reconstruct_raw_prediction": "",
        f"{prefix}_prediction_masks_count": 0,
        f"{prefix}_pred_mask_sum": None,
        f"{prefix}_reconstruct_iou": 0.0,
        f"{prefix}_success_iou_gt_0_5": False,
        f"{prefix}_route": route,
    }
    if not predict_caption:
        return metrics

    reconstruction = model.reconstruct_mask(
        image=image,
        caption=predict_caption,
        description_status=caption_result.status,
        spatial_hint=model._coarse_spatial_hint(gt_mask),
        gt_mask=gt_mask,
    )
    reconstruct_iou = model._compute_iou(gt_mask, reconstruction.pred_mask)
    metrics.update(
        {
            f"{prefix}_reconstruct_status": reconstruction.status,
            f"{prefix}_reconstruct_question": reconstruction.question,
            f"{prefix}_reconstruct_raw_prediction": reconstruction.raw_prediction,
            f"{prefix}_prediction_masks_count": reconstruction.prediction_masks_count,
            f"{prefix}_pred_mask_sum": None
            if reconstruction.pred_mask is None
            else int(np.asarray(model._to_numpy_mask(reconstruction.pred_mask)).sum()),
            f"{prefix}_reconstruct_iou": float(reconstruct_iou),
            f"{prefix}_success_iou_gt_0_5": bool(
                caption_to_mask_seg_correct(iou=reconstruct_iou)
            ),
        }
    )
    return metrics


def _build_teacher_predict_reconstruction_metrics(*, model, image, gt_mask, student_question, student_caption):
    teacher_predict = model.predict_teacher_caption_on_student_trajectory(
        image=image,
        mask_prompts=gt_mask,
        student_question=student_question,
        student_caption=student_caption,
        apply_mask_focus=True,
    )
    metrics = _build_caption_reconstruction_metrics(
        prefix="teacher_predict",
        model=model,
        image=image,
        gt_mask=gt_mask,
        caption_result=teacher_predict,
        prompt_text=student_question,
        route=ON_POLICY_DISTILL_ROUTE,
    )
    metrics["teacher_predict_caption_self_iou"] = metrics["teacher_predict_reconstruct_iou"]
    return metrics


def _build_teacher_regenerate_reconstruction_metrics(
    *,
    model,
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
    teacher_regenerate = model.generate_teacher_caption_with_privileged_prompt(
        image=image,
        gt_mask=gt_mask,
        ref_mask=ref_mask,
        student_question=student_question,
        student_caption=student_caption,
        description_status=description_status,
        reconstruction=reconstruction,
        iou=iou,
        teacher_fields=teacher_fields,
    )
    return _build_caption_reconstruction_metrics(
        prefix="teacher_regenerate",
        model=model,
        image=image,
        gt_mask=gt_mask,
        caption_result=teacher_regenerate,
        prompt_text=teacher_fields.get("teacher_regenerate_prompt", ""),
        route=TEACHER_REGENERATE_ROUTE,
    )


def _build_teacher_diagnosis_fields_staged(
    *,
    model,
    sample,
    reconstruction,
    gt_mask,
    iou,
    include_teacher_metrics=False,
    low_threshold=LOW_IOU_TEACHER_REGENERATE_THRESHOLD,
    high_threshold=HIGH_QUALITY_IOU_THRESHOLD,
):
    ref_mask = gt_mask * 0 if reconstruction.pred_mask is None else model._to_numpy_mask(reconstruction.pred_mask)
    sample_with_model = {**sample, "model": model}
    require_teacher_model = getattr(model, "require_teacher_model", None)
    if callable(require_teacher_model):
        region_model = require_teacher_model("Teacher diagnosis field construction")
    else:
        region_model = getattr(model, "teacher_model", None)
        if region_model is None:
            raise RuntimeError(
                "Teacher diagnosis field construction requires an initialized teacher model."
            )

    gt_desc_result = describe_region_with_retries(
        model=model,
        region_model=region_model,
        image=sample["image"],
        mask=gt_mask,
        student_question=sample.get("student_question", ""),
    )
    ref_desc_result = describe_region_with_retries(
        model=model,
        region_model=region_model,
        image=sample["image"],
        mask=ref_mask,
        student_question=sample.get("student_question", ""),
    )

    gt_desc = gt_desc_result.clean_caption.strip()
    ref_desc = ref_desc_result.clean_caption.strip()
    gt_desc_usable = _is_region_description_usable(model, gt_desc_result)
    ref_desc_usable = _is_region_description_usable(model, ref_desc_result)

    teacher_gtmask = (
        gt_desc
        if gt_desc_usable
        else build_region_description_fallback(model=model, mask=gt_mask, role_name="region1")
    )
    teacher_refmask = (
        ref_desc
        if ref_desc_usable
        else build_region_description_fallback(
            model=model,
            mask=ref_mask,
            role_name="region2",
            paired_region_text=teacher_gtmask,
            iou=iou,
        )
    )
    route = classify_teacher_route(
        iou=iou,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    seg_correct = caption_to_mask_seg_correct(iou=iou)

    teacher_mask_prompts = np.stack([gt_mask.astype(np.float32), ref_mask.astype(np.float32)], axis=0)
    best_prompt = ""
    best_output = ""
    best_parsed = {}
    best_score = None
    generation_overrides = {
        "max_new_tokens": 128,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 5,
    }

    for teacher_prompt in build_compare_prompt_variants(
        sample=sample_with_model,
        reconstruction=reconstruction,
        iou=iou,
        gt_desc=teacher_gtmask,
        ref_desc=teacher_refmask,
        gt_mask=gt_mask,
        ref_mask=ref_mask,
        route=route,
    ):
        teacher_predict = model.predict_text_with_masks(
            region_model,
            image=sample["image"],
            text=teacher_prompt,
            mask_prompts=teacher_mask_prompts,
            apply_mask_focus=True,
            generation_overrides=generation_overrides,
        )
        teacher_output = normalize_teacher_output(teacher_predict.get("prediction", ""))
        parsed = parse_labeled_sections(teacher_output, THREE_LABELS)
        score = score_labeled_output(teacher_output, parsed, THREE_LABELS)
        if best_score is None or score > best_score:
            best_prompt = teacher_prompt
            best_output = teacher_output
            best_parsed = parsed
            best_score = score
        if labeled_field_count(parsed, THREE_LABELS) == len(THREE_LABELS) and not is_degenerate_teacher_output(teacher_output):
            break

    fallback_compare = build_compare_fallback(
        model=model,
        sample=sample_with_model,
        reconstruction=reconstruction,
        iou=iou,
        gt_mask=gt_mask,
        ref_mask=ref_mask,
        gt_desc=teacher_gtmask,
        ref_desc=teacher_refmask,
        route=route,
    )
    best_output_is_degenerate = is_degenerate_teacher_output(best_output)
    usable_best_parsed = best_parsed if not best_output_is_degenerate else {}
    merged_compare = {
        key: (usable_best_parsed.get(key) or fallback_compare[key]).strip()
        for key in ("caption_problem", "correction_direction", "reason")
    }

    parse_status = "staged_compare_parsed"
    if labeled_field_count(best_parsed, THREE_LABELS) == 0:
        parse_status = "staged_compare_fallback_full"
    elif labeled_field_count(best_parsed, THREE_LABELS) < len(THREE_LABELS):
        parse_status = "staged_compare_fallback_partial"
    if best_output_is_degenerate:
        parse_status = f"{parse_status}_degenerate_output"

    output_source = "staged_model"
    if labeled_field_count(best_parsed, THREE_LABELS) < len(THREE_LABELS) or best_output_is_degenerate:
        output_source = "fallback_structured"

    final_fields = {
        "gtmask": teacher_gtmask,
        "refmask": teacher_refmask,
        "caption_problem": merged_compare["caption_problem"],
        "correction_direction": merged_compare["correction_direction"],
        "reason": merged_compare["reason"],
    }

    result = {
        "teacher_prompt": best_prompt,
        "teacher_raw_output": best_output,
        "teacher_output": format_teacher_structured_output(final_fields),
        "teacher_output_source": output_source,
        "teacher_parse_status": parse_status,
        "teacher_gtmask": final_fields["gtmask"],
        "teacher_refmask": final_fields["refmask"],
        "teacher_caption_problem": final_fields["caption_problem"],
        "teacher_correction_direction": final_fields["correction_direction"],
        "teacher_reason": final_fields["reason"],
        "teacher_route": route,
        "caption_to_mask_seg_correct": bool(seg_correct),
        "teacher_gtmask_raw_output": normalize_teacher_output(gt_desc_result.raw_prediction),
        "teacher_gtmask_status": gt_desc_result.status,
        "teacher_refmask_raw_output": normalize_teacher_output(ref_desc_result.raw_prediction),
        "teacher_refmask_status": ref_desc_result.status,
    }
    if include_teacher_metrics:
        result.update(_empty_teacher_metrics("teacher_predict", route=route))
        result.update(_empty_teacher_metrics("teacher_regenerate", route=route))
        if route == ON_POLICY_DISTILL_ROUTE:
            result.update(
                _build_teacher_predict_reconstruction_metrics(
                    model=model,
                    image=sample["image"],
                    gt_mask=gt_mask,
                    student_question=sample.get("student_question", ""),
                    student_caption=sample.get("caption", ""),
                )
            )
        elif route == TEACHER_REGENERATE_ROUTE:
            result.update(
                _build_teacher_regenerate_reconstruction_metrics(
                    model=model,
                    image=sample["image"],
                    gt_mask=gt_mask,
                    ref_mask=ref_mask,
                    student_question=sample.get("student_question", ""),
                    student_caption=sample.get("caption", ""),
                    description_status=sample.get("description_status", ""),
                    reconstruction=reconstruction,
                    iou=iou,
                    teacher_fields=result,
                )
            )
    return result


def build_teacher_diagnosis_fields_staged(
    *,
    model,
    sample,
    reconstruction,
    gt_mask,
    iou,
    low_threshold=LOW_IOU_TEACHER_REGENERATE_THRESHOLD,
    high_threshold=HIGH_QUALITY_IOU_THRESHOLD,
):
    return _build_teacher_diagnosis_fields_staged(
        model=model,
        sample=sample,
        reconstruction=reconstruction,
        gt_mask=gt_mask,
        iou=iou,
        include_teacher_metrics=False,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )


def build_teacher_context_validation_fields(
    *,
    model,
    sample,
    reconstruction,
    gt_mask,
    iou,
    low_threshold=LOW_IOU_TEACHER_REGENERATE_THRESHOLD,
    high_threshold=HIGH_QUALITY_IOU_THRESHOLD,
):
    return _build_teacher_diagnosis_fields_staged(
        model=model,
        sample=sample,
        reconstruction=reconstruction,
        gt_mask=gt_mask,
        iou=iou,
        include_teacher_metrics=True,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
