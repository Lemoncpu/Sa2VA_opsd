import argparse
import json
import os
import re
import runpy
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.sa2va.datasets.refcoco_opsd import build_refcoco_opsd_records
from projects.sa2va.evaluation.caption_to_mask_common import run_caption_to_mask_eval
from projects.sa2va.evaluation.teacher_diagnosis_common import build_teacher_diagnosis_fields_staged
from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3


def parse_args():
    parser = argparse.ArgumentParser(description="Run RefCOCO mask->caption->mask closure evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--annotation-file", default=None, help="Reuse an existing annotations.json and rerun stage 2 only.")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(path):
    return runpy.run_path(path)


def resolve_image_root(cfg, cli_image_root=None):
    if cli_image_root:
        return cli_image_root
    if cfg.get("image_root"):
        return cfg["image_root"]
    for candidate in cfg.get("image_root_candidates", []):
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "No valid RefCOCO image_root found. "
        "Pass --image-root explicitly or update image_root_candidates in the config."
    )


def mask_to_rle(mask):
    encoded = mask_utils.encode(np.asfortranarray(np.asarray(mask).astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode()
    return encoded


def parse_teacher_diagnosis_output(text):
    text = normalize_teacher_output(text)
    normalized = text
    replacements = {
        r"(?im)^\s*gt[\s_-]*mask\s*:": "GTMASK:",
        r"(?im)^\s*ref[\s_-]*mask\s*:": "REFMASK:",
        r"(?im)^\s*caption[\s_-]*problem\s*:": "CAPTION_PROBLEM:",
        r"(?im)^\s*correction[\s_-]*direction\s*:": "CORRECTION_DIRECTION:",
        r"(?im)^\s*recorrection[\s_-]*direction\s*:": "CORRECTION_DIRECTION:",
        r"(?im)^\s*reason\s*:": "REASON:",
    }
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized)

    fields = {}
    labels = ["GTMASK", "REFMASK", "CAPTION_PROBLEM", "CORRECTION_DIRECTION", "REASON"]
    for label in labels:
        pattern = rf"(?ms)^\s*{label}:\s*(.*?)(?=^\s*(?:{'|'.join(labels)}):|\Z)"
        match = re.search(pattern, normalized)
        fields[label.lower()] = match.group(1).strip() if match else ""
    return fields


def normalize_teacher_output(text):
    text = (text or "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def teacher_parsed_field_count(fields):
    return sum(bool((fields.get(key) or "").strip()) for key in (
        "gtmask",
        "refmask",
        "caption_problem",
        "correction_direction",
        "reason",
    ))


def is_degenerate_teacher_output(text):
    normalized = re.sub(r"\s+", " ", normalize_teacher_output(text).lower()).strip()
    if not normalized:
        return True
    repeated_phrases = (
        "the caption is not clear",
        "ouut format",
        "the bounding box is",
        "the reason is that the image is blurry",
    )
    if any(normalized.count(phrase) >= 3 for phrase in repeated_phrases):
        return True
    tokens = re.findall(r"[a-z0-9']+", normalized)
    if len(tokens) >= 40:
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        if unique_ratio < 0.4:
            return True
    return False


def score_teacher_output(text, fields):
    score = teacher_parsed_field_count(fields) * 100
    normalized = normalize_teacher_output(text)
    if normalized.startswith("GTMASK:"):
        score += 10
    if not is_degenerate_teacher_output(normalized):
        score += 5
    score -= min(len(normalized) // 400, 5)
    return score


def build_teacher_diagnosis_prompt_variants(sample, reconstruction, iou, gt_mask, ref_mask):
    student_question = sample.get("student_question", "")
    clean_question = student_question.replace("<image>", "").strip()
    caption = sample.get("caption", "") or ""
    verifier_caption = sample.get("verifier_caption", "") or caption
    reconstruct_question = reconstruction.question or ""
    description_status = sample.get("description_status", "ok")
    reconstruct_status = reconstruction.status
    gt_summary = sample["model"]._mask_summary(gt_mask) if "model" in sample else ""
    ref_summary = sample["model"]._mask_summary(ref_mask) if "model" in sample else ""
    seg_correct = "true" if iou >= 0.5 else "false"
    base_context = (
        "You are optimizing the following task: given a gtmask, generate a caption that describes it. "
        "You are now given the original input, the student question, and privileged verification information. "
        "Use these privileged signals to improve the caption generation.\n"
        f"Student prompt: {clean_question}\n"
        f"Student caption: {caption}\n"
        f"Verifier caption: {verifier_caption}\n"
        f"Reconstruction question: {reconstruct_question}\n"
        f"Description status: {description_status}\n"
        f"Reconstruction status: {reconstruct_status}\n"
        f"caption_to_mask_seg_correct: {seg_correct}\n"
        f"IoU: {iou:.4f}\n"
        f"gtmask stats: {gt_summary}\n"
        f"refmask stats: {ref_summary}\n"
    )
    return [
        f"""<image>
You are optimizing the following task: given a gtmask, generate a caption that describes it. You are now given the original input, the student question, and privileged verification information. Use these privileged signals to improve the caption generation.

Compare the TWO marked regions in the image:
- region1 = gtmask = the true target
- region2 = refmask = the mask reconstructed from the student's caption

Use the visual difference between region1 and region2. Do not judge from text alone.
If region2 is empty, nearly empty, wrong object, or wrong extent, say so directly.
Do not add any intro, apology, bullets, or extra explanation before the labels.
Judge what problem the current IoU indicates from the facts above. Do not rely on any pre-labeled failure category.

{base_context}
Output exactly 5 lines and nothing else:
GTMASK: one short sentence describing region1.
REFMASK: one short sentence describing region2 or saying it is empty/wrong.
CAPTION_PROBLEM: the specific caption error.
CORRECTION_DIRECTION: how the caption should change.
REASON: why that change should move region2 toward region1.
""",
        f"""<image>
You are optimizing the following task: given a gtmask, generate a caption that describes it. You are now given the original input, the student question, and privileged verification information. Use these privileged signals to improve the caption generation.
Compare region1(gtmask) and region2(refmask) visually in the same image.
Judge what problem the current IoU indicates from the facts above. Do not rely on any pre-labeled failure category.
{base_context}
Return only these five lines, one line each, short and concrete:
GTMASK:
REFMASK:
CAPTION_PROBLEM:
CORRECTION_DIRECTION:
REASON:
""",
    ]


def build_teacher_diagnosis_fallback(sample, reconstruction, iou, gt_mask, ref_mask):
    model = sample["model"]
    caption = (sample.get("caption") or "").strip()
    token_count = model._caption_token_count(caption)
    generic_caption = model._is_overly_generic_caption(caption)
    if sample.get("description_status") != "ok":
        problem = "The first-stage caption was empty or malformed, so reconstruction had no stable target."
        correction = "Produce a valid noun phrase that directly names the masked target."
    elif reconstruction.status != "ok" or int(np.asarray(ref_mask).sum()) == 0:
        problem = "The caption did not reconstruct a usable target mask."
        correction = "Add the target category plus one local appearance or spatial cue."
    elif iou < 0.2:
        problem = "The caption points to the wrong object or wrong image extent."
        correction = "Name the masked target itself and add a stronger local disambiguation cue."
    elif iou < 0.5:
        problem = "The caption is partially correct but still misses key disambiguation for the target extent."
        correction = "Keep the target noun and tighten the local attribute or relation."
    else:
        problem = "The caption is close, but it still underspecifies some local detail of the target extent."
        correction = "Keep the current target noun and refine it with one precise local detail."

    if generic_caption or token_count <= 2:
        correction += " Avoid single-word or overly generic captions."

    return {
        "gtmask": f"Target region summary: {model._mask_summary(gt_mask)}.",
        "refmask": (
            f"Reconstructed region summary: {model._mask_summary(ref_mask)}."
            if int(np.asarray(ref_mask).sum()) > 0
            else "Reconstructed region is empty or nearly empty."
        ),
        "caption_problem": problem,
        "correction_direction": correction,
        "reason": (
            "These changes add target-specific evidence so reconstruction is less likely to drift away from region1."
        ),
    }


def format_teacher_structured_output(fields):
    return "\n".join([
        f"GTMASK: {fields['gtmask']}",
        f"REFMASK: {fields['refmask']}",
        f"CAPTION_PROBLEM: {fields['caption_problem']}",
        f"CORRECTION_DIRECTION: {fields['correction_direction']}",
        f"REASON: {fields['reason']}",
    ])


def build_teacher_diagnosis_prompt(sample, reconstruction, iou, gt_mask, ref_mask):
    return build_teacher_diagnosis_prompt_variants(sample, reconstruction, iou, gt_mask, ref_mask)[0]


def build_teacher_diagnosis_fields(*, model, sample, reconstruction, gt_mask, iou):
    return build_teacher_diagnosis_fields_staged(
        model=model,
        sample=sample,
        reconstruction=reconstruction,
        gt_mask=gt_mask,
        iou=iou,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    image_root = resolve_image_root(cfg, args.image_root) if args.annotation_file is None else None
    device = args.device or cfg.get("device", "cuda:0")
    limit = args.limit or cfg.get("limit", 50)
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Sa2VAOPSDModelV3(
        model_path=cfg["model_path"],
        teacher_model_path=cfg.get("teacher_model_path"),
        enable_teacher=cfg.get("enable_teacher", True),
        grpo_group_size=0,
        tokenizer_path=cfg.get("tokenizer_path", cfg["model_path"]),
        device=device,
        torch_dtype="auto",
        use_flash_attn=True,
        min_caption_tokens=4,
        teacher_summary_template=cfg.get("teacher_summary_template"),
    )
    model.eval()

    if args.annotation_file is None:
        ref_samples, resolved_image_root = build_refcoco_opsd_records(
            data_root=cfg["data_root"],
            dataset_name=cfg.get("dataset_name", "refcoco"),
            split=cfg.get("split", "val"),
            image_root=image_root,
            image_root_candidates=cfg.get("image_root_candidates", []),
        )

        annotations = []
        attempted = 0
        exported = 0
        valid_caption_count = 0
        with torch.no_grad():
            for sample in ref_samples:
                if exported >= limit:
                    break
                image_path = sample["image_path"]
                if not os.path.exists(image_path):
                    continue
                attempted += 1
                image = Image.open(image_path).convert("RGB")
                gt_mask = model._to_numpy_mask(sample["gt_mask"])
                description = model.generate_description(
                    image=image,
                    mask_prompts=np.expand_dims(gt_mask.astype(np.float32), axis=0),
                    student_question=cfg["student_question"],
                )
                caption = description.clean_caption.strip()
                if description.status == "ok" and caption:
                    valid_caption_count += 1
                annotations.append(
                    {
                        "sample_id": exported,
                        "source_index": attempted - 1,
                        "npz_path": None,
                        "image": image_path,
                        "image_info": {
                            "id": exported,
                            "height": int(image.size[1]),
                            "width": int(image.size[0]),
                            "file_name": image_path,
                        },
                        "selected_labels": [caption] if caption else [],
                        "gt_masks": [mask_to_rle(gt_mask)],
                        "description_status": description.status,
                        "raw_prediction": description.raw_prediction,
                        "caption": caption,
                        "caption_storage_format": "clean_caption_strip_preserve_case",
                        "caption_token_count": model._caption_token_count(caption),
                        **sample["meta"],
                    }
                )
                exported += 1
                print(f"[{exported:03d}] status={description.status:<18} caption={caption!r}")

        annotations_payload = {
            "meta": {
                "dataset_name": cfg.get("dataset_name", "refcoco"),
                "split": cfg.get("split", "val"),
                "image_root": resolved_image_root,
                "model_path": cfg["model_path"],
                "student_question": cfg["student_question"],
                "attempted": attempted,
                "exported": exported,
                "valid_caption_count": valid_caption_count,
                "valid_caption_rate": valid_caption_count / max(exported, 1),
            },
            "items": annotations,
        }
        annotations_path = output_dir / "annotations.json"
        annotations_path.write_text(json.dumps(annotations_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        annotations_path = Path(args.annotation_file)
        annotations_payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        annotations = annotations_payload["items"]

    eval_samples = []
    student_question = annotations_payload["meta"].get("student_question", cfg["student_question"])
    for item in annotations:
        encoded_mask = item["gt_masks"][0]
        decoded = mask_utils.decode(
            {"size": encoded_mask["size"], "counts": encoded_mask["counts"].encode()}
        )
        if decoded.ndim == 3:
            decoded = np.sum(decoded, axis=2)
        eval_samples.append(
            {
                "image": Image.open(item["image"]).convert("RGB"),
                "gt_mask": (decoded > 0).astype(np.uint8),
                "caption": item["caption"],
                "verifier_caption": item["caption"],
                "description_status": item["description_status"],
                "student_question": student_question,
                "meta": {
                    "ref_id": item["ref_id"],
                    "ann_id": item["ann_id"],
                    "image_id": item["image_id"],
                    "image_path": item["image_path"],
                    "ref_sentences": item["ref_sentences"],
                },
            }
        )

    raw_eval_path = output_dir / "caption_to_mask_eval_raw_caption.json"
    run_caption_to_mask_eval(
        model=model,
        samples=eval_samples,
        limit=len(eval_samples),
        output=str(raw_eval_path),
        summary_prefix={"annotation_file": str(annotations_path)},
        sample_extra_builder=build_teacher_diagnosis_fields,
    )


if __name__ == "__main__":
    main()
