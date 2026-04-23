import argparse
import json
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer


def normalize_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0) if frame.max() <= 1.0 else np.clip(frame, 0.0, 255.0)
        if frame.max() <= 1.0:
            frame = frame * 255.0
    frame = frame.astype(np.uint8)
    return Image.fromarray(frame).convert("RGB")


def normalize_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def clean_caption_text(caption):
    caption = caption.replace("<|im_end|>", "")
    caption = caption.replace("<|endoftext|>", "")
    caption = re.sub(r"</?p>", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\[SEG\]\.?", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"<[^>]+>", " ", caption)
    caption = re.sub(r"(assistant|bot)\s*[:：]\s*", "", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\s+", " ", caption)
    caption = re.sub(r"\s+([,.;:!?])", r"\1", caption)
    caption = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", caption)
    return caption.strip(" .,")


def coarse_spatial_hint(mask):
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


def compute_iou(gt_mask, pred_mask):
    if pred_mask is None:
        return 0.0
    gt_mask = gt_mask.astype(np.uint8)
    pred_mask = pred_mask.astype(np.uint8)
    if gt_mask.shape != pred_mask.shape:
        pred_mask_t = torch.from_numpy(pred_mask[None, None].astype(np.float32))
        pred_mask_t = F.interpolate(pred_mask_t, size=gt_mask.shape, mode="nearest")[0, 0]
        pred_mask = (pred_mask_t.numpy() > 0).astype(np.uint8)
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return 0.0 if union == 0 else float(intersection / union)


def canonicalize_caption(caption):
    caption = re.sub(r"\s+", " ", caption.strip())
    caption = re.sub(r"^(a|an)\s+", "the ", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\bis\s+([a-z]+ing)\b", r"\1", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\bare\s+([a-z]+ing)\b", r"\1", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\bis on\b", " on", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\bis in\b", " in", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\bis at\b", " at", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\bis with\b", " with", caption, flags=re.IGNORECASE)
    return caption.strip(" .,")


def compress_caption(caption):
    caption = re.sub(r"\s+", " ", (caption or "").strip())
    if not caption:
        return caption
    caption = re.sub(r"^(a|an)\s+", "the ", caption, flags=re.IGNORECASE)
    match = re.match(r"^(the\s+.+?)\s+(is|are|was|were)\s+([a-z]+ing\b.*)$", caption, flags=re.IGNORECASE)
    if match:
        subject = match.group(1).strip(" ,.")
        predicate = match.group(3).strip(" ,.")
        predicate = re.sub(
            r"\b(on|in|at)\s+(the\s+)?(street|road|sidewalk|floor|ground|room|store|mall|aisle|kitchen|bed|table)\b.*$",
            "",
            predicate,
            flags=re.IGNORECASE,
        ).strip(" ,.")
        return f"{subject} {predicate}".strip(" ,.") if predicate else subject
    return canonicalize_caption(caption)


def first_mask(prediction_masks):
    if not prediction_masks:
        return None
    mask = prediction_masks[0]
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return (mask > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--caption", default=None)
    args = parser.parse_args()

    data = np.load(args.npz_path)
    image = normalize_frame(data["frame1"])
    gt_mask = normalize_mask(data["mask1"])
    mask_prompts = [np.expand_dims(gt_mask.astype(np.float32), axis=0)]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    caption_prompts = [
        "<image>Can you provide me with a detailed description of the region in the picture marked by region1.",
        "<image>Can you provide me with a detailed description of the region in the picture marked by region1? Please answer with plain natural language only, and do not output segmentation tokens such as [SEG].",
        "<image>Describe the region marked by region1 with a clear natural-language caption. Identify the target itself and mention its category, distinctive appearance, color, and only the minimum necessary spatial cue. Answer with one complete sentence only. Do not output [SEG], masks, tags, placeholder tokens, or any segmentation format.",
        "<image>Provide a single descriptive sentence for the object or region marked by region1. Focus on the target's visible attributes so that it can be localized again from the text. Do not output segmentation tokens such as [SEG].",
    ]

    caption_results = []
    with torch.no_grad():
        for prompt in caption_prompts:
            out = model.predict_forward(
                image=image,
                text=prompt,
                past_text="",
                mask_prompts=mask_prompts,
                tokenizer=tokenizer,
            )
            caption_results.append(
                {
                    "prompt": prompt,
                    "raw_prediction": out.get("prediction", ""),
                    "clean_caption": clean_caption_text(out.get("prediction", "")),
                    "compressed_caption": compress_caption(clean_caption_text(out.get("prediction", ""))),
                    "prediction_masks_count": len(out.get("prediction_masks") or []),
                }
            )

        base_captions = []
        if args.caption:
            base_captions.append(args.caption)
        base_captions.extend(
            item["compressed_caption"] for item in caption_results if item["compressed_caption"]
        )
        dedup_captions = []
        for caption in base_captions:
            if caption not in dedup_captions:
                dedup_captions.append(caption)

        recon_results = []
        spatial_hint = coarse_spatial_hint(gt_mask)
        for caption in dedup_captions:
            variants = [caption, canonicalize_caption(caption)]
            if spatial_hint:
                variants.extend([f"{variant.rstrip('.')} {spatial_hint}".strip() for variant in list(variants)])
            seen_variants = []
            for variant in variants:
                if variant and variant not in seen_variants:
                    seen_variants.append(variant)
            prompts = []
            for variant in seen_variants:
                prompts.extend(
                    [
                        f"<image>Please segment the region described as: {variant}",
                        f"<image>\nPlease segment the region that matches this description: {variant}. Respond with the segmentation mask.",
                        f"<image>\nWhere is the target described as '{variant}' in this image? Please respond with segmentation mask.",
                        f"<image>\nCan you segment the object or region described as: {variant}? Please output segmentation mask.",
                    ]
                )
            for prompt in prompts:
                out = model.predict_forward(
                    image=image,
                    text=prompt,
                    past_text="",
                    mask_prompts=None,
                    tokenizer=tokenizer,
                )
                pred_mask = first_mask(out.get("prediction_masks"))
                recon_results.append(
                    {
                        "caption": caption,
                        "prompt": prompt,
                        "raw_prediction": out.get("prediction", ""),
                        "prediction_masks_count": len(out.get("prediction_masks") or []),
                        "pred_mask_sum": None if pred_mask is None else int(pred_mask.sum()),
                        "iou": compute_iou(gt_mask, pred_mask),
                    }
                )

    print(
        json.dumps(
            {
                "npz_path": args.npz_path,
                "gt_mask_sum": int(gt_mask.sum()),
                "caption_results": caption_results,
                "top_recon_results": sorted(recon_results, key=lambda x: x["iou"], reverse=True)[:20],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
