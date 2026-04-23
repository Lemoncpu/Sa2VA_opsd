import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def normalize_refcoco_caption(text):
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text


def run_caption_to_mask_eval(
    *,
    model,
    samples,
    limit,
    output,
    summary_prefix=None,
    sample_extra_builder=None,
):
    results = []
    iou_sum = 0.0
    success_count = 0
    reconstruct_ok_count = 0
    checked = 0

    with torch.no_grad():
        for sample in samples:
            if checked >= limit:
                break
            caption = normalize_refcoco_caption(sample.get("caption"))
            if not caption:
                continue
            description_status = sample.get("description_status", "ok")
            gt_mask = model._to_numpy_mask(sample["gt_mask"])
            reconstruction = model.reconstruct_mask(
                image=sample["image"],
                caption=caption,
                description_status=description_status,
                spatial_hint=model._coarse_spatial_hint(gt_mask),
                gt_mask=gt_mask,
            )
            iou = model._compute_iou(gt_mask, reconstruction.pred_mask)
            ok = iou >= 0.5
            checked += 1
            iou_sum += iou
            if ok:
                success_count += 1
            if reconstruction.status == "ok":
                reconstruct_ok_count += 1

            item = dict(sample.get("meta", {}))
            item.update(
                {
                    "index": checked - 1,
                    "caption": caption,
                    "description_status": description_status,
                    "reconstruct_status": reconstruction.status,
                    "reconstruct_question": reconstruction.question,
                    "reconstruct_raw_prediction": reconstruction.raw_prediction,
                    "prediction_masks_count": reconstruction.prediction_masks_count,
                    "gt_mask_sum": int(np.asarray(gt_mask).sum()),
                    "pred_mask_sum": None
                    if reconstruction.pred_mask is None
                    else int(np.asarray(model._to_numpy_mask(reconstruction.pred_mask)).sum()),
                    "iou": float(iou),
                    "success_iou_gt_0_5": bool(ok),
                }
            )
            if sample_extra_builder is not None:
                extra = sample_extra_builder(
                    model=model,
                    sample=sample,
                    reconstruction=reconstruction,
                    gt_mask=gt_mask,
                    iou=iou,
                )
                if extra:
                    item.update(extra)
            results.append(item)
            print(
                f"[{checked:03d}] iou={iou:.4f} success={int(ok)} "
                f"recon={reconstruction.status:<24} caption={caption!r}"
            )

    summary = {}
    if summary_prefix:
        summary.update(summary_prefix)
    summary.update(
        {
            "limit": checked,
            "avg_iou": iou_sum / max(checked, 1),
            "reconstruct_ok_rate": reconstruct_ok_count / max(checked, 1),
            "seg_success_count_iou_gt_0_5": success_count,
            "seg_success_rate_iou_gt_0_5": success_count / max(checked, 1),
            "results": results,
        }
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    return summary


def _load_eval_image(sample):
    image = sample.get("image")
    if image is not None:
        return image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)).convert("RGB")

    image_path = sample.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return None
    return Image.open(image_path).convert("RGB")


def run_mask_to_caption_to_mask_eval(
    *,
    model,
    samples,
    limit,
    output,
    student_question,
    summary_prefix=None,
):
    results = []
    iou_sum = 0.0
    success_count = 0
    reconstruct_ok_count = 0
    checked = 0
    processed = 0

    with torch.no_grad():
        for sample in samples:
            if processed >= limit:
                break

            image = _load_eval_image(sample)
            if image is None:
                continue
            processed += 1

            gt_mask = model._to_numpy_mask(sample["gt_mask"])
            description = model.generate_description(
                image=image,
                mask_prompts=np.expand_dims(gt_mask.astype(np.float32), axis=0),
                student_question=student_question,
            )
            caption = normalize_refcoco_caption(description.clean_caption)
            if not caption:
                print(
                    f"[skip-empty-caption] desc={description.status:<18} "
                    f"image={sample.get('image_path', sample.get('meta', {}).get('image_path'))!r}"
                )
                continue

            reconstruction = model.reconstruct_mask(
                image=image,
                caption=caption,
                description_status=description.status,
                spatial_hint=model._coarse_spatial_hint(gt_mask),
                gt_mask=gt_mask,
            )
            iou = model._compute_iou(gt_mask, reconstruction.pred_mask)
            ok = iou >= 0.5
            checked += 1
            iou_sum += iou
            if ok:
                success_count += 1
            if reconstruction.status == "ok":
                reconstruct_ok_count += 1

            item = dict(sample.get("meta", {}))
            item.update(
                {
                    "index": checked - 1,
                    "caption": caption,
                    "description_status": description.status,
                    "reconstruct_status": reconstruction.status,
                    "reconstruct_question": reconstruction.question,
                    "reconstruct_raw_prediction": reconstruction.raw_prediction,
                    "prediction_masks_count": reconstruction.prediction_masks_count,
                    "gt_mask_sum": int(np.asarray(gt_mask).sum()),
                    "pred_mask_sum": None
                    if reconstruction.pred_mask is None
                    else int(np.asarray(model._to_numpy_mask(reconstruction.pred_mask)).sum()),
                    "iou": float(iou),
                    "success_iou_gt_0_5": bool(ok),
                }
            )
            results.append(item)
            print(
                f"[{checked:03d}] iou={iou:.4f} success={int(ok)} "
                f"desc={description.status:<18} recon={reconstruction.status:<24} caption={caption!r}"
            )

    summary = {}
    if summary_prefix:
        summary.update(summary_prefix)
    summary.update(
        {
            "limit": checked,
            "avg_iou": iou_sum / max(checked, 1),
            "reconstruct_ok_rate": reconstruct_ok_count / max(checked, 1),
            "seg_success_count_iou_gt_0_5": success_count,
            "seg_success_rate_iou_gt_0_5": success_count / max(checked, 1),
            "results": results,
        }
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    return summary
