import argparse
import json
import os
import runpy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.sa2va.datasets.refcoco_opsd import build_refcoco_opsd_records
from projects.sa2va.evaluation.teacher_diagnosis_common import (
    GRPO_POSITIVE_ROUTE,
    ON_POLICY_DISTILL_ROUTE,
    TEACHER_REGENERATE_ROUTE,
    build_teacher_context_validation_fields,
    caption_to_mask_seg_correct,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run teacher privileged diagnosis on RefCOCO samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(path):
    return runpy.run_path(path)


def init_distributed_if_needed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return distributed, rank, local_rank, world_size


def finalize_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def resolve_runtime_device(args_device, cfg_device, *, distributed, local_rank):
    if distributed:
        if args_device not in {None, "auto"}:
            raise ValueError(
                "Distributed mode infers device from LOCAL_RANK. "
                "Do not pass --device explicitly."
            )
        return f"cuda:{local_rank}"
    return args_device or cfg_device or "cuda:0"


def build_rank_output_path(output_path, rank):
    suffix = output_path.suffix or ".json"
    stem = output_path.name[:-len(suffix)] if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}.rank{rank}{suffix}")


def summarize_results(cfg, resolved_image_root, results):
    ordered_results = sorted(results, key=lambda item: int(item.get("index", -1)))
    caption_to_mask_seg_correct_count = sum(
        bool(item.get("caption_to_mask_seg_correct")) for item in ordered_results
    )
    route_counts = {
        TEACHER_REGENERATE_ROUTE: 0,
        ON_POLICY_DISTILL_ROUTE: 0,
        GRPO_POSITIVE_ROUTE: 0,
    }
    for item in ordered_results:
        route_counts[item.get("teacher_route", "")] = route_counts.get(item.get("teacher_route", ""), 0) + 1

    onpolicy_results = [
        item for item in ordered_results
        if item.get("teacher_route") == ON_POLICY_DISTILL_ROUTE
    ]
    regenerate_results = [
        item for item in ordered_results
        if item.get("teacher_route") == TEACHER_REGENERATE_ROUTE
    ]

    teacher_predict_available_count = sum(
        bool((item.get("teacher_predict_caption") or "").strip()) for item in onpolicy_results
    )
    teacher_predict_reconstruct_ok_count = sum(
        item.get("teacher_predict_reconstruct_status") == "ok"
        for item in onpolicy_results
        if (item.get("teacher_predict_caption") or "").strip()
    )
    teacher_predict_success_count = sum(
        bool(item.get("teacher_predict_success_iou_gt_0_5"))
        for item in onpolicy_results
        if (item.get("teacher_predict_caption") or "").strip()
    )
    teacher_predict_iou_sum = sum(
        float(item.get("teacher_predict_reconstruct_iou", 0.0))
        for item in onpolicy_results
        if (item.get("teacher_predict_caption") or "").strip()
    )

    teacher_regenerate_available_count = sum(
        bool((item.get("teacher_regenerate_caption") or "").strip()) for item in regenerate_results
    )
    teacher_regenerate_reconstruct_ok_count = sum(
        item.get("teacher_regenerate_reconstruct_status") == "ok"
        for item in regenerate_results
        if (item.get("teacher_regenerate_caption") or "").strip()
    )
    teacher_regenerate_success_count = sum(
        bool(item.get("teacher_regenerate_success_iou_gt_0_5"))
        for item in regenerate_results
        if (item.get("teacher_regenerate_caption") or "").strip()
    )
    teacher_regenerate_iou_sum = sum(
        float(item.get("teacher_regenerate_reconstruct_iou", 0.0))
        for item in regenerate_results
        if (item.get("teacher_regenerate_caption") or "").strip()
    )
    return {
        "dataset_name": cfg.get("dataset_name", "refcoco"),
        "split": cfg.get("split", "val"),
        "image_root": resolved_image_root,
        "count": len(ordered_results),
        "teacher_route_counts": route_counts,
        "caption_to_mask_seg_correct_count": caption_to_mask_seg_correct_count,
        "caption_to_mask_seg_correct_rate": caption_to_mask_seg_correct_count / max(len(ordered_results), 1),
        "caption_to_mask_seg_error_count": len(ordered_results) - caption_to_mask_seg_correct_count,
        "teacher_predict_route_count": len(onpolicy_results),
        "teacher_predict_caption_available_count": teacher_predict_available_count,
        "teacher_predict_caption_reconstruct_ok_count": teacher_predict_reconstruct_ok_count,
        "teacher_predict_caption_reconstruct_ok_rate": teacher_predict_reconstruct_ok_count / max(teacher_predict_available_count, 1),
        "teacher_predict_caption_seg_success_count_iou_gt_0_5": teacher_predict_success_count,
        "teacher_predict_caption_seg_success_rate_iou_gt_0_5": teacher_predict_success_count / max(teacher_predict_available_count, 1),
        "teacher_predict_caption_avg_reconstruct_iou": teacher_predict_iou_sum / max(teacher_predict_available_count, 1),
        "teacher_regenerate_route_count": len(regenerate_results),
        "teacher_regenerate_caption_available_count": teacher_regenerate_available_count,
        "teacher_regenerate_caption_reconstruct_ok_count": teacher_regenerate_reconstruct_ok_count,
        "teacher_regenerate_caption_reconstruct_ok_rate": teacher_regenerate_reconstruct_ok_count / max(teacher_regenerate_available_count, 1),
        "teacher_regenerate_caption_seg_success_count_iou_gt_0_5": teacher_regenerate_success_count,
        "teacher_regenerate_caption_seg_success_rate_iou_gt_0_5": teacher_regenerate_success_count / max(teacher_regenerate_available_count, 1),
        "teacher_regenerate_caption_avg_reconstruct_iou": teacher_regenerate_iou_sum / max(teacher_regenerate_available_count, 1),
        "teacher_regenerate_caption_fix_success_count_on_low_iou": teacher_regenerate_success_count,
        "teacher_regenerate_caption_fix_success_rate_on_low_iou": teacher_regenerate_success_count / max(len(regenerate_results), 1),
        "results": ordered_results,
    }


def mask_stats(mask):
    mask = np.asarray(mask).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    area = int(mask.sum())
    area_ratio = float(area) / float(max(h * w, 1))
    if len(xs) == 0 or len(ys) == 0:
        return {
            "area": area,
            "area_ratio": area_ratio,
            "bbox": None,
            "center": None,
        }
    return {
        "area": area,
        "area_ratio": area_ratio,
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "center": [round(float(xs.mean()), 2), round(float(ys.mean()), 2)],
    }


def build_teacher_diagnosis_prompt(sample, description, reconstruction, iou, gt_mask, ref_mask):
    gt_stats = mask_stats(gt_mask)
    ref_stats = mask_stats(ref_mask)
    gt_only_stats = mask_stats(np.logical_and(np.asarray(gt_mask) > 0, np.asarray(ref_mask) == 0).astype(np.uint8))
    ref_only_stats = mask_stats(np.logical_and(np.asarray(ref_mask) > 0, np.asarray(gt_mask) == 0).astype(np.uint8))
    clean_question = sample["student_question"].replace("<image>", "").strip()
    seg_correct = caption_to_mask_seg_correct(iou=iou)
    return f"""<image>
You are optimizing the following task: given a gtmask, generate a caption that describes it. You are now given the original input, the student question, and privileged verification information. Use these privileged signals to improve the caption generation.

The image shows two marked regions and you must use BOTH of them:
- region1 = gtmask = the true target mask
- region2 = refmask = the mask reconstructed from the student's caption

Do not judge the caption from text alone.
You must compare region1 and region2 visually, then explain why the caption led to region2 instead of region1.
If region2 is empty or nearly empty, say that clearly.
Do not praise the caption. Do not say it is clear. Focus on the error.

Original student task:
{clean_question}

Student caption:
{description.clean_caption}

Verifier caption used for reconstruction:
{description.clean_caption}

Reconstruction question:
{reconstruction.question}

Description status: {description.status}
Reconstruction status: {reconstruction.status}
caption_to_mask_seg_correct: {"true" if seg_correct else "false"}
IoU is the intersection-over-union between gtmask and refmask: intersection / union.
If IoU is close to 0, gtmask and refmask are weakly related or completely different.
If IoU is close to 1, gtmask and refmask are very similar.
Current IoU between gtmask and refmask: {iou:.4f}

Auxiliary mask statistics:
- gtmask area ratio: {gt_stats["area_ratio"]:.4f}, bbox: {gt_stats["bbox"]}, center: {gt_stats["center"]}
- refmask area ratio: {ref_stats["area_ratio"]:.4f}, bbox: {ref_stats["bbox"]}, center: {ref_stats["center"]}
- unique non-overlap area in gtmask: area_ratio={gt_only_stats["area_ratio"]:.4f}, bbox: {gt_only_stats["bbox"]}, center: {gt_only_stats["center"]}
- unique non-overlap area in refmask: area_ratio={ref_only_stats["area_ratio"]:.4f}, bbox: {ref_only_stats["bbox"]}, center: {ref_only_stats["center"]}

Judge what problem the current IoU indicates from the facts above. Do not rely on any pre-labeled failure category.

Required output format:
GTMASK: describe what region1 refers to in the image.
REFMASK: describe what region2 refers to, or say it is empty/wrong extent/wrong object.
CAPTION_PROBLEM: identify exactly which part of the caption is too vague, misleading, missing, or incorrect.
CORRECTION_DIRECTION: say how the caption should be changed so the mask moves from region2 toward region1.
REASON: explain why that change should help segmentation.

Your answer must contain all five labels exactly:
GTMASK:
REFMASK:
CAPTION_PROBLEM:
CORRECTION_DIRECTION:
REASON:
"""


def main():
    args = parse_args()
    from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3

    distributed, rank, local_rank, world_size = init_distributed_if_needed()
    cfg = load_config(args.config)
    image_root = args.image_root or cfg.get("image_root")
    limit = args.limit or cfg.get("limit", 100)
    device = resolve_runtime_device(
        args.device,
        cfg.get("device", "cuda:0"),
        distributed=distributed,
        local_rank=local_rank,
    )
    output = args.output
    if output is None:
        output_dir = Path(cfg.get("output_dir", Path(args.config).resolve().parent))
        output_dir.mkdir(parents=True, exist_ok=True)
        output = str(output_dir / "teacher_privileged_diagnosis_refcoco.json")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_output_path = build_rank_output_path(output_path, rank)

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
        iou_low_threshold=cfg.get("iou_low_threshold", 0.3),
        iou_high_threshold=cfg.get("iou_high_threshold", 0.8),
        low_iou_regen_max_new_tokens=cfg.get("low_iou_regen_max_new_tokens", 48),
    )
    model.eval()

    records, resolved_image_root = build_refcoco_opsd_records(
        data_root=cfg["data_root"],
        dataset_name=cfg.get("dataset_name", "refcoco"),
        split=cfg.get("split", "val"),
        image_root=image_root,
        image_root_candidates=cfg.get("image_root_candidates", []),
    )
    limited_records = records[: min(limit, len(records))]
    sharded_records = list(enumerate(limited_records))
    if distributed:
        sharded_records = sharded_records[rank::world_size]

    results = []

    with torch.no_grad():
        for idx, record in sharded_records:
            image = Image.open(record["image_path"]).convert("RGB")
            gt_mask = model._to_numpy_mask(record["gt_mask"])
            description = model.generate_description(
                image=image,
                mask_prompts=np.expand_dims(gt_mask.astype(np.float32), axis=0),
                student_question=cfg["student_question"],
            )
            reconstruction = model.reconstruct_mask(
                image=image,
                caption=description.clean_caption,
                description_status=description.status,
                spatial_hint=model._coarse_spatial_hint(gt_mask),
                gt_mask=gt_mask,
            )
            ref_mask = gt_mask * 0 if reconstruction.pred_mask is None else model._to_numpy_mask(reconstruction.pred_mask)
            iou = model._compute_iou(gt_mask, ref_mask)
            teacher_prompt = build_teacher_diagnosis_prompt(
                sample={"student_question": cfg["student_question"]},
                description=description,
                reconstruction=reconstruction,
                iou=iou,
                gt_mask=gt_mask,
                ref_mask=ref_mask,
            )
            staged_fields = build_teacher_context_validation_fields(
                model=model,
                sample={
                    "image": image,
                    "student_question": cfg["student_question"],
                    "caption": description.clean_caption,
                    "description_status": description.status,
                },
                reconstruction=reconstruction,
                gt_mask=gt_mask,
                iou=iou,
                low_threshold=model.iou_low_threshold,
                high_threshold=model.iou_high_threshold,
            )
            results.append(
                {
                    "index": idx,
                    "sample_key": record["sample_key"],
                    "student_question": cfg["student_question"],
                    "description_status": description.status,
                    "caption": description.clean_caption,
                    "reconstruct_status": reconstruction.status,
                    "reconstruct_question": reconstruction.question,
                    "reconstruct_raw_prediction": reconstruction.raw_prediction,
                    "iou": float(iou),
                    "gt_mask_stats": mask_stats(gt_mask),
                    "ref_mask_stats": mask_stats(ref_mask),
                    "teacher_legacy_prompt": teacher_prompt,
                    **record["meta"],
                    **staged_fields,
                }
            )
            print(
                f"[rank {rank}][{idx:03d}] iou={iou:.4f} desc={description.status:<18} "
                f"recon={reconstruction.status:<24} caption={description.clean_caption!r}"
            )
            print(staged_fields["teacher_output"])
            print("-" * 80)

    shard_summary = summarize_results(cfg, resolved_image_root, results)
    shard_payload = {
        **shard_summary,
        "rank": rank,
        "world_size": world_size,
    }
    shard_output_path.write_text(json.dumps(shard_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if distributed:
        dist.barrier()
        if rank == 0:
            merged_results = []
            for shard_rank in range(world_size):
                current_shard_path = build_rank_output_path(output_path, shard_rank)
                shard_data = json.loads(current_shard_path.read_text(encoding="utf-8"))
                merged_results.extend(shard_data.get("results", []))
            summary = summarize_results(cfg, resolved_image_root, merged_results)
            output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
        dist.barrier()
    else:
        output_path.write_text(json.dumps(shard_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in shard_summary.items() if k != "results"}, ensure_ascii=False, indent=2))

    finalize_distributed(distributed)


if __name__ == "__main__":
    main()
