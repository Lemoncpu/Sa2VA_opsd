import argparse
import gzip
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.sa2va.datasets.refcoco_opsd import build_refcoco_opsd_records
from third_parts.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from third_parts.sam2.build_sam import build_sam2


LOGGER = logging.getLogger("refcoco_sam_confuser_pool")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a per-image SAM confuser pool for RefCOCO style datasets."
    )
    parser.add_argument("--data-root", required=True, help="RefCOCO root or its parent directory.")
    parser.add_argument("--image-root", required=True, help="Directory containing train2014 images.")
    parser.add_argument(
        "--dataset",
        default="refcoco",
        choices=["refcoco", "refcoco_plus", "refcoco+", "refcocog"],
        help="RefCOCO family dataset name.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to export.")
    parser.add_argument("--sam2-config", default="configs/sam2/sam2_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on unique images.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.92)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--min-mask-region-area", type=int, default=300)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.90)
    parser.add_argument("--min-box-size", type=float, default=8.0)
    parser.add_argument("--sam-duplicate-iou-thresh", type=float, default=0.90)
    parser.add_argument("--gt-duplicate-iou-thresh", type=float, default=0.95)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def normalize_refcoco_data_root(data_root: str) -> Tuple[str, str]:
    root = Path(data_root).expanduser().resolve()
    if root.name == "refcoco":
        return str(root.parent), str(root)
    return str(root), str(root / "refcoco")


def normalize_sam2_config_name(config_name: str) -> str:
    config_name = str(config_name).strip()
    if not config_name:
        raise ValueError("sam2 config name cannot be empty.")
    if config_name.endswith(".yaml"):
        return Path(config_name).name
    return config_name


def decode_coco_rle(rle: Dict) -> np.ndarray:
    encoded = dict(rle)
    counts = encoded.get("counts")
    if isinstance(counts, str):
        encoded["counts"] = counts.encode()
    mask = mask_utils.decode(encoded)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return (mask > 0).astype(np.uint8)


def sanitize_coco_rle(rle: Dict) -> Dict:
    encoded = dict(rle)
    counts = encoded.get("counts")
    if isinstance(counts, bytes):
        encoded["counts"] = counts.decode()
    return encoded


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a) > 0
    mask_b = np.asarray(mask_b) > 0
    intersection = np.logical_and(mask_a, mask_b).sum(dtype=np.int64)
    union = np.logical_or(mask_a, mask_b).sum(dtype=np.int64)
    if union == 0:
        return 0.0
    return float(intersection / union)


def group_records_by_image(records: Iterable[Dict]) -> List[Dict]:
    grouped: "OrderedDict[int, Dict]" = OrderedDict()
    for record in records:
        image_id = int(record["meta"]["image_id"])
        if image_id not in grouped:
            grouped[image_id] = {
                "image_id": image_id,
                "image_path": record["image_path"],
                "ann_ids": [],
                "ref_ids": [],
                "gt_masks": [],
            }
        grouped[image_id]["ann_ids"].append(int(record["meta"]["ann_id"]))
        grouped[image_id]["ref_ids"].append(int(record["meta"]["ref_id"]))
        grouped[image_id]["gt_masks"].append(np.asarray(record["gt_mask"]).astype(np.uint8))
    return list(grouped.values())


def build_generator(args, device: torch.device) -> SAM2AutomaticMaskGenerator:
    sam2_model = build_sam2(
        normalize_sam2_config_name(args.sam2_config),
        ckpt_path=args.sam2_checkpoint,
        device=str(device),
        mode="eval",
    )
    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="coco_rle",
    )


def prepare_output_path(out_dir: Path, dataset: str, split: str, image_id: int) -> Path:
    return out_dir / dataset / split / f"{image_id}.json.gz"


def atomic_write_gzip_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        return np.asarray(rgb_image), width, height


def filter_mask_records(
    *,
    raw_masks: List[Dict],
    gt_masks: List[np.ndarray],
    image_width: int,
    image_height: int,
    min_area_ratio: float,
    max_area_ratio: float,
    min_box_size: float,
    sam_duplicate_iou_thresh: float,
    gt_duplicate_iou_thresh: float,
) -> Tuple[List[Dict], Dict[str, int]]:
    total_pixels = max(int(image_width * image_height), 1)
    stats = {
        "num_empty_filtered": 0,
        "num_area_filtered": 0,
        "num_box_filtered": 0,
        "num_duplicate_gt_filtered": 0,
        "num_duplicate_sam_filtered": 0,
    }
    filtered_records: List[Dict] = []
    kept_masks: List[np.ndarray] = []

    sorted_raw_masks = sorted(
        raw_masks,
        key=lambda item: (
            -float(item.get("predicted_iou", 0.0)),
            -float(item.get("stability_score", 0.0)),
            -float(item.get("area", 0.0)),
        ),
    )

    for item in sorted_raw_masks:
        mask = decode_coco_rle(item["segmentation"])
        area = int(mask.sum())
        if area <= 0:
            stats["num_empty_filtered"] += 1
            continue

        area_ratio = float(area / total_pixels)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            stats["num_area_filtered"] += 1
            continue

        bbox = item.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if len(bbox) != 4 or float(bbox[2]) < min_box_size or float(bbox[3]) < min_box_size:
            stats["num_box_filtered"] += 1
            continue

        if any(compute_iou(mask, gt_mask) >= gt_duplicate_iou_thresh for gt_mask in gt_masks):
            stats["num_duplicate_gt_filtered"] += 1
            continue

        if any(compute_iou(mask, selected_mask) >= sam_duplicate_iou_thresh for selected_mask in kept_masks):
            stats["num_duplicate_sam_filtered"] += 1
            continue

        kept_masks.append(mask)
        filtered_records.append(
            {
                "segmentation": sanitize_coco_rle(item["segmentation"]),
                "bbox": [float(value) for value in bbox],
                "area": area,
                "area_ratio": area_ratio,
                "predicted_iou": float(item.get("predicted_iou", 0.0)),
                "stability_score": float(item.get("stability_score", 0.0)),
                "crop_box": [float(value) for value in item.get("crop_box", [0.0, 0.0, 0.0, 0.0])],
            }
        )

    return filtered_records, stats


def process_image_record(
    *,
    image_record: Dict,
    generator: SAM2AutomaticMaskGenerator,
    args,
    output_path: Path,
) -> Dict:
    image_array, width, height = load_image(image_record["image_path"])
    raw_masks = generator.generate(image_array)
    filtered_masks, filter_stats = filter_mask_records(
        raw_masks=raw_masks,
        gt_masks=image_record["gt_masks"],
        image_width=width,
        image_height=height,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_box_size=args.min_box_size,
        sam_duplicate_iou_thresh=args.sam_duplicate_iou_thresh,
        gt_duplicate_iou_thresh=args.gt_duplicate_iou_thresh,
    )

    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "image_id": int(image_record["image_id"]),
        "image_path": image_record["image_path"],
        "refcoco_ann_ids": sorted(set(int(ann_id) for ann_id in image_record["ann_ids"])),
        "refcoco_ref_ids": sorted(set(int(ref_id) for ref_id in image_record["ref_ids"])),
        "generator": {
            "model": "sam2",
            "config": normalize_sam2_config_name(args.sam2_config),
            "checkpoint": os.path.basename(args.sam2_checkpoint),
            "points_per_side": int(args.points_per_side),
            "points_per_batch": int(args.points_per_batch),
            "pred_iou_thresh": float(args.pred_iou_thresh),
            "stability_score_thresh": float(args.stability_score_thresh),
            "box_nms_thresh": float(args.box_nms_thresh),
            "crop_n_layers": int(args.crop_n_layers),
            "crop_nms_thresh": float(args.crop_nms_thresh),
            "min_mask_region_area": int(args.min_mask_region_area),
            "min_area_ratio": float(args.min_area_ratio),
            "max_area_ratio": float(args.max_area_ratio),
            "min_box_size": float(args.min_box_size),
            "sam_duplicate_iou_thresh": float(args.sam_duplicate_iou_thresh),
            "gt_duplicate_iou_thresh": float(args.gt_duplicate_iou_thresh),
        },
        "masks": filtered_masks,
        "stats": {
            "image_width": int(width),
            "image_height": int(height),
            "num_raw_masks": int(len(raw_masks)),
            "num_saved_masks": int(len(filtered_masks)),
            **filter_stats,
        },
    }
    atomic_write_gzip_json(output_path, payload)
    return payload["stats"]


def validate_args(args) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"--limit must be positive, got {args.limit}.")
    if args.num_shards <= 0:
        raise ValueError(f"--num-shards must be positive, got {args.num_shards}.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}."
        )
    if args.log_every <= 0:
        raise ValueError(f"--log-every must be positive, got {args.log_every}.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    validate_args(args)

    data_root, refcoco_root = normalize_refcoco_data_root(args.data_root)
    if not os.path.isdir(refcoco_root):
        raise FileNotFoundError(f"Expected RefCOCO annotations under: {refcoco_root}")
    if not os.path.isdir(args.image_root):
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")
    if not os.path.exists(args.sam2_checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {args.sam2_checkpoint}")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is False.")
        torch.cuda.set_device(device)

    LOGGER.info(
        "Collecting RefCOCO records: data_root=%s dataset=%s split=%s image_root=%s",
        data_root,
        args.dataset,
        args.split,
        args.image_root,
    )
    records, resolved_image_root = build_refcoco_opsd_records(
        data_root=data_root,
        dataset_name=args.dataset,
        split=args.split,
        image_root=args.image_root,
        skip_empty_masks=True,
        skip_missing_images=True,
    )
    image_records = group_records_by_image(records)
    if args.limit is not None:
        image_records = image_records[: args.limit]
    sharded_image_records = [
        record
        for index, record in enumerate(image_records)
        if index % args.num_shards == args.shard_index
    ]

    LOGGER.info(
        "Resolved image root=%s unique_images=%d shard=%d/%d shard_images=%d",
        resolved_image_root,
        len(image_records),
        args.shard_index,
        args.num_shards,
        len(sharded_image_records),
    )

    output_root = Path(args.out_dir).expanduser().resolve()
    generator = build_generator(args, device)

    success_count = 0
    skipped_existing_count = 0
    failed_count = 0
    total_saved_masks = 0
    total_raw_masks = 0

    with torch.inference_mode():
        for index, image_record in enumerate(sharded_image_records, 1):
            output_path = prepare_output_path(
                output_root, args.dataset, args.split, int(image_record["image_id"])
            )
            if output_path.exists() and not args.overwrite:
                skipped_existing_count += 1
                if index % args.log_every == 0 or index == len(sharded_image_records):
                    LOGGER.info(
                        "[%d/%d] skipped existing image_id=%s",
                        index,
                        len(sharded_image_records),
                        image_record["image_id"],
                    )
                continue

            try:
                stats = process_image_record(
                    image_record=image_record,
                    generator=generator,
                    args=args,
                    output_path=output_path,
                )
            except Exception as exc:
                failed_count += 1
                LOGGER.warning(
                    "Failed to export image_id=%s path=%s reason=%s",
                    image_record["image_id"],
                    image_record["image_path"],
                    exc,
                )
                continue

            success_count += 1
            total_saved_masks += int(stats["num_saved_masks"])
            total_raw_masks += int(stats["num_raw_masks"])
            if index % args.log_every == 0 or index == len(sharded_image_records):
                LOGGER.info(
                    "[%d/%d] exported image_id=%s raw_masks=%d saved_masks=%d out=%s",
                    index,
                    len(sharded_image_records),
                    image_record["image_id"],
                    stats["num_raw_masks"],
                    stats["num_saved_masks"],
                    output_path,
                )

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "resolved_data_root": data_root,
        "resolved_image_root": resolved_image_root,
        "output_root": str(output_root),
        "device": str(device),
        "num_unique_images": len(image_records),
        "num_shard_images": len(sharded_image_records),
        "success_count": success_count,
        "skipped_existing_count": skipped_existing_count,
        "failed_count": failed_count,
        "avg_raw_masks_per_success": (total_raw_masks / success_count) if success_count else 0.0,
        "avg_saved_masks_per_success": (total_saved_masks / success_count) if success_count else 0.0,
    }
    LOGGER.info("Export summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
