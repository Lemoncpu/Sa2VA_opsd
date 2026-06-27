import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.sa2va.datasets.common import DEFAULT_MASK_TO_CAPTION_QUESTION
from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3


DEFAULT_DLC_BENCH_QUERY = DEFAULT_MASK_TO_CAPTION_QUESTION


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Sa2VA DLC-Bench predictions in the official NVlabs/describe-anything format."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--data-root", required=True, help="Path to the official DLC-bench directory.")
    parser.add_argument("--output", required=True, help="Path to write official pred.json.")
    parser.add_argument("--debug-output", default=None, help="Optional sidecar debug json.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--student-question", default=DEFAULT_DLC_BENCH_QUERY)
    return parser.parse_args()


def _decode_segmentation(segmentation, height, width):
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(rles)
    elif isinstance(segmentation, dict):
        encoded = dict(segmentation)
        counts = encoded.get("counts")
        if isinstance(counts, str):
            encoded["counts"] = counts.encode()
        decoded = mask_utils.decode(encoded)
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)!r}")
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return np.asarray(decoded > 0, dtype=np.uint8)


def _load_annotations(data_root: Path):
    annotations_path = data_root / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations.json: {annotations_path}")
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("annotations"), list):
            return payload["annotations"], payload
        if isinstance(payload.get("data"), list):
            return payload["data"], payload
    if isinstance(payload, list):
        return payload, None
    raise ValueError("Unsupported DLC-Bench annotations.json format.")


def _build_image_lookup(annotation_payload):
    lookup = {}
    if not isinstance(annotation_payload, dict):
        return lookup
    images = annotation_payload.get("images")
    if not isinstance(images, list):
        return lookup
    for image_item in images:
        if not isinstance(image_item, dict):
            continue
        image_id = image_item.get("id", image_item.get("image_id"))
        if image_id is None:
            continue
        lookup[str(image_id)] = image_item
    return lookup


def _resolve_image_name(annotation, image_lookup):
    for key in ("image_name", "file_name", "image", "image_path"):
        value = annotation.get(key)
        if value:
            return str(value)
    image_id = annotation.get("image_id", annotation.get("img_id"))
    if image_id is not None:
        image_item = image_lookup.get(str(image_id))
        if isinstance(image_item, dict):
            for key in ("file_name", "image_name", "coco_url", "flickr_url"):
                value = image_item.get(key)
                if value:
                    value = str(value)
                    if key in {"coco_url", "flickr_url"}:
                        return value.rsplit("/", 1)[-1]
                    return value
    raise KeyError(f"Missing image name fields in annotation ann_id={annotation.get('ann_id', annotation.get('id'))!r}")


def _resolve_ann_id(annotation):
    ann_id = annotation.get("ann_id", annotation.get("id"))
    if ann_id is None:
        raise KeyError("Annotation is missing ann_id/id.")
    return str(ann_id)


def _resolve_segmentation(annotation):
    if "segmentation" in annotation:
        return annotation["segmentation"]
    if isinstance(annotation.get("mask"), dict):
        mask_payload = annotation["mask"]
        if "segmentation" in mask_payload:
            return mask_payload["segmentation"]
    raise KeyError(f"Annotation ann_id={annotation.get('ann_id', annotation.get('id'))!r} is missing segmentation.")


def _resolve_class_name(annotation):
    for key in ("class_name", "category_name", "label", "class"):
        value = annotation.get(key)
        if value:
            return str(value)
    return ""


def _build_mask_prompts(mask):
    return np.expand_dims(mask.astype(np.float32), axis=0)


def _clean_official_eval_caption(caption: str) -> str:
    text = (caption or "").strip()
    if not text:
        return ""

    cleanup_patterns = (
        r"^\s*in\s+region1\s*,?\s*",
        r"^\s*the\s+region1\s+contains\s+",
        r"^\s*the\s+region1\s+region\s+in\s+the\s+image\s+is\s+",
        r"^\s*the\s+region1\s+region\s+is\s+",
        r"^\s*the\s+region1\s+is\s+",
        r"^\s*the\s+region\s+in\s+region1\s+shows\s+",
        r"^\s*the\s+region\s+in\s+the\s+image\s+shows\s+",
        r"^\s*the\s+masked\s+region\s+in\s+the\s+image\s+(?:is|represents)\s+",
        r"^\s*the\s+masked\s+region\s+(?:is|represents)\s+",
        r"^\s*the\s+target\s+in\s+region1\s+is\s+",
        r"^\s*the\s+target\s+marked\s+by\s+region1\s+is\s+",
    )
    for pattern in cleanup_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\bregion1\b", "region", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    if not text:
        return ""
    if text[0].isalpha():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    images_root = data_root / "images"
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_root}")
    if not (data_root / "qa.json").exists():
        raise FileNotFoundError(f"Missing qa.json: {data_root / 'qa.json'}")
    if not (data_root / "class_names.json").exists():
        raise FileNotFoundError(f"Missing class_names.json: {data_root / 'class_names.json'}")

    annotations, annotation_payload = _load_annotations(data_root)
    image_lookup = _build_image_lookup(annotation_payload)
    tokenizer_path = args.tokenizer_path or args.model_path
    model = Sa2VAOPSDModelV3(
        model_path=args.model_path,
        enable_teacher=False,
        grpo_group_size=0,
        tokenizer_path=tokenizer_path,
        device=args.device,
        torch_dtype="auto",
        use_flash_attn=True,
        min_caption_tokens=4,
    )
    model.eval()

    predictions = {"query": DEFAULT_DLC_BENCH_QUERY}
    debug_items = {}

    attempted = 0
    exported = 0
    valid_caption_count = 0
    step = max(int(args.step), 1)
    start = max(int(args.start), 0)
    limit = None if args.limit is None else max(int(args.limit), 0)

    with torch.no_grad():
        for index in range(start, len(annotations), step):
            if limit is not None and attempted >= limit:
                break
            annotation = annotations[index]
            attempted += 1
            ann_id = _resolve_ann_id(annotation)
            image_name = _resolve_image_name(annotation, image_lookup)
            image_path = images_root / image_name
            class_name = _resolve_class_name(annotation)
            if not image_path.exists():
                predictions[ann_id] = ""
                debug_items[ann_id] = {
                    "ann_id": ann_id,
                    "image_name": image_name,
                    "class_name": class_name,
                    "raw_prediction": "",
                    "clean_caption": "",
                    "description_status": "missing_image",
                    "caption_token_count": 0,
                }
                print(f"[skip-missing-image] ann_id={ann_id} image={image_name}")
                exported += 1
                continue

            with Image.open(image_path) as pil_image:
                image = pil_image.convert("RGB")
                width, height = image.size
                gt_mask = _decode_segmentation(_resolve_segmentation(annotation), height, width)

            description = model.generate_description(
                image=image,
                mask_prompts=_build_mask_prompts(gt_mask),
                student_question=args.student_question,
            )
            caption = _clean_official_eval_caption(description.clean_caption or "")
            predictions[ann_id] = caption
            token_count = model._caption_token_count(caption)
            if description.status == "ok" and caption:
                valid_caption_count += 1
            debug_items[ann_id] = {
                "ann_id": ann_id,
                "image_name": image_name,
                "class_name": class_name,
                "raw_prediction": description.raw_prediction,
                "clean_caption": caption,
                "model_clean_caption": description.clean_caption or "",
                "description_status": description.status,
                "caption_token_count": token_count,
            }
            exported += 1
            print(
                f"[{exported:04d}] ann_id={ann_id} "
                f"status={description.status:<18} caption={caption!r}"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.debug_output:
        debug_output_path = Path(args.debug_output)
        debug_output_path.parent.mkdir(parents=True, exist_ok=True)
        debug_payload = {
            "meta": {
                "data_root": str(data_root),
                "model_path": args.model_path,
                "tokenizer_path": tokenizer_path,
                "attempted": attempted,
                "exported": exported,
                "valid_caption_count": valid_caption_count,
                "valid_caption_rate": valid_caption_count / max(exported, 1),
                "query": DEFAULT_DLC_BENCH_QUERY,
                "student_question": args.student_question,
            },
            "items": debug_items,
        }
        debug_output_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "data_root": str(data_root),
        "model_path": args.model_path,
        "output": str(output_path),
        "attempted": attempted,
        "exported": exported,
        "valid_caption_count": valid_caption_count,
        "valid_caption_rate": valid_caption_count / max(exported, 1),
        "query": DEFAULT_DLC_BENCH_QUERY,
    }
    if args.debug_output:
        summary["debug_output"] = str(Path(args.debug_output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
