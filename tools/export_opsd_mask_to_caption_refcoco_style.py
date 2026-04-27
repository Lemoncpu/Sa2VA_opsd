import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.sa2va.datasets.sa2va_opsd_npz_v2 import Sa2VAOpsdNPZDatasetV2
from projects.sa2va.datasets.common import DEFAULT_MASK_TO_CAPTION_QUESTION
from projects.sa2va.evaluation.caption_to_mask_common import normalize_refcoco_caption
from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3


DEFAULT_STUDENT_QUESTION = DEFAULT_MASK_TO_CAPTION_QUESTION


def parse_args():
    parser = argparse.ArgumentParser(description="Export OPSD mask-to-caption results as RefCOCO-style annotations.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--npz-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-question", default=DEFAULT_STUDENT_QUESTION)
    return parser.parse_args()


def mask_to_rle(mask):
    mask = np.asarray(mask).astype(np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(mask))
    encoded["counts"] = encoded["counts"].decode()
    return encoded


def save_image(image, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(output_path)
        return image.size[1], image.size[0]
    pil_image = Image.fromarray(np.asarray(image))
    pil_image.save(output_path)
    return pil_image.size[1], pil_image.size[0]


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.model_path

    dataset = Sa2VAOpsdNPZDatasetV2(
        npz_dir=args.npz_dir,
        prefix="masklet_data",
        shuffle=False,
        repeats=1,
        skip_empty_masks=True,
        student_question=args.student_question,
    )

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

    output_path = Path(args.output)
    image_dir = None if args.image_dir is None else Path(args.image_dir)
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)

    items = []
    attempted = 0
    exported = 0
    valid_caption_count = 0

    target_export_count = len(dataset.npz_files) if args.limit is None else min(args.limit, len(dataset.npz_files))
    seen_npz_paths = set()
    with torch.no_grad():
        for index in range(args.start, len(dataset.npz_files), max(args.step, 1)):
            if exported >= target_export_count:
                break
            attempted += 1
            try:
                sample = dataset.prepare_data(index)
            except Exception as exc:
                print(f"[skip-invalid] index={index:06d} reason={exc}")
                continue

            if sample["npz_path"] in seen_npz_paths:
                print(f"[skip-duplicate] index={index:06d} npz={sample['npz_path']}")
                continue
            seen_npz_paths.add(sample["npz_path"])

            image_filename = None
            height = int(sample["image"].size[1])
            width = int(sample["image"].size[0])
            if image_dir is not None:
                image_filename = f"opsd_{index:06d}.png"
                image_path = image_dir / image_filename
                height, width = save_image(sample["image"], image_path)

            description = model.generate_description(
                image=sample["image"],
                mask_prompts=sample["prompt_masks"],
                student_question=sample["student_question"],
            )
            caption = normalize_refcoco_caption(description.clean_caption)
            selected_labels = [caption] if caption else []
            if description.status == "ok" and selected_labels:
                valid_caption_count += 1

            gt_mask = model._to_numpy_mask(sample["gt_mask"])
            item = {
                "sample_id": exported,
                "source_index": index,
                "npz_path": sample["npz_path"],
                "image": image_filename,
                "image_info": {
                    "id": exported,
                    "height": int(height),
                    "width": int(width),
                    "file_name": image_filename,
                },
                "selected_labels": selected_labels,
                "gt_masks": [mask_to_rle(gt_mask)],
                "description_status": description.status,
                "raw_prediction": description.raw_prediction,
                "caption": caption,
                "caption_storage_format": "clean_caption_strip_preserve_case",
                "caption_token_count": model._caption_token_count(caption),
            }
            items.append(item)
            exported += 1
            print(
                f"[{exported:03d}] export_id={exported - 1:03d} "
                f"status={description.status:<18} caption={description.clean_caption!r}"
            )

    payload = {
        "meta": {
            "npz_dir": args.npz_dir,
            "model_path": args.model_path,
            "image_dir": None if image_dir is None else str(image_dir),
            "attempted": attempted,
            "exported": exported,
            "valid_caption_count": valid_caption_count,
            "valid_caption_rate": valid_caption_count / max(exported, 1),
        },
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
