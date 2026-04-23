import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils


PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RefCOCO masks on images.")
    parser.add_argument("--ref-root", default="/data/xyc/refcoco")
    parser.add_argument("--split-by", default="unc")
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.38)
    return parser.parse_args()


def load_refs(ref_root, split_by, split):
    refs_path = Path(ref_root) / f"refs({split_by}).p"
    with refs_path.open("rb") as f:
        refs = pickle.load(f)
    if isinstance(refs, dict):
        refs = refs.get("refs", refs)
    return [ref for ref in refs if ref.get("split") == split]


def load_instances(ref_root):
    instances_path = Path(ref_root) / "instances.json"
    data = json.loads(instances_path.read_text())
    anns = {ann["id"]: ann for ann in data["annotations"]}
    imgs = {img["id"]: img for img in data["images"]}
    return anns, imgs


def decode_ref_mask(ann, height, width):
    segmentation = ann["segmentation"]
    if len(segmentation) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation[0], list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rles = segmentation
        for item in rles:
            if not isinstance(item["counts"], bytes):
                item["counts"] = item["counts"].encode()
    mask = mask_utils.decode(rles)
    if mask.ndim == 3:
        mask = np.sum(mask, axis=2)
    return (mask > 0).astype(np.uint8)


def truncate_text(text, max_len=56):
    text = " ".join((text or "").strip().split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def parse_image_id(image_name):
    return int(Path(image_name).stem.split("_")[-1])


def blend_masks(image, objects, alpha):
    arr = np.asarray(image.convert("RGBA"), dtype=np.float32).copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    for index, obj in enumerate(objects, start=1):
        color = np.asarray(PALETTE[(index - 1) % len(PALETTE)], dtype=np.float32)
        mask = obj["mask"].astype(bool)
        if not np.any(mask):
            continue
        arr[mask, :3] = arr[mask, :3] * (1.0 - alpha) + color * alpha
    arr[..., 3] = 255
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def draw_boxes_and_labels(image, objects):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, obj in enumerate(objects, start=1):
        color = PALETTE[(index - 1) % len(PALETTE)]
        x, y, w, h = obj["bbox"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = str(index)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 3
        text_x = x1
        text_y = max(0, y1 - th - pad * 2)
        draw.rectangle(
            (text_x, text_y, text_x + tw + pad * 2, text_y + th + pad * 2),
            fill=color,
        )
        draw.text((text_x + pad, text_y + pad), label, fill=(255, 255, 255), font=font)
    return image


def append_legend(image, image_name, objects):
    font = ImageFont.load_default()
    lines = [f"{image_name}  masks={len(objects)}"]
    for index, obj in enumerate(objects, start=1):
        lines.append(f"{index}. ann={obj['ann_id']}  {truncate_text(obj['label'])}")

    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_heights = []
    max_width = 0
    for line in lines:
        bbox = draw_probe.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        max_width = max(max_width, width)
        line_heights.append(height)

    padding = 8
    line_gap = 4
    legend_height = padding * 2 + sum(line_heights) + line_gap * max(len(lines) - 1, 0)
    legend_width = max(image.width, max_width + padding * 2)
    canvas = Image.new("RGBA", (legend_width, image.height + legend_height), (255, 255, 255, 255))
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (0, image.height, legend_width, image.height + legend_height),
        fill=(18, 18, 18, 255),
    )
    y = image.height + padding
    for index, line in enumerate(lines):
        fill = (255, 255, 255)
        if index > 0:
            color = PALETTE[(index - 1) % len(PALETTE)]
            draw.rectangle((padding, y + 2, padding + 12, y + 14), fill=color)
            text_x = padding + 18
        else:
            text_x = padding
        draw.text((text_x, y), line, fill=fill, font=font)
        y += line_heights[index] + line_gap
    return canvas.convert("RGB")


def build_objects_for_image(refs, anns, imgs, image_id):
    seen = {}
    for ref in refs:
        if ref["image_id"] != image_id:
            continue
        ann_id = ref["ann_id"]
        if ann_id in seen:
            continue
        ann = anns[ann_id]
        img = imgs[image_id]
        label = ""
        if ref.get("sentences"):
            label = ref["sentences"][0].get("sent", "")
        seen[ann_id] = {
            "ann_id": ann_id,
            "bbox": ann["bbox"],
            "label": label,
            "mask": decode_ref_mask(ann, img["height"], img["width"]),
        }
    return list(seen.values())


def main():
    args = parse_args()
    ref_root = Path(args.ref_root)
    image_dir = Path(args.image_dir) if args.image_dir else ref_root / "refcoco_val_100"
    output_dir = Path(args.output_dir) if args.output_dir else ref_root / "refcoco_mask"
    output_dir.mkdir(parents=True, exist_ok=True)

    refs = load_refs(ref_root, args.split_by, args.split)
    anns, imgs = load_instances(ref_root)

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    manifest = []
    for image_path in image_paths:
        image_id = parse_image_id(image_path.name)
        objects = build_objects_for_image(refs, anns, imgs, image_id)
        image = Image.open(image_path).convert("RGB")
        blended = blend_masks(image, objects, args.alpha)
        with_boxes = draw_boxes_and_labels(blended, objects)
        final_image = append_legend(with_boxes, image_path.name, objects)
        output_path = output_dir / f"{image_path.stem}.png"
        final_image.save(output_path)
        manifest.append(
            {
                "image": image_path.name,
                "output": output_path.name,
                "image_id": image_id,
                "mask_count": len(objects),
                "ann_ids": [obj["ann_id"] for obj in objects],
                "labels": [obj["label"] for obj in objects],
            }
        )
        print(f"[ok] {image_path.name} -> {output_path.name} masks={len(objects)}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": len(manifest), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
