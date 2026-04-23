import argparse
import html
import json
import math
import textwrap
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize OPSD NPZ samples referenced by an exported annotations.json file."
    )
    parser.add_argument("--annotation-file", required=True, help="Path to exported annotations.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save PNG visualizations and index.html. Default: <annotation_dir>/npz_visualizations",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to visualize")
    parser.add_argument("--panel-size", type=int, default=384, help="Max width/height per panel")
    parser.add_argument(
        "--mask-color",
        default="#ff4d4f",
        help="Overlay color for masks in hex format, e.g. '#ff4d4f'",
    )
    return parser.parse_args()


def load_items(annotation_file):
    payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload.get("items", []), payload.get("meta", {})
    return payload, {}


def normalize_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] != 3:
        raise ValueError(f"Unsupported frame channel layout: {frame.shape}")
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0) if frame.max() <= 1.0 else np.clip(frame, 0.0, 255.0)
        if frame.max() <= 1.0:
            frame = frame * 255.0
    return Image.fromarray(frame.astype(np.uint8)).convert("RGB")


def normalize_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")
    return (mask > 0).astype(np.uint8)


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def resize_to_panel(image, panel_size):
    width, height = image.size
    scale = min(panel_size / max(width, 1), panel_size / max(height, 1), 1.0)
    if scale == 1.0:
        return image
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, RESAMPLE_BILINEAR)


def blend_mask(image, mask, color_rgb, alpha=0.45):
    image_np = np.asarray(image).astype(np.float32)
    mask_bool = np.asarray(mask).astype(bool)
    overlay = image_np.copy()
    overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + np.asarray(color_rgb) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    blended = Image.fromarray(overlay)
    draw = ImageDraw.Draw(blended)
    bbox = mask_bbox(mask)
    if bbox is not None:
        draw.rectangle(bbox, outline=(255, 255, 0), width=3)
    return blended


def build_mask_only(mask, color_rgb):
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    canvas[mask > 0] = np.asarray(color_rgb, dtype=np.uint8)
    return Image.fromarray(canvas)


def panel_with_title(title, image, panel_size, font):
    image = resize_to_panel(image, panel_size)
    pad = 10
    title_h = 28
    canvas = Image.new("RGB", (panel_size, panel_size + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, panel_size - 1, title_h - 1), fill="#f2f4f7", outline="#d0d5dd")
    draw.text((pad, 7), title, fill="black", font=font)
    x = (panel_size - image.size[0]) // 2
    y = title_h + (panel_size - image.size[1]) // 2
    canvas.paste(image, (x, y))
    draw.rectangle((0, title_h, panel_size - 1, title_h + panel_size - 1), outline="#d0d5dd", width=1)
    return canvas


def footer_height(lines, width, font):
    probe = Image.new("RGB", (width, 10), "white")
    draw = ImageDraw.Draw(probe)
    joined = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), joined, font=font, spacing=4)
    return max(70, bbox[3] - bbox[1] + 20)


def compose_sheet(item, frame1, mask1, frame2, mask2, color_rgb, panel_size, font):
    gap = 16
    margin = 16
    title_font = font

    panels = [
        panel_with_title("frame1 (caption target)", frame1, panel_size, title_font),
        panel_with_title("frame1 + mask1 (caption target)", blend_mask(frame1, mask1, color_rgb), panel_size, title_font),
        panel_with_title("frame2 (context only)", frame2, panel_size, title_font),
        panel_with_title("frame2 + mask2 (not caption target)", blend_mask(frame2, mask2, color_rgb), panel_size, title_font),
        panel_with_title("mask1 only", build_mask_only(mask1, color_rgb), panel_size, title_font),
        panel_with_title("mask2 only", build_mask_only(mask2, color_rgb), panel_size, title_font),
    ]

    cols = 2
    rows = int(math.ceil(len(panels) / cols))
    panel_w, panel_h = panels[0].size
    content_w = cols * panel_w + (cols - 1) * gap
    text_lines = [
        f"sample_id={item.get('sample_id')} source_index={item.get('source_index')} status={item.get('description_status')}",
        f"npz={item.get('npz_path')}",
        f"caption={item.get('caption', '')}",
        f"compressed_caption={item.get('compressed_caption', '')}",
        f"caption_tokens={item.get('caption_token_count')} mask1_sum={int(mask1.sum())} mask2_sum={int(mask2.sum())}",
        "note=caption in annotations.json is generated from frame1/mask1 only; frame2/mask2 is shown only for extra NPZ context",
    ]
    wrapped_lines = []
    for line in text_lines:
        wrapped_lines.extend(textwrap.wrap(line, width=92) or [""])
    footer_h = footer_height(wrapped_lines, content_w, font)

    sheet_w = content_w + margin * 2
    sheet_h = rows * panel_h + (rows - 1) * gap + footer_h + margin * 2
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, panel in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = margin + col * (panel_w + gap)
        y = margin + row * (panel_h + gap)
        sheet.paste(panel, (x, y))

    footer_top = margin + rows * panel_h + (rows - 1) * gap + 10
    draw.rectangle((margin, footer_top - 8, sheet_w - margin, sheet_h - margin), fill="#f8fafc", outline="#d0d5dd")
    draw.multiline_text((margin + 10, footer_top), "\n".join(wrapped_lines), fill="black", font=font, spacing=4)
    return sheet


def render_html(records, output_dir, annotation_file, meta):
    rows = []
    for record in records:
        rows.append(
            f"""
            <div class="card">
              <a href="{html.escape(record['image_file'])}">
                <img src="{html.escape(record['image_file'])}" alt="{html.escape(record['title'])}">
              </a>
              <div class="body">
                <div><strong>{html.escape(record['title'])}</strong></div>
                <div><code>{html.escape(record['npz_path'])}</code></div>
                <div>status: <code>{html.escape(str(record['description_status']))}</code></div>
                <div>caption: {html.escape(record['caption'])}</div>
                <div>compressed: {html.escape(record['compressed_caption'])}</div>
              </div>
            </div>
            """
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OPSD NPZ Visualizations</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f5f7fa; color: #111827; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    .meta {{ margin-bottom: 20px; line-height: 1.6; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #d0d5dd; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
    .card img {{ width: 100%; display: block; background: white; }}
    .body {{ padding: 12px 14px; font-size: 14px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="meta">
    <div><strong>annotation_file</strong>: <code>{html.escape(str(annotation_file))}</code></div>
    <div><strong>npz_dir</strong>: <code>{html.escape(str(meta.get('npz_dir', '')))}</code></div>
    <div><strong>attempted/exported</strong>: {meta.get('attempted')} / {meta.get('exported')}</div>
    <div><strong>valid_caption_rate</strong>: {meta.get('valid_caption_rate')}</div>
  </div>
  <div class="grid">
    {''.join(rows)}
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main():
    args = parse_args()
    annotation_file = Path(args.annotation_file)
    output_dir = Path(args.output_dir) if args.output_dir else annotation_file.parent / "npz_visualizations"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        if args.output_dir:
            raise
        fallback_dir = Path("/tmp") / f"{annotation_file.parent.name}_visualizations"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] output dir is not writable, fallback to {fallback_dir}")
        output_dir = fallback_dir

    items, meta = load_items(annotation_file)
    if args.limit is not None:
        items = items[: args.limit]

    font = ImageFont.load_default()
    color = args.mask_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"mask-color must be a 6-digit hex string, got: {args.mask_color}")
    color_rgb = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))

    records = []
    for idx, item in enumerate(items):
        npz_path = item.get("npz_path")
        if not npz_path:
            print(f"[skip] item {idx} has no npz_path")
            continue
        npz_path = Path(npz_path)
        if not npz_path.exists():
            print(f"[skip] missing npz: {npz_path}")
            continue

        data = np.load(npz_path)
        frame1 = normalize_frame(data["frame1"])
        mask1 = normalize_mask(data["mask1"])
        frame2 = normalize_frame(data["frame2"])
        mask2 = normalize_mask(data["mask2"])

        sheet = compose_sheet(
            item=item,
            frame1=frame1,
            mask1=mask1,
            frame2=frame2,
            mask2=mask2,
            color_rgb=color_rgb,
            panel_size=args.panel_size,
            font=font,
        )

        image_file = f"{idx:03d}_{npz_path.stem}.png"
        sheet.save(output_dir / image_file)
        records.append(
            {
                "title": f"{idx:03d} | {npz_path.stem}",
                "image_file": image_file,
                "npz_path": str(npz_path),
                "description_status": item.get("description_status"),
                "caption": item.get("caption", ""),
                "compressed_caption": item.get("compressed_caption", ""),
            }
        )
        print(f"[{idx + 1:03d}] saved {image_file}")

    render_html(records, output_dir, annotation_file, meta)
    print(f"Saved {len(records)} visualizations to {output_dir}")
    print(f"Index: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
