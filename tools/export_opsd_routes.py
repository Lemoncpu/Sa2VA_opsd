import argparse
import json
import os
import runpy
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from mmengine.config import DictAction

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.sa2va.datasets.refcoco_opsd import build_refcoco_opsd_records


def parse_args():
    parser = argparse.ArgumentParser(description="Export offline OPSD route manifest for RefCOCO.")
    parser.add_argument("config", help="Training config path.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to load before route export.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--device", default=None, help="Runtime device override.")
    parser.add_argument(
        "--route-model",
        default="teacher",
        choices=["teacher", "student"],
        help="Model used for description->reconstruct route estimation.",
    )
    parser.add_argument("--global-step", type=int, default=0, help="Step value recorded in manifest.")
    parser.add_argument("--image-root", default=None, help="Optional image root override.")
    parser.add_argument(
        "--update-latest",
        action="store_true",
        help="Also update routes_latest.jsonl beside the output manifest.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config values, same format as tools/train.py.",
    )
    return parser.parse_args()


def _merge_cfg_options(cfg: dict, cfg_options: dict):
    if not cfg_options:
        return cfg
    for dotted_key, value in cfg_options.items():
        keys = dotted_key.split(".")
        cursor = cfg
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return cfg


def load_config(path: str, cfg_options: dict = None):
    cfg = runpy.run_path(path)
    return _merge_cfg_options(cfg, cfg_options)


def load_checkpoint_if_needed(model, checkpoint_path: str):
    if not checkpoint_path:
        return
    from xtuner.model.utils import guess_load_checkpoint

    state_dict = guess_load_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)


def build_model_from_cfg(cfg: dict, *, device: str = None):
    from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3

    model_cfg = dict(cfg["model"])
    model_cfg.pop("type", None)
    model_cfg["device"] = device or model_cfg.get("device", "auto")
    model = Sa2VAOPSDModelV3(**model_cfg)
    model.eval()
    return model


def iter_refcoco_samples_from_cfg(cfg: dict, *, image_root: str = None, limit: int = None):
    dataset_cfg = dict(cfg["train_dataset"])
    records, _ = build_refcoco_opsd_records(
        data_root=dataset_cfg["data_root"],
        dataset_name=dataset_cfg.get("dataset_name", "refcoco"),
        split=dataset_cfg.get("split", "train"),
        image_root=image_root or dataset_cfg.get("image_root"),
        image_root_candidates=dataset_cfg.get("image_root_candidates", []),
        skip_empty_masks=dataset_cfg.get("skip_empty_masks", True),
        skip_missing_images=dataset_cfg.get("skip_missing_images", True),
    )
    if limit is not None:
        records = records[:limit]
    student_question = dataset_cfg["student_question"]
    for record in records:
        yield {
            "sample_key": record["sample_key"],
            "image_path": record["image_path"],
            "gt_mask": record["gt_mask"],
            "student_question": student_question,
        }


def build_manifest_record(route_info: dict, *, sample_key: str, global_step: int, timestamp: str):
    return {
        "sample_key": sample_key,
        "route": route_info["route"],
        "iou": float(route_info["iou"]),
        "description_status": route_info["description_status"],
        "reconstruct_status": route_info["reconstruct_status"],
        "timestamp": timestamp,
        "global_step": int(global_step),
    }


def export_routes(
    *,
    model,
    samples,
    out_path: str,
    global_step: int = 0,
    route_model: str = "teacher",
    update_latest: bool = False,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    description_model = model.student_model
    reconstruct_model = model.student_model
    if route_model == "teacher":
        teacher_model = model.require_teacher_model("Teacher route export")
        description_model = teacher_model
        reconstruct_model = teacher_model

    route_counts = {}
    with open(out_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for item in samples:
                image = Image.open(item["image_path"]).convert("RGB")
                gt_mask = model._to_numpy_mask(item["gt_mask"])
                prompt_masks = gt_mask.astype("float32")[None, ...]
                route_info = model.estimate_opsd_route_for_sample_with_model(
                    description_model=description_model,
                    reconstruct_model=reconstruct_model,
                    image=image,
                    prompt_masks=prompt_masks,
                    student_question=item["student_question"],
                    gt_mask=gt_mask,
                    sample_key=item["sample_key"],
                    debug=False,
                )
                manifest_record = build_manifest_record(
                    route_info,
                    sample_key=item["sample_key"],
                    global_step=global_step,
                    timestamp=timestamp,
                )
                route = manifest_record["route"]
                route_counts[route] = route_counts.get(route, 0) + 1
                f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
    if update_latest:
        latest_path = out_path.parent / "routes_latest.jsonl"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        try:
            latest_path.symlink_to(out_path.name)
        except OSError:
            shutil.copyfile(out_path, latest_path)
    return route_counts


def export_routes_from_runner(
    *,
    runner,
    out_path: str,
    global_step: int,
    route_model: str = "teacher",
    limit: int = None,
):
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    cfg = runner.cfg
    samples = iter_refcoco_samples_from_cfg(cfg, limit=limit)
    route_counts = export_routes(
        model=model,
        samples=samples,
        out_path=out_path,
        global_step=global_step,
        route_model=route_model,
        update_latest=True,
    )
    runner.logger.info(
        "OPSD route export finished: step=%s route_model=%s route_counts=%s out=%s",
        global_step,
        route_model,
        route_counts,
        out_path,
    )
    return route_counts


def main():
    args = parse_args()
    cfg = load_config(args.config, cfg_options=args.cfg_options)
    model = build_model_from_cfg(cfg, device=args.device)
    load_checkpoint_if_needed(model, args.checkpoint)
    samples = iter_refcoco_samples_from_cfg(
        cfg,
        image_root=args.image_root,
        limit=args.limit,
    )
    route_counts = export_routes(
        model=model,
        samples=samples,
        out_path=args.out,
        global_step=args.global_step,
        route_model=args.route_model,
        update_latest=args.update_latest,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "global_step": args.global_step,
                "route_model": args.route_model,
                "route_counts": route_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
