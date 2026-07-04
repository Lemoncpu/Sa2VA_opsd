import argparse
import json
import os
import runpy
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from PIL import Image
from mmengine.config import DictAction

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.sa2va.evaluation.teacher_diagnosis_common import (
    GRPO_POSITIVE_ROUTE,
    ON_POLICY_DISTILL_ROUTE,
    TEACHER_REGENERATE_ROUTE,
)
from projects.sa2va.datasets.refcoco_opsd import SKIP_OPSD_ROUTE
from tools.export_opsd_routes import (
    atomic_write_json,
    atomic_write_text,
    build_confuser_dataset_from_cfg,
    build_rank_shard_path,
    build_rank_status_path,
    cleanup_shards,
    collect_refcoco_samples_from_cfg,
    dist_is_initialized,
    get_coord_dir,
    get_coord_timeout_seconds,
    get_rank,
    get_result_path,
    get_timestamp_path,
    get_world_size,
    is_rank0,
    iter_batches,
    load_checkpoint_if_needed,
    load_json,
    load_jsonl_records,
    maybe_init_distributed,
    merge_route_counts,
    merge_shards,
    shard_samples,
    wait_for_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export offline DLC choose-one OPSD route manifest for RefCOCO.")
    parser.add_argument("config", help="Training config path.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to load before route export.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-rank route export batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap.")
    parser.add_argument(
        "--route-model",
        default="student",
        choices=["teacher", "student"],
        help="Model used for DLC generation and choose-one route estimation.",
    )
    parser.add_argument("--global-step", type=int, default=0, help="Step value recorded in manifest.")
    parser.add_argument("--image-root", default=None, help="Optional image root override.")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config values, same format as tools/train.py.",
    )
    parser.add_argument("--launcher", default="none", help="Launcher mode for distributed export.")
    parser.add_argument("--deepspeed", default=None, help="Ignored compatibility argument from tools/dist.sh.")
    parser.add_argument("--local_rank", type=int, default=None, help="Local rank passed by torch.distributed.launch.")
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


def _patch_mmengine_adafactor_duplicate_registration() -> None:
    try:
        from mmengine.registry.registry import Registry
    except Exception:
        return

    if getattr(Registry, "_sa2va_adafactor_duplicate_patch", False):
        return

    original_register_module = Registry._register_module

    def patched_register_module(self, module, module_name=None, force=False):
        names = module_name
        if names is None:
            names = [module.__name__]
        elif isinstance(names, str):
            names = [names]

        if not force and self.name == "optimizer":
            for name in names:
                if name != "Adafactor":
                    continue
                existing = self._module_dict.get(name)
                if existing is None:
                    continue
                if existing is module:
                    return
                if (
                    getattr(existing, "__name__", None) == getattr(module, "__name__", None)
                    and getattr(existing, "__module__", None) == getattr(module, "__module__", None)
                ):
                    return

        return original_register_module(self, module=module, module_name=module_name, force=force)

    Registry._register_module = patched_register_module
    Registry._sa2va_adafactor_duplicate_patch = True


def load_config(path: str, cfg_options: dict = None):
    _patch_mmengine_adafactor_duplicate_registration()
    cfg = runpy.run_path(path)
    return _merge_cfg_options(cfg, cfg_options)


def build_model_from_cfg(cfg: dict):
    model_cfg = dict(cfg["model"])
    model_type = model_cfg.pop("type")
    model_cfg["device"] = model_cfg.get("device", "auto")
    model = model_type(**model_cfg)
    model.eval()
    return model


def build_manifest_record(route_info: dict, *, sample_key: str, global_step: int, timestamp: str):
    return {
        "sample_key": sample_key,
        "route": route_info["route"],
        "iou": float(route_info["dlc_choose_one_confidence"]),
        "description_status": route_info["description_status"],
        "reconstruct_status": route_info["reconstruct_status"],
        "timestamp": timestamp,
        "global_step": int(global_step),
        "route_metric_name": "dlc_choose_one_confidence",
        "dlc_choose_one_correct": bool(route_info["dlc_choose_one_correct"]),
        "dlc_choose_one_confidence": float(route_info["dlc_choose_one_confidence"]),
        "wrong_confuser_available": bool(route_info["wrong_confuser_available"]),
        "failure_reason": route_info.get("failure_reason", ""),
    }


def build_skip_manifest_record(*, sample_key: str, global_step: int, timestamp: str):
    return {
        "sample_key": sample_key,
        "route": SKIP_OPSD_ROUTE,
        "iou": 0.0,
        "description_status": "skipped",
        "reconstruct_status": "skipped",
        "timestamp": timestamp,
        "global_step": int(global_step),
        "route_metric_name": "dlc_choose_one_confidence",
        "dlc_choose_one_correct": False,
        "dlc_choose_one_confidence": 0.0,
        "wrong_confuser_available": False,
        "failure_reason": "skipped",
    }


def estimate_dlc_choose_one_route_for_sample_with_model(
    *,
    model,
    route_model,
    image,
    prompt_masks,
    student_question,
    gt_mask,
    confuser_candidate_masks,
):
    if hasattr(model, "build_dlc_student_prompt"):
        prompt_text = model.build_dlc_student_prompt(student_question)
        description = model._generate_caption_with_prompt(
            route_model,
            image=image,
            mask_prompts=prompt_masks,
            prompt_text=prompt_text,
            apply_mask_focus=True,
        )
    else:
        description = model.generate_description_with_model(
            route_model,
            image=image,
            mask_prompts=prompt_masks,
            student_question=student_question,
            apply_mask_focus=True,
        )
    if not model._is_caption_trainable_for_student_losses(description):
        return {
            "route": TEACHER_REGENERATE_ROUTE,
            "dlc_choose_one_correct": False,
            "dlc_choose_one_confidence": 0.0,
            "description_status": description.status,
            "reconstruct_status": "skipped_invalid_description",
            "wrong_confuser_available": False,
            "failure_reason": f"invalid_description:{description.status}",
        }
    confuser_masks, confuser_meta = model._select_confuser_masks(
        gt_mask=gt_mask,
        candidate_masks=confuser_candidate_masks,
    )
    if confuser_masks is None:
        return {
            "route": ON_POLICY_DISTILL_ROUTE,
            "dlc_choose_one_correct": False,
            "dlc_choose_one_confidence": 0.0,
            "description_status": description.status,
            "reconstruct_status": "missing_confuser_masks",
            "wrong_confuser_available": False,
            "failure_reason": confuser_meta.get("grpo_skip_reason") or "missing_confuser_masks",
        }
    option_masks = [model._to_numpy_mask(gt_mask), *[model._to_numpy_mask(mask) for mask in confuser_masks]]
    random_state = np.random.RandomState(0)
    random_state.shuffle(option_masks)
    correct_option_idx = next(
        idx for idx, candidate_mask in enumerate(option_masks)
        if float(model._compute_iou(gt_mask, candidate_mask)) >= model.grpo_confuser_duplicate_iou_threshold
    )
    selection = model._score_caption_against_mask_options(
        model=route_model,
        image=image,
        option_masks=np.stack(option_masks, axis=0).astype(np.float32),
        caption=description.clean_caption,
        correct_option_idx=correct_option_idx,
    )
    confidence = float(selection.correct_option_prob)
    if not selection.selected_correct:
        route = TEACHER_REGENERATE_ROUTE
        reconstruct_status = "choose_one_wrong"
        wrong_confuser_available = True
    elif confidence < float(getattr(model, "dlc_onpolicy_conf_threshold", 0.5)):
        route = ON_POLICY_DISTILL_ROUTE
        reconstruct_status = "choose_one_low_confidence"
        wrong_confuser_available = False
    else:
        route = GRPO_POSITIVE_ROUTE
        reconstruct_status = "choose_one_correct_high_confidence"
        wrong_confuser_available = False
    return {
        "route": route,
        "dlc_choose_one_correct": bool(selection.selected_correct),
        "dlc_choose_one_confidence": confidence,
        "description_status": description.status,
        "reconstruct_status": reconstruct_status,
        "wrong_confuser_available": wrong_confuser_available,
        "failure_reason": "",
    }


def export_routes_shard(
    *,
    model,
    samples: Sequence[Dict],
    confuser_dataset,
    sample_key_to_index: Dict[str, int],
    shard_out_path: Path,
    global_step: int = 0,
    route_model_name: str = "teacher",
    timestamp: str,
    batch_size: int = 1,
):
    shard_out_path.parent.mkdir(parents=True, exist_ok=True)
    description_model = model.student_model
    if route_model_name == "teacher":
        description_model = model.require_teacher_model("Teacher DLC route export")

    route_counts: Dict[str, int] = {}
    description_status_counts: Dict[str, int] = {}
    reconstruct_status_counts: Dict[str, int] = {}
    record_count = 0
    with open(shard_out_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for batch in iter_batches(samples, batch_size):
                for item in batch:
                    image = Image.open(item["image_path"]).convert("RGB")
                    gt_mask = model._to_numpy_mask(item["gt_mask"])
                    prompt_masks = gt_mask.astype("float32")[None, ...]
                    sample_index = sample_key_to_index.get(str(item["sample_key"]))
                    if sample_index is None:
                        raise KeyError(f"Missing sample_key in confuser dataset index: {item['sample_key']}")
                    prepared_item = confuser_dataset.prepare_data(sample_index)
                    route_info = estimate_dlc_choose_one_route_for_sample_with_model(
                        model=model,
                        route_model=description_model,
                        image=image,
                        prompt_masks=prompt_masks,
                        student_question=item["student_question"],
                        gt_mask=gt_mask,
                        confuser_candidate_masks=prepared_item.get("confuser_candidate_masks"),
                    )
                    manifest_record = build_manifest_record(
                        route_info,
                        sample_key=item["sample_key"],
                        global_step=global_step,
                        timestamp=timestamp,
                    )
                    route = manifest_record["route"]
                    route_counts[route] = route_counts.get(route, 0) + 1
                    description_status = manifest_record.get("description_status")
                    if description_status:
                        description_status_counts[str(description_status)] = (
                            description_status_counts.get(str(description_status), 0) + 1
                        )
                    reconstruct_status = manifest_record.get("reconstruct_status")
                    if reconstruct_status:
                        reconstruct_status_counts[str(reconstruct_status)] = (
                            reconstruct_status_counts.get(str(reconstruct_status), 0) + 1
                        )
                    f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
                    record_count += 1
    return route_counts, description_status_counts, reconstruct_status_counts, record_count


def export_routes(
    *,
    model,
    samples: Sequence[Dict],
    confuser_dataset,
    sample_key_to_index: Dict[str, int],
    out_path: str,
    global_step: int = 0,
    route_model_name: str = "teacher",
    batch_size: int = 1,
    limit: Optional[int] = None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rank = get_rank()
    world_size = get_world_size()
    samples_to_export = list(samples[:limit] if limit is not None else samples)
    coord_dir = get_coord_dir(out_path)
    if is_rank0():
        if coord_dir.exists():
            shutil.rmtree(coord_dir)
        coord_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(get_timestamp_path(coord_dir), datetime.now(timezone.utc).isoformat())
    wait_for_paths(
        [get_timestamp_path(coord_dir)],
        timeout_seconds=get_coord_timeout_seconds(),
        description=f"shared dlc route export timestamp for {out_path.name}",
    )
    timestamp = get_timestamp_path(coord_dir).read_text(encoding="utf-8").strip()
    shard_samples_local = shard_samples(samples_to_export)
    shard_out_path = build_rank_shard_path(out_path, rank, world_size)
    shard_counts = {}
    shard_description_status_counts = {}
    shard_reconstruct_status_counts = {}
    shard_record_count = 0
    shard_ok = True
    shard_error = None
    try:
        (
            shard_counts,
            shard_description_status_counts,
            shard_reconstruct_status_counts,
            shard_record_count,
        ) = export_routes_shard(
            model=model,
            samples=shard_samples_local,
            confuser_dataset=confuser_dataset,
            sample_key_to_index=sample_key_to_index,
            shard_out_path=shard_out_path,
            global_step=global_step,
            route_model_name=route_model_name,
            timestamp=timestamp,
            batch_size=batch_size,
        )
    except Exception as exc:
        shard_ok = False
        shard_error = f"{type(exc).__name__}: {exc}"
        if shard_out_path.exists():
            shard_out_path.unlink()
    status_path = build_rank_status_path(out_path, rank, world_size)
    atomic_write_json(
        status_path,
        {
            "rank": rank,
            "world_size": world_size,
            "ok": shard_ok,
            "error": shard_error,
            "record_count": shard_record_count,
            "route_counts": shard_counts,
            "description_status_counts": shard_description_status_counts,
            "reconstruct_status_counts": shard_reconstruct_status_counts,
            "shard_out_path": str(shard_out_path),
        },
    )
    shard_paths = [build_rank_shard_path(out_path, shard_rank, world_size) for shard_rank in range(world_size)]
    result_path = get_result_path(coord_dir)
    if is_rank0():
        try:
            status_paths = [build_rank_status_path(out_path, shard_rank, world_size) for shard_rank in range(world_size)]
            wait_for_paths(
                status_paths,
                timeout_seconds=get_coord_timeout_seconds(),
                description=f"dlc route export shard status files for {out_path.name}",
            )
            rank_statuses = [load_json(path) for path in status_paths]
            failed_statuses = [status for status in rank_statuses if not status.get("ok", False)]
            if failed_statuses:
                error_details = "; ".join(
                    f"rank {status['rank']}: {status.get('error') or 'unknown export error'}"
                    for status in failed_statuses
                )
                raise RuntimeError(f"Distributed DLC route export failed before merge. {error_details}")
            merged_counts = merge_route_counts([status.get("route_counts", {}) for status in rank_statuses])
            merge_shards(out_path=out_path, shard_paths=shard_paths)
            manifest_records = load_jsonl_records(out_path)
            manifest_record_count = len(manifest_records)
            cleanup_shards(shard_paths)
            atomic_write_json(
                result_path,
                {
                    "ok": True,
                    "route_counts": merged_counts,
                    "record_count": manifest_record_count,
                    "world_size": world_size,
                    "out": str(out_path),
                },
            )
        except Exception as exc:
            atomic_write_json(
                result_path,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "world_size": world_size,
                    "out": str(out_path),
                },
            )
    wait_for_paths(
        [result_path],
        timeout_seconds=get_coord_timeout_seconds(),
        description=f"dlc route export merge result for {out_path.name}",
    )
    result_payload = load_json(result_path)
    if not result_payload.get("ok", False):
        raise RuntimeError(result_payload.get("error", "Distributed DLC route export failed."))
    return {
        "route_counts": result_payload["route_counts"],
        "record_count": int(result_payload["record_count"]),
        "world_size": int(result_payload["world_size"]),
    }


def main():
    args = parse_args()
    maybe_init_distributed(args)
    print_summary = is_rank0()
    export_summary = None
    try:
        cfg = load_config(args.config, cfg_options=args.cfg_options)
        model = build_model_from_cfg(cfg)
        load_checkpoint_if_needed(model, args.checkpoint)
        samples = collect_refcoco_samples_from_cfg(
            cfg,
            image_root=args.image_root,
            limit=args.limit,
        )
        confuser_dataset, sample_key_to_index = build_confuser_dataset_from_cfg(
            cfg,
            image_root=args.image_root,
            limit=args.limit,
        )
        export_summary = export_routes(
            model=model,
            samples=samples,
            confuser_dataset=confuser_dataset,
            sample_key_to_index=sample_key_to_index,
            out_path=args.out,
            global_step=args.global_step,
            route_model_name=args.route_model,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    finally:
        if dist_is_initialized():
            torch.distributed.destroy_process_group()
    if print_summary:
        print(
            json.dumps(
                {
                    "out": args.out,
                    "global_step": args.global_step,
                    "route_model": args.route_model,
                    "batch_size": args.batch_size,
                    "world_size": export_summary["world_size"],
                    "record_count": export_summary["record_count"],
                    "route_counts": export_summary["route_counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
