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


def load_config(path: str, cfg_options: dict = None):
    cfg = runpy.run_path(path)
    return _merge_cfg_options(cfg, cfg_options)


def load_checkpoint_if_needed(model, checkpoint_path: str):
    if not checkpoint_path:
        return
    from xtuner.model.utils import guess_load_checkpoint

    state_dict = guess_load_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)


def dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if dist_is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def get_world_size() -> int:
    if dist_is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def is_rank0() -> bool:
    return get_rank() == 0


def maybe_init_distributed(args) -> None:
    if args.launcher == "none" or dist_is_initialized():
        return
    if "LOCAL_RANK" not in os.environ and args.local_rank is not None:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)


def sync_success_or_raise(ok: bool, *, device: torch.device) -> None:
    if not dist_is_initialized():
        if not ok:
            raise RuntimeError("Distributed OPSD export failed.")
        return
    status = torch.tensor([1 if ok else 0], device=device)
    torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MIN)
    if int(status.item()) != 1:
        raise RuntimeError("Distributed OPSD export failed on at least one rank.")


def barrier() -> None:
    if dist_is_initialized():
        torch.distributed.barrier()


def get_coord_dir(out_path: Path) -> Path:
    run_token = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "single"
    return out_path.parent / f".{out_path.stem}.coord.{run_token}"


def get_coord_timeout_seconds() -> float:
    return float(os.environ.get("OPSD_ROUTE_EXPORT_SYNC_TIMEOUT_SECONDS", "7200"))


def get_coord_poll_seconds() -> float:
    return float(os.environ.get("OPSD_ROUTE_EXPORT_SYNC_POLL_SECONDS", "2"))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-rank{get_rank():05d}-{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: Dict) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_records(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl_records(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def wait_for_paths(paths: Sequence[Path], *, timeout_seconds: float, description: str) -> None:
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    missing_paths = [path for path in paths if not path.exists()]
    while missing_paths:
        if time.monotonic() >= deadline:
            missing_preview = ", ".join(str(path) for path in missing_paths[:3])
            raise TimeoutError(f"Timed out waiting for {description}. Missing: {missing_preview}")
        time.sleep(max(get_coord_poll_seconds(), 0.1))
        missing_paths = [path for path in paths if not path.exists()]


def get_timestamp_path(coord_dir: Path) -> Path:
    return coord_dir / "timestamp.txt"


def get_result_path(coord_dir: Path) -> Path:
    return coord_dir / "result.json"


def build_rank_status_path(out_path: Path, rank: int, world_size: int) -> Path:
    coord_dir = get_coord_dir(out_path)
    return coord_dir / f"rank{rank:05d}-of-{world_size:05d}.status.json"


def build_model_from_cfg(cfg: dict):
    from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3

    model_cfg = dict(cfg["model"])
    model_cfg.pop("type", None)
    model_cfg["device"] = model_cfg.get("device", "auto")
    model = Sa2VAOPSDModelV3(**model_cfg)
    model.eval()
    return model


def collect_refcoco_samples_from_cfg(cfg: dict, *, image_root: str = None, limit: int = None) -> List[Dict]:
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
    return [
        {
            "sample_key": record["sample_key"],
            "image_path": record["image_path"],
            "gt_mask": record["gt_mask"],
            "student_question": student_question,
        }
        for record in records
    ]


def filter_samples_by_sample_keys(
    samples: Sequence[Dict],
    *,
    excluded_sample_keys: Optional[Set[str]] = None,
) -> List[Dict]:
    if not excluded_sample_keys:
        return list(samples)
    return [sample for sample in samples if sample["sample_key"] not in excluded_sample_keys]


def shard_samples(samples: Sequence[Dict]) -> List[Dict]:
    rank = get_rank()
    world_size = get_world_size()
    return [sample for index, sample in enumerate(samples) if index % world_size == rank]


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


def build_rank_shard_path(out_path: Path, rank: int, world_size: int) -> Path:
    return out_path.parent / f"{out_path.stem}.rank{rank:05d}-of-{world_size:05d}{out_path.suffix}"


def export_routes_shard(
    *,
    model,
    samples: Sequence[Dict],
    shard_out_path: Path,
    global_step: int = 0,
    route_model: str = "teacher",
    timestamp: str,
):
    shard_out_path.parent.mkdir(parents=True, exist_ok=True)
    description_model = model.student_model
    reconstruct_model = model.student_model
    if route_model == "teacher":
        teacher_model = model.require_teacher_model("Teacher route export")
        description_model = teacher_model
        reconstruct_model = teacher_model

    route_counts = {}
    record_count = 0
    with open(shard_out_path, "w", encoding="utf-8") as f:
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
                record_count += 1
    return route_counts, record_count


def merge_route_counts(all_counts: Sequence[Dict[str, int]]) -> Dict[str, int]:
    merged = {}
    for counts in all_counts:
        for route, value in counts.items():
            merged[route] = merged.get(route, 0) + int(value)
    return merged


def summarize_route_counts(records: Sequence[Dict]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for record in records:
        route = record.get("route")
        if not route:
            continue
        summary[str(route)] = summary.get(str(route), 0) + 1
    return summary


def merge_manifest_records(
    *,
    existing_records: Sequence[Dict],
    updated_records: Sequence[Dict],
) -> List[Dict]:
    merged_by_sample_key: Dict[str, Dict] = {}
    for record in existing_records:
        sample_key = record.get("sample_key")
        if sample_key:
            merged_by_sample_key[str(sample_key)] = record
    for record in updated_records:
        sample_key = record.get("sample_key")
        if sample_key:
            merged_by_sample_key[str(sample_key)] = record
    return [merged_by_sample_key[sample_key] for sample_key in sorted(merged_by_sample_key.keys())]


def merge_shards(
    *,
    out_path: Path,
    shard_paths: Sequence[Path],
    update_latest: bool = False,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as shard_file:
                shutil.copyfileobj(shard_file, out_file)
    if update_latest:
        latest_path = out_path.parent / "routes_latest.jsonl"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        try:
            latest_path.symlink_to(out_path.name)
        except OSError:
            shutil.copyfile(out_path, latest_path)


def cleanup_shards(shard_paths: Sequence[Path]) -> None:
    for shard_path in shard_paths:
        if shard_path.exists():
            shard_path.unlink()


def export_routes(
    *,
    model,
    samples: Sequence[Dict],
    out_path: str,
    global_step: int = 0,
    route_model: str = "teacher",
    update_latest: bool = False,
    consumed_sample_keys: Optional[Sequence[str]] = None,
    existing_manifest_path: Optional[str] = None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rank = get_rank()
    world_size = get_world_size()
    consumed_sample_key_set = {str(sample_key) for sample_key in consumed_sample_keys or [] if sample_key}
    samples_to_export = filter_samples_by_sample_keys(samples, excluded_sample_keys=consumed_sample_key_set)
    coord_dir = get_coord_dir(out_path)
    if is_rank0():
        if coord_dir.exists():
            shutil.rmtree(coord_dir)
        coord_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(get_timestamp_path(coord_dir), datetime.now(timezone.utc).isoformat())
    wait_for_paths(
        [get_timestamp_path(coord_dir)],
        timeout_seconds=get_coord_timeout_seconds(),
        description=f"shared route export timestamp for {out_path.name}",
    )
    timestamp = get_timestamp_path(coord_dir).read_text(encoding="utf-8").strip()
    shard_samples_local = shard_samples(samples_to_export)
    shard_out_path = build_rank_shard_path(out_path, rank, world_size)
    shard_counts = {}
    shard_record_count = 0
    shard_ok = True
    shard_error = None
    try:
        shard_counts, shard_record_count = export_routes_shard(
            model=model,
            samples=shard_samples_local,
            shard_out_path=shard_out_path,
            global_step=global_step,
            route_model=route_model,
            timestamp=timestamp,
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
                description=f"route export shard status files for {out_path.name}",
            )
            rank_statuses = [load_json(path) for path in status_paths]
            failed_statuses = [status for status in rank_statuses if not status.get("ok", False)]
            if failed_statuses:
                error_details = "; ".join(
                    f"rank {status['rank']}: {status.get('error') or 'unknown export error'}"
                    for status in failed_statuses
                )
                raise RuntimeError(f"Distributed OPSD export failed before merge. {error_details}")
            merged_counts = merge_route_counts([status.get("route_counts", {}) for status in rank_statuses])
            record_count = int(sum(int(status.get("record_count", 0)) for status in rank_statuses))
            merge_shards(
                out_path=out_path,
                shard_paths=shard_paths,
                update_latest=update_latest,
            )
            if existing_manifest_path:
                existing_manifest = Path(existing_manifest_path)
                if existing_manifest.exists():
                    merged_records = merge_manifest_records(
                        existing_records=load_jsonl_records(existing_manifest),
                        updated_records=load_jsonl_records(out_path),
                    )
                    write_jsonl_records(out_path, merged_records)
                    if update_latest:
                        latest_path = out_path.parent / "routes_latest.jsonl"
                        if latest_path.exists() or latest_path.is_symlink():
                            latest_path.unlink()
                        try:
                            latest_path.symlink_to(out_path.name)
                        except OSError:
                            shutil.copyfile(out_path, latest_path)
                    merged_counts = summarize_route_counts(merged_records)
                    record_count = len(merged_records)
            cleanup_shards(shard_paths)
            atomic_write_json(
                result_path,
                {
                    "ok": True,
                    "route_counts": merged_counts,
                    "record_count": record_count,
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
        description=f"route export merge result for {out_path.name}",
    )
    result_payload = load_json(result_path)
    if not result_payload.get("ok", False):
        raise RuntimeError(result_payload.get("error", "Distributed OPSD export failed."))
    return {
        "route_counts": result_payload["route_counts"],
        "record_count": int(result_payload["record_count"]),
        "world_size": int(result_payload["world_size"]),
    }


def export_routes_from_runner(
    *,
    runner,
    out_path: str,
    global_step: int,
    route_model: str = "teacher",
    limit: int = None,
    consumed_sample_keys: Optional[Sequence[str]] = None,
):
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    cfg = runner.cfg
    samples = collect_refcoco_samples_from_cfg(cfg, limit=limit)
    dataset = getattr(getattr(getattr(runner, "train_loop", None), "dataloader", None), "dataset", None)
    existing_manifest_path = None
    if dataset is not None and hasattr(dataset, "resolve_active_route_manifest_path"):
        existing_manifest_path = dataset.resolve_active_route_manifest_path()
    export_summary = export_routes(
        model=model,
        samples=samples,
        out_path=out_path,
        global_step=global_step,
        route_model=route_model,
        update_latest=True,
        consumed_sample_keys=consumed_sample_keys,
        existing_manifest_path=existing_manifest_path,
    )
    if is_rank0():
        runner.logger.info(
            "OPSD route export finished: step=%s route_model=%s world_size=%s records=%s consumed=%s route_counts=%s out=%s",
            global_step,
            route_model,
            export_summary["world_size"],
            export_summary["record_count"],
            len(consumed_sample_keys or []),
            export_summary["route_counts"],
            out_path,
        )
    return export_summary["route_counts"]


def main():
    args = parse_args()
    maybe_init_distributed(args)
    print_summary = is_rank0()
    cfg = load_config(args.config, cfg_options=args.cfg_options)
    export_summary = None
    try:
        model = build_model_from_cfg(cfg)
        load_checkpoint_if_needed(model, args.checkpoint)
        samples = collect_refcoco_samples_from_cfg(
            cfg,
            image_root=args.image_root,
            limit=args.limit,
        )
        export_summary = export_routes(
            model=model,
            samples=samples,
            out_path=args.out,
            global_step=args.global_step,
            route_model=args.route_model,
            update_latest=args.update_latest,
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
