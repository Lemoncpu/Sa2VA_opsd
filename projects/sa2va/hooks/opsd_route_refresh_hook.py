from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper


class OpsdRouteRefreshHook(Hook):
    """Refresh offline OPSD route cache every N per-rank train iterations."""

    priority = "LOW"

    def __init__(
        self,
        interval: int = 5000,
        route_cache_dir: str = "route_cache",
        route_model: str = "teacher",
        export_limit: int = None,
        restrict_manifest_to_active_window: bool = True,
    ):
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}.")
        self.interval = int(interval)
        self.route_cache_dir = route_cache_dir
        self.route_model = str(route_model)
        self.export_limit = export_limit
        self.restrict_manifest_to_active_window = bool(restrict_manifest_to_active_window)

    @staticmethod
    def _unwrap_model(runner):
        model = runner.model
        return model.module if is_model_wrapper(model) else model

    @staticmethod
    def _dist_is_initialized():
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _is_rank0(self) -> bool:
        return not self._dist_is_initialized() or torch.distributed.get_rank() == 0

    def _barrier(self):
        if self._dist_is_initialized():
            torch.distributed.barrier()

    def _get_dataset_and_sampler(self, runner):
        train_loop = getattr(runner, "train_loop", None)
        dataloader = getattr(train_loop, "dataloader", None)
        if dataloader is None:
            return train_loop, None, None
        return train_loop, getattr(dataloader, "dataset", None), getattr(dataloader, "sampler", None)

    @staticmethod
    def _extract_batch_sample_keys(data_batch):
        if not isinstance(data_batch, dict):
            return []
        data = data_batch.get("data")
        if not isinstance(data, dict):
            return []
        sample_keys = data.get("sample_keys")
        if not sample_keys:
            return []
        return [str(sample_key) for sample_key in sample_keys if sample_key]

    def _mark_batch_consumed(self, runner, data_batch) -> None:
        sample_keys = self._extract_batch_sample_keys(data_batch)
        if not sample_keys:
            return
        _, _, sampler = self._get_dataset_and_sampler(runner)
        if sampler is not None and hasattr(sampler, "mark_consumed_sample_keys"):
            sampler.mark_consumed_sample_keys(sample_keys)

    def _get_global_consumed_sample_keys(self, runner):
        _, _, sampler = self._get_dataset_and_sampler(runner)
        if sampler is None or not hasattr(sampler, "get_consumed_sample_keys"):
            return []
        local_sample_keys = list(sampler.get_consumed_sample_keys())
        if not self._dist_is_initialized():
            return local_sample_keys
        gathered_sample_keys = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_sample_keys, local_sample_keys)
        merged = set()
        for sample_keys in gathered_sample_keys:
            for sample_key in sample_keys or []:
                if sample_key:
                    merged.add(str(sample_key))
        merged_list = sorted(merged)
        if hasattr(sampler, "replace_consumed_sample_keys"):
            sampler.replace_consumed_sample_keys(merged_list)
        return merged_list

    def _build_export_paths(self, runner, global_step: int):
        cache_dir = Path(runner.work_dir) / self.route_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / f"routes_step_{global_step:07d}.jsonl"
        latest_path = cache_dir / "routes_latest.jsonl"
        return manifest_path, latest_path

    def _resolve_active_window_size(self, runner) -> int:
        train_loop, _, sampler = self._get_dataset_and_sampler(runner)
        dataloader = getattr(train_loop, "dataloader", None)
        if sampler is None and dataloader is None:
            return self.interval
        world_size = int(getattr(sampler, "world_size", 1) or 1) if sampler is not None else 1
        per_iter_batch_size = int(getattr(dataloader, "batch_size", 1) or 1) if dataloader is not None else 1
        return max(self.interval * world_size * per_iter_batch_size, 1)

    def _export_routes(self, runner, global_step: int):
        from tools.export_opsd_routes import export_routes_from_runner

        consumed_sample_keys = self._get_global_consumed_sample_keys(runner)
        manifest_path, _ = self._build_export_paths(runner, global_step)
        active_window_size = None
        if self.restrict_manifest_to_active_window:
            active_window_size = self._resolve_active_window_size(runner)
        route_counts = export_routes_from_runner(
            runner=runner,
            out_path=str(manifest_path),
            global_step=global_step,
            route_model=self.route_model,
            limit=self.export_limit,
            consumed_sample_keys=consumed_sample_keys,
            active_window_size=active_window_size,
            restrict_manifest_to_active_window=self.restrict_manifest_to_active_window,
        )
        if self._is_rank0():
            exported_route_count = int(sum(route_counts.values())) if route_counts else 0
            runner.logger.info(
                "Exported OPSD route manifest to %s with exported_routes=%s after excluding %s consumed samples; active_window_size=%s",
                manifest_path,
                exported_route_count,
                len(consumed_sample_keys),
                active_window_size,
            )

    def before_train_epoch(self, runner) -> None:
        train_loop, _, sampler = self._get_dataset_and_sampler(runner)
        if train_loop is None:
            return
        if sampler is not None and hasattr(sampler, "reset_consumed_sample_keys"):
            sampler.reset_consumed_sample_keys()
        dataloader = getattr(train_loop, "dataloader", None)
        if dataloader is None:
            return
        self._refresh_dataset_and_log(runner, train_loop, dataloader)

    def _refresh_dataset_and_log(self, runner, train_loop, dataloader) -> None:
        dataset = getattr(dataloader, "dataset", None)
        refresh_fn = getattr(dataset, "refresh_route_manifest_if_needed", None)
        if callable(refresh_fn):
            refresh_fn(force=True)
        sampler = getattr(dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(getattr(train_loop, "_epoch", 0))
        if dataset is not None and hasattr(dataset, "get_route_distribution"):
            route_distribution = dataset.get_route_distribution()
            runner.logger.info(
                "Active OPSD route manifest: %s route_counts=%s",
                getattr(dataset, "active_route_manifest_path", None),
                route_distribution,
            )

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        del batch_idx, outputs
        # runner.iter is the per-process cumulative train-iteration counter in mmengine.
        # In DDP all ranks advance it in lockstep, so interval=5000 means 5000 iterations
        # on one rank rather than 5000 optimizer steps after gradient accumulation.
        self._mark_batch_consumed(runner, data_batch)
        refresh_iter = int(getattr(runner, "iter", -1)) + 1
        if refresh_iter <= 0 or refresh_iter % self.interval != 0:
            return
        self._export_routes(runner, refresh_iter)
        train_loop = getattr(runner, "train_loop", None)
        dataloader = getattr(train_loop, "dataloader", None)
        if dataloader is not None:
            self._refresh_dataset_and_log(runner, train_loop, dataloader)
