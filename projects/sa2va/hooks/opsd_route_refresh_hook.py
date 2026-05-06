import os
import shutil
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
    ):
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}.")
        self.interval = int(interval)
        self.route_cache_dir = route_cache_dir
        self.route_model = str(route_model)
        self.export_limit = export_limit

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

    def _sync_success_or_raise(self, runner, ok: bool):
        if not self._dist_is_initialized():
            if not ok:
                raise RuntimeError("OPSD route refresh failed on rank0.")
            return
        status = torch.tensor([1 if ok else 0], device=self._unwrap_model(runner).device)
        torch.distributed.broadcast(status, src=0)
        if int(status.item()) != 1:
            raise RuntimeError("OPSD route refresh failed on rank0; aborting all ranks.")

    def _build_export_paths(self, runner, global_step: int):
        cache_dir = Path(runner.work_dir) / self.route_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / f"routes_step_{global_step:07d}.jsonl"
        latest_path = cache_dir / "routes_latest.jsonl"
        return manifest_path, latest_path

    def _export_routes(self, runner, global_step: int):
        from tools.export_opsd_routes import export_routes_from_runner

        manifest_path, latest_path = self._build_export_paths(runner, global_step)
        export_routes_from_runner(
            runner=runner,
            out_path=str(manifest_path),
            global_step=global_step,
            route_model=self.route_model,
            limit=self.export_limit,
        )
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        try:
            latest_path.symlink_to(manifest_path.name)
        except OSError:
            shutil.copyfile(manifest_path, latest_path)
        runner.logger.info(f"Exported OPSD route manifest to {manifest_path}")

    def before_train_epoch(self, runner) -> None:
        train_loop = getattr(runner, "train_loop", None)
        dataloader = getattr(train_loop, "dataloader", None)
        if dataloader is None:
            return
        self._refresh_dataset_and_log(runner, dataloader)

    def _refresh_dataset_and_log(self, runner, dataloader) -> None:
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
        del batch_idx, data_batch, outputs
        # runner.iter is the per-process cumulative train-iteration counter in mmengine.
        # In DDP all ranks advance it in lockstep, so interval=5000 means 5000 iterations
        # on one rank rather than 5000 optimizer steps after gradient accumulation.
        refresh_iter = int(getattr(runner, "iter", -1)) + 1
        if refresh_iter <= 0 or refresh_iter % self.interval != 0:
            return
        export_ok = True
        if self._is_rank0():
            try:
                self._export_routes(runner, refresh_iter)
            except Exception:
                export_ok = False
                runner.logger.exception("Failed to export OPSD route manifest at iter=%s", refresh_iter)
        self._sync_success_or_raise(runner, export_ok)
        train_loop = getattr(runner, "train_loop", None)
        dataloader = getattr(train_loop, "dataloader", None)
        if dataloader is not None:
            self._refresh_dataset_and_log(runner, dataloader)
