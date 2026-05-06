import math
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Sampler


class RouteGroupedSampler(Sampler[int]):
    """Groups indices by route so each per-rank batch contains one route only."""

    def __init__(
        self,
        dataset,
        per_device_batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        require_routes: bool = True,
        **kwargs,
    ):
        del kwargs
        if per_device_batch_size <= 0:
            raise ValueError(
                f"per_device_batch_size must be positive, got {per_device_batch_size}."
            )
        if not drop_last:
            raise ValueError(
                "RouteGroupedSampler requires drop_last=True to keep all ranks aligned."
            )
        self.dataset = dataset
        self.per_device_batch_size = int(per_device_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.require_routes = bool(require_routes)
        self.epoch = 0
        self.rank = self._resolve_rank(rank)
        self.world_size = self._resolve_world_size(world_size)

    @staticmethod
    def _dist_is_initialized() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _resolve_rank(self, rank: Optional[int]) -> int:
        if rank is not None:
            return int(rank)
        if self._dist_is_initialized():
            return int(torch.distributed.get_rank())
        return 0

    def _resolve_world_size(self, world_size: Optional[int]) -> int:
        if world_size is not None:
            return int(world_size)
        if self._dist_is_initialized():
            return int(torch.distributed.get_world_size())
        return 1

    @property
    def global_batch_size(self) -> int:
        return self.per_device_batch_size * self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _get_route(self, index: int) -> Optional[str]:
        route = None
        get_route_for_index = getattr(self.dataset, "get_route_for_index", None)
        if callable(get_route_for_index):
            route = get_route_for_index(index)
        elif hasattr(self.dataset, "route_info_by_key") and hasattr(self.dataset, "records"):
            record = self.dataset.records[index % len(self.dataset.records)]
            route = self.dataset.route_info_by_key.get(record["sample_key"], {}).get("route")
        if route == "skip":
            return None
        return route

    def _build_route_buckets(self) -> Dict[str, List[int]]:
        route_to_indices: Dict[str, List[int]] = {}
        for index in range(len(self.dataset)):
            route = self._get_route(index)
            if not route:
                if self.require_routes:
                    raise RuntimeError(
                        "RouteGroupedSampler requires dataset samples to provide a non-empty route. "
                        f"Missing route at dataset index {index}."
                    )
                continue
            route_to_indices.setdefault(str(route), []).append(index)
        return route_to_indices

    def _build_rank_indices(self) -> List[int]:
        route_to_indices = self._build_route_buckets()
        rng = random.Random(self.seed + self.epoch)
        route_batches = []

        for route, indices in route_to_indices.items():
            bucket = list(indices)
            if self.shuffle:
                rng.shuffle(bucket)
            usable_size = (len(bucket) // self.global_batch_size) * self.global_batch_size
            if usable_size <= 0:
                continue
            bucket = bucket[:usable_size]
            for start in range(0, len(bucket), self.global_batch_size):
                route_batches.append((route, bucket[start : start + self.global_batch_size]))

        if self.shuffle:
            rng.shuffle(route_batches)
        else:
            route_batches.sort(key=lambda item: item[0])

        rank_indices: List[int] = []
        local_start = self.rank * self.per_device_batch_size
        local_end = local_start + self.per_device_batch_size
        for _, global_batch in route_batches:
            rank_indices.extend(global_batch[local_start:local_end])
        return rank_indices

    def __iter__(self):
        return iter(self._build_rank_indices())

    def __len__(self) -> int:
        total_local_samples = 0
        for indices in self._build_route_buckets().values():
            total_local_samples += (
                len(indices) // self.global_batch_size
            ) * self.per_device_batch_size
        return total_local_samples
