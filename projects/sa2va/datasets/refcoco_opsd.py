import gzip
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from .sa2va_opsd_npz_v2 import DEFAULT_STUDENT_QUESTION


VALID_OPSD_ROUTES = {
    "teacher_regenerate",
    "on_policy_distill",
    "grpo_positive",
}
SKIP_OPSD_ROUTE = "skip"


DATASET_META = {
    "refcoco": {"splitBy": "unc", "dataset_name": "refcoco"},
    "refcoco_plus": {"splitBy": "unc", "dataset_name": "refcoco+"},
    "refcoco+": {"splitBy": "unc", "dataset_name": "refcoco+"},
    "refcocog": {"splitBy": "umd", "dataset_name": "refcocog"},
}


def _resolve_dataset_meta(dataset_name: str) -> Dict[str, str]:
    if dataset_name not in DATASET_META:
        raise KeyError(f"Unsupported RefCOCO dataset: {dataset_name}")
    return DATASET_META[dataset_name]


def decode_refcoco_mask(ann, height: int, width: int) -> np.ndarray:
    if len(ann["segmentation"]) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(ann["segmentation"][0], list):
        rles = mask_utils.frPyObjects(ann["segmentation"], height, width)
    else:
        rles = ann["segmentation"]
        for item in rles:
            if not isinstance(item["counts"], bytes):
                item["counts"] = item["counts"].encode()
    mask = mask_utils.decode(rles)
    if mask.ndim == 3:
        mask = np.sum(mask, axis=2)
    return (mask > 0).astype(np.uint8)


def resolve_refcoco_image_root(
    *,
    data_root: str,
    image_root: str = None,
    image_root_candidates: List[str] = None,
) -> str:
    if image_root:
        return image_root
    candidates = [
        os.path.join(data_root, "refcoco", "train2014"),
        os.path.join(data_root, "images/mscoco/images/train2014"),
        *(image_root_candidates or []),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "No valid RefCOCO image_root found. "
        "Pass image_root explicitly or provide image_root_candidates."
    )


def build_refcoco_opsd_records(
    *,
    data_root: str,
    dataset_name: str = "refcoco",
    split: str = "val",
    image_root: str = None,
    image_root_candidates: List[str] = None,
    skip_empty_masks: bool = True,
    skip_missing_images: bool = True,
) -> Tuple[List[Dict], str]:
    from projects.sa2va.evaluation.utils.refcoco_refer import REFER

    meta = _resolve_dataset_meta(dataset_name)
    resolved_image_root = resolve_refcoco_image_root(
        data_root=data_root,
        image_root=image_root,
        image_root_candidates=image_root_candidates,
    )
    refer = REFER(data_root, meta["dataset_name"], meta["splitBy"])
    ref_ids = refer.getRefIds(split=split)
    refs = refer.loadRefs(ref_ids=ref_ids)

    records = []
    for ref in refs:
        image_info = refer.loadImgs(image_ids=[ref["image_id"]])[0]
        ann = refer.refToAnn[ref["ref_id"]]
        gt_mask = decode_refcoco_mask(ann, image_info["height"], image_info["width"])
        if skip_empty_masks and int(gt_mask.sum()) == 0:
            continue
        image_path = os.path.join(resolved_image_root, image_info["file_name"])
        if skip_missing_images and not os.path.exists(image_path):
            continue
        record = {
            "sample_key": f"{dataset_name}:{split}:ref_id={ref['ref_id']}",
            "image_path": image_path,
            "gt_mask": gt_mask,
            "meta": {
                "ref_id": ref["ref_id"],
                "ann_id": ref["ann_id"],
                "image_id": ref["image_id"],
                "image_path": image_path,
                "ref_sentences": [sent["sent"] for sent in ref.get("sentences", [])],
            },
        }
        records.append(record)
    return records, resolved_image_root


class Sa2VAOpsdRefCocoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_name: str = "refcoco",
        split: str = "train",
        image_root: str = None,
        image_root_candidates: List[str] = None,
        student_question: str = DEFAULT_STUDENT_QUESTION,
        repeats: int = 1,
        max_records: Optional[int] = None,
        shuffle: bool = False,
        max_refetch: int = 20,
        skip_empty_masks: bool = True,
        skip_missing_images: bool = True,
        route_manifest_path: str = None,
        route_manifest_latest_path: str = None,
        route_manifest_required: bool = False,
        skip_route_manifest_skip_samples: bool = True,
        sam_confuser_pool_dir: str = None,
        min_confuser_candidate_count: int = 3,
        sam_confuser_duplicate_iou_threshold: float = 0.95,
        sam_confuser_min_area_ratio: float = 0.001,
        sam_confuser_max_area_ratio: float = 0.95,
        sam_confuser_nearby_center_weight: float = 0.25,
        sam_confuser_overlap_weight: float = 1.0,
        **kwargs,
    ):
        del kwargs
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split = split
        self.student_question = student_question
        self.repeats = repeats
        self.max_records = max_records
        self.shuffle = shuffle
        self.max_refetch = max_refetch
        self.skip_empty_masks = skip_empty_masks
        self.skip_missing_images = skip_missing_images
        self.route_manifest_path = route_manifest_path
        self.route_manifest_latest_path = route_manifest_latest_path
        self.route_manifest_required = route_manifest_required
        self.skip_route_manifest_skip_samples = skip_route_manifest_skip_samples
        self.sam_confuser_pool_dir = sam_confuser_pool_dir
        self.min_confuser_candidate_count = max(int(min_confuser_candidate_count), 0)
        self.sam_confuser_duplicate_iou_threshold = float(sam_confuser_duplicate_iou_threshold)
        self.sam_confuser_min_area_ratio = float(sam_confuser_min_area_ratio)
        self.sam_confuser_max_area_ratio = float(sam_confuser_max_area_ratio)
        self.sam_confuser_nearby_center_weight = float(sam_confuser_nearby_center_weight)
        self.sam_confuser_overlap_weight = float(sam_confuser_overlap_weight)
        self.active_route_manifest_path = None
        self.route_manifest_mtime = None
        self.route_manifest_version = 0
        self.route_info_by_key = {}
        self._sam_confuser_pool_cache: Dict[int, List[Dict]] = {}
        self._base_records, self.image_root = build_refcoco_opsd_records(
            data_root=data_root,
            dataset_name=dataset_name,
            split=split,
            image_root=image_root,
            image_root_candidates=image_root_candidates,
            skip_empty_masks=skip_empty_masks,
            skip_missing_images=skip_missing_images,
        )
        if self.max_records is not None:
            if self.max_records <= 0:
                raise ValueError(f"max_records must be positive, got {self.max_records}.")
            self._base_records = self._base_records[: self.max_records]
        if not self._base_records:
            raise ValueError(
                f"No RefCOCO OPSD records found for dataset={dataset_name}, split={split}, image_root={self.image_root}."
            )
        if shuffle:
            random.shuffle(self._base_records)
        self._records_by_image_id = defaultdict(list)
        for record in self._base_records:
            self._records_by_image_id[record["meta"]["image_id"]].append(record)
        self.records = list(self._base_records)
        self.load_route_manifest(required=self.route_manifest_required)
        self._apply_route_manifest_filter()

    @property
    def modality_length(self):
        return [100] * len(self)

    def resolve_active_route_manifest_path(self) -> Optional[str]:
        latest_path = self.route_manifest_latest_path
        if latest_path and os.path.exists(latest_path):
            return latest_path
        return self.route_manifest_path

    @staticmethod
    def load_route_manifest_file(path: str) -> Dict[str, Dict]:
        info_by_key = {}
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                sample_key = item.get("sample_key")
                if not sample_key:
                    raise ValueError(f"Missing sample_key in route manifest {path}:{line_no}")
                route = item.get("route")
                if route not in VALID_OPSD_ROUTES and route != "skip":
                    raise ValueError(
                        f"Invalid route {route!r} for sample_key={sample_key!r} in {path}:{line_no}"
                    )
                info_by_key[sample_key] = item
        return info_by_key

    def load_route_manifest(self, required: bool = False):
        path = self.resolve_active_route_manifest_path()
        self.route_info_by_key = {}
        self.active_route_manifest_path = path
        self.route_manifest_mtime = None
        if not path:
            if required:
                raise FileNotFoundError("route_manifest_path is required but was not set.")
            return
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Route manifest does not exist: {path}")
            return
        self.route_info_by_key = self.load_route_manifest_file(path)
        self.route_manifest_mtime = os.path.getmtime(path)
        self.route_manifest_version += 1

    def _apply_route_manifest_filter(self):
        if self.skip_route_manifest_skip_samples and self.route_info_by_key:
            self.records = [
                record
                for record in self._base_records
                if self.get_route_for_sample_key(record["sample_key"]) != SKIP_OPSD_ROUTE
            ]
            if not self.records:
                raise ValueError(
                    "All RefCOCO OPSD records were filtered by route manifest: "
                    f"{self.active_route_manifest_path or self.route_manifest_path}"
                )
            return
        self.records = list(self._base_records)

    def refresh_route_manifest_if_needed(self, force: bool = False):
        path = self.resolve_active_route_manifest_path()
        if not path or not os.path.exists(path):
            return
        mtime = os.path.getmtime(path)
        if (
            force
            or self.active_route_manifest_path != path
            or self.route_manifest_mtime is None
            or mtime != self.route_manifest_mtime
        ):
            self.load_route_manifest(required=self.route_manifest_required)
            self._apply_route_manifest_filter()

    def get_route_manifest_version(self) -> int:
        return int(self.route_manifest_version)

    def get_route_for_sample_key(self, sample_key: str):
        info = self.route_info_by_key.get(sample_key, {})
        return info.get("route")

    def get_sample_key_for_index(self, index: int) -> str:
        record = self.records[index % len(self.records)]
        return record["sample_key"]

    def get_route_info_for_index(self, index: int) -> Dict:
        record = self.records[index % len(self.records)]
        return self.route_info_by_key.get(record["sample_key"], {})

    def get_route_for_index(self, index: int):
        return self.get_route_info_for_index(index).get("route")

    def get_route_distribution(self) -> Dict[str, int]:
        route_counts: Dict[str, int] = {}
        for record in self.records:
            route = self.get_route_for_sample_key(record["sample_key"])
            route_key = route or "missing"
            route_counts[route_key] = route_counts.get(route_key, 0) + 1
        return route_counts

    def __len__(self):
        return len(self.records) * self.repeats

    @staticmethod
    def _to_prompt_masks(mask: torch.Tensor) -> np.ndarray:
        return np.expand_dims(mask.cpu().numpy().astype(np.float32), axis=0)

    @staticmethod
    def _decode_coco_rle(rle: Dict) -> np.ndarray:
        encoded = dict(rle)
        counts = encoded.get("counts")
        if isinstance(counts, str):
            encoded["counts"] = counts.encode()
        mask = mask_utils.decode(encoded)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        return (mask > 0).astype(np.uint8)

    @staticmethod
    def _mask_area_ratio(mask: np.ndarray) -> float:
        mask = np.asarray(mask)
        if mask.ndim != 2:
            return 0.0
        return float((mask > 0).sum()) / float(max(mask.shape[0] * mask.shape[1], 1))

    @staticmethod
    def _mask_center(mask: np.ndarray):
        mask = np.asarray(mask)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        height, width = mask.shape
        return (float(xs.mean()) / max(width, 1), float(ys.mean()) / max(height, 1))

    @staticmethod
    def _mask_center_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        center_a = Sa2VAOpsdRefCocoDataset._mask_center(mask_a)
        center_b = Sa2VAOpsdRefCocoDataset._mask_center(mask_b)
        if center_a is None or center_b is None:
            return 1.0
        return float(((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5)

    @staticmethod
    def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        mask_a = np.asarray(mask_a) > 0
        mask_b = np.asarray(mask_b) > 0
        intersection = np.logical_and(mask_a, mask_b).sum(dtype=np.int64)
        union = np.logical_or(mask_a, mask_b).sum(dtype=np.int64)
        if union == 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def _prepare_mask_like_gt(gt_mask: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
        gt_mask = (np.asarray(gt_mask) > 0).astype(np.uint8)
        candidate_mask = (np.asarray(candidate_mask) > 0).astype(np.uint8)
        if gt_mask.shape != candidate_mask.shape:
            candidate_mask_t = torch.from_numpy(candidate_mask[None, None].astype(np.float32))
            candidate_mask_t = torch.nn.functional.interpolate(
                candidate_mask_t, size=gt_mask.shape, mode="nearest"
            )[0, 0]
            candidate_mask = (candidate_mask_t.numpy() > 0).astype(np.uint8)
        return candidate_mask

    def _score_confuser_candidate(self, gt_mask: np.ndarray, candidate_mask: np.ndarray) -> float:
        overlap_iou = self._compute_iou(gt_mask, candidate_mask)
        center_distance = self._mask_center_distance(gt_mask, candidate_mask)
        return (
            self.sam_confuser_overlap_weight * overlap_iou
            - self.sam_confuser_nearby_center_weight * center_distance
        )

    def _candidate_pool_dataset_names(self) -> List[str]:
        meta_dataset_name = _resolve_dataset_meta(self.dataset_name)["dataset_name"]
        candidates = [self.dataset_name, meta_dataset_name]
        normalized_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                normalized_candidates.append(candidate)
        return normalized_candidates

    def _resolve_sam_confuser_pool_path(self, image_id: int) -> Optional[str]:
        if not self.sam_confuser_pool_dir:
            return None
        for dataset_dir in self._candidate_pool_dataset_names():
            candidate = os.path.join(
                self.sam_confuser_pool_dir, dataset_dir, self.split, f"{int(image_id)}.json.gz"
            )
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_sam_confuser_pool(self, image_id: int) -> List[Dict]:
        image_id = int(image_id)
        if image_id in self._sam_confuser_pool_cache:
            return self._sam_confuser_pool_cache[image_id]
        pool_path = self._resolve_sam_confuser_pool_path(image_id)
        if not pool_path:
            self._sam_confuser_pool_cache[image_id] = []
            return self._sam_confuser_pool_cache[image_id]
        with gzip.open(pool_path, "rt", encoding="utf-8") as file:
            payload = json.load(file)
        masks = payload.get("masks", [])
        if not isinstance(masks, list):
            raise ValueError(f"Invalid SAM confuser pool format in {pool_path}: 'masks' must be a list.")
        self._sam_confuser_pool_cache[image_id] = masks
        return self._sam_confuser_pool_cache[image_id]

    def _select_sam_pool_confusers(
        self,
        *,
        image_id: int,
        gt_mask: np.ndarray,
        existing_candidate_masks: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[str]]:
        needed_count = max(self.min_confuser_candidate_count - len(existing_candidate_masks), 0)
        if needed_count <= 0:
            return [], []
        pool_masks = self._load_sam_confuser_pool(image_id)
        if not pool_masks:
            return [], []

        gt_mask = self._prepare_mask_like_gt(gt_mask, gt_mask)
        existing_prepared_masks = []
        for existing_mask in existing_candidate_masks:
            prepared_existing_mask = self._prepare_mask_like_gt(gt_mask, existing_mask)
            if int(prepared_existing_mask.sum()) == 0:
                continue
            existing_prepared_masks.append(prepared_existing_mask)

        scored_candidates = []
        for pool_index, pool_item in enumerate(pool_masks):
            segmentation = pool_item.get("segmentation")
            if not isinstance(segmentation, dict):
                continue
            prepared_mask = self._prepare_mask_like_gt(gt_mask, self._decode_coco_rle(segmentation))
            if int(prepared_mask.sum()) == 0:
                continue
            area_ratio = self._mask_area_ratio(prepared_mask)
            if area_ratio < self.sam_confuser_min_area_ratio or area_ratio > self.sam_confuser_max_area_ratio:
                continue
            if self._compute_iou(gt_mask, prepared_mask) >= self.sam_confuser_duplicate_iou_threshold:
                continue
            is_duplicate_with_existing = any(
                self._compute_iou(existing_mask, prepared_mask) >= self.sam_confuser_duplicate_iou_threshold
                for existing_mask in existing_prepared_masks
            )
            if is_duplicate_with_existing:
                continue
            scored_candidates.append(
                (
                    self._score_confuser_candidate(gt_mask, prepared_mask),
                    float(pool_item.get("predicted_iou", 0.0)),
                    float(pool_item.get("stability_score", 0.0)),
                    float(pool_item.get("area", 0.0)),
                    pool_index,
                    prepared_mask,
                )
            )

        scored_candidates.sort(reverse=True)
        selected_masks = []
        selected_ref_ids = []
        for _, _, _, _, pool_index, prepared_mask in scored_candidates:
            is_duplicate = any(
                self._compute_iou(existing_mask, prepared_mask) >= self.sam_confuser_duplicate_iou_threshold
                for existing_mask in selected_masks
            )
            if is_duplicate:
                continue
            selected_masks.append(prepared_mask)
            selected_ref_ids.append(f"sam_pool:image_id={int(image_id)}:idx={int(pool_index)}")
            if len(selected_masks) >= needed_count:
                break
        return selected_masks, selected_ref_ids

    def prepare_data(self, index: int):
        record = self.records[index % len(self.records)]
        route_info = self.route_info_by_key.get(record["sample_key"], {})
        with Image.open(record["image_path"]) as pil_image:
            image = pil_image.convert("RGB")
        gt_mask = torch.from_numpy(np.asarray(record["gt_mask"]).astype(np.uint8))
        if self.skip_empty_masks and int(gt_mask.sum().item()) == 0:
            raise ValueError(f"Empty ground-truth mask in {record['sample_key']}")
        confuser_candidate_masks = []
        confuser_candidate_ref_ids = []
        for peer_record in self._records_by_image_id.get(record["meta"]["image_id"], []):
            if peer_record["sample_key"] == record["sample_key"]:
                continue
            confuser_candidate_masks.append(np.asarray(peer_record["gt_mask"]).astype(np.uint8))
            confuser_candidate_ref_ids.append(peer_record["meta"]["ref_id"])
        sam_pool_confuser_masks = []
        sam_pool_confuser_ref_ids = []
        if len(confuser_candidate_masks) < self.min_confuser_candidate_count:
            sam_pool_confuser_masks, sam_pool_confuser_ref_ids = self._select_sam_pool_confusers(
                image_id=record["meta"]["image_id"],
                gt_mask=np.asarray(record["gt_mask"]).astype(np.uint8),
                existing_candidate_masks=confuser_candidate_masks,
            )
        confuser_candidate_masks.extend(sam_pool_confuser_masks)
        confuser_candidate_ref_ids.extend(sam_pool_confuser_ref_ids)
        return {
            "image": image,
            "prompt_masks": self._to_prompt_masks(gt_mask),
            "student_question": self.student_question,
            "gt_mask": gt_mask,
            "confuser_candidate_masks": confuser_candidate_masks,
            "confuser_candidate_ref_ids": confuser_candidate_ref_ids,
            "num_original_confuser_candidates": len(confuser_candidate_masks) - len(sam_pool_confuser_masks),
            "num_sam_pool_confuser_candidates_added": len(sam_pool_confuser_masks),
            "sam_confuser_pool_dir": self.sam_confuser_pool_dir,
            "sample_key": record["sample_key"],
            "route": route_info.get("route"),
            "route_iou": route_info.get("iou"),
            "route_global_step": route_info.get("global_step"),
            "route_timestamp": route_info.get("timestamp"),
            "route_manifest_path": self.active_route_manifest_path or self.route_manifest_path,
            "npz_path": None,
            "frame1": None,
            "mask1": None,
            "frame2": None,
            "mask2": None,
            **record["meta"],
        }

    def _refetch_index(self, index, attempt):
        if self.shuffle:
            return random.randint(0, len(self.records) - 1)
        return (index + attempt + 1) % len(self.records)

    def __getitem__(self, index):
        for attempt in range(self.max_refetch + 1):
            try:
                return self.prepare_data(index)
            except Exception:
                index = self._refetch_index(index, attempt)
        raise RuntimeError(
            f"Failed to read a valid RefCOCO OPSD sample after {self.max_refetch + 1} retries."
        )
