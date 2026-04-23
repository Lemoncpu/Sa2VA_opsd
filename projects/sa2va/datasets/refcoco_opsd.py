import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from .sa2va_opsd_npz_v2 import DEFAULT_STUDENT_QUESTION


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
        records.append(
            {
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
        )
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
        self.records, self.image_root = build_refcoco_opsd_records(
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
            self.records = self.records[: self.max_records]
        if not self.records:
            raise ValueError(
                f"No RefCOCO OPSD records found for dataset={dataset_name}, split={split}, image_root={self.image_root}."
            )
        if shuffle:
            random.shuffle(self.records)

    @property
    def modality_length(self):
        return [100] * len(self)

    def __len__(self):
        return len(self.records) * self.repeats

    @staticmethod
    def _to_prompt_masks(mask: torch.Tensor) -> np.ndarray:
        return np.expand_dims(mask.cpu().numpy().astype(np.float32), axis=0)

    def prepare_data(self, index: int):
        record = self.records[index % len(self.records)]
        image = Image.open(record["image_path"]).convert("RGB")
        gt_mask = torch.from_numpy(np.asarray(record["gt_mask"]).astype(np.uint8))
        if self.skip_empty_masks and int(gt_mask.sum().item()) == 0:
            raise ValueError(f"Empty ground-truth mask in {record['sample_key']}")
        return {
            "image": image,
            "prompt_masks": self._to_prompt_masks(gt_mask),
            "student_question": self.student_question,
            "gt_mask": gt_mask,
            "sample_key": record["sample_key"],
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
