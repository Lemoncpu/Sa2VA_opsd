import glob
import os
import random
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_STUDENT_QUESTION = (
    "<image>"
    "Can you provide me with a concise description of the region in the picture marked by region1? "
    "Answer with a short referring expression in RefCOCO style, usually 2 to 6 words, naming the target itself "
    "with only the most necessary attribute or location cue. Prefer forms like category plus color, size, "
    "left/right, top/bottom, or a nearby relation. Do not describe the whole scene. Do not write a full sentence "
    "and do not start with 'it is'. Do not output [SEG], masks, tags, or placeholder tokens."
)


class Sa2VAOpsdNPZDataset(Dataset):
    """Single-frame NPZ dataset for InternVL3-2B Sa2VA OPSD training."""

    def __init__(
        self,
        npz_dir: str,
        prefix: str = "masklet_data",
        student_question: str = DEFAULT_STUDENT_QUESTION,
        transform=None,
        shuffle: bool = False,
        repeats: int = 1,
        max_refetch: int = 20,
        **kwargs,
    ):
        del kwargs
        self.npz_dir = npz_dir
        self.prefix = prefix
        self.student_question = student_question
        self.transform = transform
        self.repeats = repeats
        self._max_refetch = max_refetch

        pattern = os.path.join(npz_dir, "**", f"{prefix}_*.npz")
        self.npz_files: List[str] = glob.glob(pattern, recursive=True)
        if not self.npz_files:
            raise ValueError(f"No NPZ files matching '{pattern}' were found.")

        if shuffle:
            random.shuffle(self.npz_files)

    @property
    def modality_length(self):
        return [100] * len(self)

    def __len__(self):
        return len(self.npz_files) * self.repeats

    @staticmethod
    def _load_raw_npz(npz_path: str):
        data = np.load(npz_path)
        frame1 = torch.from_numpy(data["frame1"]).float()
        mask1 = torch.from_numpy(data["mask1"]).float()
        frame2 = torch.from_numpy(data["frame2"]).float()
        mask2 = torch.from_numpy(data["mask2"]).float()
        return (frame1, mask1), (frame2, mask2)

    def _normalize_frame(self, frame: np.ndarray) -> Image.Image:
        frame = np.asarray(frame)
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        if frame.ndim != 3:
            raise ValueError(f"Unsupported frame1 shape: {frame.shape}")

        if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))

        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        if frame.shape[-1] != 3:
            raise ValueError(f"Unsupported frame1 channel layout: {frame.shape}")

        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0) if frame.max() <= 1.0 else np.clip(frame, 0.0, 255.0)
            if frame.max() <= 1.0:
                frame = frame * 255.0
        frame = frame.astype(np.uint8)
        return Image.fromarray(frame).convert("RGB")

    def _normalize_mask(self, mask: np.ndarray) -> torch.Tensor:
        mask = np.asarray(mask)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise ValueError(f"Unsupported mask1 shape: {mask.shape}")
        return torch.from_numpy((mask > 0).astype(np.uint8))

    def prepare_data(self, index: int):
        index = index % len(self.npz_files)
        npz_path = self.npz_files[index]
        (frame1, mask1), (frame2, mask2) = self._load_raw_npz(npz_path)

        if self.transform:
            frame1, mask1 = self.transform(frame1, mask1)
            frame2, mask2 = self.transform(frame2, mask2)

        image = self._normalize_frame(frame1.cpu().numpy())
        mask = self._normalize_mask(mask1.cpu().numpy())
        next_frame = self._normalize_frame(frame2.cpu().numpy())
        next_mask = self._normalize_mask(mask2.cpu().numpy())

        return {
            "image": image,
            "prompt_masks": [mask],
            "student_question": self.student_question,
            "gt_mask": mask,
            "frame1": frame1,
            "mask1": mask1,
            "frame2": frame2,
            "mask2": mask2,
            "next_image": next_frame,
            "next_gt_mask": next_mask,
            "npz_path": npz_path,
        }

    def _rand_another(self):
        return random.randint(0, len(self.npz_files) - 1)

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            try:
                return self.prepare_data(index)
            except Exception:
                index = self._rand_another()
        raise RuntimeError(f"Failed to read a valid NPZ sample after {self._max_refetch + 1} retries.")
