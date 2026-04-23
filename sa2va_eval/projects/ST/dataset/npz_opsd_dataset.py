import glob
import os
import random
from typing import List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_STUDENT_QUESTION = (
    "<image>\nPlease provide a detailed description of the region marked by <Prompt0>."
)


class Sa2VAOpsdNPZDataset(Dataset):
    """Single-frame NPZ dataset for Sa2VA OPSD training.

    Each sample uses only frame1 and mask1. frame2/mask2 are intentionally ignored.
    """

    def __init__(
        self,
        npz_dir: str,
        prefix: str = "masklet_data",
        student_question: str = DEFAULT_STUDENT_QUESTION,
        shuffle: bool = False,
        repeats: int = 1,
        max_refetch: int = 20,
        **kwargs,
    ):
        del kwargs
        self.npz_dir = npz_dir
        self.prefix = prefix
        self.student_question = student_question
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

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise ValueError(f"Unsupported mask1 shape: {mask.shape}")
        return (mask > 0).astype(np.uint8)

    def prepare_data(self, index: int):
        index = index % len(self.npz_files)
        npz_path = self.npz_files[index]
        data = np.load(npz_path)

        image = self._normalize_frame(data["frame1"])
        mask = self._normalize_mask(data["mask1"])

        return {
            "image": image,
            "prompt_masks": [mask],
            "prompt_ids": [0],
            "student_question": self.student_question,
            "npz_path": npz_path,
            "gt_mask": mask,
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
