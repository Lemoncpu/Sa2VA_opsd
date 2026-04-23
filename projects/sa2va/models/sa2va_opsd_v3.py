import os

import torch

from projects.sa2va.models.sa2va_opsd_v2 import Sa2VAOPSDModelV2


class Sa2VAOPSDModelV3(Sa2VAOPSDModelV2):
    """DDP-friendly OPSD wrapper built on top of the v2 implementation."""

    @staticmethod
    def _resolve_runtime_device(device):
        if isinstance(device, torch.device):
            return device

        if device is None or device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for Sa2VA_OPSD V3 training.")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            return torch.device(f"cuda:{local_rank}")

        return torch.device(device)

    def __init__(self, *args, device="auto", disable_gradient_checkpointing_for_ddp=False, **kwargs):
        resolved_device = self._resolve_runtime_device(device)
        if resolved_device.type == "cuda":
            torch.cuda.set_device(resolved_device)
        super().__init__(*args, device=resolved_device, **kwargs)
        self.disable_gradient_checkpointing_for_ddp = bool(disable_gradient_checkpointing_for_ddp)
        if self.disable_gradient_checkpointing_for_ddp:
            self._disable_gradient_checkpointing_for_ddp()

    def _disable_gradient_checkpointing_for_ddp(self):
        for model in (self.student_model, self.teacher_model):
            if model is None:
                continue
            disable_fn = getattr(model, "gradient_checkpointing_disable", None)
            if callable(disable_fn):
                disable_fn()

            language_model = getattr(model, "language_model", None)
            disable_fn = getattr(language_model, "gradient_checkpointing_disable", None)
            if callable(disable_fn):
                disable_fn()

            vision_model = getattr(model, "vision_model", None)
            disable_fn = getattr(vision_model, "gradient_checkpointing_disable", None)
            if callable(disable_fn):
                disable_fn()
