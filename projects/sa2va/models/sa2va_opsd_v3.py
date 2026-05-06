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

    def __init__(
        self,
        *args,
        device="auto",
        disable_gradient_checkpointing_for_ddp=False,
        enable_ddp_route_safety_loss=True,
        **kwargs,
    ):
        resolved_device = self._resolve_runtime_device(device)
        if resolved_device.type == "cuda":
            torch.cuda.set_device(resolved_device)
        super().__init__(*args, device=resolved_device, **kwargs)
        self.disable_gradient_checkpointing_for_ddp = bool(disable_gradient_checkpointing_for_ddp)
        self.enable_ddp_route_safety_loss = bool(enable_ddp_route_safety_loss)
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

    @staticmethod
    def _ddp_is_initialized():
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _ddp_route_safety_loss(self):
        safety_loss = None
        for parameter in self.parameters():
            if not parameter.requires_grad or parameter.numel() == 0:
                continue
            term = parameter.reshape(-1)[0] * 0.0
            safety_loss = term if safety_loss is None else safety_loss + term
        if safety_loss is None:
            safety_loss = next(self.student_model.parameters()).sum() * 0.0
        return safety_loss

    def forward(self, data, data_samples=None, mode="loss"):
        batch_route = self._resolve_batch_route(data.get("routes", []))
        metrics = super().forward(data, data_samples=data_samples, mode=mode)
        if isinstance(metrics, dict):
            metrics.setdefault("batch_route", batch_route or "")
        if (
            self.enable_ddp_route_safety_loss
            and self.training
            and self._ddp_is_initialized()
            and isinstance(metrics, dict)
            and "loss_opsd_total" in metrics
        ):
            metrics["loss_opsd_total"] = metrics["loss_opsd_total"] + self._ddp_route_safety_loss()
        return metrics
