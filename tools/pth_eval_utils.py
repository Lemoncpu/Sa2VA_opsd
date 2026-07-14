from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_CONFIG = (
    ROOT / "projects/sa2va/configs/sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3.py"
)
DEFAULT_REFERRING_MODEL_CONFIG = (
    ROOT / "projects/sa2va/configs/sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3_referring.py"
)


def _infer_model_config_path(config_path: str, checkpoint_path: str) -> str:
    override = os.environ.get("SA2VA_PTH_EVAL_MODEL_CONFIG")
    if override:
        return override

    config_name = Path(config_path).name.lower()
    checkpoint_hint = str(checkpoint_path).lower()
    joined_hint = f"{config_name} {checkpoint_hint}"
    if "referring" in joined_hint or "fault_report" in joined_hint:
        return str(DEFAULT_REFERRING_MODEL_CONFIG)
    return str(DEFAULT_MODEL_CONFIG)


def load_opsd_model_from_pth(
    *,
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda:0",
    base_model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    enable_teacher: bool = False,
    grpo_group_size: int = 0,
    use_flash_attn: bool = True,
    min_caption_tokens: int = 4,
):
    from tools.train_fallback_compat import (
        guess_load_checkpoint as fallback_guess_load_checkpoint,
        install_xtuner_fallback_modules,
        patch_mmengine_adafactor_duplicate_registration,
    )

    patch_mmengine_adafactor_duplicate_registration()
    try:
        from xtuner.registry import BUILDER
    except Exception:
        install_xtuner_fallback_modules()
        from xtuner.registry import BUILDER

    try:
        from xtuner.model.utils import guess_load_checkpoint
    except Exception:
        guess_load_checkpoint = fallback_guess_load_checkpoint

    cfg = Config.fromfile(config_path)
    if not cfg.get("model", None):
        model_config_path = _infer_model_config_path(config_path, checkpoint_path)
        cfg = Config.fromfile(model_config_path)
    model_cfg = cfg.model.copy()

    if base_model_path:
        model_cfg["model_path"] = base_model_path
        if hasattr(cfg, "path"):
            cfg.path = base_model_path
    if tokenizer_path:
        model_cfg["tokenizer_path"] = tokenizer_path
    elif base_model_path and model_cfg.get("tokenizer_path") == cfg.get("path", None):
        model_cfg["tokenizer_path"] = base_model_path

    model_cfg["enable_teacher"] = enable_teacher
    model_cfg["grpo_group_size"] = grpo_group_size
    model_cfg["device"] = device
    model_cfg["use_flash_attn"] = use_flash_attn
    model_cfg["min_caption_tokens"] = min_caption_tokens
    model_cfg["torch_dtype"] = "auto"

    model = BUILDER.build(model_cfg)
    state_dict = guess_load_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
