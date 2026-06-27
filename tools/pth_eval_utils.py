from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from mmengine.config import Config
from xtuner.registry import BUILDER


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    from xtuner.model.utils import guess_load_checkpoint

    cfg = Config.fromfile(config_path)
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
