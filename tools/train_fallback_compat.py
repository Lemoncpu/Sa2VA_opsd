import importlib
import importlib.machinery
import logging
import os
import os.path as osp
import sys
import types
from copy import deepcopy
from typing import Any

import torch


TARGET_FALLBACK_CHAIN = "refcoco_opsd_4b"
TARGET_CONFIG_BASENAME = "sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3.py"


def is_supported_fallback_target(config_path: str) -> bool:
    if os.environ.get("SA2VA_TRAIN_FALLBACK_CHAIN") == TARGET_FALLBACK_CHAIN:
        return True
    return osp.basename(str(config_path)) == TARGET_CONFIG_BASENAME


def patch_mmengine_adafactor_duplicate_registration() -> None:
    try:
        from mmengine.registry.registry import Registry
    except Exception:
        return

    if getattr(Registry, "_sa2va_adafactor_duplicate_patch", False):
        return

    original_register_module = Registry._register_module

    def patched_register_module(self, module, module_name=None, force=False):
        names = module_name
        if names is None:
            names = [module.__name__]
        elif isinstance(names, str):
            names = [names]

        if not force and self.name == "optimizer":
            for name in names:
                if name != "Adafactor":
                    continue
                for existing in (
                    self._module_dict.get(name),
                    getattr(self, "get", lambda _: None)(name),
                ):
                    if existing is None:
                        continue
                    if existing is module:
                        return
                    if (
                        getattr(existing, "__name__", None) == getattr(module, "__name__", None)
                        and getattr(existing, "__module__", None) == getattr(module, "__module__", None)
                    ):
                        return

        try:
            return original_register_module(self, module=module, module_name=module_name, force=force)
        except KeyError as exc:
            if (
                not force
                and self.name == "optimizer"
                and any(name == "Adafactor" for name in names)
                and "Adafactor is already registered in optimizer" in str(exc)
            ):
                return
            raise exc

    Registry._register_module = patched_register_module
    Registry._sa2va_adafactor_duplicate_patch = True


def _torch_load_compat(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def guess_load_checkpoint(path: str):
    checkpoint = _torch_load_compat(path)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def get_peft_model_state_dict(model, state_dict=None):
    if state_dict is not None:
        return state_dict
    if hasattr(model, "state_dict"):
        return model.state_dict()
    return {}


def get_bos_eos_token_ids(tokenizer):
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    bos_ids = [] if bos_token_id is None else [int(bos_token_id)]
    eos_ids = [] if eos_token_id is None else [int(eos_token_id)]
    return bos_ids, eos_ids


class _MapFuncRegistry(dict):
    def register_module(self, module=None, name=None, force=False):
        if module is None:
            raise ValueError("module must not be None")
        key = name or getattr(module, "__name__", None)
        if not key:
            raise ValueError("module name must not be empty")
        if not force and key in self:
            return module
        self[key] = module
        return module


class _MinimalBuilder:
    def build(self, cfg):
        if cfg is None:
            return None
        if isinstance(cfg, (list, tuple)):
            return [self.build(item) for item in cfg]
        if not isinstance(cfg, dict):
            return cfg

        cfg = deepcopy(cfg)
        obj_type = cfg.pop("type", None)
        if obj_type is None:
            return cfg

        obj = self._resolve_type(obj_type)
        if isinstance(obj, type):
            return obj(**cfg)
        if callable(obj):
            return obj(**cfg)
        raise TypeError(f"Unsupported build target type: {obj!r}")

    @staticmethod
    def _resolve_type(obj_type: Any):
        if not isinstance(obj_type, str):
            return obj_type
        if "." not in obj_type:
            raise KeyError(
                f"Minimal xtuner BUILDER cannot resolve non-dotted type name {obj_type!r}. "
                "This fallback only supports the current RefCOCO OPSD training chain."
            )
        module_name, attr_name = obj_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)


def install_xtuner_fallback_modules() -> None:
    from mmengine.runner.loops import EpochBasedTrainLoop

    xtuner_module = sys.modules.setdefault("xtuner", types.ModuleType("xtuner"))
    if getattr(xtuner_module, "__spec__", None) is None:
        xtuner_module.__spec__ = importlib.machinery.ModuleSpec("xtuner", loader=None, is_package=True)
    if not hasattr(xtuner_module, "__path__"):
        xtuner_module.__path__ = []

    engine_module = sys.modules.setdefault("xtuner.engine", types.ModuleType("xtuner.engine"))
    runner_module = sys.modules.setdefault("xtuner.engine.runner", types.ModuleType("xtuner.engine.runner"))
    if getattr(engine_module, "__spec__", None) is None:
        engine_module.__spec__ = importlib.machinery.ModuleSpec("xtuner.engine", loader=None, is_package=True)
    if not hasattr(engine_module, "__path__"):
        engine_module.__path__ = []
    if getattr(runner_module, "__spec__", None) is None:
        runner_module.__spec__ = importlib.machinery.ModuleSpec("xtuner.engine.runner", loader=None)

    class TrainLoop(EpochBasedTrainLoop):
        pass

    runner_module.TrainLoop = TrainLoop
    engine_module.runner = runner_module
    xtuner_module.engine = engine_module

    model_module = sys.modules.setdefault("xtuner.model", types.ModuleType("xtuner.model"))
    model_utils_module = sys.modules.setdefault(
        "xtuner.model.utils", types.ModuleType("xtuner.model.utils")
    )
    if getattr(model_module, "__spec__", None) is None:
        model_module.__spec__ = importlib.machinery.ModuleSpec("xtuner.model", loader=None, is_package=True)
    if not hasattr(model_module, "__path__"):
        model_module.__path__ = []
    if getattr(model_utils_module, "__spec__", None) is None:
        model_utils_module.__spec__ = importlib.machinery.ModuleSpec("xtuner.model.utils", loader=None)
    model_utils_module.guess_load_checkpoint = guess_load_checkpoint
    model_utils_module.get_peft_model_state_dict = get_peft_model_state_dict
    model_module.utils = model_utils_module
    xtuner_module.model = model_module

    registry_module = sys.modules.setdefault("xtuner.registry", types.ModuleType("xtuner.registry"))
    if getattr(registry_module, "__spec__", None) is None:
        registry_module.__spec__ = importlib.machinery.ModuleSpec("xtuner.registry", loader=None)
    if not hasattr(registry_module, "BUILDER"):
        registry_module.BUILDER = _MinimalBuilder()
    if not hasattr(registry_module, "MAP_FUNC"):
        registry_module.MAP_FUNC = _MapFuncRegistry()
    xtuner_module.registry = registry_module

    dataset_module = sys.modules.setdefault("xtuner.dataset", types.ModuleType("xtuner.dataset"))
    dataset_utils_module = sys.modules.setdefault(
        "xtuner.dataset.utils", types.ModuleType("xtuner.dataset.utils")
    )
    if getattr(dataset_module, "__spec__", None) is None:
        dataset_module.__spec__ = importlib.machinery.ModuleSpec("xtuner.dataset", loader=None, is_package=True)
    if not hasattr(dataset_module, "__path__"):
        dataset_module.__path__ = []
    if getattr(dataset_utils_module, "__spec__", None) is None:
        dataset_utils_module.__spec__ = importlib.machinery.ModuleSpec("xtuner.dataset.utils", loader=None)
    dataset_utils_module.get_bos_eos_token_ids = get_bos_eos_token_ids
    dataset_module.utils = dataset_utils_module
    xtuner_module.dataset = dataset_module

    utils_module = sys.modules.setdefault("xtuner.utils", types.ModuleType("xtuner.utils"))
    if getattr(utils_module, "__spec__", None) is None:
        utils_module.__spec__ = importlib.machinery.ModuleSpec("xtuner.utils", loader=None)
    utils_module.IGNORE_INDEX = -100
    utils_module.DEFAULT_PAD_TOKEN_INDEX = 0
    xtuner_module.utils = utils_module

    configs_module = sys.modules.setdefault("xtuner.configs", types.ModuleType("xtuner.configs"))
    if getattr(configs_module, "__spec__", None) is None:
        configs_module.__spec__ = importlib.machinery.ModuleSpec("xtuner.configs", loader=None)
    if not hasattr(configs_module, "cfgs_name_path"):
        configs_module.cfgs_name_path = {}
    xtuner_module.configs = configs_module


def load_fallback_config(config_path: str, args):
    from mmengine.config import Config

    if not osp.isfile(config_path):
        raise FileNotFoundError(
            f"Fallback training only supports an explicit config path for the current RefCOCO OPSD chain, got: {config_path}"
        )

    patch_mmengine_adafactor_duplicate_registration()
    install_xtuner_fallback_modules()

    cfg = Config.fromfile(config_path)
    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(config_path))[0])

    if args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    normalize_train_cfg_for_fallback(cfg)
    return cfg


def normalize_train_cfg_for_fallback(cfg) -> None:
    from mmengine.runner.loops import EpochBasedTrainLoop

    train_cfg = cfg.get("train_cfg")
    if not isinstance(train_cfg, dict):
        return

    loop_type = train_cfg.get("type")
    loop_name = getattr(loop_type, "__name__", None)
    if loop_name == "TrainLoop":
        train_cfg["type"] = EpochBasedTrainLoop


def build_runner_from_cfg(cfg):
    from mmengine.registry import RUNNERS
    from mmengine.runner import Runner

    if "runner_type" not in cfg:
        return Runner.from_cfg(cfg)
    return RUNNERS.build(cfg)


def log_fallback_banner(config_path: str, import_error: Exception) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("sa2va.train").info(
        "xtuner train launcher unavailable, switching to local fallback for current RefCOCO OPSD chain. config=%s error=%s",
        config_path,
        import_error,
    )
