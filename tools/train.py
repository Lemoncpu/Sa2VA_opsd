import argparse
import os
import sys

from mmengine.config import DictAction

from tools.train_fallback_compat import (
    build_runner_from_cfg,
    is_supported_fallback_target,
    load_fallback_config,
    log_fallback_banner,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM")
    parser.add_argument("config", help="config file name or path.")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="the path to the .json file for deepspeed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="specify checkpoint path to be resumed from.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for the training"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def enable_short_torch_repr():
    import torch

    ori_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = (
        lambda self: f"Shape of Tensor({self.size()})"
        if self.numel() > 100
        else ori_repr(self)
    )


def run_xtuner_train(args):
    import xtuner.tools.train as train

    train.parse_args = parse_args
    enable_short_torch_repr()
    train.main()


def run_local_fallback(args, import_error):
    if not is_supported_fallback_target(args.config):
        raise RuntimeError(
            "xtuner is unavailable, and the local fallback launcher only supports the current "
            "RefCOCO OPSD 4B training chain. "
            f"config={args.config}"
        ) from import_error

    log_fallback_banner(args.config, import_error)
    if args.deepspeed is not None:
        print(
            f"[INFO] Ignoring --deepspeed={args.deepspeed} in local fallback mode and using the mmengine runner path.",
            file=sys.stderr,
        )
    cfg = load_fallback_config(args.config, args)
    enable_short_torch_repr()
    runner = build_runner_from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_xtuner_train(args)
    except ModuleNotFoundError as exc:
        if not (
            exc.name == "xtuner"
            or (exc.name or "").startswith("xtuner.")
            or "xtuner" in str(exc)
        ):
            raise
        run_local_fallback(args, exc)
