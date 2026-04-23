import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import RESDataset

from refcoco_eval_common import run_refcoco_style_eval


def parse_args():
    parser = argparse.ArgumentParser(description="RefCocoSeg")
    parser.add_argument("model_path", help="hf model path.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS_ATTRIBUTES.keys(),
        default="refcoco",
        help="Specify a ref dataset",
    )
    parser.add_argument("--split", default="val", help="Specify a split")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--vis", action="store_true", help="whether to visualize the results")
    parser.add_argument(
        "--with-thinking",
        action="store_true",
        help="whether to enable the reasoning process",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument(
        "--data_root",
        default="/mnt/bn/zilongdata-us/xiangtai/Sa2VA/data",
        help="Root directory for all datasets.",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


DATASETS_ATTRIBUTES = {
    "refcoco": {"splitBy": "unc", "dataset_name": "refcoco"},
    "refcoco_plus": {"splitBy": "unc", "dataset_name": "refcoco_plus"},
    "refcocog": {"splitBy": "umd", "dataset_name": "refcocog"},
    "grefcoco": {"splitBy": "unc", "dataset_name": "grefcoco"},
}


def main():
    args = parse_args()
    image_folder = os.path.join(args.data_root, "glamm_data/images/coco2014/train2014/")
    data_path = os.path.join(args.data_root, "ref_seg/")
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]
    dataset = RESDataset(
        image_folder=image_folder,
        dataset_name=dataset_info["dataset_name"],
        data_path=data_path,
        split=args.split,
        reasoning=args.with_thinking,
    )
    run_refcoco_style_eval(
        model_path=args.model_path,
        dataset=dataset,
        launcher=args.launcher,
    )


if __name__ == "__main__":
    main()
