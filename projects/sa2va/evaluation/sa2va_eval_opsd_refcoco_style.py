import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.sa2va.evaluation.dataset.opsd_refcoco_style import OPSDRefcocoStyleDataset
from projects.sa2va.evaluation.caption_to_mask_common import run_caption_to_mask_eval
from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3


def parse_args():
    parser = argparse.ArgumentParser(description="OPSD Caption-to-Mask Eval Aligned With sa2va_opsd_v2.py")
    parser.add_argument("model_path", help="hf model path.")
    parser.add_argument("--annotation-file", required=True, help="Exported OPSD caption annotations.")
    parser.add_argument("--image-folder", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.model_path

    dataset = OPSDRefcocoStyleDataset(
        annotation_file=args.annotation_file,
        image_folder=args.image_folder,
        reasoning=False,
    )
    model = Sa2VAOPSDModelV3(
        model_path=args.model_path,
        enable_teacher=False,
        grpo_group_size=0,
        tokenizer_path=tokenizer_path,
        device=args.device,
        torch_dtype="auto",
        use_flash_attn=True,
        min_caption_tokens=4,
    )
    model.eval()

    limit = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.annotation_file).with_name("caption_to_mask_eval.json"))
    samples = []
    for idx in range(limit):
        sample = dataset[idx]
        ann = dataset.json_datas[idx]
        raw_caption = ann.get("caption") or ann["selected_labels"][0]
        samples.append(
            {
                "image": sample["image"],
                "gt_mask": sample["gt_masks"][0],
                "caption": raw_caption,
                "description_status": ann.get("description_status", "ok"),
                "meta": {
                    "idx": idx,
                    "npz_path": ann.get("npz_path"),
                    "caption_storage_format": ann.get("caption_storage_format"),
                },
            }
        )

    run_caption_to_mask_eval(
        model=model,
        samples=samples,
        limit=limit,
        output=output_path,
        summary_prefix={
            "annotation_file": args.annotation_file,
        },
    )


if __name__ == "__main__":
    main()
