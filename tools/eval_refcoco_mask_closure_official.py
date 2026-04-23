import argparse
import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.sa2va.datasets.refcoco_opsd import build_refcoco_opsd_records
from projects.sa2va.evaluation.caption_to_mask_common import run_mask_to_caption_to_mask_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run official RefCOCO mask->caption->mask closure evaluation without intermediate annotation files."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(path):
    return runpy.run_path(path)


def resolve_image_root(cfg, cli_image_root=None):
    if cli_image_root:
        return cli_image_root
    if cfg.get("image_root"):
        return cfg["image_root"]
    for candidate in cfg.get("image_root_candidates", []):
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "No valid RefCOCO image_root found. "
        "Pass --image-root explicitly or update image_root_candidates in the config."
    )


def main():
    args = parse_args()
    from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3

    cfg = load_config(args.config)

    image_root = resolve_image_root(cfg, args.image_root)
    device = args.device or cfg.get("device", "cuda:0")
    limit = args.limit or cfg.get("limit", 50)
    output = args.output
    if output is None:
        output_dir = Path(cfg.get("output_dir", Path(args.config).resolve().parent))
        output_dir.mkdir(parents=True, exist_ok=True)
        output = str(output_dir / "caption_to_mask_eval_official.json")

    model = Sa2VAOPSDModelV3(
        model_path=cfg["model_path"],
        enable_teacher=cfg.get("enable_teacher", False),
        teacher_model_path=cfg.get("teacher_model_path"),
        grpo_group_size=0,
        tokenizer_path=cfg.get("tokenizer_path", cfg["model_path"]),
        device=device,
        torch_dtype="auto",
        use_flash_attn=True,
        min_caption_tokens=4,
    )
    model.eval()

    samples, _ = build_refcoco_opsd_records(
        data_root=cfg["data_root"],
        dataset_name=cfg.get("dataset_name", "refcoco"),
        split=cfg.get("split", "val"),
        image_root=image_root,
        image_root_candidates=cfg.get("image_root_candidates", []),
    )

    run_mask_to_caption_to_mask_eval(
        model=model,
        samples=samples,
        limit=min(limit, len(samples)),
        output=output,
        student_question=cfg["student_question"],
        summary_prefix={
            "annotation_file": None,
        },
    )


if __name__ == "__main__":
    main()
