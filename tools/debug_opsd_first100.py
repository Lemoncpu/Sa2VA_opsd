import argparse
import json
from pathlib import Path

import torch

from projects.sa2va.datasets.sa2va_opsd_npz_v2 import Sa2VAOpsdNPZDatasetV2
from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--npz-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.model_path

    dataset = Sa2VAOpsdNPZDatasetV2(
        npz_dir=args.npz_dir,
        prefix="masklet_data",
        shuffle=False,
        repeats=1,
        skip_empty_masks=True,
        student_question=(
            "<image>"
            "Can you provide me with a concise description of the region in the picture marked by region1? "
            "Answer with a short referring expression in RefCOCO style, usually 2 to 6 words, naming the target itself "
            "with only the most necessary attribute or location cue. Prefer forms like category plus color, size, "
            "left/right, top/bottom, or a nearby relation. Do not describe the whole scene. Do not write a full sentence "
            "and do not start with 'it is'. Do not output [SEG], masks, tags, or placeholder tokens."
        ),
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

    results = []
    ok_caption = 0
    ok_reconstruct = 0
    iou_sum = 0.0

    with torch.no_grad():
        max_items = min(args.limit, len(dataset))
        indices = list(range(args.start, max_items, max(args.step, 1)))
        for idx in indices:
            sample = dataset[idx]
            description = model.generate_description(
                image=sample["image"],
                mask_prompts=sample["prompt_masks"],
                student_question=sample["student_question"],
            )
            reconstruction = model.reconstruct_mask(
                image=sample["image"],
                caption=description.clean_caption,
                description_status=description.status,
                spatial_hint=model._coarse_spatial_hint(model._to_numpy_mask(sample["gt_mask"])),
                gt_mask=model._to_numpy_mask(sample["gt_mask"]),
            )
            iou = model._compute_iou(sample["gt_mask"], reconstruction.pred_mask)
            reconstruct_candidates = []
            if description.status == "ok":
                spatial_caption = description.clean_caption
                for question in model._resolve_reconstruct_questions(spatial_caption):
                    predict_dict = model._predict_forward_eval(
                        model.student_model,
                        image=sample["image"],
                        text=question,
                        past_text="",
                        mask_prompts=None,
                        tokenizer=model.tokenizer,
                    )
                    prediction_masks = predict_dict.get("prediction_masks")
                    pred_mask = None
                    if prediction_masks:
                        first_mask = prediction_masks[0]
                        if isinstance(first_mask, torch.Tensor):
                            first_mask = first_mask.detach().cpu().numpy()
                        first_mask = torch.as_tensor(first_mask)
                        if first_mask.ndim == 3 and first_mask.shape[0] == 1:
                            first_mask = first_mask[0]
                        pred_mask = (first_mask > 0).to(torch.uint8).cpu().numpy()
                    reconstruct_candidates.append(
                        {
                            "question": question,
                            "raw_prediction": predict_dict.get("prediction", ""),
                            "prediction_masks_count": 0 if prediction_masks is None else len(prediction_masks),
                            "pred_mask_sum": None
                            if pred_mask is None
                            else int(torch.as_tensor(pred_mask).sum().item()),
                            "iou": float(model._compute_iou(sample["gt_mask"], pred_mask)),
                        }
                    )
            best_candidate_iou = max([x["iou"] for x in reconstruct_candidates], default=iou)
            if description.status == "ok":
                ok_caption += 1
            if reconstruction.status == "ok":
                ok_reconstruct += 1
            iou_sum += iou
            results.append(
                {
                    "index": idx,
                    "npz_path": sample["npz_path"],
                    "description_status": description.status,
                    "raw_prediction": description.raw_prediction,
                    "caption": description.clean_caption,
                    "caption_token_count": model._caption_token_count(description.clean_caption),
                    "reconstruct_status": reconstruction.status,
                    "reconstruct_question": reconstruction.question,
                    "reconstruct_raw_prediction": reconstruction.raw_prediction,
                    "prediction_masks_count": reconstruction.prediction_masks_count,
                    "gt_mask_sum": int(sample["gt_mask"].sum().item()),
                    "pred_mask_sum": None
                    if reconstruction.pred_mask is None
                    else int(torch.as_tensor(reconstruction.pred_mask).sum().item()),
                    "iou": float(iou),
                    "best_candidate_iou": float(best_candidate_iou),
                    "reconstruct_candidates": reconstruct_candidates,
                }
            )
            print(
                f"[{idx:03d}] desc={description.status:<18} recon={reconstruction.status:<24} "
                f"iou={iou:.4f} caption={description.clean_caption!r}"
            )

    summary = {
        "limit": len(results),
        "start": args.start,
        "step": args.step,
        "ok_caption_rate": ok_caption / max(len(results), 1),
        "ok_reconstruct_rate": ok_reconstruct / max(len(results), 1),
        "avg_iou": iou_sum / max(len(results), 1),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
