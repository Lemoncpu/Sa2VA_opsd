from typing import Dict, Sequence


def opsd_collate_fn(instances: Sequence[Dict]):
    images = [instance["image"] for instance in instances]
    prompt_masks = [instance["prompt_masks"] for instance in instances]
    prompt_ids = [instance["prompt_ids"] for instance in instances]
    student_questions = [instance["student_question"] for instance in instances]
    gt_masks = [instance["gt_mask"] for instance in instances]
    npz_paths = [instance.get("npz_path") for instance in instances]

    return {
        "data": {
            "images": images,
            "prompt_masks": prompt_masks,
            "prompt_ids": prompt_ids,
            "student_questions": student_questions,
            "gt_masks": gt_masks,
            "npz_paths": npz_paths,
        },
        "data_samples": None,
    }
