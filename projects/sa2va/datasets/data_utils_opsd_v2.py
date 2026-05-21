from typing import Dict, Sequence


def sa2va_opsd_collect_fn_v2(
        instances: Sequence[Dict],
        return_hf_format: bool = False,
        use_varlen_attn: bool = False):
    assert not return_hf_format, "return_hf_format is not supported."
    assert not use_varlen_attn, "use_varlen_attn is not supported."

    return {
        "data": {
            "images": [instance["image"] for instance in instances],
            "prompt_masks": [instance["prompt_masks"] for instance in instances],
            "student_questions": [instance["student_question"] for instance in instances],
            "gt_masks": [instance["gt_mask"] for instance in instances],
            "confuser_candidate_masks": [instance.get("confuser_candidate_masks") for instance in instances],
            "confuser_candidate_ref_ids": [instance.get("confuser_candidate_ref_ids") for instance in instances],
            "sample_keys": [instance.get("sample_key", instance.get("npz_path")) for instance in instances],
            "routes": [instance.get("route") for instance in instances],
            "route_ious": [instance.get("route_iou") for instance in instances],
            "route_global_steps": [instance.get("route_global_step") for instance in instances],
            "route_manifest_paths": [instance.get("route_manifest_path") for instance in instances],
            "npz_paths": [instance.get("npz_path") for instance in instances],
            "frame1s": [instance.get("frame1") for instance in instances],
            "mask1s": [instance.get("mask1") for instance in instances],
            "frame2s": [instance.get("frame2") for instance in instances],
            "mask2s": [instance.get("mask2") for instance in instances],
        },
        "data_samples": None,
    }
