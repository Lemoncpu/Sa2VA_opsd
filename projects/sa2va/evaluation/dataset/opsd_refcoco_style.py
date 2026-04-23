import json
import os

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as _mask

from projects.sa2va.datasets.sa2va_opsd_npz_v2 import Sa2VAOpsdNPZDatasetV2
from projects.sa2va.evaluation.utils import AverageMeter, Summary, intersectionAndUnionGPU, master_only


class OPSDRefcocoStyleDataset:
    METAINFO = dict(name="OPSD RefCOCO-Style Segmentation")

    def __init__(self, annotation_file, image_folder=None, reasoning=False):
        self.annotation_file = annotation_file
        self.image_folder = image_folder or os.path.dirname(annotation_file)
        self.reasoning = reasoning
        self.split = "opsd"
        self.dataset_name = "opsd_refcoco_style"
        self.json_datas = self._load_items(annotation_file)

    @classmethod
    def metainfo_name(cls):
        return cls.METAINFO["name"].replace(" ", "_")

    def _load_items(self, annotation_file):
        with open(annotation_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        items = payload["items"] if isinstance(payload, dict) else payload
        filtered = []
        for item in items:
            caption = str(item.get("caption", "")).strip()
            labels = [label for label in item.get("selected_labels", []) if str(label).strip()]
            if not caption and not labels:
                continue
            if not labels and caption:
                item = dict(item)
                item["selected_labels"] = [caption]
            filtered.append(item)
        return filtered

    def __len__(self):
        return len(self.json_datas)

    def real_len(self):
        return len(self.json_datas)

    def _resolve_image_path(self, image_name):
        if image_name is None:
            return None
        if os.path.isabs(image_name):
            return image_name
        return os.path.join(self.image_folder, image_name)

    def _load_npz_sample(self, npz_path):
        (frame1, mask1), _ = Sa2VAOpsdNPZDatasetV2._load_raw_npz(npz_path)
        image = Sa2VAOpsdNPZDatasetV2._normalize_frame(frame1.cpu().numpy())
        gt_mask = Sa2VAOpsdNPZDatasetV2._normalize_mask(mask1.cpu().numpy()).unsqueeze(0)
        return image, gt_mask

    def decode_mask(self, encoded_masks):
        masks = []
        for item in encoded_masks:
            decoded = _mask.decode(item)
            if decoded.ndim == 3:
                decoded = np.sum(decoded, axis=2)
            masks.append(np.uint8(decoded > 0))
        return torch.from_numpy(np.stack(masks, axis=0))

    def only_get_text_infos(self, json_data):
        return {"sampled_sents": json_data["selected_labels"]}

    def get_questions(self, text_require_infos):
        sampled_sents = text_require_infos["sampled_sents"]
        ret = []
        for sent in sampled_sents:
            if self.reasoning:
                question = "Please segment {} in this image.".format(sent)
                think_prompt = (
                    "You should first think about the reasoning process in the mind and then provides "
                    "the user with the answer. Please respond with segmentation mask in both the thinking "
                    "process and the answer."
                )
                template_prompt = (
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> "
                    "</answer> tags, respectively, i.e., <think> reasoning process here </think> "
                    "<answer> answers here </answer>."
                )
                ret.append("<image>\n{}\n\n{}\n\n{}".format(question, think_prompt, template_prompt))
            else:
                ret.append("<image>\n Please segment {} in this image.".format(sent))
        return ret

    def filter_data_dict(self, data_dict):
        names = ["image", "text", "gt_masks", "img_id", "image_path"]
        return {name: data_dict[name] for name in names}

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = dict(self.json_datas[index])
        text_require_infos = self.only_get_text_infos(data_dict)
        questions = self.get_questions(text_require_infos)
        npz_path = data_dict.get("npz_path")
        if npz_path:
            image, masks = self._load_npz_sample(npz_path)
            image_path = npz_path
        else:
            image_path = self._resolve_image_path(data_dict["image"])
            image = Image.open(image_path).convert("RGB")
            masks = self.decode_mask(data_dict["gt_masks"])
        data_dict["gt_masks"] = masks
        data_dict["image"] = image
        data_dict["text"] = questions
        data_dict["img_id"] = str(data_dict.get("sample_id", index))
        data_dict["image_path"] = image_path
        return self.filter_data_dict(data_dict)

    @master_only
    def evaluate(self, result):
        trackers = {
            "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union": AverageMeter("Union", ":6.3f", Summary.SUM),
            "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM),
        }
        for pred_dict in result:
            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            masks = pred_dict["prediction_masks"]
            pred_masks = []
            for mask in masks:
                if mask is not None:
                    mask = rle_to_mask(mask)
                pred_masks.append(mask)
            targets = pred_dict["gt_masks"]
            gt_masks = rle_to_mask(targets)

            for item_idx, pred_mask in enumerate(pred_masks):
                gt_mask = gt_masks[item_idx : item_idx + 1]
                if pred_mask is None:
                    pred_mask = gt_mask * 0
                for prediction, target in zip(pred_mask, gt_mask):
                    prediction = torch.from_numpy(prediction).int().cuda()
                    target = torch.from_numpy(target).int().cuda()
                    intersect, union_, _ = intersectionAndUnionGPU(
                        prediction.contiguous().clone(),
                        target.contiguous(),
                        2,
                        ignore_index=255,
                    )
                    intersection += intersect
                    union += union_
                    accuracy_iou += intersect / (union_ + 1e-5)
                    accuracy_iou[union_ == 0] += 1.0

            if not isinstance(intersection, float):
                intersection = intersection.cpu().numpy()
            if not isinstance(union, float):
                union = union.cpu().numpy()
            if not isinstance(accuracy_iou, float):
                accuracy_iou = accuracy_iou.cpu().numpy()
            accuracy_iou = accuracy_iou / gt_masks.shape[0]
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=gt_masks.shape[0])

        cur_results = {
            "pixel_intersection": trackers["intersection"].sum[1],
            "pixel_union": trackers["union"].sum[1],
            "gIoU": trackers["gIoU"].avg[1],
            "mask_counts": trackers["gIoU"].count,
        }
        class_iou = cur_results["pixel_intersection"] / (cur_results["pixel_union"] + 1e-10)
        global_iou = cur_results["gIoU"]

        print("============================================", "current")
        print("CIoU: {}, GIoU: {}".format(class_iou, global_iou), "current")
        print("============================================", "current")
        print("{} successfully finished evaluating".format(self.dataset_name), "current")
        return {"Acc": class_iou}


def rle_to_mask(rle):
    mask = []
    for item in rle:
        decoded = _mask.decode(item)
        decoded = np.uint8(decoded)
        mask.append(decoded)
    return np.stack(mask, axis=0)
