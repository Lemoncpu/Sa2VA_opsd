import copy
import os

import numpy as np
import torch
import tqdm
from pycocotools import mask as _mask
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from utils import collect_results_cpu, get_dist_info, get_rank, _init_dist_pytorch
from projects.sa2va.models.utils import find_seg_indices


def mask_to_rle(mask):
    rle = []
    for m in mask:
        encoded = _mask.encode(np.asfortranarray(m.astype(np.uint8)))
        encoded["counts"] = encoded["counts"].decode()
        rle.append(encoded)
    return rle


def build_model_components(model_path):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    processor = None
    if "qwen" in model_path.lower():
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer, processor


def build_dataloader(dataset, rank, world_size, batch_size=1, num_workers=8):
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda x: x[0],
    )


def run_refcoco_style_eval(
    *,
    model_path,
    dataset,
    launcher="none",
    batch_size=1,
    num_workers=8,
):
    if launcher != "none":
        import datetime

        _init_dist_pytorch("nccl", timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    model, tokenizer, processor = build_model_components(model_path)
    dataloader = build_dataloader(
        dataset,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    results = []
    for data_batch in tqdm.tqdm(dataloader):
        prediction = {
            "img_id": data_batch["img_id"],
            "gt_masks": mask_to_rle(data_batch["gt_masks"].cpu().numpy()),
        }
        texts = data_batch["text"]
        del data_batch["img_id"], data_batch["gt_masks"], data_batch["image_path"], data_batch["text"]

        pred_masks = []
        pred_texts = []
        pred_n_masks = []
        for text in texts:
            batch_for_text = copy.deepcopy(data_batch)
            batch_for_text["text"] = text
            predict_kwargs = dict(batch_for_text)
            predict_kwargs["tokenizer"] = tokenizer
            if processor is not None:
                predict_kwargs["processor"] = processor
            try:
                pred = model.predict_forward(**predict_kwargs)
            except TypeError as exc:
                if processor is None or "processor" not in str(exc):
                    raise
                predict_kwargs.pop("processor", None)
                pred = model.predict_forward(**predict_kwargs)
            pred_mask = pred["prediction_masks"]
            pred_text = pred["prediction"]
            cleaned_pred_text = pred_text.replace("<|im_end|>", "").replace("<|end|>", "").strip()
            _, answer_seg_idx = find_seg_indices(cleaned_pred_text)
            if len(answer_seg_idx) == 0:
                _, answer_seg_idx = find_seg_indices(pred_text)

            pred_texts.append(pred_text)

            if len(pred_mask) == 0:
                pred_masks.append(None)
                pred_n_masks.append(None)
                continue

            pred_n_masks.append(np.concatenate(pred_mask, axis=0))
            if len(answer_seg_idx) > 0:
                selected_masks = [pred_mask[idx] for idx in answer_seg_idx if idx < len(pred_mask)]
                if len(selected_masks) > 0:
                    final_mask = selected_masks[0]
                    for mask in selected_masks[1:]:
                        final_mask = final_mask | mask
                    pred_masks.append(mask_to_rle(final_mask))
                else:
                    pred_masks.append(None)
            else:
                pred_masks.append(None)

        prediction.update(
            {
                "prediction_masks": pred_masks,
                "prediction_texts": pred_texts,
                "prediction_all_masks": pred_n_masks,
            }
        )
        results.append(prediction)

    tmpdir = "./dist_test_temp_res_" + dataset.metainfo_name() + model_path.replace("/", "").replace(".", "")
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        metric = dataset.evaluate(results)
        print(metric)
        return metric, results
    return None, None
