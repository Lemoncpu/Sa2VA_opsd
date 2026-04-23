import argparse
import os
import runpy

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from projects.sa2va.datasets.common import SEG_QUESTIONS
from projects.sa2va.evaluation.caption_to_mask_common import normalize_refcoco_caption, run_caption_to_mask_eval
from projects.sa2va.evaluation.utils.refcoco_refer import REFER
from projects.sa2va.models.sa2va_opsd_v3 import Sa2VAOPSDModelV3


DATASET_META = {
    'refcoco': {'splitBy': 'unc', 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': 'unc', 'dataset_name': 'refcoco+'},
    'refcocog': {'splitBy': 'umd', 'dataset_name': 'refcocog'},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--image-root', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def load_config(path):
    return runpy.run_path(path)


def resolve_image_root(cfg, cli_image_root=None):
    if cli_image_root:
        return cli_image_root
    if cfg.get('image_root'):
        return cfg['image_root']
    for candidate in cfg.get('image_root_candidates', []):
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        'No valid RefCOCO image_root found. '
        'Pass --image-root explicitly or update image_root_candidates in the config.'
    )


def decode_ref_mask(ann, height, width):
    if len(ann['segmentation']) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(ann['segmentation'][0], list):
        rles = mask_utils.frPyObjects(ann['segmentation'], height, width)
    else:
        rles = ann['segmentation']
        for item in rles:
            if not isinstance(item['counts'], bytes):
                item['counts'] = item['counts'].encode()
    mask = mask_utils.decode(rles)
    if mask.ndim == 3:
        mask = np.sum(mask, axis=2)
    return (mask > 0).astype(np.uint8)


def build_samples(data_root, dataset_name, split, image_root):
    meta = DATASET_META[dataset_name]
    refer = REFER(data_root, meta['dataset_name'], meta['splitBy'])
    ref_ids = refer.getRefIds(split=split)
    refs = refer.loadRefs(ref_ids=ref_ids)
    samples = []
    for ref in refs:
        image_info = refer.loadImgs(image_ids=[ref['image_id']])[0]
        ann = refer.refToAnn[ref['ref_id']]
        gt_mask = decode_ref_mask(ann, image_info['height'], image_info['width'])
        image_path = os.path.join(image_root, image_info['file_name'])
        for sent in ref['sentences']:
            caption = normalize_refcoco_caption(sent['sent'])
            if not caption:
                continue
            samples.append(
                {
                    'image': image_path,
                    'caption': caption,
                    'gt_mask': gt_mask,
                    'meta': {
                        'ref_id': ref['ref_id'],
                        'ann_id': ref['ann_id'],
                        'image_id': ref['image_id'],
                        'image_path': image_path,
                    },
                }
            )
    return samples


def main():
    args = parse_args()
    cfg = load_config(args.config)

    image_root = resolve_image_root(cfg, args.image_root)
    limit = args.limit or cfg.get('limit', 50)
    output = args.output or cfg['output']
    device = args.device or cfg.get('device', 'cuda:0')

    model = Sa2VAOPSDModelV3(
        model_path=cfg['model_path'],
        teacher_model_path=cfg.get('teacher_model_path'),
        enable_teacher=cfg.get('enable_teacher', False),
        grpo_group_size=0,
        tokenizer_path=cfg.get('tokenizer_path', cfg['model_path']),
        device=device,
        torch_dtype='auto',
        use_flash_attn=True,
        min_caption_tokens=4,
    )
    model.eval()

    samples = build_samples(
        data_root=cfg['data_root'],
        dataset_name=cfg.get('dataset_name', 'refcoco'),
        split=cfg.get('split', 'val'),
        image_root=image_root,
    )

    prepared_samples = []
    for sample in samples:
        image_path = sample["image"]
        if not os.path.exists(image_path):
            continue
        prepared = dict(sample)
        prepared["image"] = Image.open(image_path).convert("RGB")
        prepared_samples.append(prepared)

    run_caption_to_mask_eval(
        model=model,
        samples=prepared_samples,
        limit=limit,
        output=output,
        summary_prefix={
            'dataset_name': cfg.get('dataset_name', 'refcoco'),
            'split': cfg.get('split', 'val'),
            'image_root': image_root,
            'seg_questions_reference': SEG_QUESTIONS[:3],
        },
    )


if __name__ == '__main__':
    main()
