model_path = '/data/xyc/Sa2va_opsd/Sa2VA/pretrained/Sa2VA-4B'
tokenizer_path = model_path
enable_teacher = False

dataset_name = 'refcoco'
split = 'val'

# Annotation root follows REFER convention: <data_root>/<dataset_name>/{instances.json, refs(...).p}
data_root = '/data/xyc'

# Image root is kept explicit because server layout may differ from the default REFER path.
image_root = None
image_root_candidates = [
    '/data/xyc/images/mscoco/images/train2014',
    '/data/xyc/glamm_data/images/coco2014/train2014',
    '/data/xyc/osprey-724k/coco/train2014',
    '/data/coco/train2014',
]

device = 'cuda:0'
limit = 50
output = '/data/xyc/Sa2va_opsd/Sa2VA/work_dirs/refcoco_caption_to_mask_eval_50.json'
