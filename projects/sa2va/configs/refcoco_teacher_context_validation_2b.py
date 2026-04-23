_os = __import__("os")
_getenv = _os.getenv
_join = _os.path.join


ROOT_DIR = _os.path.abspath(_getenv("SA2VA_ROOT_DIR", _os.getcwd()))

data_root = _getenv("SA2VA_REFCOCO_DATA_ROOT", "/data/xiaoyicheng")
image_root = _getenv("SA2VA_REFCOCO_IMAGE_ROOT") or None

model_path = _getenv(
    "SA2VA_REFCOCO_TEACHER_2B_MODEL_PATH",
    _join(ROOT_DIR, "pretrained", "public_hf", "Sa2VA-InternVL3-2B"),
)
tokenizer_path = _getenv("SA2VA_REFCOCO_TEACHER_2B_TOKENIZER_PATH", model_path)
enable_teacher = True

dataset_name = _getenv("SA2VA_REFCOCO_DATASET", "refcoco")
split = _getenv("SA2VA_REFCOCO_EVAL_SPLIT", "val")

image_root_candidates = [
    candidate
    for candidate in [
        image_root,
        f"{data_root}/refcoco/train2014",
        f"{data_root}/images/mscoco/images/train2014",
        "/data/coco/train2014",
    ]
    if candidate
]

student_question = (
    "<image>"
    "Can you provide me with a concise description of the region in the picture marked by region1? "
    "Answer with a short referring expression in RefCOCO style, usually 2 to 6 words, naming the target itself "
    "with only the most necessary attribute or location cue. Prefer forms like category plus color, size, "
    "left/right, top/bottom, or a nearby relation. Do not describe the whole scene. Do not write a full sentence "
    "and do not start with 'it is'. Do not output [SEG], masks, tags, or placeholder tokens."
)

device = _getenv("SA2VA_REFCOCO_DEVICE", "cuda:0")
limit = int(_getenv("SA2VA_REFCOCO_LIMIT", "100"))
iou_low_threshold = float(_getenv("SA2VA_REFCOCO_IOU_LOW_THRESHOLD", "0.3"))
iou_high_threshold = float(_getenv("SA2VA_REFCOCO_IOU_HIGH_THRESHOLD", "0.8"))
low_iou_regen_max_new_tokens = int(_getenv("SA2VA_REFCOCO_LOW_IOU_REGEN_MAX_NEW_TOKENS", "48"))
output_dir = _getenv(
    "SA2VA_REFCOCO_TEACHER_2B_OUTPUT_DIR",
    _join(ROOT_DIR, "work_dirs", "refcoco_teacher_context_validation_2b"),
)
