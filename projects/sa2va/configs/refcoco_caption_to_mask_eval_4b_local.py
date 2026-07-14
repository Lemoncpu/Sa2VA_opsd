_os = __import__("os")
_getenv = _os.getenv

model_path = _getenv(
    "SA2VA_REFCOCO_EVAL_MODEL_PATH",
    "/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd/work_dirs/hf_iter_400",
)
tokenizer_path = _getenv("SA2VA_REFCOCO_EVAL_TOKENIZER_PATH", model_path)
enable_teacher = False

dataset_name = "refcoco"
split = _getenv("SA2VA_REFCOCO_EVAL_SPLIT", "val")

# REFER-style root. The evaluator will load from <data_root>/refcoco
data_root = _getenv(
    "SA2VA_REFCOCO_EVAL_DATA_ROOT",
    "/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco",
)

image_root = None
image_root_candidates = [
    _getenv(
        "SA2VA_REFCOCO_EVAL_IMAGE_ROOT",
        "/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco/train2014",
    ),
]

device = _getenv("SA2VA_REFCOCO_EVAL_DEVICE", "cuda:0")
limit = int(_getenv("SA2VA_REFCOCO_EVAL_LIMIT", "0")) or None
output = _getenv(
    "SA2VA_REFCOCO_EVAL_OUTPUT",
    "/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd/work_dirs/refcoco_caption_to_mask_eval_hf_iter_400.json",
)
