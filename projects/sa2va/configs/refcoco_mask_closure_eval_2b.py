from projects.sa2va.datasets.common import DEFAULT_MASK_TO_CAPTION_QUESTION

model_path = "/data/xyc/Sa2va_opsd/Sa2VA/pretrained/Sa2VA-4B"
tokenizer_path = model_path
enable_teacher = True

dataset_name = "refcoco"
split = "val"
data_root = "/data/xyc"

image_root = None
image_root_candidates = [
    "/data/xyc/images/mscoco/images/train2014",
    "/data/xyc/glamm_data/images/coco2014/train2014",
    "/data/xyc/osprey-724k/coco/train2014",
    "/data/coco/train2014",
]

student_question = DEFAULT_MASK_TO_CAPTION_QUESTION

teacher_summary_template = (
    "You are optimizing the following task: given a gtmask, generate a caption that describes it. "
    "You are now given the original input, the student question, and privileged verification information. "
    "Use these privileged signals to improve the caption generation.\n"
    "Original student prompt: {student_question}\n"
    "Student caption: {student_caption}\n"
    "Verifier caption used for reconstruction: {verifier_caption}\n"
    "Reconstruction question: {reconstruct_question}\n"
    "Description generation status: {description_status}\n"
    "Reconstruction status: {reconstruct_status}\n"
    "caption_to_mask_seg_correct: {caption_to_mask_seg_correct}\n"
    "IoU between gtmask (region1) and refmask (region2): {iou:.4f}\n"
    "Reconstruction produced a valid mask: {has_mask}\n"
    "region1 = gtmask summary: {gtmask_summary}\n"
    "region2 = refmask summary: {refmask_summary}\n"
    "Compare region1 and region2, then infer how the caption should be revised so the reconstruction moves from region2 toward region1.\n"
    "Use that strategy to better model the student's caption tokens."
)

device = "cuda:0"
limit = 100
output_dir = "/data/xyc/Sa2va_opsd/Sa2VA/work_dirs/refcoco_mask_closure_eval_2b"
