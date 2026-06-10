# Code Change Log

## Project Overview
- This project is optimizing a mask-to-caption task: given a target mask as a visual prompt, train the model to generate a detailed localized caption for the masked object or region.
- The primary training goal is caption quality, not segmentation quality by itself.
- The evaluation target is a detailed localized caption that identifies the masked target with enough visible, local, and distinguishing detail.
- Caption-to-mask reconstruction IoU is used only as a verifier signal during training.
- The working hypothesis is: if a caption describes the target object in sufficiently detailed and localized visual terms, reconstructing a mask from that caption should return to the original target region.
- Based on verifier IoU, training enters different supervision routes:
  - `teacher_regenerate`: low-IoU failures use privileged teacher regeneration to provide a better caption target.
  - `on_policy_distill`: mid-IoU samples use on-policy self-distillation to refine the current caption while staying close to the student trajectory.
  - `GRPO confuser`: high-IoU or relatively strong samples use confuser-based GRPO to improve discrimination against nearby distractors.

## Usage Rule
- Read this document before any future code modification in this repository.
- When a code path is changed, append a new entry with the problem, root cause, chosen fix, and any rejected alternatives.
- Keep entries chronological so later work does not revert earlier design decisions by accident.

## 2026-06-10 OPSD 4B Caption Collapse Investigation

### Problem
- OPSD 4B training produced a normal first caption, then later captions degenerated into repeated `!`.
- Two high-priority causes were identified:
  1. The OPSD V2/V3 path loaded the student model directly with `AutoModel.from_pretrained(...)`, so the student language model was effectively trained in full-parameter mode instead of using an explicit freeze or LoRA policy.
  2. The Qwen3VL caption generation path accepted `mask_prompts` but did not inject them into the language prompt in a region-aware way, and it ignored configured caption generation controls such as `max_new_tokens`, repetition penalty, and no-repeat-ngram.

### Root Cause Notes
- `projects/sa2va/models/sa2va_opsd_v2.py` bypassed the project Qwen wrapper and had no explicit student freezing or LoRA preparation path.
- `projects/sa2va/hf/models_qwen3vl/modeling_sa2va_qwen.py` built Qwen chat messages using only image plus text, so `region1`-style supervision was not grounded by prompt-time region tokens.
- The same generation path hardcoded greedy decoding with `max_new_tokens=2048`.

### Chosen Fix Direction
- Add explicit OPSD-side model training policy controls so the student model can be frozen or adapted with LoRA by configuration instead of relying on hidden defaults.
- Extend the Qwen3VL HF model `predict_forward()` interface so OPSD can pass region-aware prompt text and generation kwargs through a stable API.

### Rejected Direction
- Do not continue patching around the symptom with more caption post-processing. The raw prediction is already malformed, so cleanup-only changes would not address the failure source.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - Added explicit OPSD student training controls: `student_freeze_llm`, `student_freeze_visual_encoder`, and `student_llm_lora`.
  - Applied those controls during model load instead of leaving the HF student model in implicit full-parameter training mode.
  - Passed caption generation constraints through the OPSD `predict_forward()` call path so description generation now uses configured decoding bounds.
- Updated `projects/sa2va/hf/models/modeling_sa2va_chat.py`:
  - Extended the real `Sa2VA-4B` HF loading path (`Sa2VAChatModel`) so OPSD can override generation bounds and decoding controls without changing the model family.
- Added `projects/sa2va/configs/sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3.py` and its online variant:
  - Renamed the 4B OPSD config to reflect the actual checkpoint composition (`Sa2VA-4B = InternVL2.5-4B + Qwen2.5-3B-Instruct`) instead of the misleading `internvl3_4b` name.
  - Switched the default training recipe to a more stable mask-to-caption setup: frozen student backbones plus LoRA, shorter caption decoding, and weaker GRPO pressure.
- Updated 4B helper scripts to point at the renamed config and matching work directory.
- Updated remaining `tools/` launcher defaults (`train.sh`, `train1.sh`, `train_resume.sh`, `export.sh`, `testconf.sh`) to use the renamed `Sa2VA-4B` OPSD work directory instead of the misleading `internvl3_4b` path.

### Correction Note
- An intermediate change was briefly applied to `projects/sa2va/hf/models_qwen3vl/modeling_sa2va_qwen.py`, but that path is not used by `Sa2VA-4B`.
- After confirming the real `config.json` for `Sa2VA-4B`, the effective interface change was moved to `projects/sa2va/hf/models/modeling_sa2va_chat.py`, which is the actual `AutoModel` target for this checkpoint.

## 2026-06-10 Train Launcher Startup Failure On Fresh Server

### Problem
- `tools/train1.sh` failed on a fresh server before real training started.
- The launcher also started `tools/plot_training_metrics.py` in watch mode, and that helper crashed immediately when no training metrics had been written yet.

### Root Cause Notes
- The launcher scripts defaulted `SAM_CONFUSER_POOL_DIR` to `${WORK_DIR}/sam_confuser_pool`, but the training configs and export flow use the shared default `./work_dirs/refcoco_sam_confuser_pool`.
- On a fresh work directory this per-run path does not exist, so `tools/train_refcoco_opsd_impl.sh` exited during argument validation.
- `tools/plot_training_metrics.py` treated "no metrics yet" as a fatal error even in `--watch` mode, which is normal at startup.

### Chosen Fix Direction
- Align launcher defaults with the shared confuser-pool directory used by the configs.
- Make the metric plot watcher tolerate empty logs until the first training metrics arrive.

### Rejected Direction
- Do not auto-create an empty confuser-pool directory just to pass validation. That would hide a real missing-data problem and shift the failure deeper into training.

### Implemented Changes
- Updated `tools/train.sh`, `tools/train1.sh`, `tools/train_resume.sh`, `tools/export.sh`, and `tools/testconf.sh` so `SAM_CONFUSER_POOL_DIR` defaults to `${PROJECT_ROOT}/work_dirs/refcoco_sam_confuser_pool`.
- Updated `tools/plot_training_metrics.py` so `--watch` prints a waiting message instead of raising when metrics are not available yet.

### Follow-up Correction
- The earlier attempt to align launchers to a shared `refcoco_sam_confuser_pool` directory was the wrong design for the current workflow.
- The actual expected layout is a per-work-dir `sam_confuser_pool`, so the configs were updated to point at `.../sam_confuser_pool` under each training work directory, and the launcher defaults were reverted to `${WORK_DIR}/sam_confuser_pool`.
