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

### Second Follow-up
- A fresh server still failed because the per-work-dir `sam_confuser_pool` was expected to exist before training.
- Updated `tools/train.sh`, `tools/train1.sh`, and `tools/train_resume.sh` so they generate the confuser pool in-place with `tools/conf.sh` when the directory is missing, using the requested GPU shard layout before starting training.

## 2026-06-10 Sa2VAChatModel LoRA Injection Failure

### Problem
- Training failed during model construction with `AttributeError: 'Sa2VAChatModel' object has no attribute 'model'`.

### Root Cause Notes
- `projects/sa2va/models/sa2va_opsd_v2.py` applied PEFT preparation and LoRA wrapping to `model.model`, assuming a generic wrapper layout.
- The actual `Sa2VAChatModel` exposes the trainable LLM as `language_model`, not `model`.
- The auto-generated LoRA target module names were also scoped as `language_model.<name>`, which is incorrect when PEFT is attached directly to the language model object.

### Chosen Fix Direction
- Apply `prepare_model_for_kbit_training` and `get_peft_model` directly to `language_model`.
- Generate target module names relative to `language_model` itself.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so student LoRA preparation targets `language_model`, enables input grads when supported, and writes the wrapped module back to `model.language_model`.

### Follow-up Correction
- The server PEFT version does not support the newer `use_activation_checkpointing` keyword in `prepare_model_for_kbit_training`.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` to inspect the function signature and only pass that keyword when the installed PEFT version supports it.

## 2026-06-10 Remote Predict Forward Signature Mismatch

### Problem
- Training reached the first forward pass, but generation failed with `TypeError: Sa2VAChatModel.predict_forward() got an unexpected keyword argument 'max_new_tokens'`.

### Root Cause Notes
- The local repository version of `projects/sa2va/hf/models/modeling_sa2va_chat.py` accepts decoding kwargs in `predict_forward()`.
- The runtime model loaded through `trust_remote_code=True` on the server exposed an older `predict_forward()` signature that does not accept those kwargs.
- OPSD passed decoding controls unconditionally once generation started.

### Chosen Fix Direction
- Filter `predict_forward()` kwargs against the runtime method signature before calling it.
- Keep the OPSD caller-side decoding controls for newer model versions, but degrade gracefully on older remote-code snapshots.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so `_predict_forward_eval()` inspects the loaded model signature and drops unsupported kwargs unless the method accepts arbitrary `**kwargs`.

## 2026-06-10 Remote Generate DType Mismatch In GRPO

### Problem
- Training reached the `grpo_positive` route, then failed inside the runtime `Sa2VA-4B` `generate()` method with `RuntimeError: Index put requires the source and destination dtypes match`.

### Root Cause Notes
- The `trust_remote_code` runtime model under Hugging Face cache used a `generate()` implementation that writes `vp_embeds` into `input_embeds` after only moving devices, not aligning dtypes.
- Under bf16 training, `vp_embeds` could be `BFloat16` while `input_embeds` stayed `Float`, causing the indexed assignment to fail.
- Patching only the repository copy of `modeling_sa2va_chat.py` would not help because runtime execution uses the cached remote-code module.

### Chosen Fix Direction
- Monkey patch the loaded runtime model instance after load so `generate()` always casts visual prompt embeddings to `input_embeds.dtype` before indexed assignment.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so `_ensure_generation_ready()` patches the loaded `Sa2VAChatModel.generate()` implementation in-memory and enforces dtype alignment for `vit_embeds` and `vp_embeds`.

### Follow-up Correction
- After fixing the indexed assignment dtype mismatch, the runtime generation path still reached FlashAttention with `inputs_embeds` in `float32`.
- FlashAttention in the Qwen2 stack only accepts `fp16` or `bf16`, so the runtime patch was extended to cast `inputs_embeds` to the language model compute dtype immediately before `language_model.generate()`.

### Second Follow-up Correction
- The remaining `float32` upcast came from applying `prepare_model_for_kbit_training()` on a non-quantized `Sa2VA-4B` language model before LoRA wrapping.
- That helper is intended for k-bit preparation and can upcast embeddings or normalization-related states to `float32`, which then conflicts with FlashAttention in the Qwen2 generation path.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` to skip `prepare_model_for_kbit_training()` entirely for the current LoRA path and wrap the already-loaded bf16/fp16 language model directly with PEFT.

## 2026-06-11 OPSD Rolling Metrics And Training Diagnostics

### Problem
- Training logs and plots reported global cumulative metrics, which hid short-horizon regressions and made route quality harder to diagnose.
- Runtime logs showed anomalous `completion_len=2048` and `avg_caption_tokens≈414`, while teacher regenerate acceptance and GRPO reward signals stayed unexpectedly sparse.

### Root Cause Notes
- `projects/sa2va/models/sa2va_opsd_v2.py` emitted most training metrics from lifetime accumulators instead of a recent-iteration window, so later bad behavior was diluted by early good samples.
- Older remote-code `predict_forward()` snapshots can silently ignore caller-side generation kwargs; when that happens, caption generation can fall back to the model default `max_new_tokens=2048`.
- OPSD also re-tokenized cleaned captions without a local post-check, so a runaway raw generation could still become an oversized training completion.
- The existing logs exposed only coarse GRPO and teacher outcomes, not the key rates needed to separate gate strictness from confuser/reward sparsity.

### Chosen Fix Direction
- Keep existing cumulative counters for compatibility, but switch the surfaced training metrics to a rolling window over the last 10 optimization updates.
- Enforce generation limits even when remote `predict_forward()` does not accept decoding kwargs, and add a local completion truncation guard before captions enter training losses.
- Add rolling diagnostics for teacher positive-IoU gain and GRPO zero-variance / nonzero-reward / missing-confuser rates.

### Rejected Direction
- Do not remove the cumulative counters entirely. They are still useful for offline analysis and changing every internal counter would add unnecessary migration risk.
- Do not immediately relax teacher gates or redesign GRPO reward in the same patch; first improve observability and stop the runaway generation path.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - Added a configurable `rolling_metric_window_iters` window and switched surfaced training metrics to use the recent window instead of lifetime averages.
  - Added fallback generation-config overrides inside `_predict_forward_eval()` so old remote-code models still honor OPSD decoding limits.
  - Added a local caption completion truncation guard based on `description_max_new_tokens`.
  - Added rolling diagnostics for `teacher_positive_gain_rate`, `teacher_iou_gain_mean`, `grpo_zero_reward_variance_rate`, `grpo_nonzero_reward_rate`, and `grpo_missing_confuser_rate`.

## 2026-06-11 Teacher Gate Relaxation And GRPO Rollout Expansion

### Problem
- Recent rolling diagnostics showed that teacher regenerate frequently improved IoU but still almost never entered CE supervision.
- GRPO was the dominant route, while the rollout group size had been temporarily reduced to 2 for debugging and produced relatively sparse pairwise reward diversity.

### Root Cause Notes
- The teacher regenerate gate required an IoU gain greater than `0.5`, which rejected many practically useful teacher captions with moderate but real improvement.
- The 4B RefCOCO training config still set `grpo_group_size=2`, limiting within-group reward variation.

### Chosen Fix Direction
- Restore `grpo_group_size` to 4 in the active 4B RefCOCO OPSD config.
- Relax teacher CE admission so regeneration is accepted either when IoU gain is greater than `0.5`, or when the teacher reaches at least `0.6` IoU and improves the student by at least `0.1`.

### Rejected Direction
- Do not remove the strong `>0.5` gate path entirely. Keep it as a high-confidence fast path and add the moderate-improvement clause alongside it.

### Implemented Changes
- Updated `projects/sa2va/configs/sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3.py` to set `grpo_group_size=4`.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so `_teacher_regenerate_gate_passed()` now returns true for either:
  - `teacher_iou - student_iou > 0.5`, or
  - `teacher_iou >= 0.6 and teacher_iou - student_iou >= 0.1`.

## 2026-06-11 Dense GRPO Reward For Confuser MCQ

### Problem
- The GRPO confuser reward was sparse: wrong argmax predictions usually received `0`, so many rollout groups produced low-variance or zero-variance reward signals.
- Recent diagnostics showed GRPO remained the dominant route, so sparse MCQ reward limited how often this route could provide useful policy gradients.

### Root Cause Notes
- `_score_caption_against_mask_options()` only rewarded the GT option probability when the predicted option matched the GT, ignoring the rest of the option distribution.
- The reward did not account for how much probability mass the model placed on confusers that are visually similar to the GT mask.

### Chosen Fix Direction
- Replace the sparse argmax-style reward with a dense expected reward over the full option distribution:
  - positive term: `p(gt)`
  - negative term: `sum_i p(confuser_i) * IoU(confuser_i, gt)`
- Keep PPO/GRPO rollout, clipping, and route logic unchanged.
- Add explicit logging for GT probability, confuser penalty, and unclipped reward mean.

### Rejected Direction
- Do not mix caption-quality reward or reconstruction-IoU reward into this change. Keep the dense reward local to the confuser MCQ path so its effect is interpretable.
- Do not only penalize the chosen wrong option; use the full distribution so reward stays dense even when argmax is correct but confuser mass is high.

## 2026-06-12 Teacher Diagnosis Specificity Upgrade

### Problem
- Teacher regenerate diagnosis often produced structurally present but semantically weak fields, especially `teacher_caption_problem`, which frequently collapsed into generic geometry restatements such as "misses a broad area".
- Even when diagnosis fields were logged, they were not specific enough to explain which target-only cue was missing or which distractor-only cue was pulling reconstruction away from the target.

### Root Cause Notes
- The upstream difference compression layer exposed mostly symmetric geometric summaries, so teacher could safely paraphrase them without producing phrase-level or cue-level diagnosis.
- The diagnosis prompt emphasized field format more than failure localization, encouraging generic restatements instead of concrete target-vs-distractor error attribution.
- The parser and validator treated any non-empty diagnosis text as potentially acceptable, but they did not push the model toward cue-grounded problem descriptions.
- Empty or malformed `REASON` fields caused hard failures, while weak `CAPTION_PROBLEM` fields still slipped through when present.

### Chosen Fix Direction
- Upgrade the program-side difference context into finer, asymmetric evidence bullets that expose target-only and distractor-only anchor cues separately.
- Strengthen the diagnosis prompt so `CAPTION_PROBLEM` must name a concrete missing target cue or wrong distractor cue rather than only paraphrasing broad spatial differences.
- Tighten diagnosis validation to require cue-grounded wording and minimum semantic content, while adding a parser-side fallback that backfills `REASON` from `likely_drift_reason` when needed.

### Rejected Direction
- Do not revert to the older heavy multi-stage fault-report pipeline just to get more structured text. The current issue is specificity and robustness of the light diagnosis path, not lack of available labels.
- Do not loosen validation into accepting any generic diagnosis sentence, because that would reintroduce low-value teacher supervision into regenerate.

### Implemented Changes
- Updated `projects/sa2va/evaluation/teacher_diagnosis_common.py`:
  - Added finer asymmetric evidence construction helpers for target-only and distractor-only regions.
  - Reformatted `target_only_evidence` and `distractor_only_evidence` into bullet-like cue lists with explicit target-vs-distractor anchors.
  - Rebuilt `likely_drift_reason` so it summarizes missing target cues versus distractor pull in more diagnosis-friendly natural language.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - Strengthened the `teacher_light_diagnosis` prompt to require concrete cue-level problem statements instead of generic broad-area paraphrases.
  - Added parser-side fallback that fills empty `REASON` from `likely_drift_reason`.
  - Added normalization that enriches overly generic `CAPTION_PROBLEM` with the first target-only cue when necessary.
  - Tightened diagnosis validation with cue-grounding, semantic-anchor, and minimum-length checks so successful diagnoses better reflect actual target/distractor differences.

## 2026-06-12 Teacher Regenerate Cumulative Rate Fix And Verification Caption Logging

### Problem
- The newly added cumulative/window teacher regenerate dual-output metrics produced impossible values greater than `1.0`, so the logged rates could not be interpreted.
- When inspecting the `teacher_regenerate` route, the logs did not surface the actual verification caption text used for gate reconstruction, which made it hard to audit why gate passes were rare.

### Root Cause Notes
- `teacher_regenerate_dual_output_rate`, `teacher_regenerate_verification_caption_valid_rate`, and `teacher_regenerate_verification_iou_mean` used counts collected from every teacher analysis attempt gated by `allow_teacher_ce`, but divided them by `teacher_regenerate_count`, which only counts samples whose final loss branch is `teacher_regenerate`.
- The per-sample pre-return debug summary already carried teacher verification caption fields in memory, but the formatted log line did not print them.

### Chosen Fix Direction
- Add a dedicated `teacher_regenerate_analysis_count` denominator for the dual-output / verification-caption metrics and use it for both rolling and cumulative log-only summaries.
- Extend the per-sample debug log text so `teacher_regenerate` route inspection includes `teacher_verification_caption` and its status directly in the emitted record.

### Rejected Direction
- Do not redefine these metrics against `teacher_regenerate_count`, because that would keep mixing route-assignment counts with teacher-analysis counts and continue to skew rates whenever teacher analysis runs outside the final regenerate branch.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - Extended `ConfuserSelectionResult` with dense reward metadata.
  - Replaced sparse MCQ reward with `clip(p_gt - sum_i p_confuser_i * IoU(confuser_i, gt), -1, 1)`.
  - Added GRPO aggregation and rolling metrics for:
    - `grpo_reward_raw_mean`
    - `grpo_gt_prob_mean`
    - `grpo_confuser_penalty_mean`
  - Extended GRPO per-rollout debug logging to print dense reward components.

## 2026-06-11 Caption/Segmentation Mode Drift Guardrails

### Problem
- Mid-training caption generation could drift into segmentation-answer templates such as `Sure, the segmentation result is [SEG].`, after which student captions became invalid and the main optimization routes stopped providing useful training signal.

### Root Cause Notes
- Caption generation and segmentation generation share the same model, special tokens, and region-prompt injection path, so the model can fall back to the strong `[SEG]` answer prior when caption-mode control weakens.
- The previous status pipeline cleaned `[SEG]` out of raw text before classification, so seg-style failures were often misreported as `truncated_caption`.
- On-policy distillation and GRPO only checked the cleaned status, which made early caption degradation hard to distinguish from normal short captions and obscured route-level blocking reasons.

### Chosen Fix Direction
- Detect seg-style failures on the raw caption output before cleanup.
- Add caption-only decode guardrails that ban core segmentation tokens where the runtime generation interface allows it.
- Block seg-style and truncated captions from on-policy / GRPO self-reinforcement, and give seg-style failures a dedicated teacher-recovery escape hatch without changing the normal teacher gate for other samples.

### Rejected Direction
- Do not change reconstruction prompts, GRPO reward formulas, or the normal teacher gate semantics for non-caption-mode-failure samples in this patch.
- Do not rely on cleanup-only handling, because once `[SEG]` is stripped from the text the failure mode becomes ambiguous in logs and metrics.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - Added raw caption failure-mode detection and carried it through `DescriptionResult`.
  - Passed caption-only `bad_words_ids` for `[SEG]` / segmentation phrases through the caption generation path, with existing generation-config fallback handling preserved for older remote-code runtimes.
  - Blocked `seg_style_answer`, `truncated_caption`, `empty`, and `decode_error` captions from on-policy and GRPO entries via a unified trainability check.
  - Allowed seg-style caption failures to use teacher regenerate as a recovery path when the teacher returns a valid caption, without applying the normal IoU gate to that specific failure class.
  - Added rolling metrics and debug fields for raw seg-style rate, caption-mode failure rate, seg-style route blocking counts, and teacher recovery counts.

### Follow-up Correction
- The first version initialized caption `bad_words_ids` before `self.tokenizer` was constructed, which caused model build to fail with `AttributeError: 'Sa2VAOPSDModelV3' object has no attribute 'tokenizer'`.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so caption bad-word tokenization happens immediately after tokenizer initialization instead of earlier in `__init__`.

## 2026-06-11 Teacher Regenerate DLC + Verification Caption

### Problem
- Teacher regenerate was expected to use privileged mask information to produce better long DLC targets, but long-caption reconstruction gate pass remained too low to serve as a reliable supervision source.
- The verifier path appears to prefer shorter, more directly referential descriptions than the full detailed caption needed by the main task.

### Root Cause Notes
- Teacher regenerate previously used a single long caption for both CE supervision and reconstruction IoU gate validation.
- This coupled the DLC training target to a verifier that is more stable on shorter verifier-friendly phrases, making teacher usefulness look near-random even when the long caption still carried useful detail.

### Chosen Fix Direction
- Split teacher regenerate outputs into:
  - `DLC`: the long detailed localized caption used for CE
  - `VERIFICATION_CAPTION`: a shorter, still discriminative caption used only for reconstruction IoU gate validation
- Keep student route IoU, on-policy, and GRPO tied to long captions so the main task does not collapse into short referring expressions.
- Add matching window metrics plus cumulative log-only diagnostics for teacher dual-output quality.

### Rejected Direction
- Do not switch student route IoU, on-policy, or GRPO to short captions in this patch.
- Do not train CE on the verification caption. It is a gate-only verifier text, not the primary caption target.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - Added a teacher regenerate dual-output prompt contract with strict `DLC:` and `VERIFICATION_CAPTION:` fields.
  - Added dedicated parsing and status handling for teacher dual-output captions.
  - Switched teacher regenerate gate reconstruction to use only the generated verification caption.
  - Kept regenerate CE supervision on the detailed caption only.
  - Added window metrics and cumulative log fields for dual-output parse rate, verification-caption validity, verification IoU, and DLC CE application counts.

## 2026-06-12 Teacher Regenerate Four-Stage Diagnosis Pipeline

### Problem
- The single-step teacher regenerate prompt still let the teacher jump straight to rewriting a caption, so privileged mask information was not being converted into an explicit, auditable diagnosis of why the student caption first failed.
- Low regenerate gate pass rate remained hard to interpret because logs did not distinguish whether failure came from poor diagnosis, weak rewrite planning, bad DLC generation, or weak verification caption generation.

### Root Cause Notes
- The previous teacher regenerate flow combined diagnosis, rewrite planning, DLC generation, and verification caption generation in one response, so there was no structured intermediate signal to validate or filter before CE.
- Regenerate logging tracked only end-state caption validity and gate outcomes, which hid where the teacher pipeline broke down.

### Chosen Fix Direction
- Replace the single-step teacher regenerate output with a four-stage pipeline:
  - structured fault report
  - structured repair plan
  - DLC generation
  - verification-caption generation
- Validate each stage before allowing the next one to run, and stop early with a stage-specific failure reason when diagnosis or rewrite structure is not usable.
- Keep teacher regenerate CE on the DLC only and keep gate reconstruction on the verification caption only.

### Rejected Direction
- Do not keep the old dual-output regenerate prompt as the main path, because it still mixes diagnosis and generation too early.
- Do not add new training losses in this patch; the new structure is used only for teacher-side generation, filtering, gating, and diagnostics.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - Replaced the old teacher dual-output result with a richer four-stage pipeline result object.
  - Added structured parsing and validation for fault-report and repair-plan outputs.
  - Added dedicated generation paths for fault report, repair plan, DLC, and verification caption.
  - Switched teacher regenerate analysis to run the staged pipeline, stop early on invalid intermediate outputs, and surface stage-specific failure reasons.
  - Added rolling metrics and cumulative log-only diagnostics for fault-report validity, repair-plan validity, DLC validity, verification gate pass rate, low-confidence diagnosis rate, and non-empty missing/distractor evidence rates.
  - Extended debug logs to print teacher diagnosis fields, rewrite fields, verification caption, and pipeline stop stage/failure reason.

### Follow-up Cleanup
- Removed teacher regenerate code paths that became unreachable after the four-stage migration:
  - the old `teacher_summary_template` storage
  - the unused `_mask_summary()` helper
  - the obsolete `generation_mode="regenerate_caption"` prompt branch
- Kept compatibility-facing counters and log names that are still read by training dashboards, even when their internal semantics now reflect the staged pipeline.

### Follow-up Fix
- The first four-stage training run crashed in rolling metric aggregation with `KeyError: 'teacher_regenerate_analysis_count'`.
- Root cause: `_window_metric_counts()` was not updated to include the new teacher pipeline counter keys, so `window_totals` omitted them when old and new metric windows were aggregated together.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` to register all new teacher pipeline count metrics in `_window_metric_counts()` so rolling stats remain backward-compatible during live training.

### Second Follow-up Fix
- A later cleanup removed `_mask_summary()` from `projects/sa2va/models/sa2va_opsd_v2.py`, but `projects/sa2va/evaluation/teacher_diagnosis_common.py` still calls `model._mask_summary(...)` while building teacher privileged relation context.
- This caused teacher regenerate to fail immediately at fault-report prompt construction with `AttributeError: 'Sa2VAOPSDModelV3' object has no attribute '_mask_summary'`.
- Restored `_mask_summary()` as a compatibility helper because it remains part of the active teacher diagnosis call chain.

### Third Follow-up Fix
- After the four-stage pipeline started running, logs showed every teacher sample stopping at `fault_report`, with common failures `missing_ref_summary` and `missing_evidence_for_failure`.
- Root cause: the first-stage fault report prompt was too verbose and the parser was too brittle for real model outputs, so labels could bleed into the next field and many required fields came back empty.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` to:
  - parse fault-report fields with a multi-label section parser instead of one-label-at-a-time extraction
  - shorten and harden the fault-report prompt with explicit single-line field formatting
  - backfill missing fault-report fields from mask relation context and the student caption so stage 1 does not collapse when the teacher output is partially structured

## 2026-06-12 Teacher Regenerate Difference-Driven Compression Migration

### Problem
- The four-stage `fault_report -> repair_plan -> DLC -> verification` teacher regenerate pipeline stayed structurally valid in code but failed almost entirely in real training, with the teacher stopping at the first structured stage and never producing useful regenerate CE targets.
- The main training path still depended on heavy schema fields that the teacher did not follow reliably, so teacher regenerate observability improved while actual teacher utility regressed.

### Root Cause Notes
- The teacher was being asked to do too much structure induction itself: compare masks, classify failure types, fill phrase-level repair forms, then write captions.
- Even after parser hardening, the first structured stage remained the bottleneck because the model interface was mismatched to the task.
- The old historically successful version used a much lighter diagnosis interface, so the current main path had drifted away from the most robust teacher behavior.

### Chosen Fix Direction
- Keep the long DLC target and separate verification caption gate text, but move mask comparison back to the program side.
- Introduce a difference-driven natural-language compression layer that summarizes `gtmask` vs `refmask` as `target/distractor/shared/target-only/distractor-only` evidence plus localization hints.
- Replace the heavy main-path teacher schema with a light three-field diagnosis:
  - `CAPTION_PROBLEM`
  - `CORRECTION_DIRECTION`
  - `REASON`

### Rejected Direction
- Do not continue expanding the four-stage fault-report and repair-plan schema in the training main path. Logs already showed that more structure was decreasing teacher usability.
- Do not revert the DLC target back to short RefCOCO expressions. Keep the training target as DLC and only make the gate text verifier-friendly.

### Implemented Changes
- Updated `projects/sa2va/evaluation/teacher_diagnosis_common.py`:
  - added `build_teacher_regenerate_difference_context(...)`
  - compressed mask relation information into `target_summary`, `distractor_summary`, `shared_evidence`, `target_only_evidence`, `distractor_only_evidence`, localization hints, and a template-built `likely_drift_reason`
  - added a de-homogenization guard so target/distractor summaries are not left identical when unique evidence exists
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - redefined the active `TeacherRegeneratePipelineResult` around difference context, lightweight diagnosis, DLC, and verification caption
  - added `generate_teacher_light_diagnosis(...)` and `validate_teacher_light_diagnosis(...)`
  - switched the active regenerate pipeline to:
    1. program-side difference compression
    2. teacher lightweight diagnosis
    3. teacher DLC generation
    4. teacher verification caption generation
    5. verification-caption reconstruction gate
  - changed DLC validation to penalize distractor-heavy overlap and require target-only evidence when available
  - changed verification validation to require shorter verifier text, preserve target-only evidence, and reject distractor overlap
  - updated debug sample logging, rolling metrics, and cumulative log-only stats to report the new difference/diagnosis fields instead of the overloaded fault-report/repair-plan counters

### Compatibility Note
- Legacy heavy fault-report and repair-plan helper functions are still present temporarily for compatibility, but the training main path no longer depends on them.

### Follow-up Logging Fix
- The first difference-driven version still only exposed verification caption and diagnosis fields in per-sample debug records, which made quick training-log inspection inconvenient.
- The same main log line also retained accidental `0.0=...` placeholder fragments after the metric rename pass.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so the standard training log line now prints:
  - `teacher_verification_caption`
  - `teacher_caption_problem`
  - `teacher_correction_direction`
  - `teacher_reason`
- Removed the stray `0.0=...` placeholders from the same log output.

### Follow-up Fix For Natural-Language Difference Evidence
- The first difference-driven compression still exposed `target_only_evidence` and `distractor_only_evidence` mainly as `area_ratio / bbox / center` strings.
- In logs, the teacher often produced semantically correct diagnosis text, but not by copying those numeric mask summaries, so `teacher_diagnosis_valid_rate` remained at `0.0`.
- Updated the active difference-compression path so `projects/sa2va/evaluation/teacher_diagnosis_common.py` now converts mask-only differences into natural-language spatial evidence such as broad-area vs small-patch and left/right or upper/lower cues, while retaining the coarse localization hint.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so diagnosis validation now accepts either:
  - direct phrase overlap with those natural-language evidence snippets, or
  - semantic spatial anchors in `reason` / `correction_direction`
- This keeps the validator meaningful while no longer requiring the teacher to parrot raw numeric mask summaries.

### Follow-up Confirmation For Main Log Text Fields
- Verified that the active main training log in `projects/sa2va/models/sa2va_opsd_v2.py` no longer contains the accidental `0.0=0.0000` placeholder fragments.
- The standard `[Sa2VA_OPSD_V2]` line now prints:
  - `teacher_verification_caption`
  - `teacher_caption_problem`
  - `teacher_correction_direction`
  - `teacher_reason`
- If future remote logs still show `0.0=0.0000`, that indicates the server is running an older synced copy rather than the current workspace version.

### Follow-up Fix For Empty / Interleaved REASON And Weak Difference Asymmetry
- After the first natural-language difference patch, training logs showed two remaining bottlenecks:
  - `REASON` was still often empty or partially leaked into `CORRECTION_DIRECTION`
  - `target_only_evidence` and `distractor_only_evidence` were still too symmetric, so the teacher often produced generic diagnosis text even when the validator accepted spatial language
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` to:
  - strengthen the `teacher_light_diagnosis` prompt so `REASON` must be exactly one complete sentence
  - force `REASON` to mention target-only or distractor-only cues with concrete spatial or size language
  - add a parser-side recovery path that extracts `REASON` text if it was accidentally appended to `CORRECTION_DIRECTION`
- Updated `projects/sa2va/evaluation/teacher_diagnosis_common.py` to:
  - add stronger asymmetric difference descriptors based on relative size and relative location
  - enrich `target_only_evidence` / `distractor_only_evidence` with area contrast and offset wording so the two sides are less likely to collapse into the same sentence template
- The goal of this patch is to move the regenerate pipeline past `diagnosis_invalid:missing_reason` and reduce near-identical target/distractor evidence text.
