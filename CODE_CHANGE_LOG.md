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

## 2026-06-13 DLC RJob Launcher Missing Wrapper Environment

### Problem
- `tools/traindlc.sh` submitted the DLC training job successfully, but the remote run failed immediately with:
  - `SA2VA_REFCOCO_OPSD_CONFIG must be set by the entry wrapper.`

### Root Cause Notes
- `tools/train_refcoco_opsd_impl.sh` is not a standalone launcher. It requires an outer wrapper to inject the `SA2VA_REFCOCO_OPSD_*` default environment variables before argument parsing.
- `tools/traindlc.sh` called `train_refcoco_opsd_impl.sh` directly, unlike the standard `tools/train_refcoco_opsd_4b.sh` wrapper path.

### Chosen Fix Direction
- Keep `traindlc.sh` as the rjob submit wrapper, but inject the required `SA2VA_REFCOCO_OPSD_*` defaults inline before calling `train_refcoco_opsd_impl.sh`.

### Rejected Direction
- Do not weaken `train_refcoco_opsd_impl.sh` into a partially standalone script. That would duplicate launcher-default logic and drift from the existing 2B/4B entry design.

### Implemented Changes
- Updated `tools/traindlc.sh` so the remote training command now exports:
  - `SA2VA_REFCOCO_OPSD_CONFIG`

## 2026-07-05 RefCOCO Route Export MMEngine Optimizer Collision

### Problem
- RefCOCO route export failed before model loading in the host venv with:
  - `KeyError: 'Adafactor is already registered in optimizer at torch.optim'`
- The DLC route wrapper also still rewrote `DATA_ROOT` back to the parent directory, which conflicted with the now-supported flat RefCOCO snapshot layout where `refs(unc).p`, `instances.json`, and `train2014/` live directly under the passed directory.

### Root Cause Notes
- `tools/export_opsd_dlc_routes.py` and `tools/export_opsd_routes.py` load configs via `runpy.run_path(...)`, which imports `mmengine.optim.optimizer.builder` during config evaluation.
- In the host environment, `Adafactor` was already discoverable through the optimizer registry scope, so MMEngine's transformer optimizer registration raised on the duplicate instead of treating it as idempotent.
- The route shell wrappers still contained older "annotation root or its parent directory" logic and silently rewrote `train_dataset.data_root`, unlike the already-fixed confuser export path.

### Chosen Fix Direction
- Patch the route exporters before config load so MMEngine ignores duplicate `Adafactor` registration when the existing registry entry already matches the same optimizer class.
- Align both RefCOCO route wrapper scripts with the confuser exporter: the `--data-root` argument is now treated as the final RefCOCO annotation root and is passed through unchanged.

### Rejected Direction
- Do not pin or downgrade MMEngine/Transformers just to work around this host-only registry collision. The export tools should be robust across the current server environments.
- Do not keep mixed `DATA_ROOT` semantics between confuser export and route export, because that keeps reintroducing flat-layout path bugs.

### Implemented Changes
- Updated `tools/export_opsd_dlc_routes.py` and `tools/export_opsd_routes.py` to install a narrow MMEngine registry compatibility patch before config evaluation, skipping duplicate optimizer registration only for matching `Adafactor` entries.
- Updated `tools/export_refcoco_opsd_dlc_routes_impl.sh` and `tools/export_refcoco_opsd_routes_impl.sh` so they no longer rewrite `DATA_ROOT` to a parent directory or append `/refcoco`.
- Updated the wrapper help text to describe `--data-root` as the final RefCOCO annotation root.

### Follow-up Correction
- The first `Adafactor` compatibility patch only checked the current registry's local module table, but the actual host failure came from a duplicate already visible through the parent `torch.optim` scope.
- Updated both route exporters so the patch also checks `Registry.get("Adafactor")` and suppresses the duplicate even when MMEngine resolves the existing optimizer from a parent scope and raises later inside `_register_module`.

### Second Follow-up Correction
- The parent-scope equality check was still too strict for the host MMEngine/Transformers combination, because the duplicate registration is harmless for route export and may not expose a directly comparable existing module object before `_register_module` raises.
- Simplified the compatibility layer to a targeted message-based bypass: when optimizer registration raises the known `Adafactor is already registered in optimizer` duplicate error, the route exporters now skip it unconditionally and continue config loading.

## 2026-07-05 Route Export XTuner Optionality For Host Eval Venv

### Problem
- After bypassing the MMEngine optimizer collision, RefCOCO route export still failed during config import in the host evaluation venv with:
  - `ModuleNotFoundError: No module named 'xtuner'`
- Installing `xtuner` directly was blocked because its dependency chain pulled `tiktoken==0.13.0`, which fell back to source build and required a Rust compiler not available on the host.

### Root Cause Notes
- The route exporters load the training config with `runpy.run_path(...)`, so even export-only jobs execute top-level config imports.
- `projects/sa2va/configs/sa2va_opsd_combine_4b_dlc.py` imports `TrainLoop` from `xtuner.engine.runner`, but route export only needs the config object graph and never instantiates the training runner.

### Chosen Fix Direction
- Make the route exporters tolerant to a missing XTuner runtime by injecting a minimal `xtuner.engine.runner.TrainLoop` stub before config loading when XTuner is unavailable.

### Rejected Direction
- Do not require the host eval venv to install the full XTuner training stack just to export routes.
- Do not require Rust installation only to satisfy `tiktoken` during an export-only workflow.

### Implemented Changes
- Updated `tools/export_opsd_dlc_routes.py` and `tools/export_opsd_routes.py` to provide an XTuner `TrainLoop` stub during config import when `xtuner.engine.runner` is missing.

## 2026-06-28 HF Conversion RJob Empty Dedicated Log

### Problem
- `tools/converthf_ckpt.sh` submitted the HF export job, but the dedicated log file such as `work_dirs/convert_hf_iter_400.log` could stay empty even when the outer `rjob` log showed the container had started.

### Root Cause Notes
- The remote script only started appending to `LOG_FILE` when it reached the later `convert_to_hf.py | tee -a "${LOG_FILE}"` step.
- Any earlier failure during environment unpack, `apt update`, or dependency install happened before that tee pipeline, so the dedicated log remained empty and hid the real failure point.

### Chosen Fix Direction
- Start full-process log redirection immediately after the remote script initializes `LOG_FILE`, so every later command writes to the dedicated export log.

### Rejected Direction
- Do not rely on the outer `rjob` log alone. It is often truncated and does not give a stable per-export file for debugging repeated runs.

### Implemented Changes
- Updated `tools/converthf_ckpt.sh` so the remote bash process now runs with `exec > >(tee -a "${LOG_FILE}") 2>&1` immediately after log initialization.
- Added early `[convert] remote_job_started` and `[convert] log_file=...` markers so the dedicated log proves whether the remote script entered the export body at all.

## 2026-06-28 HF Conversion Blocked In Apt Install Stage

### Problem
- After fixing dedicated logging, the full HF conversion `rjob` log still ended during `apt install` output and never reached the Python environment check or `convert_to_hf.py`.

### Root Cause Notes
- `tools/converthf_ckpt.sh` installed extra Ubuntu system packages inside the remote job before running the actual conversion.
- In the current `rjob` image, that extra apt stage became the blocking point, so the export chain never reached the model conversion step.

### Chosen Fix Direction
- Default the conversion wrapper to use the `rjob` image plus unpacked `vlm_env` as-is, and skip the in-job apt installation unless explicitly requested for debugging.

### Rejected Direction
- Do not keep apt installation as the default path. That hides whether the real conversion logic works and repeatedly fails before model export even starts.

### Implemented Changes
- Updated `tools/converthf_ckpt.sh` to add `SKIP_APT_INSTALL`, defaulting to `1`.
- Added `--skip-apt-install 0|1` CLI control and remote log output for the chosen mode.
- Wrapped the previous apt source rewrite and `apt install` block behind `SKIP_APT_INSTALL != 1`; default behavior now skips apt and uses image-provided system libraries.

## 2026-06-28 DLC Export Prompt Alignment With Training

### Problem
- DLC-Bench exports from both the HF-path and `.pth`-path scripts were producing unusually short, generic captions and very low positive judge scores.

### Root Cause Notes
- The export scripts used the weak official-style query `Describe the masked region in detail.`, while OPSD training uses a much stronger mask-to-caption instruction that explicitly asks for a detailed, localized target description.
- This prompt mismatch made the export path a poor diagnostic for whether the trained model had actually learned the training task behavior.

### Chosen Fix Direction
- Align both DLC export scripts to use the same `DEFAULT_MASK_TO_CAPTION_QUESTION` prompt as training so export diagnostics reflect the trained caption objective.

### Rejected Direction
- Do not keep the weaker export prompt for this diagnostic run. It is still useful for strict official comparability, but it hides whether short captions are caused by prompt mismatch.

### Implemented Changes
- Updated `tools/eval_dlc_bench_official.py` to import `DEFAULT_MASK_TO_CAPTION_QUESTION` from `projects/sa2va/datasets/common.py` and use it as the default DLC export query.
- Updated `tools/eval_dlc_bench_official_pth.py` the same way so HF and `.pth` export paths stay aligned.

## 2026-06-30 DLC Judge Resume Loop For Flaky API Endpoint

### Problem
- Official DLC judge runs against the configured `gpt-5.5` endpoint were repeatedly timing out mid-run, leaving only partially populated `pred_eval_*.json` files and forcing manual restart.

### Root Cause Notes
- `tools/run_dlc_bench_judge_only.sh` invoked the official `eval_model_outputs.py` only once.
- The official evaluator already supports loading partial eval files and resuming, but the wrapper did not automate that behavior when the remote API failed partway through.

### Chosen Fix Direction
- Keep using the official evaluator and its saved partial JSON format, but wrap it in a bounded retry/resume loop in the local launcher so a flaky endpoint can eventually finish a 100-sample run.

### Rejected Direction
- Do not fork or heavily rewrite the official judge logic just to add retry behavior. The safer change is to retry at the wrapper level and let the official script resume from the saved eval file.

### Implemented Changes
- Updated `tools/run_dlc_bench_judge_only.sh` to:
  - derive the target eval JSON path from `pred_output` and `eval_suffix`
  - count completed entries in the saved eval file
  - rerun the official evaluator until the saved progress reaches the QA count or a configurable attempt limit is hit
  - expose `--resume-attempts` and `--resume-sleep-seconds` for tuning retry behavior

## 2026-07-04 Full Legacy Docker Environment For Sa2VA-4B

### Problem
- The target server could not install the full `uv sync --extra=legacy` environment on the host because the host `glibc` and compiler toolchain were too old for packages such as `decord2` and `contourpy`.

### Root Cause Notes
- The project pins packages that expect newer manylinux wheels or newer C/C++ build support than the host environment provides.
- Installing only inference dependencies was a workaround, but the user explicitly needed the full legacy environment that matches repository setup.

### Chosen Fix Direction
- Add a dedicated Dockerfile that uses a newer CUDA Ubuntu base image and runs `uv sync --extra=legacy` fully inside the container, isolating the project from host toolchain limitations.

### Rejected Direction
- Do not keep trying to force the host environment to compile or install the full dependency graph. That path is brittle and diverges from the repository's intended `uv` workflow.

### Implemented Changes
- Added `Dockerfile.legacy`:
  - based on `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
  - installs required system libraries and `uv`
  - installs Python 3.11 via `uv`
  - runs `uv sync --python 3.11 --extra=legacy --frozen`
  - places the synced virtual environment at `/opt/sa2va-venv`
  - routes Hugging Face and Torch caches to `/cache`

## 2026-07-04 Empty EXTRA_ARGS Expansion In conf.sh

### Problem
- `tools/conf.sh` failed before launching the exporter on some bash environments with `EXTRA_ARGS[@]: unbound variable` when no extra passthrough arguments were provided.

### Root Cause Notes
- The script runs with `set -u` and always expanded `"${EXTRA_ARGS[@]}"` even when the array was empty.
- On the target server's bash behavior, that empty-array expansion still triggered an unbound-variable failure.

### Chosen Fix Direction
- Only append `EXTRA_ARGS` to the exporter command when the array has at least one element.

### Rejected Direction
- Do not require callers to pass a dummy argument just to avoid the empty-array bug. The shell wrapper should handle the zero-extra-argument case itself.

### Implemented Changes
- Updated `tools/conf.sh` so the final exporter invocation branches on `EXTRA_ARGS` length and avoids expanding the empty array under `set -u`.

## 2026-07-04 RefCOCO Data Root Should Be Treated As Final Directory

### Problem
- The SAM confuser export path kept failing on datasets stored under a snapshot directory because the shell wrapper and exporter both tried to append an extra `/refcoco` layer automatically.

### Root Cause Notes
- `tools/conf.sh` treated any `DATA_ROOT` whose basename was not literally `refcoco` as a parent directory and rewrote it to `${DATA_ROOT}/refcoco`.
- `tools/export_refcoco_sam_confuser_pool.py` had the same fallback assumption in `normalize_refcoco_data_root`.
- The user's actual dataset layout already places `instances.json`, `refs(...).p`, and `train2014/` directly inside the provided snapshot directory.

### Chosen Fix Direction
- Stop guessing parent directories for this export flow.
- Treat the passed `DATA_ROOT` as the actual RefCOCO root directory unconditionally.

### Rejected Direction
- Do not keep the fallback-to-`/refcoco` behavior for `confuser` export. It makes snapshot-based layouts fail and hides the real directory contract.

### Implemented Changes
- Updated `tools/conf.sh` to use the provided `DATA_ROOT` directly and to document it as the RefCOCO root directory itself.
- Updated `tools/export_refcoco_sam_confuser_pool.py` so `normalize_refcoco_data_root()` no longer appends `/refcoco`.

## 2026-07-04 REFER Loader Support For Flat Snapshot Layout

### Problem
- Even after the SAM confuser wrapper stopped appending `/refcoco`, the downstream RefCOCO loader still tried to read files from `data_root/refcoco/...` and `data_root/images/mscoco/images/train2014`.

### Root Cause Notes
- `projects/sa2va/evaluation/utils/refcoco_refer.py` hardcoded `self.DATA_DIR = os.path.join(data_root, dataset)`.
- `projects/sa2va/datasets/refcoco_opsd.py` also preferred nested image locations and did not check `data_root/train2014`.
- The user's snapshot directory already stores `refs(unc).p`, `instances.json`, and `train2014/` directly at the passed root.

### Chosen Fix Direction
- Teach the core REFER loader and RefCOCO image-root resolver to recognize a flat snapshot-style dataset root directly, while preserving the older nested layout as fallback.

### Rejected Direction
- Do not keep relying on symlink wrappers around the dataset just to satisfy old path assumptions. The loaders should support the real on-disk layout directly.

### Implemented Changes
- Updated `projects/sa2va/evaluation/utils/refcoco_refer.py` so it uses the passed `data_root` directly when `refs(splitBy).p` and `instances.json` exist there.
- Updated the same loader to prefer `data_root/train2014` when present for RefCOCO-family datasets.
- Updated `projects/sa2va/datasets/refcoco_opsd.py` so `resolve_refcoco_image_root()` now checks `data_root/train2014` before the older nested fallback paths.

## 2026-07-04 SAM Confuser Export Still Passed Parent Directory

### Problem
- Even after fixing the shell wrapper and REFER loader, SAM confuser export could still resolve `refs(unc).p` under `.../snapshots/refcoco/...` instead of the provided snapshot directory.

### Root Cause Notes
- `tools/export_refcoco_sam_confuser_pool.py` still normalized `data_root` to `(root.parent, root)` and then passed the first value into `build_refcoco_opsd_records()`.
- That reintroduced the parent-directory assumption one layer deeper in the call chain.

### Chosen Fix Direction
- Make the exporter pass the provided snapshot directory itself as the canonical `data_root`.

### Rejected Direction
- Do not rely on the extra `refcoco_root` return value to preserve the older parent-directory contract. The explicit requirement now is that the provided directory is already the final dataset root.

### Implemented Changes
- Updated `tools/export_refcoco_sam_confuser_pool.py` so `normalize_refcoco_data_root()` returns `(root, root)` and the exporter no longer passes a parent directory into `build_refcoco_opsd_records()`.

## 2026-06-28 HF Conversion Remote Shell Here-Doc Quoting Failure

### Problem
- After the apt stage was skipped successfully, the remote conversion job still ended immediately with `here-document ... delimited by end-of-file (wanted 'EOF')`.

### Root Cause Notes
- `tools/converthf_ckpt.sh` embeds a long remote `bash -lc '...'` script inside a single-quoted shell string.
- The temporary Python config here-doc inside that remote script still contained single-quoted Python string literals, which prematurely terminated the outer single-quoted shell payload before the `EOF` marker could be reached.

### Chosen Fix Direction
- Remove single quotes from the embedded Python replacement lines so the full remote shell payload remains syntactically intact.

### Rejected Direction
- Do not keep debugging this as a here-doc formatting issue alone. The real breakage came from nested shell quoting, not from the `EOF` markers themselves.

### Implemented Changes
- Updated `tools/converthf_ckpt.sh` so the generated temporary Python config now uses double-quoted Python string literals in the `_text.replace(...)` lines, avoiding conflicts with the outer remote `bash -lc` single-quoted script body.

## 2026-06-28 Direct `.pth` Evaluation Entry Points For DLC And RefCOCO

### Problem
- The existing DLC-Bench and RefCOCO evaluation entry points were primarily built around HF-exported model directories, which forced an extra conversion step before evaluation.

### Root Cause Notes
- Current evaluation scripts instantiate `Sa2VAOPSDModelV3` from `model_path` / `tokenizer_path` directly.
- A plain training checkpoint such as `iter_400.pth` is only a state dict payload and cannot be consumed by those HF-style loaders without first rebuilding the model from the training config.

### Chosen Fix Direction
- Add dedicated evaluation scripts that rebuild the OPSD model from config, restore the `.pth` checkpoint, and then reuse the existing DLC export and RefCOCO caption-to-mask evaluation logic.

### Rejected Direction
- Do not overload the HF-oriented scripts with multiple incompatible loading modes in one patch. Separate `.pth` entry points are clearer and reduce regression risk for the existing HF path.

### Implemented Changes
- Added `tools/pth_eval_utils.py` with a shared `load_opsd_model_from_pth(...)` helper that:
  - loads the config
  - overrides `model_path` / `tokenizer_path` / runtime flags
  - builds the model
  - restores checkpoint weights with `guess_load_checkpoint`
- Added `tools/eval_dlc_bench_official_pth.py`:
  - exports official DLC-Bench `pred.json` directly from `config + checkpoint + base_model_path`
  - keeps the same official-format output contract and sidecar debug support
- Added `tools/eval_refcoco_caption_to_mask_pth.py`:
  - runs the existing RefCOCO caption-to-mask evaluation flow from `config + checkpoint + base_model_path`
  - preserves the current REFER sample-building and evaluator summary logic
- Added `tools/evaldlc_pth.sh` and `tools/evalrefcoco_pth.sh` as 1-GPU `rjob` wrappers for the new direct-`.pth` evaluation paths.

## 2026-06-28 Direct `.pth` Evaluation Missing `libGL.so.1`

### Problem
- The new direct-`.pth` DLC and RefCOCO evaluation `rjob` wrappers failed during Python import with `ImportError: libGL.so.1: cannot open shared object file`.

### Root Cause Notes
- The new wrappers reused the simplified no-apt startup path, but these evaluation flows import `transformers` -> `cv2`, which still requires system OpenGL-related shared libraries in the current runtime image.
- Skipping those libraries works for some pure-Python checks, but not for the actual evaluator import stack.

### Chosen Fix Direction
- Restore the small system-library install block for the direct-`.pth` evaluation wrappers by default, while keeping an explicit switch to disable it if the image is later fixed upstream.

### Rejected Direction
- Do not patch around this with Python-only import hacks. The failure is a missing native dependency in the container runtime, so the wrapper should provision the expected libraries explicitly.

### Implemented Changes
- Updated `tools/evaldlc_pth.sh` and `tools/evalrefcoco_pth.sh` to add `INSTALL_SYSTEM_LIBS`, defaulting to `1`.
- Added `--install-system-libs 0|1` CLI flags to both wrappers.
- Restored the apt source setup and `apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1` block in both wrappers when `INSTALL_SYSTEM_LIBS=1`.

## 2026-06-28 Direct `.pth` Evaluation Logs Mixed With Stale Failures

### Problem
- The refreshed `.pth` evaluation runs still appeared to show the old `libGL.so.1` traceback even after the wrapper had been changed to install system libraries first.

### Root Cause Notes
- The evaluation wrappers appended to their existing log files without truncating them first, so old failure traces remained at the top and made the newest run look broken before the fresh apt-install section.
- The lightweight startup probe only checked `import torch, transformers`, which could pass before the later `transformers -> PreTrainedModel -> cv2` import path failed.

### Chosen Fix Direction
- Make each evaluation run start from a clean log file and strengthen the startup import probe so it matches the real dependency path used by checkpoint evaluation.

### Rejected Direction
- Do not keep diagnosing from mixed old/new logs. The wrappers should make each run self-contained so the first visible traceback always belongs to the current job.

### Implemented Changes
- Updated `tools/evaldlc_pth.sh` and `tools/evalrefcoco_pth.sh` to truncate their log files at startup with `: >"${LOG_FILE}"`.
- Strengthened the startup check in both wrappers from `import torch, transformers` to `import cv2; import torch; from transformers import PreTrainedModel`, so missing `libGL` or related import issues fail immediately in the controlled preflight step.

## 2026-06-28 Switch `.pth` Eval Defaults To `iter_300` And Tighten DLC Cleaning

### Problem
- The direct `.pth` evaluation wrappers still defaulted to the later `iter_400` checkpoint, while the next debugging round needed both RefCOCO and DLC evaluation to target the same `iter_300` checkpoint in the training work directory.
- Exported DLC captions also contained a large amount of refusal text, mask-as-occluder explanations, and scene-level template prefixes that polluted downstream analysis.

### Root Cause Notes
- The wrappers only accepted checkpoint overrides but did not provide a shared default target checkpoint for the current round of investigation.
- The existing DLC export cleaner removed only shallow prompt prefixes and punctuation noise, but it did not filter refusal/occlusion captions or cut off low-signal trailing explanation clauses.

### Chosen Fix Direction
- Change both direct `.pth` evaluation wrappers to default to the same `iter_300.pth` checkpoint under the active training work directory.
- Tighten the DLC export cleaner so obviously invalid captions collapse to empty strings and weak multi-sentence explanations are reduced to a cleaner first sentence.

### Rejected Direction
- Do not keep `iter_400` as the default and rely on long command-line overrides for every run. The current debugging workflow benefits from aligned defaults across both evaluation paths.
- Do not leave refusal and black-box captions untouched in `pred.json`; those strings are not useful mask-to-caption outputs and make the exported DLC quality harder to inspect.

### Implemented Changes
- Updated `tools/evalrefcoco_pth.sh` and `tools/evaldlc_pth.sh` so `CHECKPOINT_PATH` now defaults to `.../work_dirs/sa2va_opsd_refcoco_sa2va4b_in25_qwen25_3b_v3_manifest/iter_300.pth`.
- Relaxed the wrapper usage text so `--checkpoint` remains supported but is no longer required when the shared default path is desired.
- Updated `tools/eval_dlc_bench_official_pth.py` cleaning logic to:
  - drop refusal-style captions
  - drop black-box / occlusion-explanation captions
  - strip more `The region in the image ...` style templates
  - keep only the first sentence and trim low-signal trailing clauses when possible

## 2026-06-28 Align Direct `.pth` Eval Wrappers With `train1.sh` Startup Shape

### Problem
- The direct `.pth` evaluation wrappers had drifted from the `train1.sh` remote startup shape by using full-process `exec > >(tee ...)` logging, which made the runtime behavior and log layout differ from the training launcher the user was comparing against.

### Root Cause Notes
- `train1.sh` prepares the environment, runs apt installation, and only tees the final training command into its dedicated log file.
- The direct `.pth` wrappers instead redirected the entire remote shell through `tee`, creating a different logging flow and making side-by-side comparison with the training launcher harder.

### Chosen Fix Direction
- Keep the same environment unpack and apt-install sequence, but change the direct `.pth` wrappers to follow the `train1.sh` pattern: initialize the log file and only pipe the final evaluation command through `tee -a`.

### Rejected Direction
- Do not keep the full-shell tee redirection when the explicit goal is to align behavior with `train1.sh`. Matching the training launcher shape makes the evaluation wrappers easier to reason about during debugging.

### Implemented Changes
- Updated `tools/evalrefcoco_pth.sh` and `tools/evaldlc_pth.sh` to remove the full-shell `exec > >(tee -a "${LOG_FILE}") 2>&1` redirection.
- Both wrappers now match `train1.sh` more closely by keeping the setup steps plain and piping only the final `stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"` evaluation command into the dedicated log.

## 2026-06-27 Teacher Regenerate DLC Reconstruction Should Not Be Pre-Blocked

### Problem
- The teacher regenerate path had already been simplified to a DLC-only single-stage generation flow, but training behavior still did not match the intended semantics.
- In practice, many teacher samples still stopped at `stop_stage=dlc`, so the generated DLC never reached reconstruction and gate evaluation.

### Root Cause Notes
- `projects/sa2va/models/sa2va_opsd_v2.py` still treated `_validate_teacher_dlc(...)` as a hard return condition inside `run_teacher_regenerate_pipeline(...)`.
- That meant local caption-shape heuristics such as `distractor_overlap` or `missing_target_only_evidence` could block reconstruction entirely, even when the user wanted the real decision to come from caption-to-mask reconstruction and gate improvement.

### Chosen Fix Direction
- Keep local DLC validation as a logging/debug signal, but remove it as a pre-reconstruction stop condition.
- Once the teacher raw output is non-empty, always reconstruct directly from the generated DLC and let the existing teacher gate decide whether CE should be applied.

### Rejected Direction
- Do not delete `_validate_teacher_dlc(...)` entirely in this patch. Its failure reasons are still useful to understand why a teacher DLC may be weak, even though they should no longer block reconstruction.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so `run_teacher_regenerate_pipeline(...)` now:
  - keeps local DLC validation as a debug signal instead of a hard pre-reconstruction stop
  - reconstructs directly from the generated teacher DLC once raw output is non-empty
  - lets the existing gate decide CE admission instead of pre-blocking on local caption heuristics

## 2026-06-27 DLC Official Judge Offline Wheel Support

### Problem
- The official `NVlabs/describe-anything` DLC-Bench judge path depended on runtime `pip install` from an online index.
- In the target training/eval environment, public package install was unreliable, and the local mirror did not consistently provide all required judge dependencies such as `inflect`.

### Root Cause Notes
- `tools/run_dlc_bench_judge_only.sh` only had an online-install fallback and did not support a user-prepared offline wheel repository.
- There was also no dedicated rjob submit wrapper for judge-only execution, so even after exporting `pred.json`, running the official judge in the remote environment still required manual shell work.

### Chosen Fix Direction
- Add explicit `--wheel-dir` support to the judge-only wrapper and install judge dependencies from a local wheel directory when provided.
- Add a dedicated `tools/judgedlc.sh` rjob wrapper so exported predictions can be judged remotely with the offline wheel repo and without hand-assembling commands inside the container.

### Rejected Direction
- Do not fold more fallback package logic into the export script. Export and judge are intentionally split so caption export remains usable even when official judge dependencies are unavailable.

### Implemented Changes
- Updated `tools/run_dlc_bench_judge_only.sh`:
  - validates `--wheel-dir` when provided
  - installs the official judge dependencies from the offline wheel directory with `pip --no-index --find-links`
  - installs the full small dependency set required by the official evaluator path: `inflect`, `typeguard`, `more_itertools`, `tqdm`, and `openai`
- Updated `tools/run_dlc_bench_official_eval.sh` usage text to document the offline `--wheel-dir` judge path.
- Added `tools/judgedlc.sh` as an rjob submit wrapper for judge-only execution with:
  - mounted project paths
  - wheel directory passthrough
  - container bootstrap
  - judge log output to `${OUTPUT_DIR}/judge_dlc.log`

### Follow-up
- Added `tools/judgedlc_baseline.sh` as a one-command baseline wrapper that pins:
  - `OUTPUT_DIR=${PROJECT_ROOT}/work_dirs/dlc_bench_eval_baseline_sa2va4b`
  - `PRED_OUTPUT=${OUTPUT_DIR}/pred.json`
  - the downloaded offline wheel repo path
- Updated `tools/evaldlc_ckpt.sh` to export the same offline `WHEEL_DIR` default used by the judge wrappers, so checkpoint-based export/eval entrypoints share one consistent DLC-Bench environment setup.
- Added `tools/judgedlc_ckpt.sh` as the checkpoint-parallel companion to `tools/evaldlc_ckpt.sh`:
  - requires `--model-path`
  - derives `OUTPUT_DIR=${PROJECT_ROOT}/work_dirs/dlc_bench_eval_<ckpt_name>` by default
  - derives `PRED_OUTPUT=${OUTPUT_DIR}/pred.json`
  - forwards the remaining judge options to the shared `tools/judgedlc.sh` rjob wrapper
- Added `tools/converthf_ckpt.sh` as an rjob submit wrapper for `tools/convert_to_hf.py`:
  - requires `--pth-model`
  - defaults the config to the active 4B RefCOCO OPSD config
  - derives `SAVE_PATH=${PROJECT_ROOT}/work_dirs/hf_<checkpoint_stem>` by default
  - writes conversion logs to `$(dirname SAVE_PATH)/convert_<save_dir_name>.log`
- Follow-up correction:
  - `tools/convert_to_hf.py` accepts `pth_model` as a positional argument, not `--pth-model`
  - updated `tools/converthf_ckpt.sh` to call `convert_to_hf.py <config> <pth_model> --save-path ...` so the rjob wrapper matches the real CLI signature
- Updated `tools/judgedlc.sh`, `tools/judgedlc_baseline.sh`, and `tools/judgedlc_ckpt.sh` so the rjob submit path now includes explicit judge-service placeholders in the script body:
  - `LLM_ENGINE_PATH=https://api.openai.com/v1`
  - `API_KEY=YOUR_OPENAI_API_KEY_HERE`
  - baseline/ckpt wrappers now export those defaults directly, so the user can modify the script-local values without remembering extra environment variable names
- Renamed the judge submit wrapper variable from the misleading `API_KEY_PATH` to `API_KEY` in the rjob path, because the official judge expects a raw API key string rather than a filesystem path
  - keeps `teacher_dlc_invalid:*` as a logged failure reason only,
  - always attempts reconstruction from `pipeline_result.detailed_caption` after single-stage DLC generation,
  - uses gate/reconstruction outcome as the real stop condition,
  - records `teacher_gate_failed:reconstruct_failed` or `teacher_gate_failed:iou_not_improved_enough` when CE is rejected after reconstruction.

## 2026-06-27 DLC-Bench Official Export COCO Image Lookup Compatibility

### Problem
- The new official DLC-Bench export script failed immediately on downloaded official data with:
  - `KeyError: Missing image name fields in annotation ann_id=...`

### Root Cause Notes
- `tools/eval_dlc_bench_official.py` assumed each annotation directly carried `image_name` or `file_name`.
- The actual downloaded DLC-Bench annotations can be COCO-style, where annotations reference `image_id` and the filename lives in the top-level `images` table.

### Chosen Fix Direction
- Keep direct per-annotation filename fields as the first preference.
- Add COCO-style fallback by building an `image_id -> image metadata` lookup from top-level `images` and resolving filenames from there.

### Implemented Changes
- Updated `tools/eval_dlc_bench_official.py` so it now:
  - returns both annotation rows and the raw annotation payload from `_load_annotations(...)`,
  - builds a top-level `images` lookup when present,
  - resolves image names from `annotation.image_id` when the annotation itself does not include a direct filename field.

## 2026-06-27 Remote Predict-Forward Needs Explicit Image Placeholder

### Problem
- DLC-Bench evaluation progressed past data loading but then crashed inside the remote `Sa2VA-4B` generation path with:
  - `AssertionError` at `assert selected.sum() != 0`

### Root Cause Notes
- The runtime `trust_remote_code` implementation of `predict_forward()` expects the caller text to contain an explicit `<image>` placeholder so it can inject image tokens before tokenization.
- The local OPSD helper path can build image-token prompts itself, but `_predict_forward_eval(...)` was still forwarding plain text such as `Describe the masked region in detail.` to older remote-code implementations.
- As a result, the final `input_ids` contained no image-context token slots, and the patched `generate()` path failed when trying to place visual embeddings.

### Chosen Fix Direction
- Add a caller-side compatibility shim in `_predict_forward_eval(...)`.
- Whenever visual input or `mask_prompts` are present and the text does not already contain `<image>`, prepend `<image>` automatically before calling the remote `predict_forward()`.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so `_predict_forward_eval(...)` now auto-prepends `<image>\n` to `text` for visual or mask-prompt inference calls when the placeholder is missing.

## 2026-06-27 DLC-Bench Official Judge Missing Runtime Dependencies

### Problem
- DLC-Bench prediction export succeeded, but the second evaluation stage failed immediately inside the official `describe-anything` judge script with:
  - `ModuleNotFoundError: No module named 'inflect'`

### Root Cause Notes
- `tools/run_dlc_bench_official_eval.sh` intentionally delegates scoring to the official `evaluation/eval_model_outputs.py`.
- The remote job environment used by the wrapper did not guarantee that the official judge's Python-side dependencies were preinstalled.

### Chosen Fix Direction
- Keep using the official judge entrypoint unchanged.
- Add lightweight dependency checks in the wrapper and install missing judge packages on demand before launching the official script.

### Implemented Changes
- Updated `tools/run_dlc_bench_official_eval.sh` so it now checks and installs missing `inflect`, `tqdm`, and `openai` packages via `${PYTHON_BIN} -m pip install ...` before running the official evaluation.

## 2026-06-27 DLC-Bench Official Export Caption Cleanup For Region Markers

### Problem
- Official DLC-Bench export was producing many captions with task-specific marker phrasing such as:
  - `In region1, ...`
  - `The region1 contains ...`
  - `The masked region in the image is ...`
- These are artifacts of OPSD training/task prompts, not desirable final DLC-style captions for official evaluation.

### Root Cause Notes
- The model's native `clean_caption` is shared with training behavior and intentionally preserves much of the generated sentence content.
- For official DLC-Bench scoring, that shared cleanup is not sufficient because it does not strip benchmark-irrelevant task markers like `region1`.

### Chosen Fix Direction
- Add a dedicated official-eval-only cleanup layer in the export script rather than changing the training-time caption normalization.
- Remove leading `region1` / `masked region` template phrases while leaving the rest of the sentence intact.

### Implemented Changes
- Updated `tools/eval_dlc_bench_official.py` so it now:
  - applies `_clean_official_eval_caption(...)` before writing `pred.json`,
  - strips common leading patterns such as `In region1,`, `The region1 contains`, `The target in region1 is`, and `The masked region in the image is`,
  - preserves the original model-cleaned caption separately in the debug sidecar as `model_clean_caption`.

## 2026-06-27 Training Plot Should Show Teacher CE Application Counts

### Problem
- The training plot could show non-zero `teacher_regenerate_gate_pass_rate` while `opsd_regen_ce` stayed at zero on many sampled iterations.
- That made it hard to tell whether regen CE was genuinely absent or simply not applied on the plotted logger steps.

### Root Cause Notes
- `opsd_regen_ce` is already recorded by the training loop, but it only reflects the actual regen loss value for iterations that contain regen entries.
- The plotting script did not include the teacher CE application counters that would explain when gate-passed teacher outputs were really admitted into CE supervision.

### Chosen Fix Direction
- Keep the existing `opsd_regen_ce` loss plot as the true regen-loss signal.
- Expand the plotting defaults to also visualize teacher regenerate CE application and verified/rejected counts.

### Implemented Changes
- Updated `tools/plot_training_metrics.py` so the default plotted metrics now also include:
  - `teacher_regenerate_ce_applied_count`
  - `teacher_regenerate_verified_count`
  - `teacher_regenerate_rejected_count`
  - `teacher_regenerate_dlc_ce_applied_count`

## 2026-06-27 DLC Eval Remote Image Setup Should Preinstall Judge Dependencies

### Problem
- The DLC-Bench rjob wrapper could finish model export and only then fail inside the official judge because the remote container was missing packages like `inflect`.

### Root Cause Notes
- Dependency checks had been added in the local wrapper path, but the remote image bootstrap in `tools/evaldlc.sh` still only guaranteed base system libs plus `torch`/`transformers`.
- That meant the remote job environment was not explicitly provisioned with the official judge's Python dependencies during startup.

### Chosen Fix Direction
- Mirror the `train1.sh` style of environment preparation directly in the remote image command.
- Install the official DLC judge dependencies as part of the rjob bootstrap, before launching the evaluation wrapper.

### Implemented Changes
- Updated `tools/evaldlc.sh` so the remote job bootstrap now runs:
  - `/opt/vlm/bin/python -m pip install -q inflect tqdm openai`
  after unpacking the environment and verifying the base Python stack.

### Follow-up Correction
- The first dependency-install patch still used the default public `pypi.org` index, which timed out inside the rjob container.
- Updated both `tools/evaldlc.sh` and `tools/run_dlc_bench_official_eval.sh` to install Python dependencies through the PJLab mirror:
  - `https://mirrors.h.pjlab.org.cn/pypi/web/simple`

### Second Follow-up Correction
- The first mirror-based patch still failed in the rjob container because the `https://mirrors.h.pjlab.org.cn/...` endpoint returned connection refused during pip install.
- Updated the DLC evaluation wrappers to:
  - use the mirror via `http://mirrors.h.pjlab.org.cn/pypi/web/simple`,
  - add `--trusted-host mirrors.h.pjlab.org.cn`,
  - remove the unconditional pip install from the remote bootstrap and keep dependency installation only in the wrapper's missing-package check path.

## 2026-06-27 Split DLC Export And Official Judge Into Two Scripts

### Problem
- The combined DLC-Bench workflow kept mixing two separate concerns:
  - generating `pred.json`
  - running the official LLM judge
- In the current environment, export is stable but the judge side is fragile because Python dependency availability depends on network/mirror state.

### Root Cause Notes
- A single all-in-one script made the full evaluation job fail even when the useful artifact (`pred.json`) had already been generated correctly.
- The existing rjob wrapper also defaulted to running both stages together, which is not a good fit for the current constrained runtime.

### Chosen Fix Direction
- Split the workflow into two first-class scripts:
  - export only
  - judge only
- Keep a compatibility wrapper that can still run both sequentially when the environment supports it.
- Change the rjob export wrapper to default to the export-only stage.

### Implemented Changes
- Added `tools/run_dlc_bench_export_only.sh` to generate official-format `pred.json` and optional debug output only.
- Added `tools/run_dlc_bench_judge_only.sh` to score an existing `pred.json` with the official `describe-anything` judge.
- Replaced `tools/run_dlc_bench_official_eval.sh` with a thin two-step wrapper that simply chains the new export-only and judge-only scripts.
- Updated `tools/evaldlc.sh` so the remote rjob path now runs export-only by default and writes `export_dlc_<gpu>.log`.

## 2026-06-26 Teacher Regenerate Single-Prompt Refactor

### Problem
- The staged teacher regenerate path bound success too tightly to intermediate diagnosis text fields.
- Even when the teacher could potentially write a usable regenerated caption, the pipeline often stopped early at `problem`, `direction`, or `reason`.

### Root Cause Notes
- The active training path called the teacher multiple times for diagnosis before generating the final DLC and verification caption.
- Those intermediate validators became hard blockers for regenerate CE, reducing the practical usability of teacher supervision.

### Chosen Fix Direction
- Collapse teacher regenerate into a single privileged teacher prompt that internally describes the full workflow: analyze `gtmask/refmask`, diagnose the caption drift, then directly output `DLC` and `VERIFICATION_CAPTION`.
- Keep diagnosis fields only as optional parsed log signals, not as routing gates.

### Rejected Direction
- Do not delete the old staged helper functions immediately. Keep them as compatibility helpers until the single-prompt path is validated in training logs.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - Added shared single-stage teacher prompt builders for privileged context and full regenerate generation.
  - Added `generate_teacher_regenerate_single_stage(...)` and switched the active teacher regenerate pipeline to call the teacher only once.
  - Kept DLC and verification validation plus reconstruction gate unchanged.
  - Downgraded `caption_problem`, `correction_direction`, and `reason` to optional parsed logging fields from the single raw teacher output.
  - Added `single_stage_raw` logging so the one-shot teacher output is visible in debug and batch logs.

### Follow-up Correction
- Training logs showed that most single-stage teacher outputs were natural-language captions without the required `DLC:` / `VERIFICATION_CAPTION:` labels, so the parser treated them as empty and stopped at `teacher_dlc_invalid:empty`.
- Tightened the single-stage prompt to require that the answer starts immediately with the two caption labels and forbids any prefatory text.
- Added a parser fallback: when `DLC:` is missing but the raw teacher output cleans into a valid caption, reuse that cleaned text as the DLC instead of discarding the sample outright. When `VERIFICATION_CAPTION:` is missing but the DLC is valid, temporarily reuse the DLC text as the verification caption fallback.

## 2026-06-27 Teacher Regenerate DLC-Only Reconstruction Gate

### Problem
- The verification-caption subpath became an extra failure layer for teacher regenerate and still did not produce meaningful reconstruction gains.
- Teacher supervision was split between generating a DLC target and separately generating a verification caption used only for CE admission.

### Root Cause Notes
- The main training goal remains the regenerated DLC, but the pipeline required a second caption artifact before CE could be applied.
- Even when verification captions were syntactically valid, they were often generic or noisy, so reconstruction gate pass stayed near zero.

### Chosen Fix Direction
- Collapse teacher regenerate to a single output: `DLC`.
- Always allow teacher regenerate to attempt DLC generation, even when the difference context is trivial.
- Reconstruct directly from the generated DLC and keep the existing IoU gate only for deciding whether CE is applied.

### Rejected Direction
- Do not keep the verification-caption stage as a soft fallback. That would preserve the same split objective and continue to obscure whether the DLC itself is useful.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - Rewrote the single-stage teacher prompt so it only asks for `DLC: ...`.
  - Removed verification-caption parsing from the active single-stage regenerate path.
  - Added stronger DLC cleanup for leftover labels and template noise.
  - Changed the teacher gate reconstruction input from `verification_caption` to the generated DLC itself.
  - Stopped treating trivial difference context as a hard early exit; it is now log-only context.
  - Updated teacher regenerate analysis bookkeeping so reconstruction success and gate status now reflect DLC reconstruction rather than verification-caption reconstruction.

## 2026-06-27 DLC-Bench RJob Evaluation Wrapper

### Problem
- DLC-Bench evaluation could be run locally through the official wrapper, but there was no remote `rjob` submission script matching the training launch style.

### Root Cause Notes
- The repository already had `tools/run_dlc_bench_official_eval.sh`, but no cluster-friendly wrapper that mounted shared storage, unpacked the runtime environment, and allowed selecting a custom checkpoint path.

### Chosen Fix Direction
- Add a dedicated `rjob` submission wrapper for DLC-Bench evaluation that mirrors the style of `tools/train1.sh` while targeting the official NVlabs evaluation flow.

### Implemented Changes
- Added `tools/evaldlc.sh`:
  - Submits an `rjob` with 1-GPU defaults suitable for evaluation.
  - Accepts overridable `MODEL_PATH` / `TOKENIZER_PATH` so different weights can be evaluated directly.
  - Uses the downloaded `DLC-bench` data root and the cloned `describe-anything` repo root.
  - Calls `tools/run_dlc_bench_official_eval.sh` inside the job and writes logs under the chosen output directory.

### Follow-up CLI Convenience Wrapper
- Added `tools/evaldlc_ckpt.sh` as a more ergonomic entrypoint for checkpoint evaluation.
- It accepts explicit flags such as `--model-path` and `--output-dir`, derives defaults like tokenizer path and output directory when omitted, and then forwards everything to `tools/evaldlc.sh`.

### Follow-up Baseline Wrapper
- Added `tools/evaldlc_baseline.sh` for the fixed baseline checkpoint at `/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA-4B`.
- It forwards to `tools/evaldlc_ckpt.sh` while pre-filling the baseline model/tokenizer paths and a dedicated default output directory.
  - `SA2VA_REFCOCO_OPSD_DEFAULT_WORK_DIR`
  - `SA2VA_REFCOCO_OPSD_DEFAULT_MODEL_PATH`
  - `SA2VA_REFCOCO_OPSD_DEFAULT_TOKENIZER_PATH`
  - `SA2VA_REFCOCO_OPSD_MODEL_FLAVOR`
  - `SA2VA_REFCOCO_OPSD_ENTRY_NAME`
- The injected defaults point to the DLC combine config and the rjob-provided work/model/tokenizer paths, restoring the expected wrapper contract for the shared training implementation.

### Follow-up Compatibility Fix
- After the wrapper env fix, DLC training still failed during dataloader construction with:
  - `TypeError: DefaultSampler.__init__() got an unexpected keyword argument 'per_device_batch_size'`
- Root cause:
  - `tools/train_refcoco_opsd_impl.sh` always injected `train_dataloader.sampler.per_device_batch_size` whenever batch size or accumulation overrides were present.
  - That override is only valid for `RouteGroupedSampler`, but the new DLC combine config uses `mmengine.dataset.sampler.DefaultSampler`.
- Chosen fix:
- Added a launcher-level compatibility switch `SA2VA_REFCOCO_OPSD_OVERRIDE_SAMPLER_PER_DEVICE_BATCH_SIZE`.
- Kept the shared implementation default as enabled for legacy OPSD configs.
- Disabled the override explicitly in `tools/traindlc.sh`, so the DLC config can keep using `DefaultSampler` without receiving unsupported sampler kwargs.

## 2026-06-25 Tighten Confuser-vs-GT Duplicate IoU Threshold

### Problem
- The existing SAM confuser selection logic treated masks with IoU up to `0.95` against the `gtmask` as valid confusers.
- In practice, those masks are often near-duplicates of the same object rather than meaningful distractors, which weakens choose-one supervision.

### Root Cause Notes
- Three stages shared the same overly permissive duplicate threshold:
  - SAM pool export against RefCOCO `gtmask`
  - dataset-time SAM pool supplementation
  - model-time final confuser selection
- This made the system retain masks that still overlapped the target too heavily.

### Chosen Fix Direction
- Tighten the duplicate-vs-target IoU threshold from `0.95` to `0.7` in all target-facing confuser filters.
- Keep the confuser selection score formula unchanged for now so only the admissible candidate set becomes stricter.

### Rejected Direction
- Do not change only one stage. Leaving export / dataset / model thresholds inconsistent would make confuser behavior harder to reason about and debug.

### Implemented Changes
- Updated `projects/sa2va/datasets/refcoco_opsd.py`:
  - `sam_confuser_duplicate_iou_threshold` default `0.95 -> 0.7`
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - `grpo_confuser_duplicate_iou_threshold` default `0.95 -> 0.7`
- Updated `tools/export_refcoco_sam_confuser_pool.py`:
  - `--gt-duplicate-iou-thresh` default `0.95 -> 0.7`

## 2026-06-26 Route Refresh Did Not Switch Active Manifest

### Problem
- During OPSD training, route refresh exported new manifests such as `routes_step_0006500.jsonl`, but the active dataset manifest stayed at the initial `routes_step_0000000.jsonl`.
- As a result, later training batches continued to consume stale static routes even after refresh succeeded.

### Root Cause Notes
- `OpsdRouteRefreshHook` correctly computed a new manifest path and exported it.
- However, `Sa2VAOpsdRefCocoDataset.resolve_active_route_manifest_path()` always returned the original `route_manifest_path`, so `refresh_route_manifest_if_needed()` only reloaded the old file.
- The dataset had an `active_route_manifest_path` field, but it was not actually used as the active source of truth.

### Chosen Fix Direction
- Make the dataset treat `active_route_manifest_path` as the first-priority manifest path.
- After each successful route export, have `OpsdRouteRefreshHook` explicitly switch the dataset to the freshly written manifest before forcing a refresh.

### Rejected Direction
- Do not rely on filename timestamp heuristics or directory scanning inside the dataset. The hook already knows the exact new manifest path, so the handoff should be explicit.

### Implemented Changes
- Updated `projects/sa2va/datasets/refcoco_opsd.py`:
  - `resolve_active_route_manifest_path()` now returns `active_route_manifest_path` first, then falls back to `route_manifest_path`
  - added `set_active_route_manifest_path(path)`
  - adjusted `load_route_manifest()` so `active_route_manifest_path` is maintained consistently across missing/existing manifest cases
- Updated `projects/sa2va/hooks/opsd_route_refresh_hook.py`:
  - after successful export, the hook now calls `dataset.set_active_route_manifest_path(str(manifest_path))`
  - the subsequent refresh/log path now reloads the newly exported manifest instead of the initial startup manifest

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

## 2026-06-13 Three-Mode Combined OPSD Training Skeleton

### Problem
- The existing OPSD training code binds the whole route system to a single caption task and a single verifier path.
- New experiments need one training implementation that can switch between:
  - DLC-only training with choose-one routing
  - referring-caption-only training with reconstruction IoU routing
  - combined dual-caption training with choose-one plus IoU routing

### Root Cause Notes
- `projects/sa2va/models/sa2va_opsd_v2.py` and `v3.py` assume one student caption path and one main verifier path per sample.
- The current configs also point at the V3 single-task model and manifest-oriented route setup, which is awkward for a first combined online-routing prototype.

### Chosen Fix Direction
- Add a new model file `projects/sa2va/models/sa2va_opsd_combine.py` as a V3-derived training skeleton with a `train_mode` switch.
- Keep the existing three route families unchanged (`teacher_regenerate`, `on_policy_distill`, `grpo_positive`) so DDP route alignment and loss-family shape stay stable.
- Start with online routing configs for the new modes, and keep combined GRPO conservative by using DLC as the only RL branch while supervising referring captions with auxiliary teacher/on-policy losses.

### Rejected Direction
- Do not replace the current V2/V3 implementation in-place. The new three-mode behavior is experimental and should land beside the current training path first.
- Do not introduce a fourth or fifth route family for combined mode in the first patch, because that would immediately complicate sampler and DDP alignment.

### Implemented Changes
- Added `projects/sa2va/models/sa2va_opsd_combine.py`:
  - Introduced `train_mode` with values `dlc`, `referring`, and `combine`.
  - Added mode-specific student prompt builders and output parsing, including dual-output parsing for `DLC:` and `REFERRING:` in combined mode.
  - Added choose-one based routing for DLC and IoU-based routing for referring captions, plus combined routing where choose-one determines main qualification and IoU gates GRPO promotion.
  - Added mode-specific teacher prompt builders using `gtmask + wrong confuser mask` for DLC and `gtmask + refmask` for referring supervision.
  - Added a lightweight referring-caption GRPO path based on reconstruction IoU rewards.
- Added configs:
  - `projects/sa2va/configs/sa2va_opsd_combine_4b_dlc.py`
  - `projects/sa2va/configs/sa2va_opsd_combine_4b_referring.py`
  - `projects/sa2va/configs/sa2va_opsd_combine_4b_combine.py`
  - These start in `route_mode="online"` and point to the new model with per-mode `train_mode` selection.

## 2026-06-13 DLC Choose-One Route Export

### Problem
- The existing offline route exporter only estimates routes from caption-to-mask reconstruction IoU.
- The new DLC-only training mode needs an offline manifest derived from DLC generation quality measured by confuser choose-one, not reconstruction IoU.

### Root Cause Notes
- `tools/export_opsd_routes.py` is built around `estimate_opsd_route_for_sample_with_model(...)`, which assumes one caption and one reconstruction verifier.
- DLC routing instead needs:
  - generate DLC
  - score it against GT vs confuser options
  - map choose-one correctness and confidence to the existing three route families

### Chosen Fix Direction
- Add a separate exporter `tools/export_opsd_dlc_routes.py` instead of overloading the current IoU exporter.
- Keep the manifest `route` values unchanged so the sampler and training loader can consume the output immediately.
- Store choose-one confidence in the manifest `iou` slot for route-summary compatibility, while also writing explicit DLC-specific fields to avoid ambiguity in downstream inspection.

### Rejected Direction
- Do not replace the existing IoU exporter. Reconstruction-based export is still needed for the referring-only path and other legacy analyses.
- Do not introduce new manifest route labels; reuse the current three-route family to stay compatible with route-grouped sampling.

### Implemented Changes
- Added `tools/export_opsd_dlc_routes.py`:
  - Dynamically builds the configured model class from `cfg["model"]["type"]`.
  - Generates DLC captions with the selected route model (`student` or `teacher`).
  - Scores captions with confuser choose-one and maps results to `teacher_regenerate`, `on_policy_distill`, or `grpo_positive`.
  - Writes manifest records with DLC-specific fields such as `dlc_choose_one_correct`, `dlc_choose_one_confidence`, and `wrong_confuser_available`.
- Added shell launchers:
  - `tools/export_refcoco_opsd_dlc_routes_impl.sh`
  - `tools/export_refcoco_opsd_dlc_routes_4b.sh`
  - These mirror the existing RefCOCO export launcher style while routing execution to `tools/export_opsd_dlc_routes.py` and forcing `model.train_mode=dlc`.

## 2026-06-12 Teacher Regenerate Difference Context Switched To Natural-Language Summaries

### Problem
- The active teacher regenerate diagnosis pipeline was partially migrated away from cue-based fields, but `projects/sa2va/models/sa2va_opsd_v2.py` still required `primary_target_cue` and related fields in prompts, validators, and metrics.
- As a result, the teacher diagnosis flow became internally inconsistent: the difference-context builder returned natural-language summaries for `gtmask`, `refmask`, and overlap, while the downstream validation path still rejected outputs for not matching now-empty cue placeholders.

### Root Cause Notes
- `projects/sa2va/evaluation/teacher_diagnosis_common.py` had already started generating natural-language summaries for the target, distractor, and overlap regions, but still exposed cue compatibility placeholders.
- `projects/sa2va/models/sa2va_opsd_v2.py` continued to:
  - build prompts around `primary_target_cue` and `cue_conflict_summary`
  - validate diagnosis stages by requiring cue hits
  - log and aggregate cue-hit metrics that no longer reflected real training behavior
- The teacher regenerate pipeline also failed to pass `image`, `student_question`, and a teacher region model into the new summary builder, so the intended region-level natural-language descriptions were not actually being used.

### Chosen Fix Direction
- Remove cue-based fields from the active teacher regenerate path instead of keeping dead compatibility logic in the main loop.
- Make the teacher diagnose from:
  - natural-language target summary (`gtmask`)
  - natural-language distractor summary (`refmask`)
  - natural-language shared overlap summary
  - program-built target-only and distractor-only difference evidence
- Update prompts, validators, and logging to judge whether teacher outputs consume these summaries and difference-evidence fields rather than deprecated cue fields.

### Rejected Direction
- Do not keep empty cue placeholders as a first-class interface in the active training path. That makes metrics misleading and keeps diagnosis blocked for the wrong reason.
- Do not ask the program side to over-compress all differences into a single primary cue before teacher analysis. The new direction is to let teacher reason over richer natural-language summaries and then analyze only-difference evidence.

### Implemented Changes
- Updated `projects/sa2va/evaluation/teacher_diagnosis_common.py`:
  - removed the now-unused cue selection helper from the active difference-context path
  - kept `target_summary`, `distractor_summary`, and `shared_evidence` as natural-language region analyses
  - kept `target_only_evidence` and `distractor_only_evidence` as the structured only-difference evidence used downstream
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - removed cue fields from `TeacherRegeneratePipelineResult`
  - passed `image`, `student_question`, and the teacher model into `build_teacher_regenerate_difference_context(...)` so the natural-language summary path is actually exercised
  - rewrote the three diagnosis-stage prompts to consume `target_summary`, `distractor_summary`, `shared_evidence`, and the two only-difference evidence fields
  - relaxed stage validators so they now require fine-grained semantic anchors and failure-mechanism wording instead of cue-string hits
  - removed cue-hit counters and surfaced metrics from the active training metrics path
  - added `shared_evidence` to debug logging so the new summary-driven diagnosis can be inspected directly from logs

### Follow-up Adjustment
- A remaining layer of cue-style validation still survived in the post-diagnosis stages: DLC and verification-caption validation were checking whether outputs literally reused phrase fragments split from `target_only_evidence`.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` again so these later stages no longer require phrase-level evidence hits.
- The active validation now checks for:
  - semantic difference anchors
  - target-only / distractor-only / overlap-style distinction language
  - coarse-vs-fine failure quality
  rather than requiring the model to copy specific cue-like substrings from the program-generated evidence text.

### Follow-up Adjustment For Asymmetric Difference Compression And Direction Actions
- New logs showed the diagnosis flow had moved slightly forward, but two bottlenecks remained:
  - the difference context still produced too many symmetric `target_only_evidence` / `distractor_only_evidence` bullets
  - `teacher_correction_direction` often repeated the target summary instead of producing an explicit `add/avoid` edit instruction
- Updated `projects/sa2va/evaluation/teacher_diagnosis_common.py` to:
  - normalize and de-duplicate target/distractor bullets across sides
  - drop weak symmetric bullets from the main evidence path when stronger asymmetric ones exist
  - cap the active only-difference evidence to at most 3 bullets per side
  - add a new `difference_focus` summary sentence that names the leading target-side vs distractor-side difference
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` to:
  - carry `difference_focus` through the teacher regenerate pipeline and debug logs
  - rewrite the `teacher_correction_direction` prompt so it asks only for an edit instruction, not an object restatement
  - require explicit target-side add action words and distractor-side avoid action words in `validate_teacher_correction_direction(...)`
  - reject direction outputs that are mostly summary repetition or problem restatement
- The goal of this patch is to raise `teacher_direction_valid_rate` from zero by making the upstream difference context less symmetric and the direction stage more action-oriented.

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

## 2026-06-12 Teacher Reason Primary-Cue Grounding

### Problem
- Teacher diagnosis started passing structurally, but `teacher_reason` still consumed only coarse spatial summaries and largely ignored finer target-vs-distractor differences.
- Logs showed `teacher_reason` repeatedly collapsing to broad-area wording even when program-side evidence already exposed more asymmetric local cues.

### Root Cause Notes
- The difference context exposed multiple bullets, but there was no program-side notion of which cue pair mattered most, so fallback text and prompt conditioning kept drifting back to the first coarse cue.
- `REASON` prompt requirements still allowed generic spatial summaries as long as they referenced target/distractor evidence.
- Validator logic did not force `REASON` to consume the highest-value cue pair or reject coarse-only explanations.

### Chosen Fix Direction
- Add explicit primary and secondary cue selection in the difference compression layer, then drive `cue_conflict_summary` and `likely_drift_reason` from the primary cue pair.
- Feed those primary cues directly into the teacher light-diagnosis prompt and require `REASON` to explain why the caption misses the target cue but still fits the distractor cue.
- Strengthen parser fallback and validation so coarse-only `REASON` text is rejected and primary-cue grounding becomes mandatory.

### Rejected Direction
- Do not keep relying on the first bullet as the implicit primary cue; that was the main source of broad-area collapse.
- Do not only add more evidence bullets without selecting a primary cue pair, because that increases verbosity without changing what the teacher actually uses.

### Implemented Changes
- Updated `projects/sa2va/evaluation/teacher_diagnosis_common.py`:
  - Added primary/secondary cue selection and cue-conflict summarization on top of the existing bullet evidence.
  - Rewrote `likely_drift_reason` to be driven by the selected primary cue pair instead of the first coarse bullet.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - Extended `TeacherRegeneratePipelineResult` and teacher analysis/debug payloads with primary/secondary cue fields and a coarse-reason flag.
  - Strengthened teacher light-diagnosis prompting, parser fallback, and validation so `REASON` must ground on the selected primary cues and is rejected when it stays coarse.
  - Extended debug logging so future runs can directly inspect primary cues, cue-conflict summary, and whether the resulting `REASON` was still coarse.

## 2026-06-12 Teacher DLC Log Visibility

### Problem
- Logs printed teacher verification caption and diagnosis fields, but they did not print the actual teacher-regenerated DLC text in the main training log line or pre-return debug summary.
- This made it harder to tell whether regenerate failed because DLC generation was empty, malformed, or simply later rejected by validation.

### Root Cause Notes
- `teacher_dlc` was already carried through teacher analysis and sample debug records, but it was omitted from the formatted log strings.

### Chosen Fix Direction
- Surface `teacher_dlc` alongside `teacher_verification_caption` in both the main `[Sa2VA_OPSD_V2]` log line and the detailed pre-return debug record output.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so:
  - pre-return debug record formatting includes `teacher_dlc=...`
  - the main training log line prints `teacher_dlc=...` next to `teacher_verification_caption=...`

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

## 2026-06-12 Teacher Regenerate Three-Stage Diagnosis Refactor

### Problem
- The active teacher regenerate path still used a single diagnosis stage in training, even though staged prompts and helpers had already been added.
- After tightening `REASON` grounding, logs showed diagnosis first failing at `missing_reason`, then collapsing further to `missing_caption_problem`, which meant later constraints were starving the earlier problem-identification behavior.

### Root Cause Notes
- `run_teacher_regenerate_pipeline(...)` still called the legacy `generate_teacher_light_diagnosis(...)` path, so the real training chain never enforced the intended `problem -> direction -> reason` order.
- Diagnosis metrics and failure reasons were still aggregated too coarsely, which hid which stage actually failed.
- Because all diagnosis duties were effectively still coupled, stronger `reason` constraints could suppress `caption_problem` output instead of only affecting the final explanation stage.

### Chosen Fix Direction
- Rewire the active teacher regenerate pipeline to a strict staged sequence:
  1. `problem`
  2. `direction`
  3. `reason`
  4. `dlc`
  5. `verification`
  6. `gate`
- Stop immediately when a stage fails and report the stage-specific failure reason in both logs and metrics.
- Keep DLC target supervision, verification-caption gating, teacher gate formula, student main caption path, on-policy, and GRPO unchanged.

### Rejected Direction
- Do not keep stretching the old single diagnosis prompt with more constraints or fallback text. That design was exactly what made `caption_problem` and `reason` compete with each other.
- Do not add permissive fallback-continue behavior across failed diagnosis stages in this patch. The current priority is observability and stage separation.

### Implemented Changes
- Updated `projects/sa2va/models/sa2va_opsd_v2.py`:
  - switched `run_teacher_regenerate_pipeline(...)` to strict `problem -> direction -> reason` execution before DLC generation
  - propagated `problem_valid`, `direction_valid`, `reason_valid`, and their failure reasons through `TeacherRegeneratePipelineResult`
  - updated teacher regenerate analysis export so per-sample records now carry staged validity flags
  - added rolling and cumulative diagnostics for three-stage diagnosis, including stage valid rates, primary-target cue hit rates, and `teacher_reason_coarse_rate`
  - updated main logs and pre-return debug output to print staged validity alongside `teacher_caption_problem`, `teacher_correction_direction`, `teacher_reason`, `teacher_dlc`, and `teacher_verification_caption`

### Follow-up Logging Upgrade For Three-Stage Raw Outputs
- After the staged pipeline was wired in, runtime logs still only showed parsed fields like `teacher_caption_problem`, `teacher_correction_direction`, and `teacher_reason`.
- That was not enough to distinguish:
  - the teacher truly generating nothing
  - the teacher generating free-form text that parser could not align to labels
  - the parser extracting the wrong slice from an otherwise non-empty raw response
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so teacher regenerate logs now also print:
  - `teacher_problem_raw`
  - `teacher_direction_raw`
  - `teacher_reason_raw`
- These raw fields are exported through teacher analysis, included in per-sample debug records, and printed in the main training log so future diagnosis can distinguish generation failure from parsing failure.

### Follow-up Simplification For Three-Stage Natural-Language Outputs
- The new raw-output logs showed that the teacher was already generating meaningful natural-language diagnosis sentences in stage 1, but not in the strict `CAPTION_PROBLEM:` schema expected by the parser.
- That meant the actual failure was no longer teacher generation quality, but a format mismatch between staged prompts and parser assumptions.
- Updated `projects/sa2va/models/sa2va_opsd_v2.py` so the active three-stage prompts no longer require `CAPTION_PROBLEM:`, `CORRECTION_DIRECTION:`, or `REASON:` labels from the teacher.
- Each stage now asks for exactly one natural-language sentence, and the pipeline stores the raw sentence directly as:
  - `caption_problem`
  - `correction_direction`
  - `reason`
- The later DLC prompt still receives those three fields as program-side structured context, so regeneration remains explicitly conditioned on the staged diagnosis without requiring the teacher to emit schema-formatted labels.

## 2026-06-13 DLC Export rjob Wrapper

### Problem
- The new DLC choose-one route exporter was available as a Python entry and shell launcher, but there was no cluster submission wrapper matching the existing `tools/export.sh` workflow.

### Root Cause Notes
- Existing export automation on the remote cluster assumes an `rjob submit` wrapper that prepares the runtime image, unpacks the Python environment, installs system libraries, and then calls the appropriate route-export shell launcher.
- Without a DLC-specific wrapper, running DLC route export would require manually reconstructing that full submission command.

### Chosen Fix Direction
- Add `tools/export_dlc.sh` by mirroring the structure of `tools/export.sh` and swapping only the export target and default work-dir path.
- Keep the rest of the remote image/bootstrap flow identical so DLC export behaves like the existing route export job.

### Rejected Direction
- Do not fold DLC export into the existing `tools/export.sh` yet. Keeping a separate wrapper is clearer while the new route exporter is still experimental.

### Implemented Changes
- Added `tools/export_dlc.sh`:
  - submits the same remote image/bootstrap job shape as `tools/export.sh`
  - defaults `WORK_DIR` to `${PROJECT_ROOT}/work_dirs/sa2va_opsd_combine_4b_dlc_manifest`
  - calls `tools/export_refcoco_opsd_dlc_routes_4b.sh` inside the job
  - forwards the same common parameters (`gpus`, `cuda-devices`, data/model/tokenizer/work-dir/confuser-pool paths)

### Follow-up Default Resource Adjustment
- The first `tools/export_dlc.sh` draft inherited the 4-GPU resource defaults from the old generic export wrapper, which was unnecessarily large for the intended single-GPU DLC route export.
- Updated `tools/export_dlc.sh` defaults to a proportional 1-GPU profile:
  - `JOB_GPU=1`
  - `JOB_CPU=20`
  - `JOB_MEMORY=102400`
  - `CUDA_DEVICES=0`

## 2026-06-13 DLC Train rjob Wrapper

### Problem
- The new DLC training path had model/config support, but there was no dedicated rjob launcher matching the existing `tools/train1.sh` workflow and the requested remote work directory.

### Root Cause Notes
- Existing training submission wrappers still target the legacy OPSD 4B config and work directories.
- For the new DLC manifest workflow, the remote launcher must point at the combined-model DLC path while preserving:
  - single-GPU resource defaults
  - SAM confuser pool auto-generation
  - plot watcher
  - resume/load argument forwarding

### Chosen Fix Direction
- Add `tools/traindlc.sh` by mirroring `tools/train1.sh`.
- Keep the same rjob bootstrap flow, but:
  - default `WORK_DIR` to `/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Sa2VA_opsd/work_dirs/sa2va_opsd_combine_4b_dlc_manifest`
  - invoke `tools/train_refcoco_opsd_impl.sh`
  - inject `--cfg-options model.type=Sa2VAOPSDCombineModel model.train_mode=dlc`

### Rejected Direction
- Do not modify `tools/train1.sh` in place. The legacy launcher is still useful for the old OPSD path.

### Implemented Changes
- Added `tools/traindlc.sh`:
  - mirrors the single-GPU `train1.sh` resource profile and remote bootstrap flow
  - uses the requested DLC manifest work directory by default
  - auto-builds the confuser pool if missing
  - starts training through `tools/train_refcoco_opsd_impl.sh` with DLC-specific config overrides

## 2026-06-28 HF Conversion And DLC Judge Environment Fixups

### Problem
- The new `tools/converthf_ckpt.sh` rjob wrapper reached `tools/convert_to_hf.py`, but failed with `No module named 'projects'` when MMEngine tried to import `projects.sa2va...` from the config.
- The DLC judge rjob wrapper also ended up with an accidentally embedded real API key and had no first-class proxy passthrough, which made the judge path both unsafe and harder to run in restricted-network environments.

### Root Cause Notes
- The conversion rjob shell invoked `tools/convert_to_hf.py` from the repository root, but did not export `PYTHONPATH=${PROJECT_ROOT}`, so Python could not resolve the repository-local `projects` package during lazy config import.
- `tools/judgedlc.sh` had been edited during earlier debugging and retained a real API key plus a non-placeholder default base URL in the script body.
- The judge wrapper also did not forward `HTTP_PROXY` / `HTTPS_PROXY` into the job environment, so even a correctly configured proxy could not be used by the OpenAI client inside the container.

### Chosen Fix Direction
- Make the conversion rjob self-sufficient by exporting `PYTHONPATH` before running `tools/convert_to_hf.py`.
- Restore judge script defaults to safe placeholders and explicitly support proxy passthrough for restricted-network clusters.

### Rejected Direction
- Do not patch `tools/convert_to_hf.py` itself just to fix the remote job environment. The failure was specific to the rjob wrapper setup, not the conversion utility CLI.
- Do not leave the real API key in the repository even temporarily. Use placeholders in code and pass real secrets only through environment overrides at runtime.

### Implemented Changes
- Updated `tools/converthf_ckpt.sh` to export `PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}` before invoking `tools/convert_to_hf.py`, fixing `No module named 'projects'` in the remote conversion job.
- Updated `tools/judgedlc.sh` to:
  - restore safe defaults:
    - `LLM_ENGINE_PATH=https://api.openai.com/v1`
    - `API_KEY=YOUR_OPENAI_API_KEY_HERE`
  - remove the accidentally embedded real API key
  - accept and forward `HTTP_PROXY` / `HTTPS_PROXY`
  - export lowercase proxy variants inside the container for clients that read `http_proxy` / `https_proxy`
- Follow-up conversion fix:
  - `tools/convert_to_hf.py` still failed when the training config pointed `path` at the relative placeholder `./pretrained/Sa2VA-4B`
  - updated `tools/converthf_ckpt.sh` to accept `--base-model-path` and generate a temporary config with `path` rewritten to the actual mounted base model directory before calling `convert_to_hf.py`
- Follow-up conversion observability fix:
  - the conversion wrapper could exit before removing its temporary config file, which made failed runs look ambiguous and left no explicit success/failure marker in the log
  - updated `tools/converthf_ckpt.sh` to:
    - clean the temporary config via `trap` on every exit path
    - log the effective config, checkpoint, base-model path, and save path
    - capture the real `convert_to_hf.py` exit code from the pipeline
    - emit explicit `[convert] FAILED ...` or `[convert] SUCCESS ...` markers
    - fail loudly if the expected output directory was not created

## 2026-06-28 RefCOCO Caption-to-Mask Eval rjob Wrappers

### Problem
- The repository already had Python evaluators for RefCOCO metrics, but there was no cluster-friendly `rjob` wrapper equivalent to the DLC export/eval entrypoints.
- The user needed a one-command RefCOCO evaluation path for the current `iter_400` checkpoint and the dataset root `/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco`.

### Root Cause Notes
- `tools/eval_refcoco_caption_to_mask.py` can directly evaluate RefCOCO short-referring-expression segmentation quality, but it expects a config and runtime environment rather than a ready-to-submit remote wrapper.
- Existing example configs hardcode legacy local paths, so they were not suitable as-is for the current cluster paths and HF-exported checkpoint directory.

### Chosen Fix Direction
- Add a local-path config dedicated to RefCOCO caption-to-mask evaluation against HF model directories.
- Add one generic checkpoint-based `rjob` wrapper plus one fixed `iter_400` convenience wrapper, mirroring the existing DLC helper style.

### Rejected Direction
- Do not start with the mask-closure evaluator for this wrapper. The user asked for RefCOCO指标, and the standard `caption_to_mask` evaluator is the more direct match for short-referring-expression segmentation quality.

### Implemented Changes
- Added `projects/sa2va/configs/refcoco_caption_to_mask_eval_4b_local.py`:
  - defaults model path to `${PROJECT_ROOT}/work_dirs/hf_iter_400`
  - defaults data root to `/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco`
  - defaults image root to `${data_root}/train2014`
  - reads overrides from `SA2VA_REFCOCO_EVAL_*` environment variables
- Added `tools/evalrefcoco_ckpt.sh`:
  - submits a 1-GPU `rjob`
  - exports the evaluation env vars
  - runs `tools/eval_refcoco_caption_to_mask.py`
  - writes logs to `$(dirname OUTPUT_PATH)/eval_refcoco_<gpu>gpu.log`
- Added `tools/evalrefcoco_iter400.sh`:
  - defaults `MODEL_PATH=${PROJECT_ROOT}/work_dirs/hf_iter_400`
  - defaults `OUTPUT_PATH=${PROJECT_ROOT}/work_dirs/refcoco_caption_to_mask_eval_hf_iter_400.json`
  - forwards to the generic checkpoint wrapper

## 2026-06-28 Local DLC Judge Default Model Update

### Problem
- The local DLC judge path was still defaulting to `meta-llama/Meta-Llama-3.1-8B-Instruct`, which the configured proxy backend rejected with `model_not_found`.

### Root Cause Notes
- The common judge wrapper `tools/judgedlc.sh` still carried the earlier backend-specific default model name, even after the local baseline judge flow had been switched to an OpenAI-compatible proxy path.

### Chosen Fix Direction
- Update the common judge wrapper default model to `gpt-5.5`, which matches the intended OpenAI-style judge backend more closely, while still allowing explicit overrides through `LLM_ENGINE`.

### Rejected Direction
- Do not hardcode the backend-specific model name only in one ad hoc local command. Keeping the shared wrapper default aligned avoids repeating the override for later baseline or checkpoint judge runs.

### Implemented Changes
- Updated `tools/judgedlc.sh` so the default `LLM_ENGINE` is now `gpt-5.5`.
