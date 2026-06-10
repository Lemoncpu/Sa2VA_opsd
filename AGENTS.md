# Repository Guidelines

## Workflow Rules
- Read `CODE_CHANGE_LOG.md` before making any code change.
- After each code change, append the problem, root cause, and chosen fix to `CODE_CHANGE_LOG.md`.
- Do not replace an earlier design decision by switching to a different modification strategy without first recording the reason in `CODE_CHANGE_LOG.md`.

## Project Structure & Module Organization
Core source lives under `projects/`, with the main Sa2VA implementation in `projects/sa2va/`. Use:
- `projects/sa2va/models/` for model wrappers, OPSD variants, and SAM2 integration
- `projects/sa2va/datasets/` for dataset loaders and collate logic
- `projects/sa2va/configs/` for training/eval configs
- `projects/sa2va/hooks/` and `projects/sa2va/samplers/` for training-loop extensions
- `tools/` for train, export, debug, and evaluation entry scripts
- `assets/`, `demo/`, and `sa2va_eval/` for demos, docs, and external eval utilities

## Build, Test, and Development Commands
- `uv sync --extra=latest` or `uv sync --extra=legacy`: install Python 3.11 dependencies.
- `bash tools/dist.sh train projects/sa2va/configs/sa2va_in30_8b.py 8`: launch distributed training.
- `python3 tools/test.py <CONFIG> <CKPT>`: run model evaluation through the XTuner/MMEngine entrypoint.
- `python3 tools/convert_to_hf.py <CONFIG> --pth-model <CKPT> --save-path <DIR>`: export a trained checkpoint to Hugging Face format.
- `python3 -m py_compile <files...>`: quick syntax validation for edited Python files.

## Coding Style & Naming Conventions
Use 4-space indentation and follow existing Python style. Prefer `snake_case` for functions, variables, and config names; `PascalCase` for classes; keep config filenames descriptive, e.g. `sa2va_opsd_refcoco_internvl3_2b_v3.py`. Match the local style in large model files rather than reformatting unrelated code. No repository-wide formatter is enforced at root; keep edits minimal and consistent.

## Testing Guidelines
There is no small unit-test suite at the repository root. Validate changes with:
- `python3 -m py_compile` for touched files
- the smallest relevant smoke config, e.g. `projects/sa2va/configs/sa2va_opsd_refval100_v2_grpo_smoke.py`
- targeted scripts in `tools/` for exports, route generation, or eval flows

Name new test or debug scripts by task, e.g. `tools/debug_opsd_<scope>.py`.

## Commit & Pull Request Guidelines
Recent commits use short imperative summaries such as `update grpo compute graph` and `add loss plot and log output`. Keep commit subjects concise, lowercase is acceptable, and describe the behavioral change directly. PRs should include:
- what changed and why
- affected configs/scripts
- validation performed
- sample logs, metrics, or screenshots for training/eval/UI changes

## Security & Configuration Tips
Avoid hardcoding dataset roots, model paths, or secrets. Prefer environment variables already used by configs, such as `SA2VA_PUBLIC_2B_MODEL_PATH`. Large checkpoints, generated artifacts, and `work_dirs/` outputs should stay out of commits.
