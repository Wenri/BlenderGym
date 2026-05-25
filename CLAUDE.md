# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BlenderGym (CVPR 2025) benchmarks VLM "systems" on 3D graphics editing in Blender across five tasks: `geometry`, `material`, `blendshape`, `placement`, `lighting`. A run drives a VLM (or generator+verifier pair) to edit a `start.py` bpy script so the rendered scene matches a `goal.py` reference, then scores the proposals.

## Setup

```bash
conda create -n blendergym python=3.10 && conda activate blendergym
bash starter_setup.sh        # installs torch, the local `tasksolver` pkg, runs generate_benchdata.py, then clones+builds infinigen blender at ./infinigen/blender/blender
```

`starter_setup.sh` does not install `point-cloud-utils` (needed for Chamfer evaluation on the 3D tasks) or `huggingface_hub` (used by `generate_benchdata.py`) — `pip install` them if missing.

API keys live in plain text files at `system/credentials/{openai,claude,gemini}_api.txt` (loaded via `KeyChain` — file path or literal key both work).

## Common commands

All commands run from the repo root.

```bash
# Full generator+verifier inference (default mode)
python inference.py --task placement --generator_type [model_id] --verifier_type [model_id]

# Single-step / no-verifier
python inference_oneshot.py --task placement --generator_type [model_id]

# Retry only the failures produced by a previous run (errors JSON is written automatically into info_saved/)
python inference_retry.py --errors_json [path] --generator_type [m] --verifier_type [m]
python inference_oneshot_retry.py --errors_json [path] --generator_type [m]

# Rebuild metadata from existing system/outputs/outputs_* if the JSON was lost
python reconstruct_metadata.py --outputs_dir system/outputs/outputs_<task>_<ts>

# Score a finished run
python evaluation.py --inference_metadata_saved_path info_saved/intermediate_metadata_*.json
# Baseline (start vs goal) scoring — useful for sanity-checking the benchmark itself, not a VLM:
python start_evaluation.py --inference_metadata_saved_path [path]

# Smallest possible smoke test: 3 instances per task, default `test` mode
python inference.py --task test --generator_type qwen --verifier_type qwen
```

`--task` accepts a single task name, a comma-separated list, or one of the special modes `all` (everything), `subset` (first 10 per task), `test` (first 3 per task). Special modes cannot be mixed with concrete task names.

`--render_device {auto,cpu,gpu}` on both `inference.py` and `evaluation.py` controls Cycles. Passing `cpu` exports `BLENDERGYM_FORCE_CPU=1` to the child Blender processes.

There is no test suite, lint, or build step beyond `pip install -e ./TaskSolver`.

## Architecture

**Two-process design.** Top-level scripts in the repo root (`inference*.py`, `evaluation.py`, `utils.py`) iterate over task instances and shell out to a second Python process under `system/` for each instance. The two halves communicate via a temporary YAML config (`temp.yml` written by `utils.BlenderAlchemy_run*`) and via the filesystem layout under `system/outputs/`.

- **`inference.py` / `inference_oneshot.py`**: per-instance loop, resumability, error classification. Builds `task_signature`, writes `info_saved/resume_*.json` and `info_saved/intermediate_metadata_*.json`, and calls `utils.BlenderAlchemy_run` (full tree) or `BlenderAlchemy_run_oneshot` (one shot).
- **`utils.BlenderAlchemy_run*`**: writes `temp.yml` and `subprocess.run`s `cd system && python main.py ...` (or `main_oneshot.py`). On `returncode != 0`, looks for the sentinel `FATAL_LLM_RESPONSE_LIMIT:` in stderr/stdout — when present, raises a runtime error that the outer loop classifies as "stop and persist resume state" rather than "skip this instance".
- **`system/main.py` → `system/refinement_process.py:refinement`**: the actual tree-of-edits loop (`depth` iterations × `breadth` candidates), implemented with threads. Per iteration: `tree_branch` generates `breadth` candidate scripts via `agent.think` + `agent.act` (which runs Blender to render), `get_top_candidate` runs a single-elimination tournament of pairwise judge calls via `prompting/<task>.py:craft_eval_question`. Concurrency is bounded by three semaphores set from the YAML config (`max_concurrent_rendering_processes`, `max_concurrent_generator_requests`, `max_concurrent_evaluation_requests`).
- **`system/agents.py`**: defines `EditCodeAgent` (default: brainstorm-then-diff edit style) and `GeneralAgent` (rewrite-from-scratch). `EditCodeAgent` is the one normally used and explicitly supports split brainstormer/coder models via the suffix-`llama` model_ids (`qwenllama`, `phillama`, `minicpmllama`, `internllama` → vision model brainstorms, llama writes the diff).
- **`TaskSolver/` (installable as `tasksolver`)**: the model-agnostic agent layer. `tasksolver.agent.Agent.__init__` dispatches `vision_model` strings to one of `gpt4v.py`, `claude.py`, `claude_code.py`, `gemini.py`, `qwen.py`, `phi.py`, `llama.py`, `minicpm.py`, `intern.py`. Each model class implements `prepare_payload`, `ask`, and `rough_guess` / `run_once`.

**Task name translation.** The top-level uses user-facing names (`geometry`, `blendshape`, …); inside `system/` they are translated to module names (`geonodes`, `shapekey`, …) — see `task_translate` in `utils.py` and `TASKSETTING2PROMPTMODULE` in `refinement_process.py`. The `prompting/` modules are looked up by the *translated* name.

**Claude Code CLI as a model.** `claude-code*` model_ids do not use an API key — `tasksolver/claude_code.py` shells out to the local `claude` CLI (requires `npm install -g @anthropic-ai/claude-code` and `claude auth login`). `system/agents.py` and `tasksolver/agent.py` both maintain alias maps (`claude-code-sonnet-4-6` → `claude-sonnet-4-6`, etc.) — keep them in sync when adding new aliases.

**Output / metadata layout** (one full run):
```
system/outputs/outputs_<task>_<MM-DD-HH-MM-SS>/<task_instance_id>/instance0/<variant>_d<D>_b<B>/
  scripts/         # proposal .py files
  renders/         # one merged PNG per proposal
  thought_process/iteration_<i>.json   # winner_code / winner_image are read from the last iteration's last entry
  failed_scripts/  # candidates that failed to generate or execute
  failed_responses/
info_saved/
  intermediate_metadata_<task>_<ts>.json   # consumed by evaluation.py
  resume_<task>_<gen>_<ver>_<mode>.json    # state for crash-resume
  errors_<...>.json                        # written when run finishes with failures; feed to *_retry.py
```
The oneshot variant writes `tune_leap_d1_b1/` and `intermediate_metadata_oneshot_*.json`.

**Resume / retry semantics.** `inference.py` writes a `task_signature` (generator/verifier/render_device/tree_dims/script paths). If the resume file's signature doesn't match the current CLI args, it refuses to resume — delete the resume file or rerun with matching args. The classifier `should_stop_on_error` (also mirrored in `refinement_process.is_response_limit_error`) treats rate-limit / quota / token-limit / `FATAL_LLM_RESPONSE_LIMIT` errors as fatal-to-this-run (persist and re-raise), and all other exceptions as per-instance failures (record in `failed_instances`, continue).

**Evaluation metrics.** `evaluation.py` always computes `n_clip = 1 − CLIPsim` and photometric loss `pl` (CLIP model `openai/clip-vit-base-patch32`, loaded fresh per call — keep this in mind for perf). For `geometry`, `blendshape`, `placement` (set `CHAMFER_TASKS`) it also exports geometry from each proposal via `bench_data/export_geometry_script.py` and computes Chamfer Distance through `point_cloud_utils.chamfer_distance`. Note: `evaluation.task_instance_count_dict` has different counts than `inference.py`'s (45/40/75/40/40 vs 55/45/85/50/50) — the inference counts are the source of truth for what gets generated; the evaluation counts are not used for gating (the loop iterates whatever's in the metadata JSON), so changing them won't affect runs.

## Adding a new model

Two-step plug-in:

1. Add a class to `TaskSolver/tasksolver/<your_model>.py` mirroring `claude.py` (API) or `intern.py` (local). Only `prepare_payload` and `ask` are model-specific.
2. Register a `vision_model` id in the dispatcher in `TaskSolver/tasksolver/agent.py:Agent.__init__` (look for the `# TODO: Add your own model here` block) so the new id routes to your class.

The `model_id` you choose is what users pass to `--generator_type` / `--verifier_type`. The supported-model table lives in `README.md`.

## Things to watch out for

- Heavy `subprocess.run(..., capture_output=True)` use — every Blender render and every nested `cd system && python main.py` invocation buffers all output. Don't print huge things from inside Blender scripts.
- `temp.yml` at the repo root is rewritten on every instance — concurrent `inference.py` invocations from the same checkout will trample each other.
- `system/outputs/` and `info_saved/` are not in `.gitignore` (only `system/outputs` via `system/.gitignore` and a top-level `info_saved/`). Keep generated artifacts out of commits.
- `evaluation.py` re-renders proposals into `eval_renders/<output_dir_name>/<instance>/<proposal_name>/` and reuses cached renders if the directory is non-empty — delete the cache if you change the render script.
