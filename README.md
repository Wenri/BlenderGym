# BlenderGym Benchmark [CVPR 2025 Highlight] 


[**🌐 Homepage**](https://blendergym.github.io/) | [**📖 arXiv**](https://arxiv.org/abs/2504.01786) | [**🏆 Leaderboard**](https://blendergym.github.io/#leaderboard) | [**🤗 Hugging Face**](https://huggingface.co/datasets/richard-guyunqi/BG_bench_data) 

This repo contains the evaluation code for the paper "[BlenderGym: Benchmarking Foundational Model Systems for 3D Graphics](https://arxiv.org/abs/2504.01786)".

## 🔔News
1. 2025-4-11: We release BlenderGym, the first VLM System benchmark on 3D graphics editing!
2. 2025-4-4: BlenderGym is accepted for **Highlight** at **CVPR 2025** (top 387 over 13,008 valid submissions)!
3. 2025-4-2: Our paper is now accessible at [https://arxiv.org/abs/2504.01786](https://arxiv.org/abs/2504.01786)! 
4. 2025-2-26: BlenderGym is accepted to **CVPR 2025**!

# BlenderGym Usage
1. Jump to [**Installation**](#installation) to setup the Conda environment and download benchmark data.
2. After installation, jump to [**VLM Evaluation**](#vlm-evaluation) right away to benchmark your VLM!

# Installation
```
# Clone BlenderGym
git clone https://github.com/richard-guyunqi/BlenderGym-Open.git
cd BlenderGym-Open

# Creates conda environment for BlenderGym
conda create -n blendergym python=3.10
conda activate blendergym

# Install environment and download benchmark data
bash starter_setup.sh
```

# VLM Evaluation
## VLM Setup
This section sets your VLM up for inference on BlenderGym.

We provide a list of BlenderGym-suppoted models in [Supported Models](#supported-models). To run the open-source ones among them, jump directly to [**Inference on BlenderGym**](#inference-on-blendergym). To run the API-required ones among them, jump to [API Key Plug-in](#api-key-plug-in) to enter your API first.

If the VLM you want to test is not supported, jump to [Custom VLM Plug-in](#custom-vlm-plug-in).

## [Optional] Test your VLM setup
To sanity-check your API / local implementation, you can optionally jump to [Testing VLM Setup](#testing-vlm-setup) for some simple tests.

## Inference on BlenderGym
This section introduces how to run your VLM on BlenderGym data to generate output edits.
```
python inference.py --task placement --generator_type [model_id] --verifier_type [model_id]

# Backward-compatible legacy form:
# python inference.py --task placement --vlm_only --generator_type [model_id] --evaluator_type [model_id]

# Minimal example:
# python inference.py --task test --generator_type qwen --verifier_type qwen
```
where:
* `--task`: the task your VLM is evaluated on. You may enter task names, or one of "all", "subset", "test."
* `--generator_type`: `model_id` for generator VLM.
* `--verifier_type`: `model_id` for verifier VLM.
* `--vlm_only`: optional legacy flag; VLM-only inference is now the default unless `--custom_vlm_system` is set.

More details about the format of those arguments can be found in `inference.py`. Models in [Supported Models](#supported-models) are provided with their `model_id`. For custom VLMs, you may use the `model_id` you defined in [Custom VLM Plug-in](#custom-vlm-plug-in). 

Running the command above will generate all the proposal edits and renders in `system/outputs/`. Metadata to all those edit scripts will be saved at `info_saved/` by default.

## One-shot Inference
This section introduces a one-shot inference mode that does not use a verifier. It generates exactly one runnable edit and directly returns that result as the final output.
```
python inference_oneshot.py --task placement --generator_type [model_id]

# Minimal example:
# python inference_oneshot.py --task test --generator_type claude-code-sonnet-4-6
```
where:
* `--task`: the task your VLM is evaluated on. You may enter task names, or one of "all", "subset", "test."
* `--generator_type`: `model_id` for the one-shot VLM generator.
* `--tree_dims`: kept only for explicitness and must remain `1x1` in one-shot mode.

This mode does not call a verifier or evaluator model. It always runs with one-step, one-candidate generation semantics. Outputs will be saved under `system/outputs/` in directories named like `tune_leap_d1_b1`, and metadata will be saved under `info_saved/` by default.

## Retrying Failed Instances

If some instances fail during inference (e.g., due to transient errors), you can rerun only the failed ones using the retry scripts. Both scripts accept an `errors.json` file — a list of failed instances produced automatically during inference or by `reconstruct_metadata.py`.

**Generator-verifier mode:**
```
python inference_retry.py \
  --errors_json [path_to_errors.json] \
  --generator_type [model_id] \
  --verifier_type [model_id]
```

**One-shot mode:**
```
python inference_oneshot_retry.py \
  --errors_json [path_to_errors.json] \
  --generator_type [model_id]
```

The `errors.json` format is a JSON array where each entry has `task`, `task_instance_id`, `instance_dir_path`, and `error` fields. These files are written to `info_saved/` automatically when a run finishes with failures, or can be generated with `reconstruct_metadata.py` (see below).

Results are saved to a new timestamped `intermediate_metadata_retry_*.json` (or `intermediate_metadata_oneshot_retry_*.json`) under `info_saved/`, which can then be passed to `evaluation.py`.

## Reconstructing Metadata from Outputs

If inference completed but the metadata JSON was lost or corrupted, you can reconstruct it from the outputs directory:
```
python reconstruct_metadata.py --outputs_dir system/outputs/outputs_[task]_[timestamp]

# With optional overrides:
python reconstruct_metadata.py \
  --outputs_dir system/outputs/outputs_material_05-10-23-55-43 \
  --generator_type [model_id] \
  --bench_data_dir bench_data \
  --info_saving_dir info_saved
```
where:
* `--outputs_dir`: path to an existing outputs directory under `system/outputs/`.
* `--generator_type`: model identifier to embed in the reconstructed metadata (default: `unknown`).
* `--bench_data_dir`: path to the benchmark data directory (default: `bench_data`).
* `--info_saving_dir`: directory where the reconstructed metadata and errors JSON are written (default: `info_saved`).

The script writes a reconstructed `intermediate_metadata_oneshot_*.json` and, if any instances are missing or incomplete, an `errors_oneshot_*.json` file — both under `info_saved/`. The errors file can be passed directly to `inference_oneshot_retry.py`.

## Evaluation of VLM-generated results
This section introduces how to evaluate the output from your VLM after generating the output edits.
```
python evaluation.py --inference_metadata_saved_path [path_to_the_json_metadata] 

# Example with Chamfer settings exposed:
# python evaluation.py \
#   --inference_metadata_saved_path info_saved/intermediate_metadata_oneshot_xxx.json \
#   --chamfer_num_points 4096 \
#   --chamfer_chunk_size 512
```
where
* `--inference_metadata_saved_path`: path to the inference metadata(paths of proposal edit scripts, winner information, etc.) By default, it's a json file under `infosaved/`.
* `--chamfer_num_points`: number of surface points sampled from each exported scene mesh when computing Chamfer Distance. Default: `4096`.
* `--chamfer_chunk_size`: chunk size used for pairwise nearest-neighbor distance computation during Chamfer evaluation. Default: `512`.

The evaluation script always computes the image-based metrics already used in BlenderGym:
* `n_clip`
* `pl` (photometric loss)

It now also computes Chamfer Distance for the 3D-sensitive tasks:
* `geometry`
* `blendshape`
* `placement`

For those tasks, the evaluator executes each proposal script in Blender, exports the resulting scene geometry, samples points from the mesh surface, and reports Chamfer scores in addition to the render-based metrics.

More details about the arguments can be found in `evaluation.py`.

You can check `eval_renders/overall_scores.json` for the evaluation scores. Evaluation renders should be saved under `eval_renders/`.

When Chamfer is available for a task, the output JSONs include:
* `best_chamfer`: best Chamfer score among executable proposals for that instance/task
* `selected_chamfer`: Chamfer score of the verifier-selected proposal, when selection is available

# Utilities

## Testing VLM setup
This section provides unit tests for your VLM setup. We recommend you test your API plug-in or custom VLM plug-in. It starts by testing text-only or vision-language input for your VLM, and then test your VLM on a single instance of BlenderGym.

1. We recommend start testing a single query for your VLM pipeline. To do that, we offer two test scripts saved under `Tasksolver/test_scripts/`: `text_only.py` and `vlm.py`. With them, you can test text-only input and vision-language input for your VLM, respectively. Follow the todos in them to set up the tests.
```
python Tasksolver/test_scripts/text_only.py     # Test on language-only inputs
python Tasksolver/test_scripts/vision_language.py       # Test on vision-language inputs
```
2. After testing a single query of your VLM, you can try to run your VLM on one instance of BlenderGym by 
```
cd system
python vlm_single_edit.py --model_id [your_model_id]
```
You should see one tree of edits for one task instance under `system/outputs`.

If you want a single-step run without any verifier at the top level, use:
```
python inference_oneshot.py --task test --generator_type [your_model_id]
```
This will save outputs under `system/outputs/` with a `d1_b1` suffix and will not call a verifier model.

After you are done with testing the VLM setup, you may jump direclty into to [Inference on BlenderGym](#inference-on-blendergym).


##  API Key Plug-in
This section introduces how to plug in your API for proprietary models.

You can add your OpenAI/Anthropic/Google/Other API to `system/credentials/{API-name}.txt`. An example is attached in `system/credentials/api_exmaple.txt`.

After you are done with entering your APIs, you may jump back to [VLM Setup](#vlm-setup).


##  Custom VLM Plug-in
This section introduces how to plug in your VLM if it's not listed on [Supported Models](#supported-models). It's applicable for models either with API calls or local inference.

1. Create a `TaskSolver/tasksolver/{your_model}.py` which contains a class `YourModel`, similar to `ClaudeModel` in `TaskSolver/tasksolver/claude.py`(with API calling) or `InternModel` in `TaskSolver/tasksolver/intern.py`(local inference). You only have to change `self.ask()` and `self.prepare_payload()` to fit the format. 

2. Add `YourModel` to `Tasksolver/tasksolver/agent.py` by following the `TODO`s. Note that you will be required to name your model by a `model_id`, which is crucial for later usage.

After you are done with the two steps above, you may jump back to [VLM Setup](#vlm-setup).

## Supported Models
| Supported Model Name         | model_id                          |
|--------------------|--------------------------------------|
| GPT-4o              | gpt-4o     |
| GPT4-Turbo  |  gpt-4-turbo |
| GPT-4o-mini |  gpt-4o-mini |
| Claude-3.7-Sonnet | claude-3.7-sonnet-latest  |
| Claude 3.5 Sonnet(v2)  | claude-3-5-sonnet-latest  |
| Claude 3.5 Sonnet | claude-3-5-sonnet-20240620  |
| Claude 3.5 Haiku  |  claude-3-5-haiku-latest |
| Claude 3 Opus | claude-3-opus-latest  |
| Claude Code (default CLI model) | claude-code |
| Claude Code Sonnet | claude-code-sonnet |
| Claude Code Haiku 4.5 | claude-code-haiku-4-5 |
| Claude Code Sonnet 4.6 | claude-code-sonnet-4-6 |
| Claude Code Opus 4.7 | claude-code-opus-4-7 |
| Gemini 2.0 Flash | gemini-2.0-flash |
| Gemini 1.5 Flash | gemini-1.5-flash |
| InternVL2(8B)  |  intern |
| InternLlama | internllama |
| MiniCPM V2.6  |  minicpm |
| MiniCPMLlama  |  minicpmllama |
| Phi 3.5 vision  | phi  |
| PhiLlama | phillama |
| Qwen2-VL(7B AWQ) | qwen |
| QwenLlama  | qwenllama |
| Llama 3.1(8B) | llama |

# Citation
If you find our work useful, please cite the Bibtex below:
```
@misc{gu2025blendergymbenchmarkingfoundationalmodel,
      title={BlenderGym: Benchmarking Foundational Model Systems for Graphics Editing}, 
      author={Yunqi Gu and Ian Huang and Jihyeon Je and Guandao Yang and Leonidas Guibas},
      year={2025},
      eprint={2504.01786},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2504.01786}, 
}
```






