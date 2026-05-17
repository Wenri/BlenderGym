import os
import argparse
import time
import json

from utils import BlenderAlchemy_run, tree_dim_parse
from tasksolver.exceptions import GPTMaxTriesExceededException


def VLMSystem_run(blender_file_path, start_script, start_render, goal_render, blender_render_script_path, task_instance_id, task, infinigen_installation_path):
    proposal_edits_paths = None
    proposal_renders_paths = None
    selected_edit_path = None
    selected_render_path = None
    return proposal_edits_paths, proposal_renders_paths, selected_edit_path, selected_render_path


def save_json(path, payload):
    with open(path, 'w') as file:
        json.dump(payload, file, indent=4)


def normalize_tree_dims(tree_dims):
    if isinstance(tree_dims, str):
        parsed_tree_dims = tree_dim_parse(tree_dims)
        return [int(parsed_tree_dims[0]), int(parsed_tree_dims[1])]
    if isinstance(tree_dims, (list, tuple)) and len(tree_dims) == 2:
        return [int(tree_dims[0]), int(tree_dims[1])]
    return tree_dims


def should_stop_on_error(exc):
    if isinstance(exc, GPTMaxTriesExceededException):
        return True

    msg = str(exc).lower()
    indicators = [
        "fatal_llm_response_limit",
        "model limit",
        "rate limit",
        "usage limit",
        "quota",
        "429",
        "exceeded your current quota",
        "credit balance",
        "token limit",
        "context length",
        "too many tokens",
        "no response",
        "claude code cli call failed",
        "claude auth status --text",
    ]
    return any(indicator in msg for indicator in indicators)


def load_errors(errors_json_path):
    with open(errors_json_path, 'r') as f:
        return json.load(f)


def build_pending_items_from_errors(errors):
    pending = []
    for entry in errors:
        pending.append({
            "task": entry["task"],
            "instance_dir_path": os.path.abspath(entry["instance_dir_path"]),
        })
    return pending


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retry failed instances from errors.json (generator-verifier mode)')

    parser.add_argument(
        '--errors_json',
        type=str,
        required=True,
        help='Path to errors.json containing failed instances to retry.'
    )

    parser.add_argument(
        '--info_saving_dir_path',
        type=str,
        default='info_saved',
        help='Directory where intermediate inference metadata is saved.'
    )

    parser.add_argument(
        '--blender_render_script_path',
        type=str,
        default=f"{os.path.abspath(os.path.join('bench_data', 'pipeline_render_script.py'))}",
        help='The Blender render script.'
    )

    parser.add_argument(
        '--infinigen_installation_path',
        type=str,
        default=f"{os.path.abspath('infinigen/blender/blender')}",
        help='The installation path of blender executable file.'
    )

    parser.add_argument(
        '--render_device',
        type=str,
        choices=['auto', 'cpu', 'gpu'],
        default='auto',
        help="Cycles render device selection. Use 'cpu' to skip GPU initialization entirely.",
    )

    parser.add_argument(
        '--custom_vlm_system',
        action='store_true',
        help='Use custom VLM system instead of default generator-verifier.'
    )

    parser.add_argument(
        '--generator_type',
        type=str,
        default=None,
        help='model_id of VLM generator.'
    )

    parser.add_argument(
        '--verifier_type', '--evaluator_type',
        dest='verifier_type', type=str, default=None,
        help='model_id of VLM verifier.'
    )

    parser.add_argument(
        '--tree_dims',
        type=str,
        default='3x4',
        help='Tree dimension for generation-verification tree.'
    )

    args = parser.parse_args()

    if args.render_device == 'cpu':
        os.environ["BLENDERGYM_FORCE_CPU"] = "1"
    elif args.render_device == 'gpu':
        os.environ.pop("BLENDERGYM_FORCE_CPU", None)

    if not os.path.isfile(args.errors_json):
        raise ValueError(f'errors_json not found: {args.errors_json}')

    if not os.path.isfile(args.blender_render_script_path):
        raise ValueError(f'Invalid blender_render_script_path: {args.blender_render_script_path}')

    if not os.path.exists(args.infinigen_installation_path):
        raise ValueError(f'Invalid infinigen_installation_path: {args.infinigen_installation_path}')

    if not args.custom_vlm_system:
        if not args.generator_type or not args.verifier_type:
            raise ValueError("Please provide both --generator_type and --verifier_type.")

    os.makedirs(args.info_saving_dir_path, exist_ok=True)

    errors = load_errors(args.errors_json)
    pending_items = build_pending_items_from_errors(errors)

    print(f'Retrying {len(pending_items)} failed instances from {args.errors_json}')

    starter_time = time.strftime("%m-%d-%H-%M-%S")
    output_dir_name = f"outputs_retry_{starter_time}"
    info_saving_json_path = os.path.join(
        args.info_saving_dir_path,
        f'intermediate_metadata_retry_{starter_time}.json'
    )
    tree_dims = normalize_tree_dims(args.tree_dims)

    generation_results = {
        "output_dir_name": output_dir_name,
        "generator_type": args.generator_type,
        "verifier_type": args.verifier_type,
        "tree_dims": tree_dims,
        "mode": "custom_vlm_system" if args.custom_vlm_system else "generator_verifier",
        "status": "running",
        "errors_json": os.path.abspath(args.errors_json),
    }
    save_json(info_saving_json_path, generation_results)

    for pending_index, pending_item in enumerate(pending_items):
        instance_dir_path = pending_item["instance_dir_path"]
        task = pending_item["task"]
        task_instance_id = os.path.basename(instance_dir_path)

        generation_results.setdefault(task, {})
        generation_results[task].setdefault(task_instance_id, {})

        blender_file_path = os.path.join(instance_dir_path, 'blender_file.blend')
        start_file_path = os.path.join(instance_dir_path, 'start.py')
        start_render_path = os.path.join(instance_dir_path, 'renders/start')
        goal_file_path = os.path.join(instance_dir_path, 'goal.py')
        goal_render_path = os.path.join(instance_dir_path, 'renders/goal')

        print(f'[{pending_index + 1}/{len(pending_items)}] Retrying {task_instance_id}')

        try:
            if not args.custom_vlm_system:
                proposal_edits_paths, proposal_renders_paths, selected_edit_path, selected_render_path = BlenderAlchemy_run(
                    blender_file_path,
                    start_file_path,
                    start_render_path,
                    goal_render_path,
                    args.blender_render_script_path,
                    task_instance_id,
                    task,
                    args.infinigen_installation_path,
                    args.generator_type,
                    args.verifier_type,
                    starter_time=starter_time,
                    tree_dims=tree_dims,
                    output_dir_name=output_dir_name,
                )
            else:
                proposal_edits_paths, proposal_renders_paths, selected_edit_path, selected_render_path = VLMSystem_run(
                    blender_file_path,
                    start_file_path,
                    start_render_path,
                    goal_render_path,
                    args.blender_render_script_path,
                    task_instance_id,
                    task,
                    args.infinigen_installation_path,
                )
        except Exception as exc:
            generation_results[task][task_instance_id]['instance_dir_path'] = instance_dir_path
            generation_results[task][task_instance_id]['blender_file_path'] = blender_file_path
            generation_results[task][task_instance_id]['start_script_path'] = start_file_path
            generation_results[task][task_instance_id]['goal_script_path'] = goal_file_path
            generation_results[task][task_instance_id]['error'] = str(exc)
            if should_stop_on_error(exc):
                generation_results["status"] = "paused_on_error"
                generation_results["last_error"] = {
                    "task": task,
                    "task_instance_id": task_instance_id,
                    "instance_dir_path": instance_dir_path,
                    "error": str(exc),
                }
                save_json(info_saving_json_path, generation_results)
                raise RuntimeError(
                    f"Retry stopped on {task_instance_id}: {exc}\n"
                    f"Metadata saved to {info_saving_json_path}"
                ) from exc

            generation_results.setdefault("failed_instances", [])
            generation_results["failed_instances"].append({
                "task": task,
                "task_instance_id": task_instance_id,
                "instance_dir_path": instance_dir_path,
                "error": str(exc),
            })
            generation_results["status"] = "running_with_failures"
            generation_results["last_error"] = generation_results["failed_instances"][-1]
            save_json(info_saving_json_path, generation_results)
            print(f"Skipping failed instance {task_instance_id}: {exc}")
            continue

        print(f'  proposal_edits_paths: {proposal_edits_paths}')
        print(f'  selected_edit_path: {selected_edit_path}')

        generation_results[task][task_instance_id]['instance_dir_path'] = instance_dir_path
        generation_results[task][task_instance_id]['blender_file_path'] = blender_file_path
        generation_results[task][task_instance_id]['start_script_path'] = start_file_path
        generation_results[task][task_instance_id]['goal_script_path'] = goal_file_path
        generation_results[task][task_instance_id]['proposal_edits_paths'] = proposal_edits_paths
        generation_results[task][task_instance_id]['proposal_renders_paths'] = proposal_renders_paths
        generation_results[task][task_instance_id]['selected_edit_path'] = selected_edit_path
        generation_results[task][task_instance_id]['selected_render_path'] = selected_render_path
        save_json(info_saving_json_path, generation_results)

    if generation_results.get("failed_instances"):
        generation_results["status"] = "completed_with_failures"
    else:
        generation_results["status"] = "completed"
    save_json(info_saving_json_path, generation_results)
    print(f'Done. Metadata saved to {info_saving_json_path}')
