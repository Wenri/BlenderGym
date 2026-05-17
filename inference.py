# Input modules
import os
import argparse
import time
import json

from utils import BlenderAlchemy_run, tree_dim_parse
from tasksolver.exceptions import GPTMaxTriesExceededException


task_instance_count_dict = {
    'geometry': 55,
    'material': 45,
    'blendshape': 85,
    'placement': 50,
    'lighting': 50
}


def VLMSystem_run(blender_file_path, start_script, start_render, goal_render, blender_render_script_path, task_instance_id, task, infinigen_installation_path):
    '''
    API for user-implemented VLM-system. With only a VLM, rather than a system of VLMs, the user is encouraged to use our implementation of BlenderAlchemy. Check out guide on readme.md.

    Generation and potentially selection process of the VLM system.

    Inputs:
        blender_file_path: file path to the .blend base file
        start_file_path: file path to the start.py, the script for start scene
        start_render_path: dir path to the rendered images of start scene
        goal_render: dir path to the rendered images of goal scene
        blender_render_script_path: file path to the render script of blender scene
        task: name of the task, like `geometry`, `placement`
        task_instance_id: f'{task}{i}', like `placement1`, `geometry2`
        infinigen_installation_path: file/dir path to infinigen blender executable file for background rendering

    Outputs:
        proposal_edits_paths: a list of file paths to proposal scripts from the VLM system
        selected_edit_path[optional]: if applicable, the file path to the VLM-system-selected proposal script
    '''

    proposal_edits_paths = None
    proposal_renders_paths = None
    selected_edit_path = None
    selected_render_path = None

    return proposal_edits_paths, proposal_renders_paths, selected_edit_path, selected_render_path


def build_task_instance_dir_paths(tasks):
    if 'all' in tasks:
        return {
            task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, task_instance_count_dict[task] + 1)]
            for task in task_instance_count_dict.keys()
        }
    if 'test' in tasks:
        return {
            task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, 4)]
            for task in task_instance_count_dict.keys()
        }
    if 'subset' in tasks:
        return {
            task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, 11)]
            for task in task_instance_count_dict.keys()
        }
    return {
        task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, task_instance_count_dict[task] + 1)]
        for task in tasks
    }


def build_pending_items(task_instance_dir_paths):
    pending = []
    for task, instance_dir_paths in task_instance_dir_paths.items():
        for instance_dir_path in instance_dir_paths:
            pending.append({
                "task": task,
                "instance_dir_path": os.path.abspath(instance_dir_path),
            })
    return pending


def save_json(path, payload):
    with open(path, 'w') as file:
        json.dump(payload, file, indent=4)


def persist_resume_state(resume_state_path, task_signature, starter_time, output_dir_name, info_saving_json_path, pending_items, generation_results):
    resume_state = {
        "task_signature": task_signature,
        "starter_time": starter_time,
        "output_dir_name": output_dir_name,
        "info_saving_json_path": info_saving_json_path,
        "pending_items": pending_items,
        "generation_results": generation_results,
    }
    save_json(resume_state_path, resume_state)


def normalize_task_tag(task_arg):
    return task_arg.strip().replace(',', '_').replace(' ', '')


def normalize_tree_dims(tree_dims):
    if isinstance(tree_dims, str):
        parsed_tree_dims = tree_dim_parse(tree_dims)
        return [int(parsed_tree_dims[0]), int(parsed_tree_dims[1])]
    if isinstance(tree_dims, (list, tuple)) and len(tree_dims) == 2:
        return [int(tree_dims[0]), int(tree_dims[1])]
    return tree_dims


def normalize_signature_path(path):
    if not path:
        return ""
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def build_task_signature(task, args):
    return {
        "task": task,
        "task_tag": normalize_task_tag(task),
        "custom_vlm_system": args.custom_vlm_system,
        "generator_type": args.generator_type,
        "verifier_type": args.verifier_type,
        "render_device": args.render_device,
        "tree_dims": normalize_tree_dims(args.tree_dims),
        "blender_render_script_path": normalize_signature_path(args.blender_render_script_path),
        "infinigen_installation_path": normalize_signature_path(args.infinigen_installation_path),
    }


def normalize_resume_signature(resume_signature):
    return {
        "task": resume_signature.get("task"),
        "task_tag": resume_signature.get("task_tag", normalize_task_tag(resume_signature.get("task", ""))),
        "custom_vlm_system": resume_signature.get("custom_vlm_system", False),
        "generator_type": resume_signature.get("generator_type"),
        "verifier_type": resume_signature.get("verifier_type"),
        "render_device": resume_signature.get("render_device", "auto"),
        "tree_dims": normalize_tree_dims(resume_signature.get("tree_dims")),
        "blender_render_script_path": normalize_signature_path(resume_signature.get("blender_render_script_path", "")),
        "infinigen_installation_path": normalize_signature_path(resume_signature.get("infinigen_installation_path", "")),
    }


def describe_signature_mismatch(expected_signature, actual_signature):
    differing_fields = []
    for key in expected_signature.keys():
        if expected_signature.get(key) != actual_signature.get(key):
            differing_fields.append(
                f"{key}: resume={actual_signature.get(key)!r}, current={expected_signature.get(key)!r}"
            )
    return "; ".join(differing_fields)


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


def resolve_resume_state_path(info_saving_dir_path, task_name, generator_type, verifier_type, custom_vlm_system):
    mode_tag = "custom" if custom_vlm_system else "default"
    generator_tag = generator_type or "none"
    verifier_tag = verifier_type or "none"
    canonical_path = os.path.join(
        info_saving_dir_path,
        f"resume_{task_name}_{generator_tag}_{verifier_tag}_{mode_tag}.json",
    )
    legacy_path = os.path.join(
        info_saving_dir_path,
        f"resume_{generator_tag}_{verifier_tag}_{task_name}_{mode_tag}.json",
    )

    # Reuse the legacy filename when present so older interrupted runs still resume.
    if os.path.exists(legacy_path):
        return legacy_path
    return canonical_path


def run_single_task(args, task):
    task_tag = normalize_task_tag(task)
    task_instance_dir_paths = build_task_instance_dir_paths([task])
    pending_items = build_pending_items(task_instance_dir_paths)

    print(f'task_instance_dir_paths: {task_instance_dir_paths}')

    task_signature = build_task_signature(task, args)
    resume_state_path = resolve_resume_state_path(
        args.info_saving_dir_path,
        task,
        args.generator_type,
        args.verifier_type,
        args.custom_vlm_system,
    )

    if os.path.exists(resume_state_path):
        with open(resume_state_path, 'r') as file:
            resume_state = json.load(file)

        resume_signature = resume_state.get("task_signature", {})
        normalized_resume_signature = normalize_resume_signature(resume_signature)
        if normalized_resume_signature != task_signature:
            mismatch_details = describe_signature_mismatch(task_signature, normalized_resume_signature)
            raise ValueError(
                f"Found resume state at {resume_state_path}, but it does not match the current arguments. "
                f"Mismatched fields: {mismatch_details}. "
                "Use the same command to resume, or remove that resume file first."
            )

        starter_time = resume_state["starter_time"]
        output_dir_name = resume_state.get("output_dir_name", f"outputs_{starter_time}")
        info_saving_json_path = resume_state["info_saving_json_path"]
        pending_items = resume_state["pending_items"]
        generation_results = resume_state["generation_results"]
        print(f"Resuming interrupted inference from {resume_state_path}")
    else:
        starter_time = time.strftime("%m-%d-%H-%M-%S")
        output_dir_name = f"outputs_{task_tag}_{starter_time}"
        info_saving_json_path = os.path.join(
            args.info_saving_dir_path,
            f'intermediate_metadata_{task_tag}_{starter_time}.json'
        )
        generation_results = {
            "output_dir_name": output_dir_name,
            "generator_type": args.generator_type,
            "verifier_type": args.verifier_type,
            "tree_dims": normalize_tree_dims(args.tree_dims),
            "mode": "custom_vlm_system" if args.custom_vlm_system else "generator_verifier",
            "status": "running",
            "resume_state_path": resume_state_path,
        }

    save_json(info_saving_json_path, generation_results)
    persist_resume_state(
        resume_state_path,
        task_signature,
        starter_time,
        output_dir_name,
        info_saving_json_path,
        pending_items,
        generation_results,
    )

    for pending_index, pending_item in enumerate(pending_items):
        instance_dir_path = pending_item["instance_dir_path"]
        task_instance_id = os.path.basename(instance_dir_path)

        generation_results.setdefault(task, {})
        generation_results[task].setdefault(task_instance_id, {})

        blender_file_path = os.path.join(instance_dir_path, 'blender_file.blend')
        start_file_path = os.path.join(instance_dir_path, 'start.py')
        start_render_path = os.path.join(instance_dir_path, 'renders/start')
        goal_file_path = os.path.join(instance_dir_path, 'goal.py')
        goal_render_path = os.path.join(instance_dir_path, 'renders/goal')

        try:
            if not args.custom_vlm_system:
                if not args.generator_type or not args.verifier_type:
                    raise ValueError("For VLM-only usage, please indicate both generator and evaluator model.")
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
                    tree_dims=normalize_tree_dims(args.tree_dims),
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

                remaining_pending_items = pending_items[pending_index:]
                save_json(info_saving_json_path, generation_results)
                persist_resume_state(
                    resume_state_path,
                    task_signature,
                    starter_time,
                    output_dir_name,
                    info_saving_json_path,
                    remaining_pending_items,
                    generation_results,
                )
                raise RuntimeError(
                    f"Inference stopped on {task_instance_id}: {exc}\n"
                    f"Resume state saved to {resume_state_path}"
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

            remaining_pending_items = pending_items[pending_index + 1:]
            save_json(info_saving_json_path, generation_results)
            persist_resume_state(
                resume_state_path,
                task_signature,
                starter_time,
                output_dir_name,
                info_saving_json_path,
                remaining_pending_items,
                generation_results,
            )
            print(
                f"Skipping failed instance {task_instance_id}: {exc}\n"
                f"Resume state updated at {resume_state_path}"
            )
            continue

        print(f'proposal_edits_paths: {proposal_edits_paths}')
        print(f'proposal_renders_paths: {proposal_renders_paths}')
        print(f'selected_edit_path: {selected_edit_path}')
        print(f'selected_render_path: {selected_render_path}')

        generation_results[task][task_instance_id]['instance_dir_path'] = instance_dir_path
        generation_results[task][task_instance_id]['blender_file_path'] = blender_file_path
        generation_results[task][task_instance_id]['start_script_path'] = start_file_path
        generation_results[task][task_instance_id]['goal_script_path'] = goal_file_path
        generation_results[task][task_instance_id]['proposal_edits_paths'] = proposal_edits_paths
        generation_results[task][task_instance_id]['proposal_renders_paths'] = proposal_renders_paths
        generation_results[task][task_instance_id]['selected_edit_path'] = selected_edit_path
        generation_results[task][task_instance_id]['selected_render_path'] = selected_render_path
        save_json(info_saving_json_path, generation_results)
        persist_resume_state(
            resume_state_path,
            task_signature,
            starter_time,
            output_dir_name,
            info_saving_json_path,
            pending_items[pending_index + 1:],
            generation_results,
        )

    if generation_results.get("failed_instances"):
        generation_results["status"] = "completed_with_failures"
    else:
        generation_results["status"] = "completed"
    save_json(info_saving_json_path, generation_results)
    persist_resume_state(
        resume_state_path,
        task_signature,
        starter_time,
        output_dir_name,
        info_saving_json_path,
        [],
        generation_results,
    )


if __name__ == '__main__':
    '''
    Input args are listed here.
    '''
    parser = argparse.ArgumentParser(description='Image-based program edits')

    parser.add_argument(
        '--task',
        type=str,
        default="test",
        help="`all`, `test`, `subset`, or comma-separated list of the following: `material`, `geometry`, `blendshape`, `placement`,`lighting`"
    )

    parser.add_argument(
        '--info_saving_dir_path',
        type=str,
        default="info_saved",
        help='''Directory that intermediate inference metadata, such as path to all edit scripts and images and the final output from VLM, is saved.
       The json file that pass all BlenderGym inference metadata to evaluation(calculation of performance scores) is saved here. By default, this is info_saved/'''
    )

    parser.add_argument(
        '--blender_render_script_path',
        type=str,
        default=f"{os.path.abspath(os.path.join('bench_data', 'pipeline_render_script.py'))}",
        help="The Blender render script. By default, it's bench_data/pipeline_render_script.py, which uses two views for VLM generation/verification."
    )

    parser.add_argument(
        '--infinigen_installation_path',
        type=str,
        default=f"{os.path.abspath('infinigen/blender/blender')}",
        help="The installation path of blender executable file. It's `infinigen/blender/blender` by default."
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
        help='''Whether change our VLM setup. If you want to change our VLM system (i.e. rewire/delete our generator-verifier structure)
        If you choose to use your custom_vlm_system, please implement the function VLMSystem_run().'''
    )

    parser.add_argument(
        '--vlm_only',
        action='store_true',
        help='''Backward-compatible alias for the default BlenderAlchemy VLM-only flow.
        This flag is no longer required because VLM-only is the default unless --custom_vlm_system is set.'''
    )

    parser.add_argument(
        '--generator_type',
        type=str,
        default=None,
        help="model_id of VLM generator. Note this is the specific id listed in Supported Models or named by you."
    )

    parser.add_argument(
        '--verifier_type', '--evaluator_type',
        dest='verifier_type', type=str, default=None,
        help="model_id of VLM verifier. --evaluator_type is kept as a backward-compatible alias."
    )

    parser.add_argument(
        '--tree_dims',
        type=str,
        default='3x4',
        help="Tree dimension for generation-verification tree. We set the default to 3x4, aligned with BlenderGym configuration."
    )

    args = parser.parse_args()

    if args.render_device == 'cpu':
        os.environ["BLENDERGYM_FORCE_CPU"] = "1"
    elif args.render_device == 'gpu':
        os.environ.pop("BLENDERGYM_FORCE_CPU", None)

    tasks = args.task.strip().split(',')
    if len(tasks) == 0:
        raise ValueError('Invalid input for --task: no task input detected.')

    for task in tasks:
        if task not in ('all', 'test', 'subset') and task not in task_instance_count_dict.keys():
            raise ValueError(f'Invalid input for --task: {task} is not a valid input for "--task".')

    if not os.path.isfile(args.blender_render_script_path):
        raise ValueError(f'Invalid input for blender_render_script_path: {args.blender_render_script_path}')

    if not os.path.exists(args.infinigen_installation_path):
        raise ValueError(f'Invalid input for infinigen_installation_path: {args.infinigen_installation_path}')

    os.makedirs(args.info_saving_dir_path, exist_ok=True)

    concrete_tasks = [task for task in tasks if task in task_instance_count_dict]
    special_modes = [task for task in tasks if task in ('all', 'test', 'subset')]

    if len(special_modes) > 0 and len(tasks) > 1:
        raise ValueError('Do not mix `all`, `test`, or `subset` with concrete task names.')

    if len(special_modes) == 1:
        expanded_tasks = list(build_task_instance_dir_paths(tasks).keys())
        for task in expanded_tasks:
            run_single_task(args, task)
    else:
        for task in concrete_tasks:
            run_single_task(args, task)
