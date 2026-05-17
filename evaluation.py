import os
import subprocess
import argparse
import json
import hashlib

import numpy as np
from PIL import Image
from utils import photometric_loss, img2img_clip_similarity, blender_step, clip_similarity
from tqdm import tqdm

try:
    import point_cloud_utils as pcu
except ModuleNotFoundError:
    pcu = None


CHAMFER_TASKS = {"geometry", "blendshape", "placement"}


def _record_non_executable_failure(failures, task_instance, proposal_name, proposal_path, stage, error):
    failures.append(
        {
            "task_instance": task_instance,
            "proposal_name": proposal_name,
            "proposal_path": proposal_path,
            "stage": stage,
            "error": error,
        }
    )


def _geometry_npz_has_mesh(geometry_path):
    try:
        geometry = np.load(geometry_path)
        vertices = np.asarray(geometry["vertices"], dtype=np.float32).reshape(-1, 3)
        triangles = np.asarray(geometry["triangles"], dtype=np.int32).reshape(-1, 3)
    except Exception:
        return False

    return len(vertices) > 0 and len(triangles) > 0


def export_scene_geometry(
    blender_executable_path,
    blender_file_path,
    blender_export_script_path,
    proposal_script_path,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    geometry_path = os.path.join(output_dir, "geometry.npz")
    if os.path.isfile(geometry_path):
        if _geometry_npz_has_mesh(geometry_path):
            return geometry_path
        print(f"Regenerating empty or invalid geometry cache: {geometry_path}")
        os.remove(geometry_path)

    command = [
        blender_executable_path,
        "--background",
        blender_file_path,
        "--python",
        blender_export_script_path,
        "--",
        proposal_script_path,
        output_dir,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    if not os.path.isfile(geometry_path):
        raise RuntimeError(f"Geometry export did not produce {geometry_path}.")
    return geometry_path


def _stable_seed_from_path(path):
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def sample_point_cloud_from_npz(geometry_path, num_points):
    geometry = np.load(geometry_path)
    vertices = np.asarray(geometry["vertices"], dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(geometry["triangles"], dtype=np.int32).reshape(-1, 3)

    if len(vertices) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if len(triangles) == 0:
        if len(vertices) >= num_points:
            return vertices[:num_points]
        repeat_count = int(np.ceil(num_points / len(vertices)))
        return np.tile(vertices, (repeat_count, 1))[:num_points]

    tri_vertices = vertices[triangles]
    edge_1 = tri_vertices[:, 1] - tri_vertices[:, 0]
    edge_2 = tri_vertices[:, 2] - tri_vertices[:, 0]
    double_area = np.linalg.norm(np.cross(edge_1, edge_2), axis=1)
    valid_mask = double_area > 1e-12

    if not np.any(valid_mask):
        if len(vertices) >= num_points:
            return vertices[:num_points]
        repeat_count = int(np.ceil(num_points / len(vertices)))
        return np.tile(vertices, (repeat_count, 1))[:num_points]

    tri_vertices = tri_vertices[valid_mask]
    areas = double_area[valid_mask]
    probabilities = areas / areas.sum()
    rng = np.random.default_rng(_stable_seed_from_path(geometry_path))
    sampled_triangle_indices = rng.choice(
        len(tri_vertices),
        size=num_points,
        replace=True,
        p=probabilities,
    )

    chosen_triangles = tri_vertices[sampled_triangle_indices]
    barycentric_u = rng.random(num_points).astype(np.float32)
    barycentric_v = rng.random(num_points).astype(np.float32)
    sqrt_u = np.sqrt(barycentric_u)

    sampled_points = (
        (1.0 - sqrt_u)[:, None] * chosen_triangles[:, 0]
        + (sqrt_u * (1.0 - barycentric_v))[:, None] * chosen_triangles[:, 1]
        + (sqrt_u * barycentric_v)[:, None] * chosen_triangles[:, 2]
    )
    return sampled_points.astype(np.float32)


def chamfer_distance(point_cloud_a, point_cloud_b, chunk_size):
    if len(point_cloud_a) == 0 or len(point_cloud_b) == 0:
        return float("inf")
    if pcu is None:
        raise ModuleNotFoundError(
            "point_cloud_utils is required for Chamfer Distance evaluation. "
            "Install it with `pip install point-cloud-utils`."
        )

    # `point_cloud_utils.chamfer_distance` returns the bidirectional squared
    # nearest-neighbor distance on point sets, which matches the evaluator's
    # existing metric definition. `chunk_size` is kept for CLI compatibility.
    del chunk_size
    return float(
        pcu.chamfer_distance(
            np.asarray(point_cloud_a, dtype=np.float32),
            np.asarray(point_cloud_b, dtype=np.float32),
        )
    )

task_instance_count_dict = {
    'geometry': 45,
    'material': 40,
    'blendshape': 75,
    'placement': 40,
    'lighting': 40
}

# task_instance_count_dict = {
#     'geometry': 55,
#     'material': 45,
#     'blendshape': 85,
#     'placement': 50,
#     'lighting': 50
# }

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Image-based program edits')

    parser.add_argument('--inference_metadata_saved_path', 
        type=str, 
        help="Path to the inference metadata in json format (paths of proposal edit scripts, winner information, etc.)"
    )

    parser.add_argument('--eval_render_save_dir', 
        type=str, default=None, 
        help="The directory that all evaluation renders will be saved to.."
    )

    parser.add_argument('--infinigen_installation_path', 
        type=str, default=f"{os.path.abspath('infinigen/blender/blender')}", 
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
        '--chamfer_num_points',
        type=int,
        default=4096,
        help="Number of surface points sampled per scene when computing Chamfer Distance.",
    )
    parser.add_argument(
        '--chamfer_chunk_size',
        type=int,
        default=512,
        help="Chunk size for pairwise distance computation during Chamfer evaluation.",
    )

    # parse, save, and validate the args
    args = parser.parse_args()
    inference_metadata_saved_path = args.inference_metadata_saved_path
    eval_render_save_dir = args.eval_render_save_dir
    infinigen_installation_path = args.infinigen_installation_path
    render_device = args.render_device
    chamfer_num_points = args.chamfer_num_points
    chamfer_chunk_size = args.chamfer_chunk_size

    if render_device == 'cpu':
        os.environ["BLENDERGYM_FORCE_CPU"] = "1"
    elif render_device == 'gpu':
        os.environ.pop("BLENDERGYM_FORCE_CPU", None)

    blender_render_script_path = "bench_data/all_render_script.py"
    blender_geometry_export_script_path = "bench_data/export_geometry_script.py"

    if not os.path.isfile(inference_metadata_saved_path):
        raise ValueError(f'Invalid input for --inference_metadata_saved_path: {inference_metadata_saved_path}.')

    # Load in the data from pipeline inference
    with open(inference_metadata_saved_path, 'r') as file:
        inference_metadata = json.load(file)

    # Derive name for eval_render_save_dir
    if not eval_render_save_dir:
        eval_render_save_dir = f"eval_renders/{inference_metadata['output_dir_name']}"
    os.makedirs(eval_render_save_dir, exist_ok=True)

    tasks = inference_metadata.keys()

    # Create eval renders for all instances

    scores_across_tasks = {}
    intermediates = {}

    for task in tasks:
        if task not in task_instance_count_dict.keys():
            continue
        
        # Iterate through each instance
        scores_across_instances = {
            'best_n_clip': [],
            'selected_n_clip': [],
            'best_pl': [],
            'selected_pl': [],
            'best_chamfer': [],
            'selected_chamfer': [],
            'non_executable_count': 0,
            'non_executable_details': [],
        }

        for task_instance, instance_info in inference_metadata[task].items():
            task_instance_dir = os.path.join(eval_render_save_dir, task_instance)
            os.makedirs(task_instance_dir, exist_ok=True)

            # Store local scores: score for each executable render
            task_instance_scores = {}

            try:
                # Iterate through each proposal_renders_path
                blender_file_path = instance_info['blender_file_path']
                start_file_path = instance_info['start_script_path']
                goal_file_path = instance_info['goal_script_path']
                proposal_edits_paths = instance_info['proposal_edits_paths']
            except:
                continue

            executable_proposal_names = []
            non_executable_count = 0
            non_executable_details = []
            proposal_geometry_paths = {}
            sampled_point_cloud_cache = {}
            proposal_script_paths_by_name = {
                os.path.basename(start_file_path).split('.')[0]: start_file_path,
                os.path.basename(goal_file_path).split('.')[0]: goal_file_path,
            }
            proposal_script_paths_by_name.update(
                {
                    os.path.basename(proposal_path).split('.')[0]: proposal_path
                    for proposal_path in proposal_edits_paths
                }
            )

            if not proposal_edits_paths:
                continue

            for proposal_path in (proposal_edits_paths + [start_file_path, goal_file_path]):
                # Render the images for that proposal_renders_path
                proposal_name = os.path.basename(proposal_path).split('.')[0] # Extract the name of py file, without suffix
                proposal_renders_dir = os.path.join(task_instance_dir, proposal_name)
                
                # Render images. "executable" checks whether the proposal is executable in Blender-Python API.
                if not os.path.exists(proposal_renders_dir) or not os.listdir(proposal_renders_dir): 
                    try:
                        executable = blender_step(infinigen_installation_path, blender_file_path, blender_render_script_path, proposal_path, proposal_renders_dir, merge_all_renders=False, replace_if_overlap=True)
                    except Exception as exc:
                        non_executable_count += 1
                        _record_non_executable_failure(
                            non_executable_details,
                            task_instance,
                            proposal_name,
                            proposal_path,
                            'render',
                            f'{type(exc).__name__}: {exc}',
                        )
                        continue
                    if executable:
                        executable_proposal_names.append((proposal_renders_dir,proposal_name))
                    else:
                        non_executable_count += 1
                        _record_non_executable_failure(
                            non_executable_details,
                            task_instance,
                            proposal_name,
                            proposal_path,
                            'render',
                            'blender_step returned False',
                        )
                else:
                    executable_proposal_names.append((proposal_renders_dir,proposal_name))

            if task in CHAMFER_TASKS:
                for proposal_renders_dir, proposal_name in executable_proposal_names:
                    proposal_script_path = proposal_script_paths_by_name.get(proposal_name)
                    geometry_output_dir = os.path.join(task_instance_dir, f"{proposal_name}_geometry")
                    if proposal_script_path is None:
                        continue
                    try:
                        proposal_geometry_paths[proposal_name] = export_scene_geometry(
                            infinigen_installation_path,
                            blender_file_path,
                            blender_geometry_export_script_path,
                            proposal_script_path,
                            geometry_output_dir,
                        )
                    except Exception:
                        continue
            
            # Loop through each executable proposal to compute their scores
            goal_point_cloud = None
            if task in CHAMFER_TASKS and 'goal' in proposal_geometry_paths:
                goal_geometry_path = proposal_geometry_paths['goal']
                goal_point_cloud = sample_point_cloud_from_npz(goal_geometry_path, chamfer_num_points)
                sampled_point_cloud_cache[goal_geometry_path] = goal_point_cloud

            for proposal_renders_dir, proposal_name in tqdm(executable_proposal_names):
                if proposal_name == 'goal':
                    continue

                task_instance_scores[proposal_name] = {}    

                n_clip_views = []
                pl_views = []    

                render_names = sorted(
                    render_name
                    for render_name in os.listdir(proposal_renders_dir)
                    if render_name.lower().endswith(".png")
                )

                for render_name in render_names:
                    task_instance_scores[proposal_name][render_name] = {}

                    # Get path for render
                    try:
                        proposal_render = Image.open(os.path.join(proposal_renders_dir, render_name))
                        gt_render = Image.open(os.path.join(task_instance_dir, 'goal', render_name))
                    except:
                        continue
                    
                    # Compute n_clip and pl
                    n_clip = float(1 - clip_similarity(proposal_render, gt_render))
                    pl = float(photometric_loss(proposal_render, gt_render))

                    # Aggregate scores across all views for a proposal edit to compute average
                    n_clip_views.append(n_clip)
                    pl_views.append(pl)

                    # Record scores for this render
                    task_instance_scores[proposal_name][render_name]['n_clip'] = n_clip
                    task_instance_scores[proposal_name][render_name]['pl'] = pl 
                
                # Compute average n_clip for this proposal
                if not n_clip_views or not pl_views:
                    task_instance_scores.pop(proposal_name, None)
                    continue

                average_n_clip_views = sum(n_clip_views) / len(n_clip_views)
                average_pl_views = sum(pl_views) / len(pl_views)

                # Record average scores for a proposal 
                task_instance_scores[proposal_name]['avg_n_clip'] = average_n_clip_views
                task_instance_scores[proposal_name]['avg_pl'] = average_pl_views 

                if task in CHAMFER_TASKS and goal_point_cloud is not None:
                    proposal_geometry_path = proposal_geometry_paths.get(proposal_name)
                    if proposal_geometry_path is not None:
                        if proposal_geometry_path not in sampled_point_cloud_cache:
                            sampled_point_cloud_cache[proposal_geometry_path] = sample_point_cloud_from_npz(
                                proposal_geometry_path,
                                chamfer_num_points,
                            )
                        task_instance_scores[proposal_name]['chamfer'] = chamfer_distance(
                            sampled_point_cloud_cache[proposal_geometry_path],
                            goal_point_cloud,
                            chamfer_chunk_size,
                        )
            
                # Save the local scores to the task_instance dir
                task_instance_scores_path = os.path.join(task_instance_dir, 'scores.json')
                with open(task_instance_scores_path, 'w') as file:
                    json.dump(task_instance_scores, file, indent=4)

            if not task_instance_scores:
                task_instance_scores['non_executable_count'] = non_executable_count
                task_instance_scores['non_executable_details'] = non_executable_details
                task_instance_scores_path = os.path.join(task_instance_dir, 'scores.json')
                with open(task_instance_scores_path, 'w') as file:
                    json.dump(task_instance_scores, file, indent=4)
                scores_across_instances['non_executable_count'] += non_executable_count
                scores_across_instances['non_executable_details'].extend(non_executable_details)
                continue

            # Extract best scores and record them
            best_n_clip_proposal_name = min(task_instance_scores, key=lambda proposal_name: task_instance_scores[proposal_name]['avg_n_clip'])            
            best_pl_proposal_name = min(task_instance_scores, key=lambda proposal_name: task_instance_scores[proposal_name]['avg_pl'])
            best_n_clip = task_instance_scores[best_n_clip_proposal_name]['avg_n_clip']
            best_pl = task_instance_scores[best_pl_proposal_name]['avg_pl']
            task_instance_scores['best_n_clip'] = (best_n_clip_proposal_name, best_n_clip)
            task_instance_scores['best_pl'] = (best_pl_proposal_name, best_pl)

            chamfer_candidates = [
                proposal_name
                for proposal_name, proposal_scores in task_instance_scores.items()
                if isinstance(proposal_scores, dict) and 'chamfer' in proposal_scores
            ]
            if chamfer_candidates:
                best_chamfer_proposal_name = min(
                    chamfer_candidates,
                    key=lambda proposal_name: task_instance_scores[proposal_name]['chamfer'],
                )
                best_chamfer = task_instance_scores[best_chamfer_proposal_name]['chamfer']
                task_instance_scores['best_chamfer'] = (best_chamfer_proposal_name, best_chamfer)
                scores_across_instances['best_chamfer'].append(best_chamfer)
            
            # Register this instance to the scores across this task
            scores_across_instances['best_n_clip'].append(best_n_clip)
            scores_across_instances['best_pl'].append(best_pl)

            # Handle selected edit if applicable
            selected_proposal_path = instance_info.get('selected_edit_path')
            if selected_proposal_path:
                selected_proposal_name = os.path.basename(selected_proposal_path).split('.')[0]
                
                try:
                    selectd_n_clip = task_instance_scores[selected_proposal_name]['avg_n_clip']
                    selected_pl = task_instance_scores[selected_proposal_name]['avg_pl']
                    selected_scores = {'avg_n_clip': selectd_n_clip, 'avg_pl': selected_pl}
                    if 'chamfer' in task_instance_scores[selected_proposal_name]:
                        selected_chamfer = task_instance_scores[selected_proposal_name]['chamfer']
                        selected_scores['chamfer'] = selected_chamfer
                    task_instance_scores['selected_scores'] = (selected_proposal_name, selected_scores)
                except:
                    selected_proposal_name = None

                if selected_proposal_name is not None:
                    # Register this instance to the scores across this task
                    scores_across_instances["selected_n_clip"].append(selectd_n_clip)
                    scores_across_instances["selected_pl"].append(selected_pl)
                    if 'chamfer' in task_instance_scores[selected_proposal_name]:
                        scores_across_instances["selected_chamfer"].append(selected_chamfer)

            # Save the local scores to the task_instance dir
            task_instance_scores['non_executable_count'] = non_executable_count
            task_instance_scores['non_executable_details'] = non_executable_details
            task_instance_scores_path = os.path.join(task_instance_dir, 'scores.json')
            with open(task_instance_scores_path, 'w') as file:
                json.dump(task_instance_scores, file, indent=4)

            scores_across_instances['non_executable_count'] += non_executable_count
            scores_across_instances['non_executable_details'].extend(non_executable_details)

            scores_across_instances_path = os.path.join(eval_render_save_dir, f'{task}_scores.json',)
            with open(scores_across_instances_path, 'w') as file:
                json.dump(scores_across_instances, file, indent=4)

        # If the model cannot provide any edit for more than 75%
        if len(scores_across_instances['best_n_clip']) < (len(inference_metadata[task]) * 0.25) :
            scores_across_tasks[task] = {
                'non_executable_count': scores_across_instances['non_executable_count'],
                'non_executable_details': scores_across_instances['non_executable_details'],
            }

        # If VLM system doesn't support selection
        elif not scores_across_instances["selected_n_clip"]:
            scores_across_tasks[task] = {
                'best_n_clip': sum(scores_across_instances['best_n_clip']) / len(scores_across_instances['best_n_clip']),
                'best_pl': sum(scores_across_instances['best_pl']) / len(scores_across_instances['best_pl']),
                'non_executable_count': scores_across_instances['non_executable_count'],
                'non_executable_details': scores_across_instances['non_executable_details'],
            }
            if scores_across_instances['best_chamfer']:
                scores_across_tasks[task]['best_chamfer'] = (
                    sum(scores_across_instances['best_chamfer']) / len(scores_across_instances['best_chamfer'])
                )

        else: 
            scores_across_tasks[task] = {
                'best_n_clip': sum(scores_across_instances['best_n_clip']) / len(scores_across_instances['best_n_clip']),
                'best_pl': sum(scores_across_instances['best_pl']) / len(scores_across_instances['best_pl']),
                'selected_n_clip': sum(scores_across_instances['selected_n_clip']) / len(scores_across_instances['selected_n_clip']),
                'selected_pl': sum(scores_across_instances['selected_pl']) / len(scores_across_instances['selected_pl']),
                'non_executable_count': scores_across_instances['non_executable_count'],
                'non_executable_details': scores_across_instances['non_executable_details'],
            }
            if scores_across_instances['best_chamfer']:
                scores_across_tasks[task]['best_chamfer'] = (
                    sum(scores_across_instances['best_chamfer']) / len(scores_across_instances['best_chamfer'])
                )
            if scores_across_instances['selected_chamfer']:
                scores_across_tasks[task]['selected_chamfer'] = (
                    sum(scores_across_instances['selected_chamfer']) / len(scores_across_instances['selected_chamfer'])
                )
        
        intermediates[task] = scores_across_instances
        
    scores_across_tasks_path = os.path.join(eval_render_save_dir, 'overall_scores.json',)
    with open(scores_across_tasks_path, 'w') as file:
        json.dump(scores_across_tasks, file, indent=4)
    
    scores_across_instances_path = os.path.join(eval_render_save_dir, 'intermediate_scores.json',)
    with open(scores_across_instances_path, 'w') as file:
        json.dump(intermediates, file, indent=4)
            
