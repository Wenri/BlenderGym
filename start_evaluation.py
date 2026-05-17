import argparse
import json
import os

from PIL import Image
from tqdm import tqdm

from evaluation import (
    CHAMFER_TASKS,
    chamfer_distance,
    export_scene_geometry,
    sample_point_cloud_from_npz,
    task_instance_count_dict,
)
from utils import blender_step, clip_similarity, photometric_loss


def _render_script(
    blender_executable_path,
    blender_file_path,
    blender_render_script_path,
    proposal_path,
    proposal_renders_dir,
):
    if os.path.exists(proposal_renders_dir) and os.listdir(proposal_renders_dir):
        return True

    return blender_step(
        blender_executable_path,
        blender_file_path,
        blender_render_script_path,
        proposal_path,
        proposal_renders_dir,
        merge_all_renders=False,
        replace_if_overlap=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the start.py baseline against goal.py."
    )

    parser.add_argument(
        "--inference_metadata_saved_path",
        type=str,
        required=True,
        help="Path to the inference metadata in json format.",
    )
    parser.add_argument(
        "--eval_render_save_dir",
        type=str,
        default=None,
        help="Directory that all evaluation renders and scores will be saved to.",
    )
    parser.add_argument(
        "--infinigen_installation_path",
        type=str,
        default=f"{os.path.abspath('infinigen/blender/blender')}",
        help="Path to the blender executable. Defaults to infinigen/blender/blender.",
    )
    parser.add_argument(
        "--render_device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Cycles render device selection. Use 'cpu' to skip GPU initialization entirely.",
    )
    parser.add_argument(
        "--chamfer_num_points",
        type=int,
        default=4096,
        help="Number of surface points sampled per scene when computing Chamfer Distance.",
    )
    parser.add_argument(
        "--chamfer_chunk_size",
        type=int,
        default=512,
        help="Chunk size for pairwise distance computation during Chamfer evaluation.",
    )

    args = parser.parse_args()

    if args.render_device == "cpu":
        os.environ["BLENDERGYM_FORCE_CPU"] = "1"
    elif args.render_device == "gpu":
        os.environ.pop("BLENDERGYM_FORCE_CPU", None)

    if not os.path.isfile(args.inference_metadata_saved_path):
        raise ValueError(
            f"Invalid input for --inference_metadata_saved_path: {args.inference_metadata_saved_path}."
        )

    blender_render_script_path = "bench_data/all_render_script.py"
    blender_geometry_export_script_path = "bench_data/export_geometry_script.py"

    with open(args.inference_metadata_saved_path, "r") as file:
        inference_metadata = json.load(file)

    if args.eval_render_save_dir:
        eval_render_save_dir = args.eval_render_save_dir
    else:
        eval_render_save_dir = os.path.join(
            "eval_renders_start", inference_metadata["output_dir_name"]
        )
    os.makedirs(eval_render_save_dir, exist_ok=True)

    tasks = inference_metadata.keys()
    scores_across_tasks = {}
    intermediates = {}

    for task in tasks:
        if task not in task_instance_count_dict:
            continue

        scores_across_instances = {
            "start_n_clip": [],
            "start_pl": [],
            "start_chamfer": [],
            "non_executable_count": 0,
        }

        for task_instance, instance_info in inference_metadata[task].items():
            task_instance_dir = os.path.join(eval_render_save_dir, task_instance)
            os.makedirs(task_instance_dir, exist_ok=True)

            task_instance_scores = {}

            try:
                blender_file_path = instance_info["blender_file_path"]
                start_file_path = instance_info["start_script_path"]
                goal_file_path = instance_info["goal_script_path"]
            except Exception:
                continue

            start_renders_dir = os.path.join(task_instance_dir, "start")
            goal_renders_dir = os.path.join(task_instance_dir, "goal")

            try:
                start_executable = _render_script(
                    args.infinigen_installation_path,
                    blender_file_path,
                    blender_render_script_path,
                    start_file_path,
                    start_renders_dir,
                )
                goal_executable = _render_script(
                    args.infinigen_installation_path,
                    blender_file_path,
                    blender_render_script_path,
                    goal_file_path,
                    goal_renders_dir,
                )
            except Exception:
                start_executable = False
                goal_executable = False

            if not start_executable or not goal_executable:
                task_instance_scores["non_executable_count"] = 1
                with open(os.path.join(task_instance_dir, "scores.json"), "w") as file:
                    json.dump(task_instance_scores, file, indent=4)
                scores_across_instances["non_executable_count"] += 1
                continue

            task_instance_scores["start"] = {}
            n_clip_views = []
            pl_views = []

            render_names = sorted(
                render_name
                for render_name in os.listdir(start_renders_dir)
                if render_name.lower().endswith(".png")
            )

            for render_name in tqdm(render_names):
                try:
                    start_render = Image.open(os.path.join(start_renders_dir, render_name))
                    goal_render = Image.open(os.path.join(goal_renders_dir, render_name))
                except Exception:
                    continue

                n_clip = float(1 - clip_similarity(start_render, goal_render))
                pl = float(photometric_loss(start_render, goal_render))

                n_clip_views.append(n_clip)
                pl_views.append(pl)
                task_instance_scores["start"][render_name] = {
                    "n_clip": n_clip,
                    "pl": pl,
                }

            if not n_clip_views or not pl_views:
                task_instance_scores["non_executable_count"] = 1
                with open(os.path.join(task_instance_dir, "scores.json"), "w") as file:
                    json.dump(task_instance_scores, file, indent=4)
                scores_across_instances["non_executable_count"] += 1
                continue

            avg_n_clip = sum(n_clip_views) / len(n_clip_views)
            avg_pl = sum(pl_views) / len(pl_views)
            task_instance_scores["start"]["avg_n_clip"] = avg_n_clip
            task_instance_scores["start"]["avg_pl"] = avg_pl

            if task in CHAMFER_TASKS:
                try:
                    start_geometry_path = export_scene_geometry(
                        args.infinigen_installation_path,
                        blender_file_path,
                        blender_geometry_export_script_path,
                        start_file_path,
                        os.path.join(task_instance_dir, "start_geometry"),
                    )
                    goal_geometry_path = export_scene_geometry(
                        args.infinigen_installation_path,
                        blender_file_path,
                        blender_geometry_export_script_path,
                        goal_file_path,
                        os.path.join(task_instance_dir, "goal_geometry"),
                    )
                    start_point_cloud = sample_point_cloud_from_npz(
                        start_geometry_path, args.chamfer_num_points
                    )
                    goal_point_cloud = sample_point_cloud_from_npz(
                        goal_geometry_path, args.chamfer_num_points
                    )
                    task_instance_scores["start"]["chamfer"] = chamfer_distance(
                        start_point_cloud,
                        goal_point_cloud,
                        args.chamfer_chunk_size,
                    )
                except Exception:
                    pass

            task_instance_scores["non_executable_count"] = 0
            with open(os.path.join(task_instance_dir, "scores.json"), "w") as file:
                json.dump(task_instance_scores, file, indent=4)

            scores_across_instances["start_n_clip"].append(avg_n_clip)
            scores_across_instances["start_pl"].append(avg_pl)
            if "chamfer" in task_instance_scores["start"]:
                scores_across_instances["start_chamfer"].append(
                    task_instance_scores["start"]["chamfer"]
                )

        if len(scores_across_instances["start_n_clip"]) < (
            len(inference_metadata[task]) * 0.25
        ):
            scores_across_tasks[task] = {
                "non_executable_count": scores_across_instances["non_executable_count"],
            }
        else:
            scores_across_tasks[task] = {
                "start_n_clip": sum(scores_across_instances["start_n_clip"])
                / len(scores_across_instances["start_n_clip"]),
                "start_pl": sum(scores_across_instances["start_pl"])
                / len(scores_across_instances["start_pl"]),
                "non_executable_count": scores_across_instances["non_executable_count"],
            }
            if scores_across_instances["start_chamfer"]:
                scores_across_tasks[task]["start_chamfer"] = sum(
                    scores_across_instances["start_chamfer"]
                ) / len(scores_across_instances["start_chamfer"])

        intermediates[task] = scores_across_instances

        with open(
            os.path.join(eval_render_save_dir, f"{task}_scores.json"), "w"
        ) as file:
            json.dump(scores_across_instances, file, indent=4)

    with open(os.path.join(eval_render_save_dir, "overall_scores.json"), "w") as file:
        json.dump(scores_across_tasks, file, indent=4)

    with open(
        os.path.join(eval_render_save_dir, "intermediate_scores.json"), "w"
    ) as file:
        json.dump(intermediates, file, indent=4)
