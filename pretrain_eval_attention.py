import json
import os
import traceback
from functools import partial
from typing import Any

import numpy as np
import torch

from pose_estimation.file_utils import parse_exp_dir, get_checkpoint_arguments, dotdict
from pose_estimation.identification_module import IdentificationModule
from pose_estimation.opt import parse_args
from pose_estimation.sampling import generate_all_possible_rays
from pose_estimation.test import test_pose_estimation
from pose_estimation.train import train_id_module
from scene import GaussianModel, load_data
from pose_estimation.distance_based_loss import DistanceBasedScoreLoss


def load_model(checkpoint_path, device, sh_degrees=3):
    # Load 3DGS gaussian model
    model = GaussianModel(sh_degrees)
    model.load_ply(checkpoint_path)
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    return model


def pretrain_single_object(
    checkpoint_filepath: str,
    checkpoint_args: dotdict[str, Any],
    exp_dir_filepath: str,
    object_id: str,
    category_name: str,
    starting_seed: int,
    lock_backbone: bool = True,
    device: str = "cuda",
):
    torch.manual_seed(starting_seed)

    print("data_path: ", checkpoint_args.source_path)

    gs_model = load_model(
        checkpoint_filepath, device, sh_degrees=checkpoint_args.sh_degree
    )

    if checkpoint_args.fps_sampling is None:
        checkpoint_args.fps_sampling = -1
    
    scene_info = load_data(checkpoint_args)

    backbone_type = "dino"
    id_module = (
        IdentificationModule(backbone_type=backbone_type)
        .to(device, non_blocking=True)
        .train()
    )

    if lock_backbone:
        for parameter in id_module.backbone_wrapper.parameters():
            parameter.require_grad = False

    start_iterations = 0
    id_module_ckpt_path = os.path.join(exp_dir_filepath, "id_module.th")
    if os.path.exists(id_module_ckpt_path):
        print("Checkpoint already exist, skip training phase")
        ckpt_dict = torch.load(id_module_ckpt_path, map_location=device)
        id_module.load_state_dict(ckpt_dict["model_state_dict"])
        start_iterations = ckpt_dict["epoch"]
    
    generator_callable = partial(explore_model, gs_model)

    train_id_module(
        id_module_ckpt_path,
        device,
        id_module,
        generator_callable,
        scene_info,
        object_id,
        category_name,
        start_iterations=start_iterations,
        lock_backbone=lock_backbone,
    )

    print("Training complete starting testing phase...")
    print("Testing overfit performances...")
    rays_ori, rays_dirs, rays_rgb = explore_model(gs_model)

    model_up_np = np.mean(
        np.asarray(
            [train_camera.R[:3, 1] for train_camera in scene_info.train_cameras],
            dtype=np.float32,
        ),
        axis=0,
    )
    model_up = torch.from_numpy(model_up_np).to(device=device, non_blocking=True)

    loss_fn = DistanceBasedScoreLoss()
    (
        _,
        overfit_avg_translation_error,
        overfit_avg_angular_error,
        overfit_avg_score,
        overfit_recall,
    ) = test_pose_estimation(
        scene_info.test_cameras,
        id_module,
        rays_ori,
        rays_dirs,
        rays_rgb,
        model_up,
        sequence_id=object_id,
        category_id=category_name,
        # inerf_refinement=inerf_refinement,
        loss_fn=loss_fn,
        # gs_model=gs_model,
        # save=True,
        # save_all=True,
    )
    #
    print("Overfit AVG translation error: ", overfit_avg_translation_error)
    print("Overfit AVG angular error: ", overfit_avg_angular_error)
    print("Overfit AVG score error: ", overfit_avg_score)
    print("Overfit recall: ", overfit_recall)

    print("Testing performances on same points...")

    (
        test_results,
        test_avg_translation_error,
        test_avg_angular_error,
        test_avg_score,
        test_recall,
    ) = test_pose_estimation(
        scene_info.test_cameras,
        id_module,
        rays_ori,
        rays_dirs,
        rays_rgb,
        model_up,
        sequence_id=object_id,
        category_id=category_name,
        save=False,
        save_all=False,
    )

    print("Test AVG translation error: ", test_avg_translation_error)
    print("Test AVG angular error: ", test_avg_angular_error)
    print("Test AVG score error: ", test_avg_score)
    print("Test recall: ", test_recall)

    return test_results


def cache_model_on_gpu(ckpt_path, device):
    model = load_model(ckpt_path, device)
    model.eval()
    model = model.to(device, non_blocking=True)


def explore_model(model: GaussianModel):
    point, dirs, rgb = generate_all_possible_rays(
        model,
        sample_quadricell_targets=50,
    )

    return point, dirs, rgb


def evaluate_single_object_in_blender(
    checkpoint_filepath: str,
    checkpoint_args: dotdict[str, Any],
    exp_dir_filepath: str,
    object_id: str,
    category_name: str,
    starting_seed: int = 55176280,
    device: str = "cuda",
    lock_backbone: bool = True,
):
    # Explore given directory
    results = pretrain_single_object(
        checkpoint_filepath,
        checkpoint_args,
        exp_dir_filepath,
        object_id,
        category_name,
        starting_seed,
        device=device,
        lock_backbone=lock_backbone,
    )

    if partial:
        print("Not all the sequences available")

    return results


def main():
    args, _ = parse_args()

    # create destination directory structure
    out_path_abs = os.path.abspath(args.out_path)
    out_dir = os.path.dirname(out_path_abs)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.data_type == "blender":
        prefix = "synthetic_"
    elif args.data_type == "mip360":
        prefix = "mip_360_"
    elif args.data_type == "tankstemple":
        prefix = "tt_"
    elif args.data_type == "cambridge_landmark":
        prefix = "cl_"
    else:
        prefix = ""

    # create the container for the results
    results = []
    experiments_to_test = parse_exp_dir(args.exp_path, prefix)
    for experiment_to_test in experiments_to_test.values():
        exp_dir_filepath = experiment_to_test["exp_dir_filepath"]
        checkpoint_filepath = experiment_to_test["checkpoint_filepath"]
        object_id = experiment_to_test["sequence_id"]
        category_name = experiment_to_test["category_name"]
        checkpoint_args = get_checkpoint_arguments(exp_dir_filepath)
        try:
            obj_results = evaluate_single_object_in_blender(
                checkpoint_filepath,
                checkpoint_args,
                exp_dir_filepath,
                object_id,
                category_name,
                starting_seed=55176280,
                device=device,
                lock_backbone=True,
            )

            results.extend(obj_results)
        except RuntimeError:
            traceback.print_exc()

    print("Saving results")
    with open(out_path_abs, "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    # torch.manual_seed(500661008)
    torch.manual_seed(71170)
    main()
