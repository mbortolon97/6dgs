import time
from typing import List

import torch
from tqdm import tqdm

from pose_estimation.error_computation import (
    compute_translation_error,
    compute_angular_error,
)
from statistics import mean

from pose_estimation.line_intersection import (
    compute_line_intersection_impl2,
    exclude_negatives,
    make_rotation_mat,
)
from scene.scene_structure import CameraInfo
from utils.graphics_utils import fov2focal
import numpy as np


def test_pose_estimation(
    cameras_info: List[CameraInfo],
    id_module,
    rays_ori,
    rays_dirs,
    rays_rgb,
    model_up,
    sequence_id="",
    category_id="",
    loss_fn=None,
    save=False,
    save_all=False,
):
    id_module.eval()

    translation_errors = []
    angular_errors = []
    model_up = torch.divide(model_up, torch.linalg.norm(model_up, dim=-1, keepdim=True))

    recalls = []
    avg_loss_scores = []
    results = []
    start_time = time.time()
    for img_idx, camera_info in tqdm(enumerate(cameras_info)):
        w2c = torch.eye(4, dtype=torch.float32, device=rays_ori.device)
        w2c[:3, :3] = torch.transpose(torch.from_numpy(camera_info.R), -1, -2).to(
            rays_ori.device, non_blocking=True
        )
        w2c[:3, -1] = torch.from_numpy(camera_info.T).to(
            rays_ori.device, non_blocking=True
        )
        c2w = torch.inverse(w2c)

        pose = c2w
        focalX = fov2focal(camera_info.FovX, camera_info.width)
        focalY = fov2focal(camera_info.FovY, camera_info.height)
        target_camera_intrinsic = torch.tensor(
            [
                [focalX, 0.0, camera_info.width / 2],
                [0.0, focalY, camera_info.height / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=rays_ori.device,
        )

        tensor_image = torch.from_numpy(np.array(camera_info.image))
        obs_img = tensor_image.to(
            device=rays_ori.device, dtype=torch.float32, non_blocking=True
        )
        obs_img = obs_img / 255.0

        if obs_img.shape[-1] == 4:
            mask_img = obs_img[..., -1] > 0.3
            obs_img = torch.multiply(obs_img[..., :3], obs_img[..., -1:]) + (
                1 - obs_img[..., -1:]
            )
        else:
            mask_img = torch.ones_like(
                obs_img[..., -1], dtype=torch.bool, device=rays_ori.device
            )

        idx, weights, pred_scores, camera_up_dir, attention_map = id_module.test_image(
            obs_img,
            mask_img,
            rays_ori,
            rays_dirs,
            rays_rgb,
            rays_to_output=100,
        )

        if save and (img_idx == 0 or save_all):
            saving_dict = {
                "gt_pose": pose.cpu(),
                "camera_intrinsic": target_camera_intrinsic.cpu(),
                "all_rays_ori": rays_ori.cpu(),
                "all_rays_dirs": rays_dirs.cpu(),
                "all_rays_rgb": rays_rgb.cpu(),
                "obs_img": obs_img.cpu(),
                "mask_img": mask_img.cpu(),
                "topk_nonunique_ray_idx": idx.cpu(),
                "topk_nonunique_weights": weights.cpu(),
                "all_predict_weights": pred_scores.cpu(),
            }

        avg_score = -1.0
        recall = -1.0
        if loss_fn is not None:
            avg_score, target_scores = loss_fn(
                pred_scores,
                pose,
                target_camera_intrinsic,
                rays_ori,
                rays_dirs,
                attention_map.shape[-2],
                id_module.backbone_wrapper.backbone_wh,
                model_up=camera_up_dir,
            )
            avg_score = avg_score.item()
            target_idx = torch.topk(weights, k=100).indices
            intersection = torch.count_nonzero(torch.isin(target_idx, idx))
            recall = intersection.item() / target_idx.shape[0]

            # from visual import display_selected_and_gt_rays
            # display_selected_and_gt_rays(
            #     rays_ori[target_idx].cpu().numpy(),
            #     (rays_ori[target_idx] + rays_dirs[target_idx] * 0.1).cpu().numpy(),
            #     rays_ori[idx].cpu().numpy(),
            #     (rays_ori[idx] + rays_dirs[idx] * 0.1).cpu().numpy(),
            #     rays_ori.cpu().numpy(),
            #     color_distances=weights.cpu().numpy(),
            #     # extrinsic_target_camera=pose.cpu().numpy(),
            #     # extrinsic_predict_camera=c2w_matrix.cpu().numpy(),
            # )
            if save and (img_idx == 0 or save_all):
                saving_dict["all_target_weights"] = target_scores.cpu()
                saving_dict["loss"] = avg_score
                saving_dict["recall"] = recall

            weights, idx = torch.topk(target_scores, k=100, largest=True)
            # breakpoint()

        avg_loss_scores.append(avg_score)
        recalls.append(recall)
        # from visual import display_select_vector_direction
        # display_select_vector_direction(
        #     rays_ori[idx].cpu().numpy(),
        #     (rays_ori[idx] + rays_dirs[idx] * 0.1).cpu().numpy(),
        #     rays_ori.cpu().numpy(),
        #     color_distances=weights.cpu().numpy(),
        #     # extrinsic_target_camera=pose.cpu().numpy(),
        #     # extrinsic_predict_camera=c2w_matrix.cpu().numpy(),
        # )

        unique_elements, counts = torch.unique(rays_ori[idx], return_counts=True, dim=0)
        mask = torch.isin(
            rays_ori[idx], unique_elements[counts == 1], assume_unique=True
        ).any(dim=1)
        idx = idx[mask]
        weights = weights[mask]

        if save and (img_idx == 0 or save_all):
            saving_dict["topk_unique_ray_idx"] = idx.cpu()
            saving_dict["topk_unique_weights"] = weights.cpu()

        weights = torch.divide(weights, torch.sum(weights))
        camera_optical_center = compute_line_intersection_impl2(
            rays_ori[idx], rays_dirs[idx]  # , weights=weights
        )
        weights = torch.multiply(
            weights,
            exclude_negatives(camera_optical_center, rays_ori[idx], rays_dirs[idx]),
        )
        weights = torch.divide(weights, torch.sum(weights))
        camera_optical_center = compute_line_intersection_impl2(
            rays_ori[idx], rays_dirs[idx]  # , weights=weights
        )
        # camera_optical_center = compute_line_intersection_impl2(
        #     rays_ori[idx], rays_dirs[idx]
        # )

        if torch.isnan(camera_optical_center).any(dim=0):
            print("camera_optical_center is nan")

        camera_watch_dir = torch.multiply(rays_dirs[idx], weights[:, None]).sum(dim=0)
        camera_watch_dir = torch.divide(
            camera_watch_dir, torch.linalg.norm(camera_watch_dir, dim=-1, keepdim=True)
        )

        c2w_matrix = torch.eye(4, dtype=rays_ori.dtype, device=rays_ori.device)
        w2c_rotation_matrix = make_rotation_mat(-camera_watch_dir, camera_up_dir)
        if torch.linalg.det(w2c_rotation_matrix) < 1.0e-7:
            print("extracted rotation matrix is singular")
            w2c_rotation_matrix = torch.eye(3)
        c2w_matrix[:3, :3] = torch.linalg.inv(w2c_rotation_matrix)
        c2w_matrix[:3, -1] = camera_optical_center
        # c2w_matrix[:3, :3] = torch.linalg.inv(R[0])
        # c2w_matrix[:3, -1] = t[0]

        if save and (img_idx == 0 or save_all):
            saving_dict["topk_unique_weights_after_exclusion"] = weights.cpu()
            saving_dict["pred_camera_optical_center"] = camera_optical_center.cpu()
            saving_dict["pred_camera_watch_dir"] = -camera_watch_dir.cpu()
            saving_dict["pred_c2w_matrix"] = c2w_matrix.cpu()
            saving_dict["model_up"] = model_up.cpu()

            torch.save(
                saving_dict,
                f"/home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/sample_results_{img_idx}.th",
            )

            print("Sample result saved")

        if torch.isnan(c2w_matrix).any():
            print("wrong c2w")
            c2w_matrix = torch.eye(4, dtype=rays_ori.dtype, device=rays_ori.device)

        # if False:
        #     ori_directions, dx, dy = get_ray_directions_Ks(
        #         obs_img.shape[0], obs_img.shape[1], target_camera_intrinsic[None]
        #     )
        #     directions = ori_directions / torch.norm(
        #         ori_directions, dim=-1, keepdim=True
        #     )
        #
        #     rays_o, rays_d, radii = get_rays(
        #         directions,
        #         c2w_matrix[None].to(directions.device),
        #         directions=ori_directions,
        #         dx=dx,
        #         dy=dy,
        #         keepdim=True,
        #     )  # both (h, w, 3)
        #
        #     all_rays = torch.cat([rays_o, rays_d, radii], -1)  # (h*w, 7)
        #
        #     rgbs = []
        #     all_rays_flat = all_rays.view(-1, all_rays.shape[-1])
        #     chunks = torch.split(all_rays_flat, split_size_or_sections=8192, dim=0)
        #     for chunk in chunks:
        #         rgb, _, _, _, _, _ = gs_model(chunk, white_bg=True)
        #         rgbs.append(rgb)
        #     rgbs = torch.cat(rgbs, dim=0)
        #     rgbs = rgbs.view(*rays_o.shape[1:])
        #
        #     blend_img = obs_img * 0.5 + rgbs * 0.5
        #     blend_rgb_cv2 = (blend_img.cpu().numpy()[..., [2, 1, 0]] * 255.0).astype(
        #         np.uint8
        #     )
        #
        #     cv2.imwrite(f"sample_iNeRF_{img_idx}.png", blend_rgb_cv2)

        if save and (img_idx == 0 or save_all) and False:
            from visual import display_select_vector_direction

            display_select_vector_direction(
                rays_ori[idx].cpu().numpy(),
                (rays_ori[idx] + rays_dirs[idx] * 0.6).cpu().numpy(),
                rays_ori.cpu().numpy(),
                color_distances=weights.cpu().numpy(),
                extrinsic_target_camera=pose.cpu().numpy(),
                extrinsic_predict_camera=c2w_matrix.cpu().numpy(),
                # lstsq_result=camera_optical_center_without_weights.cpu().numpy(),
            )

        gt_camera_position = (
            torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=pose.dtype, device=pose.device
            ).reshape(1, 4)
            @ pose[:3, :].T
        )

        pred_camera_position = (
            torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=pose.dtype, device=pose.device
            ).reshape(1, 4)
            @ c2w_matrix[:3, :].T
        )

        translation_error = compute_translation_error(
            gt_camera_position, pred_camera_position
        )
        angular_error = compute_angular_error(pose[:3, :3], c2w_matrix[:3, :3])

        translation_errors.append(translation_error.item())
        angular_errors.append(angular_error.item())

        results.append(
            {
                "sequence_id": sequence_id,
                "category_name": category_id,
                "frame_id": img_idx,
                "loss": weights.mean().item(),
                "scores_loss": avg_score,
                "recall": recall,
                "total_optimization_time_in_ms": 0.0,
                "pred_c2w": c2w_matrix.cpu().tolist(),
                "gt_c2w": pose.cpu().tolist(),
            }
        )

    total_time = time.time() - start_time
    time_per_element = total_time / len(cameras_info)

    avg_loss_score = mean(avg_loss_scores)
    avg_recall = mean(recalls)
    print("Average loss score: ", avg_loss_score)
    print("Average Recall: ", avg_recall)
    print("Time per element: ", time_per_element)

    avg_translation_error = mean(translation_errors)
    avg_angular_error = mean(angular_errors)
    print("Translation Error: ", avg_translation_error)
    print("Angular Error: ", avg_angular_error)

    print(
        "Smallest translation error: ",
        min(range(len(translation_errors)), key=translation_errors.__getitem__),
    )

    return results, avg_translation_error, avg_angular_error, avg_loss_score, avg_recall
