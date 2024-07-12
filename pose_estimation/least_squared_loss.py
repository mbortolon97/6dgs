import torch
from typing import Tuple
from pose_estimation.line_intersection import compute_line_intersection_impl2

class LeastSquaredLoss(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        pred_score: torch.Tensor,
        camera_pose: torch.Tensor,
        camera_intrinsic: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        total_number_of_features: int,
        backbone_wh: Tuple[int, int],
        obs_img_shape=(800, 800),
    ):
        with torch.no_grad():
            gt_camera_position = (
                torch.tensor(
                    [0.0, 0.0, 0.0, 1.0],
                    dtype=camera_pose.dtype,
                    device=camera_pose.device,
                ).reshape(1, 4)
                @ camera_pose[:3, :].T
            )

            (
                best_rays_idx,
                is_inside,
                target_score,
                target_score_with_distance,
            ) = best_one_to_one_rays_selector(
                camera_intrinsic,
                camera_pose,
                obs_img_shape,
                rays_dir,
                rays_ori,
                backbone_wh=backbone_wh,
                tanh_denominator=1.0,
            )

            pred_gt_solution = compute_line_intersection_impl2(
                rays_ori, -rays_dir, weights=(target_score / total_number_of_features)
            )

            print("gt_camera_position: ", gt_camera_position)
            print("pred_gt_solution: ", pred_gt_solution)
            print(
                "pred_gt_solution gt_camera_position distance: ",
                torch.linalg.norm(pred_gt_solution - gt_camera_position, dim=-1),
            )

        # if pred_score.isnan().any():
        #     breakpoint()

        # camera_optical_center = compute_line_intersection(point, -dirs, weights=weights)
        solution = compute_line_intersection_impl2(
            rays_ori, -rays_dir, weights=(pred_score / total_number_of_features)
        )

        avg_score = torch.nn.functional.smooth_l1_loss(
            solution[None], gt_camera_position
        )

        return avg_score, torch.zeros_like(rays_ori[..., 0])
