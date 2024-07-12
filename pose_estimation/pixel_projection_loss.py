import torch
from typing import Tuple
from pose_estimation.distance_based_loss import best_one_to_one_rays_selector

class SinglePixelProjectionLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()

        if weight is not None:
            weight = torch.tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5

        self.n_classes = 2

    def forward(
        self,
        pred_score: torch.Tensor,
        camera_pose: torch.Tensor,
        camera_intrinsic: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        backbone_wh: Tuple[int, int],
        obs_img_shape=(800, 800),  # W, H
    ):
        with torch.no_grad():
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
            )

            classification_target = torch.full(
                (rays_ori.shape[0],),
                1.0e-7,
                dtype=pred_score.dtype,
                device=pred_score.device,
            )
            classification_target[best_rays_idx] = 1.0 - 1.0e-7

            # from visual import display_select_vector_direction
            # values = target_score[is_inside][best_rays_idx]
            # _, top_idx = torch.topk(values, k=30)
            # values = values[top_idx]
            # index = torch.arange(
            #     0, rays_ori.shape[0], dtype=torch.long, device=rays_ori.device
            # )[is_inside][best_rays_idx][top_idx]
            # camera_optical_center = compute_line_intersection(
            #     rays_ori[index], -rays_dir[index], weights=values
            # )
            # # values.fill_(1.0)
            # camera_watch_dir = torch.multiply(rays_dir[index], values[:, None]).sum(
            #     dim=0
            # )
            # camera_watch_dir = torch.divide(
            #     camera_watch_dir,
            #     torch.linalg.norm(camera_watch_dir, dim=-1, keepdim=True),
            # )
            # model_up = torch.tensor(
            #     [0.0, 0.0, -1.0], dtype=rays_ori.dtype, device=rays_ori.device
            # )
            # model_up = torch.divide(
            #     model_up, torch.linalg.norm(model_up, dim=-1, keepdim=True)
            # )
            # c2w_matrix = torch.eye(4)
            # c2w_matrix[:3, :3] = torch.linalg.inv(
            #     make_rotation_mat(-camera_watch_dir, model_up)
            # )
            # c2w_matrix[:3, -1] = camera_optical_center
            # display_select_vector_direction(
            #     rays_ori[index].cpu().numpy(),
            #     (rays_ori[index] + 0.2 * rays_dir[index]).cpu().numpy(),
            #     rays_ori.cpu().numpy(),
            #     color_distances=target_score[index].cpu().numpy(),
            #     extrinsic_target_camera=camera_pose.cpu().numpy(),
            #     extrinsic_predict_camera=c2w_matrix.cpu().numpy(),
            # )

            # best score

        # loss = torch.nn.functional.binary_cross_entropy(pred_score, target_score),
        loss = torch.square(target_score - pred_score).mean()
        return (
            loss,
            classification_target,
        )
