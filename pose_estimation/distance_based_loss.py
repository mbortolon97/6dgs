import torch
from typing import Tuple


def best_one_to_one_rays_selector(
    camera_intrinsic,
    camera_pose,
    obs_img_shape,
    rays_dir,
    rays_ori,
    backbone_wh,
    tanh_denominator=1.0,
):
    gt_camera_position = (
        torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            dtype=camera_pose.dtype,
            device=camera_pose.device,
        ).reshape(1, 4)
        @ camera_pose[:3, :].T
    )
    vector_to_cam = gt_camera_position - rays_ori
    # batch version of torch.dot(vector_to_point, rays_dir)
    projection_length = torch.bmm(
        vector_to_cam.view(vector_to_cam.shape[0], 1, vector_to_cam.shape[-1]),
        rays_dir.view(rays_dir.shape[0], rays_dir.shape[-1], 1),
    )[
        ..., 0
    ]  # one dim leaved for avoid using none after

    closest_point_along_ray = torch.where(
        projection_length < 0,
        rays_ori,
        rays_ori + torch.multiply(projection_length, rays_dir),
    )
    distance = torch.linalg.norm(closest_point_along_ray - gt_camera_position, dim=-1)
    target_score = 1 - torch.tanh(distance / tanh_denominator)

    gt_camera_z_axis = (
        torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=camera_pose.dtype,
            device=camera_pose.device,
        ).reshape(1, 3)
        @ camera_pose[:3, :3].T
    )
    # batch version of torch.dot(vector_to_point, rays_dir)
    vector_to_point = rays_ori - gt_camera_position
    cam_projection_length = (
        vector_to_point.view(vector_to_point.shape[0], 1, vector_to_point.shape[-1])
        @ gt_camera_z_axis.view(1, gt_camera_z_axis.shape[-1], 1)
    )[
        ..., 0, 0
    ]  # one dim leaved for avoid using none after
    positive_projection_sign = (
        (cam_projection_length / torch.abs(cam_projection_length)) + 1.0
    ) / 2.0
    target_score = target_score * positive_projection_sign

    point_distance = torch.linalg.norm(vector_to_cam, dim=-1)
    # point_distance_min = point_distance.min()
    # point_distance = torch.divide(
    #     point_distance - point_distance_min,
    #     point_distance.max() - point_distance_min,
    # )
    point_distance_score = 1 - torch.tanh(point_distance / tanh_denominator)
    target_score_with_distance = torch.multiply(target_score, point_distance_score)

    projection_matrix = camera_intrinsic @ torch.linalg.inv(camera_pose)[:3, :]
    cam_pixels = (
        projection_matrix
        @ torch.cat(
            (
                rays_ori.mT,
                torch.ones(
                    1,
                    rays_ori.shape[0],
                    dtype=rays_ori.dtype,
                    device=rays_dir.device,
                ),
            ),
            dim=0,
        )
    ).mT  # this is valid because the rays_ori was moved to the object surface
    cam_pixels = torch.divide(cam_pixels[..., :2], cam_pixels[..., [-1]])
    # process to convert to feature space coordinates
    # resize
    backbone_scaling = 256
    if obs_img_shape[0] < obs_img_shape[1]:
        # width is lower
        width_scale_factor = backbone_scaling / obs_img_shape[0]
        height_scale_factor = width_scale_factor
    else:
        # height is lower
        height_scale_factor = backbone_scaling / obs_img_shape[1]
        width_scale_factor = height_scale_factor
    cam_pixels[:, 0] = cam_pixels[:, 0] * width_scale_factor
    cam_pixels[:, 1] = cam_pixels[:, 1] * height_scale_factor
    # center crop
    backbone_crop = 224
    cam_pixels[:, 0] -= ((width_scale_factor * obs_img_shape[0]) - backbone_crop) // 2
    cam_pixels[:, 1] -= ((height_scale_factor * obs_img_shape[1]) - backbone_crop) // 2
    patch_size = 14.0
    cam_pixels = cam_pixels / patch_size
    is_inside = (
        (cam_pixels[..., 1] >= 0.0)
        & (cam_pixels[..., 1] <= backbone_wh[1])
        & (cam_pixels[..., 0] >= 0.0)
        & (cam_pixels[..., 0] <= backbone_wh[0])
    )

    long_cam_pixels = cam_pixels.to(torch.long)
    unique_idx = (
        torch.multiply(
            long_cam_pixels[is_inside, 0],
            backbone_wh[1],
        )
        + long_cam_pixels[is_inside, 1]
    )
    out = torch.zeros(
        (257,),  # artificially one pixel more to avoid overflow
        dtype=target_score_with_distance.dtype,
        device=target_score_with_distance.device,
    )

    # _, best_rays_idx = scatter_max(
    #     target_score_with_distance[is_inside], unique_idx, 0, out=out
    # )
    # mask = best_rays_idx != unique_idx.shape[0]
    # best_rays_idx = best_rays_idx[mask]

    # best_rays_idx = best_rays_idx[best_rays_idx != torch.count_nonzero(is_inside)]

    # from visual import display_select_vector_direction
    # _, best_rays_idx = torch.topk(target_score, k=300)
    # display_select_vector_direction(
    #     rays_ori[best_rays_idx].cpu().numpy(),
    #     (rays_ori[best_rays_idx] + rays_dir[best_rays_idx] * 0.5).cpu().numpy(),
    #     rays_ori.cpu().numpy(),
    #     extrinsic_target_camera=camera_pose.cpu().numpy(),
    #     # camera_optical_axis=(-vector_to_cam).cpu().numpy(),
    # )

    return None, is_inside, target_score, target_score_with_distance


class DistanceBasedScoreLoss(torch.nn.Module):
    def __init__(
        self,
        reweight_method="none",
        lds=False,
        lds_kernel="gaussian",
        lds_ks=5,
        lds_sigma=2,
        total_number_of_elements: float = 256.0,
    ):
        super().__init__()

        assert reweight_method in {"none", "inverse", "sqrt_inv"}
        assert (
            reweight_method != "none" if lds else True
        ), "Set reweight to 'sqrt_inv' (default) or 'inverse' when using LDS"
        self.reweight_method = reweight_method
        self.lds = lds
        self.lds_kernel = lds_kernel
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma

    def forward(
        self,
        pred_score: torch.Tensor,
        camera_pose: torch.Tensor,
        camera_intrinsic: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        total_number_of_features: int,
        backbone_wh: Tuple[int, int],
        model_up=None,
        obs_img_shape=(800, 800),
    ):
        with torch.no_grad():
            (
                _,
                _,
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

            gt_camera_position = (
                torch.tensor(
                    [0.0, 0.0, 0.0, 1.0],
                    dtype=camera_pose.dtype,
                    device=camera_pose.device,
                ).reshape(1, 4)
                @ camera_pose[:3, :].T
            )
            vector_to_camera = gt_camera_position - rays_ori
            vector_to_camera_norm = torch.divide(
                vector_to_camera,
                torch.linalg.norm(vector_to_camera, dim=-1, keepdim=True),
            )

            cosine_similarity = (
                (rays_dir[..., None, :] @ vector_to_camera_norm[..., :, None])[
                    ..., 0, 0
                ]
                + 1.0
            ) / 2.0

            # breakpoint()

            # cosine_similarity *
            combined_score = target_score

            # if target_score.isnan().any():
            #     breakpoint()
            score_multiplier = total_number_of_features / combined_score.sum()

            # if score_multiplier.isnan().any():
            #     breakpoint()

            combined_score = torch.multiply(combined_score, score_multiplier)

            # if target_score.isnan().any():
            #     breakpoint()

            # from visual import display_select_vector_direction
            # from pose_estimation.line_intersection import (
            #     compute_line_intersection_impl3,
            #     make_rotation_mat,
            # )
            # values, index = torch.topk(combined_score, k=100)
            # camera_optical_center = compute_line_intersection_impl3(
            #     rays_ori[index], rays_dir[index], weights=combined_score[index]
            # )
            # camera_watch_dir = torch.multiply(rays_dir[index], values[:, None]).sum(
            #     dim=0
            # )
            # camera_watch_dir = torch.divide(
            #     camera_watch_dir,
            #     torch.linalg.norm(camera_watch_dir, dim=-1, keepdim=True),
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

            # import matplotlib.pyplot as plt
            # plt.hist(target_score.cpu().numpy(), bins=100)
            # plt.show()

        # if pred_score.isnan().any():
        #     breakpoint()

        score_diff = torch.square(pred_score - combined_score)

        # if score_diff.isnan().any():
        #     breakpoint()

        # weights = 1 - target_score
        # avg_score = torch.divide(torch.sum(torch.multiply(score_diff, weights)), weights.sum())
        avg_score = score_diff.mean()
        return avg_score, combined_score
