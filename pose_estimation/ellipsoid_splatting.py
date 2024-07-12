import torch

from scene import GaussianModel
from scene.scene_structure import CameraInfo
import math
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import numpy as np
from scene.ellipsis_model import EllipsisModel
from pose_estimation.sampling import evaluate_viewdirs_color
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation, cov2D_to_scale_theta


def screenToPixCov2D(cov2D: torch.Tensor, W: int, H: int):
    proj = torch.tensor(
        [
            [2 / float(W), 0.0],
            [0.0, 2 / float(H)],
        ],
        dtype=cov2D.dtype,
        device=cov2D.device,
    )
    return proj[None] @ cov2D @ proj[None].mT


def ellipsoid_splatting(gaussian_model: GaussianModel, camera_info: CameraInfo):
    means_3D = gaussian_model.get_xyz
    cov_3D = gaussian_model.get_covariance_mat()

    tan_fovx = math.tan(camera_info.FovX * 0.5)
    tan_fovy = math.tan(camera_info.FovY * 0.5)

    focal_x = camera_info.width / (2.0 * tan_fovx)
    focal_y = camera_info.height / (2.0 * tan_fovy)

    # compute the center splatted
    viewmatrix = (
        torch.tensor(
            getWorld2View2(camera_info.R, camera_info.T, np.array([0.0, 0.0, 0.0]), 1.0)
        )
        .transpose(0, 1)
        .to(device=means_3D.device)
    )

    projection_matrix = (
        getProjectionMatrix(0.01, 100.0, camera_info.FovX, camera_info.FovY)
        .transpose(0, 1)
        .to(device=means_3D.device)
    )
    P = viewmatrix @ projection_matrix

    means_2D = (
        torch.cat(
            (
                means_3D,
                torch.ones(
                    (means_3D.shape[0], 1), dtype=means_3D.dtype, device=means_3D.device
                ),
            ),
            dim=-1,
        )
        @ P
    )
    p_w = 1.0 / (means_2D[..., -1] + 0.0000001)
    means_2D = means_2D * p_w[..., None]

    # compute the cov2D spla
    t = (
        torch.cat(
            (
                means_3D,
                torch.ones(
                    (means_3D.shape[0], 1), dtype=means_3D.dtype, device=means_3D.device
                ),
            ),
            dim=-1,
        )
        @ viewmatrix[:, :3]
    )
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    t[..., :2] = t[..., :2] / t[..., [-1]]
    t[..., 0] = torch.clamp_max(torch.clamp_min(t[..., 0], -limx), limx) * t[..., -1]
    t[..., 1] = torch.clamp_max(torch.clamp_min(t[..., 1], -limy), limy) * t[..., -1]

    z_square = torch.square(t[..., -1])

    J = torch.zeros(*t.shape[:-1], 3, 3, dtype=t.dtype, device=t.device)
    J[:, 0, 0] = focal_x / t[..., -1]
    J[:, 0, -1] = -(focal_x * t[..., 0]) / z_square
    J[:, 1, 1] = focal_y / t[..., -1]
    J[:, 1, -1] = -(focal_y * t[..., 1]) / z_square

    T = J @ viewmatrix[None, :3, :3].mT

    cov = T @ cov_3D.mT @ T.mT

    # Apply low-pass filter: every Gaussian should be at least one pixel wide/high. Discard 3rd row and column.
    cov[:, 0, 0] += 0.3
    cov[:, 1, 1] += 0.3
    cov = cov[:, :2, :2]

    # convert the covariance to NDC-space (-1, 1)
    cov_2D = screenToPixCov2D(cov, camera_info.width, camera_info.height)
    K = torch.tensor(
        [
            [focal_x / (camera_info.width * 0.5), 0, 0.0],
            [0, focal_y / (camera_info.height * 0.5), 0.0],
            [0, 0.0, 1.0],
        ],
        dtype=gaussian_model.get_xyz.dtype,
        device=gaussian_model.get_xyz.device,
    )
    a, b, theta = cov2D_to_scale_theta(cov_2D, K)

    # compute the splatted color
    shs_view = gaussian_model.get_features.transpose(1, 2).view(
        -1, 3, (gaussian_model.max_sh_degree + 1) ** 2
    )
    camera_center = viewmatrix.inverse()[3, :3]
    dir_pp = means_3D - camera_center.repeat(gaussian_model.get_features.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(gaussian_model.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    model = EllipsisModel()

    # as we are in NDC bounding box size is [-1,-1],[1,1]
    model.create_from_memory(
        means_2D[..., :2], colors_precomp, gaussian_model.get_opacity, cov_2D, 1.41
    )

    # model.create_from_components(
    #     means_2D[..., :2],
    #     colors_precomp,
    #     gaussian_model.get_opacity,
    #     theta,
    #     torch.square(torch.stack((a, b), dim=-1)),
    #     1.41,
    # )

    return model


def ellipsoid_splatting_p1e(gaussian_model: GaussianModel, camera_info: CameraInfo):
    tan_fovx = math.tan(camera_info.FovX * 0.5)
    tan_fovy = math.tan(camera_info.FovY * 0.5)

    focal_x = camera_info.width / (2.0 * tan_fovx)
    focal_y = camera_info.height / (2.0 * tan_fovy)
    K = torch.tensor(
        [
            [focal_x / (camera_info.width * 0.5), 0, 0.0],
            [0, focal_y / (camera_info.height * 0.5), 0.0],
            [0, 0.0, 1.0],
        ],
        dtype=gaussian_model.get_xyz.dtype,
        device=gaussian_model.get_xyz.device,
    )
    P = K @ torch.cat(
        (
            torch.from_numpy(camera_info.R)
            .to(
                dtype=gaussian_model.get_xyz.dtype,
                device=gaussian_model.get_xyz.device,
            )
            .mT,
            torch.from_numpy(camera_info.T)
            .to(
                dtype=gaussian_model.get_xyz.dtype,
                device=gaussian_model.get_xyz.device,
            )
            .unsqueeze(1),
        ),
        dim=1,
    )
    P = P / P[..., -1, -1]
    P = P[None]

    Z = torch.cat(
        (
            torch.cat(
                (
                    build_rotation(gaussian_model.get_rotation),
                    gaussian_model.get_xyz[..., :, None],
                ),
                dim=-1,
            ),
            torch.tensor(
                [0.0, 0.0, 0.0, 1.0],
                dtype=gaussian_model.get_xyz.dtype,
                device=gaussian_model.get_xyz.device,
            )[None, None].expand(gaussian_model.get_xyz.shape[0], -1, -1),
        ),
        dim=-2,
    )

    scale = gaussian_model.get_scaling
    diag_v = torch.diag_embed(
        torch.cat(
            (
                scale,
                torch.full(
                    (scale.shape[0], 1), -1, dtype=scale.dtype, device=scale.device
                ),
            ),
            dim=-1,
        )
    )
    V = Z @ diag_v @ Z.mT
    V = V / V[..., -1, -1][..., None, None]
    C = P @ V @ P.mT
    C = C / C[..., -1, -1][..., None, None]
    center = C[..., 0:2, 2]
    T = (
        torch.eye(3, dtype=center.dtype, device=center.device)[None]
        .expand(center.shape[0], -1, -1)
        .contiguous()
    )
    T[:, :2, -1] = -center
    C_cent = T @ C @ T.mT
    a, b, theta = cov2D_to_scale_theta(C_cent[..., 0:2, 0:2], K)

    # compute the splatted color
    shs_view = gaussian_model.get_features.transpose(1, 2).view(
        -1, 3, (gaussian_model.max_sh_degree + 1) ** 2
    )
    viewmatrix = (
        torch.tensor(
            getWorld2View2(camera_info.R, camera_info.T, np.array([0.0, 0.0, 0.0]), 1.0)
        )
        .transpose(0, 1)
        .to(device=shs_view.device)
    )
    camera_center = viewmatrix.inverse()[3, :3]
    dir_pp = gaussian_model.get_xyz - camera_center.repeat(
        gaussian_model.get_features.shape[0], 1
    )
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(gaussian_model.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    model = EllipsisModel()

    # as we are in NDC bounding box size is [-1,-1],[1,1]
    model.create_from_components(
        center,
        colors_precomp,
        gaussian_model.get_opacity,
        theta,
        torch.square(torch.stack((a, b), dim=-1)),
        1.41,
    )

    return model


def ellipsoid_splatted_only_visible(
    gaussian_model: GaussianModel, camera_info: CameraInfo
):
    ellipsoid_splatting(gaussian_model, camera_info)
