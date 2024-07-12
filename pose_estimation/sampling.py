import numpy as np
import torch
import math

from pose_estimation.quadricell import (
    compute_quadricell_centers,
    mask_and_compute_rays,
    mask_degraded_ellipsoids,
)
from pose_estimation.sym_eig_3x3 import sym_eig_3x3
from scene import GaussianModel
from utils.sh_utils import eval_sh


def sampling_sphere(
    dtype: torch.dtype = torch.float32, device: torch.device = "cpu", num_viewdirs=5000
):
    sampling = torch.rand(num_viewdirs, 2, dtype=dtype, device=device)
    theta = 2 * math.pi * sampling[..., 0]
    phi = torch.arccos(1 - 2 * sampling[..., 1])
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    return torch.stack((x, y, z), dim=-1)


def batch_cov(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def disambiguate_vector_directions(
    df: torch.Tensor, vecs: torch.Tensor
) -> torch.Tensor:
    """
    Disambiguates normal directions according to [1].

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """
    # parse out K from the shape of knns
    K = df.shape[-2]

    # the difference between the mean of each neighborhood and
    # each element of the neighborhood
    # projection of the difference on the principal direction
    proj = (vecs[..., None, :] * df).sum(-1)
    # check how many projections are positive
    n_pos = torch.sum((proj > 0).to(dtype=df.dtype), dim=2, keepdim=True)
    # flip the principal directions where number of positive correlations
    flip = (n_pos < (0.5 * K)).to(dtype=df.dtype)
    vecs = (1.0 - 2.0 * flip) * vecs
    return vecs


def compute_normals(point_cloud_chunk, point_cloud, k_neighbors=20):
    """
    Compute normals for each point in a point cloud using PyTorch.

    Args:
        point_cloud_chunk (torch.Tensor): Input point cloud with shape (N, 3), where N is the number of points to generate the normal with.
        point_cloud (torch.Tensor): Input point cloud with shape (N, 3), where N is the number of points.
        k_neighbors (int): Number of nearest neighbors to consider for normal estimation.

    Returns:
        torch.Tensor: Normals for each point in the point cloud with shape (N, 3).
    """
    # Compute the pairwise distances between points
    point_cloud_chunk = point_cloud_chunk.unsqueeze(0)  # Add a batch dimension
    point_cloud = point_cloud.unsqueeze(0)  # Add a batch dimension
    pairwise_distances = torch.cdist(point_cloud_chunk, point_cloud, p=2.0)

    # Find the k-nearest neighbors for each point
    _, indices = torch.topk(pairwise_distances, k_neighbors, dim=2, largest=False)

    # Extract the coordinates of the k-nearest neighbors
    neighbour_points = torch.gather(
        point_cloud[:, None].expand(-1, point_cloud_chunk.shape[1], -1, -1),
        dim=2,
        index=indices[:, :, :, None].expand(-1, -1, -1, 3),
    )

    # Compute the covariance matrix for each point
    neighbor_points_mean = torch.mean(neighbour_points, dim=-2, keepdim=True)
    neighbor_points_centered = neighbour_points - neighbor_points_mean
    # covariance_matrix = torch.cov(neighbor_points_centered, dim=-1)
    covariance_matrix = neighbor_points_centered.mT @ neighbor_points_centered

    _, local_coord_frames = sym_eig_3x3(covariance_matrix, eigenvectors=True)
    # disambiguate the directions of individual principal vectors
    # disambiguate normal
    n = disambiguate_vector_directions(
        neighbor_points_centered, local_coord_frames[:, :, :, 0]
    )
    # disambiguate the main curvature
    z = disambiguate_vector_directions(
        neighbor_points_centered, local_coord_frames[:, :, :, 2]
    )
    # the secondary curvature is just a cross between n and z
    y = torch.torch.linalg.cross(n, z, dim=-1)
    # cat to form the set of principal directions
    local_coord_frames = torch.stack((n, y, z), dim=3)

    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    return local_coord_frames[0, :, :, 0] / torch.linalg.norm(
        local_coord_frames[0, :, :, 0], dim=-1, keepdim=True
    )


def evaluate_viewdirs_color(
    shs_view: torch.Tensor,
    viewdir: torch.Tensor,  # [n, 3]
    active_sh_degree: int,
):
    sh2rgb = eval_sh(active_sh_degree, shs_view, -viewdir)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return colors_precomp.view(*viewdir.shape)


@torch.no_grad()
def generate_all_possible_rays(
    model: GaussianModel,
    num_viewdirs_per_chunk=10240,
    sample_quadricell_targets=50,
):
    all_points = model.get_xyz
    # point_ellipsoid_idx = torch.arange(
    #     point_sampling.shape[0], dtype=torch.long, device=point_sampling.device
    # )

    # get ellipsoid subsample
    scale_all = model.get_scaling
    mask_valid = mask_degraded_ellipsoids(
        scale_all[..., 0],
        scale_all[..., 1],
        scale_all[..., 2],
    )
    valid_num_elements = torch.count_nonzero(mask_valid).item()
    point_ellipsoid_idx = torch.randperm(
        valid_num_elements, dtype=torch.long, device=all_points.device
    )[: min(1000, valid_num_elements)]
    point_sampling = all_points[mask_valid][point_ellipsoid_idx]

    # point_normals = sampling_sphere(
    #     dtype=point_sampling.dtype,
    #     device=point_sampling.device,
    #     num_viewdirs=point_sampling.shape[0],
    # )

    pts_per_chunk = 2500
    chunks = torch.split(
        torch.arange(
            0,
            point_sampling.shape[0],
            device=point_sampling.device,
            dtype=torch.long,
        ),
        pts_per_chunk,
    )
    computed_normals = []
    for chunk in chunks:
        computed_normals.append(
            compute_normals(point_sampling[chunk], point_sampling, k_neighbors=20)
        )
    point_normals = torch.cat(computed_normals, dim=0)

    scale = model.get_scaling[mask_valid][point_ellipsoid_idx]
    points, ellipsoid_id = compute_quadricell_centers(
        scale[..., 0],
        scale[..., 1],
        scale[..., 2],
        target_points=sample_quadricell_targets,
    )

    sampling_cov = model.get_covariance_mat()[mask_valid][point_ellipsoid_idx]
    sampling_rot = model.get_rotation_mat()[mask_valid][point_ellipsoid_idx]
    (
        point_sampling_broadcast,
        rotated_directions,
        point_ellipsoid_idx_broadcast,
    ) = mask_and_compute_rays(
        points,
        ellipsoid_id,
        point_normals,
        point_sampling,
        sampling_cov,
        sampling_rot,
        direction_mode="isocell",
    )

    # the function is the same but its real use is to display the selected viewdirs
    point_sampling_broadcast = point_sampling_broadcast.reshape(
        -1, point_sampling_broadcast.shape[-1]
    )
    point_ellipsoid_idx_broadcast = point_ellipsoid_idx_broadcast.reshape(-1)
    original_id_ellipsoid_idx = torch.arange(
        0,
        all_points.shape[0],
        dtype=torch.long,
        device=point_ellipsoid_idx_broadcast.device,
    )[mask_valid][point_ellipsoid_idx][point_ellipsoid_idx_broadcast]
    rays_dir = rotated_directions.reshape(-1, rotated_directions.shape[-1])

    # torch.save({
    #     'points': point_sampling.reshape(-1, 3).cpu(),
    #     'normals': point_normals.reshape(-1, 3).cpu(),
    # }, 'points_and_normals.pt')

    # breakpoint()
    # from visual import display_point_cloud_with_normals
    # display_point_cloud_with_normals(
    #     point_sampling.reshape(-1, 3).cpu().numpy(),
    #     normals=point_normals.reshape(-1, 3).cpu().numpy(),
    # )

    # rays_ori = point_sampling_broadcast.reshape(-1, point_sampling_broadcast.shape[-1])

    pts_per_chunk = num_viewdirs_per_chunk
    chunks = torch.split(
        torch.arange(
            0,
            point_sampling_broadcast.shape[0],
            device=point_sampling.device,
            dtype=torch.long,
        ),
        pts_per_chunk,
    )

    shs_view = model.get_features.transpose(1, 2).view(
        -1, 3, (model.max_sh_degree + 1) ** 2
    )[original_id_ellipsoid_idx]

    rgbs = []

    for chunk in chunks:
        # chunk_ellipsoid_id = point_ellipsoid_idx_broadcast[chunk]
        chunk_sh_view = shs_view[chunk]
        line_direction = rays_dir[chunk]

        rgb = evaluate_viewdirs_color(
            chunk_sh_view, line_direction, model.active_sh_degree
        )
        rgbs.append(rgb)
    rgbs = torch.cat(rgbs, dim=0)

    # breakpoint()

    # from visual import display_select_vector_direction
    #
    # display_select_vector_direction(
    #     point_sampling_broadcast.cpu().numpy(),
    #     (point_sampling_broadcast + rays_dir * 0.005).cpu().numpy(),
    #     point_sampling.cpu().numpy(),
    # )

    return (
        point_sampling_broadcast,
        rays_dir,
        rgbs,
    )
