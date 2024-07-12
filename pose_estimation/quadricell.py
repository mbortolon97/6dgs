from typing import Union, Tuple, List, Optional
import torch
import math

from torch.nn.functional import normalize


def coalesce(
    edge_index: torch.Tensor, value: Optional[torch.Tensor], reduce: str = "max"
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor],
]:
    edge_index = edge_index.contiguous()

    edge_subtraction = torch.min(edge_index, dim=0, keepdim=True).values

    edge_index_subtract = edge_index - edge_subtraction

    idx_cumprod = torch.cumprod(
        torch.flip(
            torch.max(edge_index_subtract, dim=0, keepdim=True).values + 1.0, dims=(-1,)
        ),
        dim=-1,
    )
    idx_multipliers = torch.cat(
        (
            torch.flip(idx_cumprod, dims=(-1,))[:, 1:],
            torch.tensor([[1.0]], dtype=idx_cumprod.dtype, device=idx_cumprod.device),
        ),
        dim=-1,
    )
    idx = torch.sum(
        torch.multiply(idx_multipliers, edge_index_subtract), dim=-1, keepdim=True
    )

    # consider maximum value permulation
    if reduce == "min":
        value = torch.max(value, dim=0, keepdim=True).values - value

    idx_with_value = (
        torch.multiply(idx, torch.max(value, dim=0, keepdim=True).values + 1.0)
        + value[..., None]
    )

    if (idx_with_value[1:] < idx_with_value[:-1]).any():
        perm = idx_with_value.argsort(dim=0).ravel()
        edge_index = edge_index[perm]
        if value is not None:
            value = value[perm]
        idx_with_value = idx_with_value[perm]
        idx = idx[perm]
    else:
        perm = torch.arange(0, idx.size(dim=0), 1, device=edge_index.device)

    if reduce == "max" or reduce == "min":
        mask = idx[1:] > idx[:-1]
        mask = mask.flatten()
        mask = torch.concat(
            (mask, torch.tensor([True], dtype=torch.bool, device=mask.device))
        )
    else:
        raise ValueError(
            "currently supported reduce operations: ['min','max'], given instead ",
            reduce,
        )

    if mask.all():  # Skip if indices are already coalesced.
        return edge_index, value, mask

    edge_index = edge_index[mask]

    if value is not None:
        value = value[mask]

    permutation_inverse = torch.zeros_like(mask, dtype=torch.long)
    permutation_inverse[perm] = torch.arange(
        0, mask.size(dim=0), 1, dtype=torch.long, device=edge_index.device
    )

    return edge_index, value, mask[permutation_inverse]


# for the formula of the ellipse and proof of the approximation check eq. 1.2 at https://arxiv.org/pdf/math/0506384.pdf
def ellipse_perimeter(b, c):
    p = math.pi * (
        (b + c)
        + (
            (3 * torch.square(b - c))
            / (
                10 * (b + c)
                + torch.sqrt(torch.square(b) + 14 * b * c + torch.square(c))
            )
        )
    )
    return p


def scale_minor_axis(a, axis, ring_pos, total_rings):
    delta_ring = (2 * a) / total_rings
    x = 0.5 * delta_ring + delta_ring * ring_pos
    return torch.sqrt(
        (1 - torch.square(x - a) / torch.square(a))[..., None] * torch.square(axis)
    )


def combined_ellipses_perimeters(a, b, c, total_rings, square_approximation_side):
    pos_tot = torch.sum(total_rings)
    cell_pos = torch.arange(0, pos_tot, dtype=a.dtype, device=a.device)
    # ellipsoid_id = torch.repeat_interleave(torch.arange(0, total_rings.shape[0], dtype=a.dtype, device=a.device), total_rings)
    cum_sum_cell = torch.repeat_interleave(
        torch.cumsum(
            torch.cat(
                (
                    torch.zeros(1, dtype=total_rings.dtype, device=total_rings.device),
                    total_rings[..., :-1],
                ),
                dim=-1,
            ),
            dim=-1,
        ),
        total_rings,
    )
    ellipse_ring_id = cell_pos - cum_sum_cell

    a_broadcast = torch.repeat_interleave(a, total_rings)
    b_broadcast = torch.repeat_interleave(b, total_rings)
    c_broadcast = torch.repeat_interleave(c, total_rings)

    minor_axis_scaled = scale_minor_axis(
        a_broadcast,
        torch.stack((b_broadcast, c_broadcast), dim=-1),
        ellipse_ring_id,
        torch.repeat_interleave(total_rings, total_rings),
    )
    ellipses_perimeter = ellipse_perimeter(
        minor_axis_scaled[..., 0], minor_axis_scaled[..., 1]
    )
    # print(minor_axis_scaled.shape)
    # print(ellipses_perimeter.shape)
    # print(target_points.shape)
    # avg_point_dists = torch.sum(ellipses_perimeter, dim=-1) / target_points
    # print(avg_point_dists.shape)
    points_per_ellipse = torch.floor(
        ellipses_perimeter
        / torch.repeat_interleave(square_approximation_side, total_rings)
    )

    ellipsoid_id = torch.repeat_interleave(
        torch.arange(0, a.shape[0], dtype=torch.long, device=a.device), total_rings
    )

    return (
        ellipsoid_id,
        ellipse_ring_id,
        points_per_ellipse,
        ellipses_perimeter,
        minor_axis_scaled,
    )


def ellipsoid_surface(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    return (4 * math.pi) * torch.pow(
        (torch.pow(a * b, 1.6075) + torch.pow(a * c, 1.6075) + torch.pow(b * c, 1.6075))
        / 3,
        1 / 1.6075,
    )


def mask_degraded_ellipsoids(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    target_points: int = 50,
):
    target_cell_surface = ellipsoid_surface(a, b, c) / float(target_points)
    square_approximation_side = torch.sqrt(target_cell_surface)

    total_rings_b = torch.floor(
        ellipse_perimeter(a, b) / (2 * square_approximation_side)
    )
    total_rings_c = torch.floor(
        ellipse_perimeter(a, c) / (2 * square_approximation_side)
    )
    total_rings = ((total_rings_b + total_rings_c) * 0.5).to(dtype=torch.long)

    return total_rings < target_points


def compute_quadricell_centers(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    target_points: int = 50,
    ellipse_interpolation_resolution: int = 1000,
):
    target_cell_surface = ellipsoid_surface(a, b, c) / float(target_points)
    square_approximation_side = torch.sqrt(target_cell_surface)

    total_rings_b = torch.floor(
        ellipse_perimeter(a, b) / (2 * square_approximation_side)
    )
    total_rings_c = torch.floor(
        ellipse_perimeter(a, c) / (2 * square_approximation_side)
    )
    total_rings = ((total_rings_b + total_rings_c) * 0.5).to(dtype=torch.long)

    (
        ellipsoid_id,
        ellipse_ring_id,
        points_per_ellipse,
        _,
        minor_axis_scaled,
    ) = combined_ellipses_perimeters(a, b, c, total_rings, square_approximation_side)

    delta_theta = 2 * math.pi / points_per_ellipse
    points_per_ellipse_long = points_per_ellipse.to(dtype=torch.long)
    points_per_ellipse_flatten = points_per_ellipse_long.view(-1)

    # broadcast delta theta
    delta_theta_broadcast = torch.repeat_interleave(
        delta_theta.view(-1), points_per_ellipse_flatten, dim=0
    )

    # compute the id of the ring inside the ellipse
    Ntot = torch.sum(points_per_ellipse_flatten)
    cell_ids = torch.arange(0, Ntot, dtype=torch.long, device=a.device)
    ellipse_ring_id_flatten = torch.repeat_interleave(
        ellipse_ring_id.view(-1), points_per_ellipse_flatten, dim=0
    )

    # print(ellipse_ring_id_flatten)

    cell_id_inside_ellipse = cell_ids - torch.repeat_interleave(
        torch.cat(
            (
                torch.zeros(
                    (1,), dtype=points_per_ellipse_flatten.dtype, device=a.device
                ),
                torch.cumsum(points_per_ellipse_flatten, dim=-1)[:-1],
            ),
            dim=-1,
        ),
        points_per_ellipse_flatten,
        dim=0,
    )
    cell_theta = cell_id_inside_ellipse * delta_theta_broadcast

    minor_axis_expanded = torch.repeat_interleave(
        minor_axis_scaled.view(-1, 2), points_per_ellipse_flatten, dim=0
    )

    # v2 (based on lookup)
    theta_idx = torch.arange(
        0,
        ellipse_interpolation_resolution - 1,
        dtype=delta_theta.dtype,
        device=delta_theta.device,
    )
    theta = theta_idx[None, :] * delta_theta[:, None]
    delta_s_val = torch.sqrt(
        minor_axis_scaled[:, [0]] * torch.square(torch.sin(theta))
        + minor_axis_scaled[:, [1]] * torch.square(torch.cos(theta))
    )
    integ_delta_s_val = delta_s_val * delta_theta[:, None]
    integ_delta_s = torch.cumsum(
        torch.cat(
            (
                torch.zeros(
                    (integ_delta_s_val.shape[0], 1),
                    dtype=delta_theta.dtype,
                    device=delta_theta.device,
                ),
                integ_delta_s_val,
            ),
            dim=-1,
        ),
        dim=-1,
    )
    integ_delta_s_norm = 2.0 * math.pi * (integ_delta_s / integ_delta_s[..., [-1]])

    integ_delta_s_norm_broadcast = torch.repeat_interleave(
        integ_delta_s_norm, points_per_ellipse_flatten, dim=0
    )
    edge_idx = torch.nonzero(integ_delta_s_norm_broadcast[:, 1:] < cell_theta[:, None])

    coalesce_edge_index, coalesce_value, _ = coalesce(
        edge_idx[..., [0]], edge_idx[..., 1], reduce="max"
    )
    pickup_location = torch.zeros(
        integ_delta_s_norm_broadcast.shape[0],
        dtype=torch.long,
        device=integ_delta_s_norm.device,
    )
    pickup_location[coalesce_edge_index[..., 0]] = coalesce_value
    theta_prime = torch.gather(
        integ_delta_s_norm_broadcast, 1, pickup_location[:, None]
    )[..., 0]

    # broadcast a
    a_broadcast = torch.repeat_interleave(
        torch.repeat_interleave(a, total_rings), points_per_ellipse_flatten, dim=0
    )
    total_rings_broadcast = torch.repeat_interleave(
        torch.repeat_interleave(total_rings, total_rings),
        points_per_ellipse_flatten,
        dim=0,
    )
    delta_ring = (2 * a_broadcast) / total_rings_broadcast
    z = 0.5 * delta_ring + delta_ring * ellipse_ring_id_flatten - a_broadcast

    x = minor_axis_expanded[:, 0] * torch.cos(theta_prime)
    y = minor_axis_expanded[:, 1] * torch.sin(theta_prime)
    # z = -a_broadcast + ellipse_ring_id_flatten * delta_ring

    points = torch.stack((x, y, z), dim=-1)

    return points, torch.repeat_interleave(ellipsoid_id, points_per_ellipse_flatten)


def rotate_quadricell(
    points: torch.Tensor,
    ellipsoid_id: torch.Tensor,
    rot_mat: torch.Tensor,
):
    rot_mat_per_point = rot_mat[ellipsoid_id]

    return (rot_mat_per_point @ points[..., :, None])[..., 0]


def mask_quadricell(
    rotated_points: torch.Tensor,
    ellipsoid_id: torch.Tensor,
    ellipsoid_normal: torch.Tensor,
):
    projection = (
        ellipsoid_normal[ellipsoid_id][..., :, None] @ rotated_points[..., None, :]
    )[..., 0, 0]
    mask = projection > 0
    return rotated_points[mask], ellipsoid_id[mask], mask


def compute_quadratic(pts, covariance):
    return (pts[..., None, :] @ covariance @ pts[..., :, None])[..., 0, 0]


def mask_and_compute_rays(
    unrotated_points: torch.Tensor,
    ellipsoid_id: torch.Tensor,
    ellipsoid_normal: torch.Tensor,
    ellipsoid_center: torch.Tensor,
    ellipsoid_covariance: torch.Tensor,
    ellipsoid_rot_mat: torch.Tensor,
    direction_mode: str = "isocell",
):
    rotated_points = rotate_quadricell(
        unrotated_points,
        ellipsoid_id,
        ellipsoid_rot_mat,
    )

    masked_points, masked_ellipsoid_idx, mask = mask_quadricell(
        rotated_points,
        ellipsoid_id,
        ellipsoid_normal,
    )

    if direction_mode == "surface":
        gradient = torch.func.vmap(torch.func.grad(compute_quadratic), in_dims=(0, 0))(
            masked_points, ellipsoid_covariance[masked_ellipsoid_idx]
        )
        ray_direction = normalize(gradient, dim=-1)
    else:
        ray_direction = unrotated_points[mask]
    rotated_ray_dir = (
        ellipsoid_rot_mat[masked_ellipsoid_idx] @ ray_direction[..., :, None]
    )[..., 0]
    ray_dir = normalize(rotated_ray_dir, dim=-1)
    # if direction_mode == "surface":
    #     ray_ori = masked_points + ellipsoid_center[masked_ellipsoid_idx]
    # else:
    #     ray_ori = ellipsoid_center[masked_ellipsoid_idx]
    ray_ori = masked_points + ellipsoid_center[masked_ellipsoid_idx]

    return ray_ori, ray_dir, masked_ellipsoid_idx
