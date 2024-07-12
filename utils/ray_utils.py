import math

import torch
import numpy as np
from kornia.geometry.liegroup import Se3


def cast_rays(ori, dirs, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dirs[..., None, :]


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing="xy",
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions


def get_ray_directions_Ks(H: int, W: int, K: torch.Tensor, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=K.device) + pixel_center,
            torch.arange(H, dtype=torch.float32, device=K.device) + pixel_center,
            indexing="xy",
        ),
        dim=-1,
    )  # WxHx2
    base_grid_dx = torch.clone(base_grid)
    base_grid_dx[..., 0] += 1
    base_grid_dy = torch.clone(base_grid)
    base_grid_dy[..., 1] += 1

    base_grid_stacked = torch.stack((base_grid, base_grid_dx, base_grid_dy))

    coords = torch.cat(
        (base_grid_stacked, torch.ones_like(base_grid_stacked[..., [0]])), -1
    )  # (H, W, 3)

    directions_stacked = (
        (torch.inverse(K) @ coords.permute(3, 0, 1, 2).view(1, K.shape[-1], -1))
        .view(K.shape[0], K.shape[-1], *base_grid_stacked.shape[0:3])
        .permute(0, 2, 3, 4, 1)
    )
    directions = directions_stacked[:, 0]
    dx = directions_stacked[:, 1]
    dy = directions_stacked[:, 2]

    return directions, dx, dy


def get_rays(
    viewdirs,
    c2w,
    keepdim=False,
    directions: torch.Tensor = None,
    dx: torch.Tensor = None,
    dy: torch.Tensor = None,
):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert viewdirs.shape[-1] == 3
    assert (dx is not None) == (dy is not None)

    rays_d = (viewdirs[..., None, :] * c2w[..., :3, :3]).sum(-1)
    if dx is not None:
        dx = (dx[..., None, :] * c2w[..., :3, :3]).sum(-1)
        dy = (dy[..., None, :] * c2w[..., :3, :3]).sum(-1)
    if directions is not None:
        directions = (directions[..., None, :] * c2w[..., :3, :3]).sum(-1)
    else:
        directions = rays_d

    rays_o = c2w[..., :3, 3].unsqueeze(-2).expand(rays_d.shape)

    # if directions.ndim == 2:  # (N_rays, 3)
    #     assert c2w.ndim == 3  # (N_rays, 4, 4) / (1, 4, 4)
    #     rays_d = (directions[None, :, None, :] * c2w[:, None, :3, :3]).sum(
    #         -1
    #     )  # (N_rays, 3)
    #     if dx is not None:
    #         dx = (dx[None, :, None, :] * c2w[:, None, :3, :3]).sum(-1)  # (N_rays, 3)
    #         dy = (dy[None, :, None, :] * c2w[:, None, :3, :3]).sum(-1)  # (N_rays, 3)
    #     rays_o = c2w[:, None, :3, 3].expand(rays_d.shape)
    # elif directions.ndim == 3:  # (H, W, 3)
    #     if c2w.ndim == 2:  # (4, 4)
    #         rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
    #             -1
    #         )  # (H, W, 3)
    #         if dx is not None:
    #             dx = (dx[:, :, None, :] * c2w[None, None, :3, :3]).sum(
    #                 -1
    #             )  # (N_rays, 3)
    #             dy = (dy[:, :, None, :] * c2w[None, None, :3, :3]).sum(
    #                 -1
    #             )  # (N_rays, 3)
    #         rays_o = c2w[None, None, :, 3].expand(rays_d.shape)
    #     elif c2w.ndim == 3:  # (B, 4, 4)
    #         rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #             -1
    #         )  # (B, H, W, 3)
    #         if dx is not None:
    #             dx = (dx[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #                 -1
    #             )  # (N_rays, 3)
    #             dy = (dy[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #                 -1
    #             )  # (N_rays, 3)
    #         rays_o = c2w[:, None, None, :, 3].expand(rays_d.shape)
    #     else:
    #         raise NotImplementedError("the given dimension is not supported")
    # elif directions.ndim == 4 and c2w.ndim == 3:  # (B, H, W, 3), (B, 4, 4)
    #     rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #         -1
    #     )  # (B, H, W, 3)
    #     if dx is not None:
    #         dx = (dx[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #             -1
    #         )  # (N_rays, 3)
    #         dy = (dy[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #             -1
    #         )  # (N_rays, 3)
    #     rays_o = c2w[:, None, None, :, 3].expand(rays_d.shape)
    # else:
    #     raise NotImplementedError("the given dimension is not supported")

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        directions = directions.reshape(-1, 3)
        if dx is not None:
            dx = dx.reshape(-1, 3)
            dy = dy.reshape(-1, 3)

    if dx is not None:
        dx_norm = torch.linalg.norm(dx - directions, axis=-1)
        dy_norm = torch.linalg.norm(dy - directions, axis=-1)
        # Cut the distance in half, multiply it to match the variance of a uniform
        # distribution the size of a pixel (1/12, see the original mipnerf paper).
        radii = (0.5 * (dx_norm + dy_norm)[..., None]) * (2 / math.sqrt(12))

        return rays_o, rays_d, radii
    return rays_o, rays_d


def get_rays_lie(
    viewdirs,
    c2w: Se3,
    keepdim=False,
    directions: torch.Tensor = None,
    dx: torch.Tensor = None,
    dy: torch.Tensor = None,
):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert viewdirs.shape[-1] == 3
    assert (dx is not None) == (dy is not None)

    c2w_rotation_matrix = c2w.rotation.matrix()

    rays_d = (viewdirs[..., None, :] * c2w_rotation_matrix).sum(-1)
    if dx is not None:
        dx = (dx[..., None, :] * c2w_rotation_matrix).sum(-1)
        dy = (dy[..., None, :] * c2w_rotation_matrix).sum(-1)
    if directions is not None:
        directions = (directions[..., None, :] * c2w_rotation_matrix).sum(-1)
    else:
        directions = rays_d

    rays_o = c2w.t.unsqueeze(-2).expand(rays_d.shape)

    # if directions.ndim == 2:  # (N_rays, 3)
    #     assert c2w.ndim == 3  # (N_rays, 4, 4) / (1, 4, 4)
    #     rays_d = (directions[None, :, None, :] * c2w[:, None, :3, :3]).sum(
    #         -1
    #     )  # (N_rays, 3)
    #     if dx is not None:
    #         dx = (dx[None, :, None, :] * c2w[:, None, :3, :3]).sum(-1)  # (N_rays, 3)
    #         dy = (dy[None, :, None, :] * c2w[:, None, :3, :3]).sum(-1)  # (N_rays, 3)
    #     rays_o = c2w[:, None, :3, 3].expand(rays_d.shape)
    # elif directions.ndim == 3:  # (H, W, 3)
    #     if c2w.ndim == 2:  # (4, 4)
    #         rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
    #             -1
    #         )  # (H, W, 3)
    #         if dx is not None:
    #             dx = (dx[:, :, None, :] * c2w[None, None, :3, :3]).sum(
    #                 -1
    #             )  # (N_rays, 3)
    #             dy = (dy[:, :, None, :] * c2w[None, None, :3, :3]).sum(
    #                 -1
    #             )  # (N_rays, 3)
    #         rays_o = c2w[None, None, :, 3].expand(rays_d.shape)
    #     elif c2w.ndim == 3:  # (B, 4, 4)
    #         rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #             -1
    #         )  # (B, H, W, 3)
    #         if dx is not None:
    #             dx = (dx[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #                 -1
    #             )  # (N_rays, 3)
    #             dy = (dy[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #                 -1
    #             )  # (N_rays, 3)
    #         rays_o = c2w[:, None, None, :, 3].expand(rays_d.shape)
    #     else:
    #         raise NotImplementedError("the given dimension is not supported")
    # elif directions.ndim == 4 and c2w.ndim == 3:  # (B, H, W, 3), (B, 4, 4)
    #     rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #         -1
    #     )  # (B, H, W, 3)
    #     if dx is not None:
    #         dx = (dx[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #             -1
    #         )  # (N_rays, 3)
    #         dy = (dy[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
    #             -1
    #         )  # (N_rays, 3)
    #     rays_o = c2w[:, None, None, :, 3].expand(rays_d.shape)
    # else:
    #     raise NotImplementedError("the given dimension is not supported")

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        directions = directions.reshape(-1, 3)
        if dx is not None:
            dx = dx.reshape(-1, 3)
            dy = dy.reshape(-1, 3)

    if dx is not None:
        dx_norm = torch.linalg.norm(dx - directions, axis=-1)
        dy_norm = torch.linalg.norm(dy - directions, axis=-1)
        # Cut the distance in half, multiply it to match the variance of a uniform
        # distribution the size of a pixel (1/12, see the original mipnerf paper).
        radii = (0.5 * (dx_norm + dy_norm)[..., None]) * (2 / math.sqrt(12))

        return rays_o, rays_d, radii
    return rays_o, rays_d


def get_rays_from_parameters(
    H: int,
    W: int,
    K: torch.Tensor,
    c2w: torch.Tensor,
    keepdim=False,
    use_pixel_centers=True,
):
    directions, dx, dy = get_ray_directions_Ks(
        H, W, K, use_pixel_centers=use_pixel_centers
    )
    return get_rays(directions, c2w, keepdim=keepdim, dx=dx, dy=dy)
