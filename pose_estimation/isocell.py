import numpy as np
import torch
import math


def isocell_distribution(
    ray_target, dtype, device, N0=1, isrand=-1, int_dtype=torch.int64
):
    # Number of divisions
    n = math.sqrt(ray_target / N0)
    n = int(math.ceil(n))
    Ntot = int(N0 * n**2)

    # cell area
    A0 = np.pi / Ntot

    # distance between circles
    dR = 1 / n

    rings_id = torch.arange(1, n + 1, dtype=int_dtype, device=device)
    nc = N0 * (2 * rings_id - 1)
    R = torch.repeat_interleave(rings_id, nc, dim=0) * dR

    dth = 2 * math.pi / nc.to(dtype=dtype)
    th0 = torch.rand(1, dtype=dtype, device=device) * dth
    if isrand == -1:
        th0 = 0

    cell_ids = torch.arange(0, Ntot, dtype=int_dtype, device=device)
    nc_shift = torch.cat(
        (
            torch.zeros(1, device=nc.device, dtype=nc.dtype),
            torch.cumsum(nc, dim=0)[:-1],
        ),
        dim=0,
    )
    ring_cell_ids = (cell_ids - torch.repeat_interleave(nc_shift, nc, dim=0)).to(
        dtype=dtype
    )

    dth_index = torch.repeat_interleave(
        torch.arange(0, n, dtype=int_dtype, device=nc.device), nc, dim=0
    )
    dth_expanded = dth[dth_index]
    th0 = th0 + ring_cell_ids * dth_expanded

    if isrand == 1:
        R = R - torch.rand(1, Ntot, dtype=dtype, device=device) * dR
        th = th0 + torch.rand(1, Ntot, dtype=dtype, device=device) * dth_expanded
    elif isrand == 2:
        R = R - torch.rand(1, Ntot, dtype=dtype, device=device) * dR
        th = th0 + dth_expanded / 2
    elif isrand == 3:
        rr = (1 + torch.randn(1, Ntot, dtype=dtype, device=device) / 6.5) / 2
        R = R - rr * dR
        rr = (1 + torch.randn(1, Ntot, dtype=dtype, device=device) / 6.5) / 2
        th = th0 + rr * dth_expanded / 2
    elif isrand == 4:
        rr = (1 + torch.randn(1, Ntot, dtype=dtype, device=device) / 6.5) / 2
        R = R - rr * dR
        th = th0 + dth_expanded / 2
    else:
        R = R - dR / 2
        th = th0 + dth_expanded / 2

    Xr = R * torch.cos(th)
    Yr = R * torch.sin(th)

    # set output values
    Zr = torch.real(
        torch.sqrt(
            1
            - torch.square(Xr.to(dtype=torch.complex64))
            - torch.square(Yr.to(dtype=torch.complex64))
        )
    )
    # python does not automatically understand the feature of the numbers so we need to
    # manually adjust the types for the computation to np.float64
    # (e.g. np.sqrt(1 - Xr.astype(np.complex) ** 2 - Yr.astype(np.complex) ** 2).astype(np.float64)
    # or get the real() part of the complex numbers

    points = torch.column_stack([Xr, Yr, Zr])

    return points


def group_by_360_isocell(dirs, ray_target, N0=3, int_dtype=torch.int64):
    # Number of divisions
    n = math.sqrt(ray_target / N0)
    n = int(math.ceil(n))

    # distance between circles
    dR = 1 / n

    rings_id = torch.arange(1, n + 1, dtype=dirs.dtype, device=dirs.device)
    nc = N0 * (2 * rings_id - 1)

    test_rotation_normals = torch.tensor(
        [[0.0, 0.0, 1.0]], dtype=dirs.dtype, device=dirs.device
    )
    # batched version of torch.dot(dirs, test_rotation_normals)
    # geometric meaning scalar projection of the direction vector over the test normals
    normal_projection = torch.bmm(
        dirs.view(dirs.shape[0], 1, dirs.shape[-1]),
        test_rotation_normals.view(
            test_rotation_normals.shape[0], test_rotation_normals.shape[-1], 1
        ).expand(dirs.shape[0], -1, -1),
    )[..., 0, 0]
    isocell_group = normal_projection < 0

    projection_onto_plane = dirs - torch.multiply(
        normal_projection[..., None], test_rotation_normals
    )
    assert torch.allclose(
        projection_onto_plane[..., -1],
        torch.tensor(
            0.0, dtype=projection_onto_plane.dtype, device=projection_onto_plane.device
        ),
    ), "after projection z should be zero"

    R = torch.linalg.norm(projection_onto_plane, dim=-1)
    ring_id = (torch.floor_divide(R, dR)).to(dtype=int_dtype).clamp(min=0, max=N0)

    th_arccos = torch.arccos(projection_onto_plane[..., 0] / R)
    th_arcsin = torch.arcsin(projection_onto_plane[..., 1] / R)

    th = torch.where(th_arcsin >= 0, th_arccos, -th_arccos) + math.pi

    # th = th - th0 # assume th0 to be zero
    dth = (2 * math.pi) / nc[ring_id]
    ring_cell_ids = torch.floor_divide(th, dth).to(dtype=int_dtype)

    return isocell_group.to(dtype=int_dtype), ring_id, ring_cell_ids


def get_dirs_group_idx(dirs, ray_target, N0=3, int_dtype=torch.int64):
    isocell_group, ring_id, ring_cell_ids = group_by_360_isocell(
        dirs, ray_target, N0=N0, int_dtype=int_dtype
    )
    comb_isocell = torch.multiply(isocell_group, ring_id.max() + 1) + ring_id
    comb_isocell_combined = (
        torch.multiply(comb_isocell, ring_cell_ids.max() + 1) + ring_cell_ids
    )
    unique_idx_group = comb_isocell_combined + ring_cell_ids

    dirs_idx = torch.arange(0, dirs.shape[0], dtype=int_dtype, device=dirs.device)
    groups = []
    for i in range(unique_idx_group.max().item()):
        group_idx = dirs_idx[unique_idx_group == i]
        groups.append(group_idx)

    groups = [group for group in groups if group.shape[0] != 0]

    return groups


def batch_vec2ss_matrix(vector):  # vector to skewsym. matrix
    ss_matrix = torch.zeros(
        (*vector.shape[:-1], 3, 3), dtype=vector.dtype, device=vector.device
    )
    ss_matrix[..., 0, 1] = -vector[..., 2]
    ss_matrix[..., 0, 2] = vector[..., 1]
    ss_matrix[..., 1, 0] = vector[..., 2]
    ss_matrix[..., 1, 2] = -vector[..., 0]
    ss_matrix[..., 2, 0] = -vector[..., 1]
    ss_matrix[..., 2, 1] = vector[..., 0]

    return ss_matrix


def rotate_isocell(isocell_directions: torch.Tensor, normal: torch.Tensor):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    isocell_original_shape = isocell_directions.shape[:-1]
    normal_original_shape = normal.shape[:-1]
    isocell_directions = isocell_directions.reshape(-1, 3)
    normal = normal.reshape(-1, 3)

    a = torch.tensor(
        [0.0, 0.0, 1.0],
        dtype=isocell_directions.dtype,
        device=isocell_directions.device,
    )
    a = a[None, None, :].expand(-1, isocell_directions.shape[0], -1)
    b = torch.divide(normal, torch.linalg.norm(normal, dim=-1, keepdim=True))[:, None]
    # make the dimension the same
    combined_size = (b.shape[0], a.shape[1], b.shape[-1])
    b = torch.broadcast_to(b, combined_size).reshape(-1, b.shape[-1])
    a = torch.broadcast_to(a, combined_size).reshape(-1, a.shape[-1])
    v = torch.linalg.cross(a, b, dim=-1)
    # batched version of torch.dot(a, b)
    c = torch.bmm(
        a.view(a.shape[0], 1, a.shape[-1]), b.view(b.shape[0], b.shape[-1], 1)
    )[..., 0, 0]
    s = torch.linalg.norm(v, dim=-1)
    # batched version of kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    kmat = batch_vec2ss_matrix(v)
    # batched version of kmat.dot(kmat)
    dotted_kmat = torch.bmm(kmat, kmat)
    rotation_matrix = (
        torch.eye(3, dtype=isocell_directions.dtype, device=isocell_directions.device)[
            None
        ]
        + kmat
        + torch.multiply(dotted_kmat, ((1 - c) / (s**2))[..., None, None])
    )

    broadcasted_isocells = torch.broadcast_to(
        isocell_directions[None, :], combined_size
    ).reshape(-1, b.shape[-1])
    rotated_dirs = torch.bmm(
        broadcasted_isocells.view(-1, 1, 3), torch.transpose(rotation_matrix, -1, -2)
    )
    rotated_dirs = rotated_dirs.reshape(*combined_size)
    return rotated_dirs.reshape(
        *normal_original_shape,
        *isocell_original_shape,
        3,
    )


if __name__ == "__main__":
    from visual import display_select_vector_direction

    device = torch.device("cpu")
    dtype = torch.float32

    isocell_directions = isocell_distribution(35, dtype, device, N0=3, isrand=-1)

    normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device)

    rotated_isocell_directions = (
        rotate_isocell(isocell_directions, normals).cpu().numpy()
    )

    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    color = np.ones(isocell_directions.shape[0], dtype=np.float32)

    # print(points + isocell_directions)
    print(isocell_directions.shape)
    print(np.broadcast_to(points, isocell_directions.shape).shape)
    display_select_vector_direction(
        np.broadcast_to(points, isocell_directions.shape),
        np.broadcast_to(points, isocell_directions.shape) + isocell_directions,
        color,
        points,
    )
