# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple
import torch.nn.functional as F

import torch


def _create_rotation_matrices(rotation_2d) -> torch.Tensor:
    """
    Compute rotations for later use in U V computation

    Args:
        rotation_2d: a π/2 rotation matrix.

    Returns:
        a (3, 3, 3) tensor containing 3 rotation matrices around each of the coordinate axes
        by π/2
    """

    rotations_3d = torch.zeros(
        (3, 3, 3), dtype=rotation_2d.dtype, device=rotation_2d.device
    )
    rotation_axes = set(range(3))
    for rotation_axis in rotation_axes:
        rest = list(rotation_axes - {rotation_axis})
        rotations_3d[rotation_axis][rest[0], rest] = rotation_2d[0]
        rotations_3d[rotation_axis][rest[1], rest] = rotation_2d[1]

    return rotations_3d


def _construct_eigenvecs_set(
    inputs: torch.Tensor, eigenvals: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Construct an orthonormal set of eigenvectors by given inputs and pre-computed eigenvalues

    Args:
        inputs: tensor of symmetric matrices of shape (..., 3, 3)
        eigenvals: tensor of pre-computed eigenvalues of of shape (..., 3, 3)

    Returns:
        Tuple of three eigenvector tensors of shape (..., 3, 3), composing an orthonormal
         set
    """
    eigenvecs_tuple_for_01 = _construct_eigenvecs(
        inputs, eigenvals[..., 0], eigenvals[..., 1], eps
    )
    eigenvecs_for_01 = torch.stack(eigenvecs_tuple_for_01, dim=-1)

    eigenvecs_tuple_for_21 = _construct_eigenvecs(
        inputs, eigenvals[..., 2], eigenvals[..., 1], eps
    )
    eigenvecs_for_21 = torch.stack(eigenvecs_tuple_for_21[::-1], dim=-1)

    # The result will be smooth here even if both parts of comparison
    # are close, because eigenvecs_01 and eigenvecs_21 would be mostly equal as well
    eigenvecs_cond = (
        eigenvals[..., 1] - eigenvals[..., 0] > eigenvals[..., 2] - eigenvals[..., 1]
    ).detach()
    eigenvecs = torch.where(
        eigenvecs_cond[..., None, None], eigenvecs_for_01, eigenvecs_for_21
    )

    return eigenvecs


def _construct_eigenvecs(
    inputs: torch.Tensor, alpha0: torch.Tensor, alpha1: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct an orthonormal set of eigenvectors by given pair of eigenvalues.

    Args:
        inputs: tensor of symmetric matrices of shape (..., 3, 3)
        alpha0: first eigenvalues of shape (..., 3)
        alpha1: second eigenvalues of shape (..., 3)

    Returns:
        Tuple of three eigenvector tensors of shape (..., 3, 3), composing an orthonormal
         set
    """

    # Find the eigenvector corresponding to alpha0, its eigenvalue is distinct
    ev0 = _get_ev0(
        inputs
        - alpha0[..., None, None]
        * torch.eye(3, dtype=inputs.dtype, device=inputs.device),
        eps,
    )
    u, v = _get_uv(ev0)
    ev1 = _get_ev1(
        inputs
        - alpha1[..., None, None]
        * torch.eye(3, dtype=inputs.dtype, device=inputs.device),
        u,
        v,
        eps,
    )
    # The Third eigenvector is computed as the cross-product of the other two
    ev2 = torch.cross(ev0, ev1, dim=-1)

    return ev0, ev1, ev2


def _get_ev0(char_poly: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Construct the first normalized eigenvector given a characteristic polynomial

    Args:
        char_poly: a characteristic polynomials of the input matrices of shape (..., 3, 3)

    Returns:
        Tensor of first eigenvectors of shape (..., 3)
    """

    r01 = torch.cross(char_poly[..., 0, :], char_poly[..., 1, :], dim=-1)
    r12 = torch.cross(char_poly[..., 1, :], char_poly[..., 2, :], dim=-1)
    r02 = torch.cross(char_poly[..., 0, :], char_poly[..., 2, :], dim=-1)

    cross_products = torch.stack((r01, r12, r02), dim=-2)
    # Regularize it with + or -eps depending on the sign of the first vector
    cross_products += eps * _sign_without_zero(cross_products[..., :1, :])

    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    norms_sq = (cross_products**2).sum(dim=-1)
    max_norms_index = norms_sq.argmax(dim=-1)

    # Pick only the cross-product with the highest squared norm for each input
    max_cross_products = _gather_by_index(
        cross_products, max_norms_index[..., None, None], -2
    )
    # Pick corresponding squared norms for each cross-product
    max_norms_sq = _gather_by_index(norms_sq, max_norms_index[..., None], -1)

    # Normalize cross-product vectors by thier norms
    return max_cross_products / torch.sqrt(max_norms_sq[..., None])


def _gather_by_index(
    source: torch.Tensor, index: torch.Tensor, dim: int
) -> torch.Tensor:
    """
    Selects elements from the given source tensor by provided index tensor.
    Number of dimensions should be the same for source and index tensors.

    Args:
        source: input tensor to gather from
        index: index tensor with indices to gather from source
        dim: dimension to gather across

    Returns:
        Tensor of shape same as the source with exception of specified dimension.
    """

    index_shape = list(source.shape)
    index_shape[dim] = 1

    return source.gather(dim, index.expand(index_shape)).squeeze(dim)


def _get_uv(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes unit-length vectors U and V such that {U, V, W} is a right-handed
    orthonormal set.

    Args:
        w: eigenvector tensor of shape (..., 3)

    Returns:
        Tuple of U and V unit-length vector tensors of shape (..., 3)
    """

    min_idx = w.abs().argmin(dim=-1)
    _rotations_3d = _create_rotation_matrices(
        torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=w.dtype, device=w.device)
    )
    rotation_2d = _rotations_3d[min_idx].to(w)

    u = F.normalize((rotation_2d @ w[..., None])[..., 0], dim=-1)
    v = torch.cross(w, u, dim=-1)
    return u, v


def _get_ev1(
    char_poly: torch.Tensor, u: torch.Tensor, v: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Computes the second normalized eigenvector given a characteristic polynomial
    and U and V vectors

    Args:
        char_poly: a characteristic polynomials of the input matrices of shape (..., 3, 3)
        u: unit-length vectors from _get_uv method
        v: unit-length vectors from _get_uv method

    Returns:
        desc
    """

    j = torch.stack((u, v), dim=-1)
    m = j.transpose(-1, -2) @ char_poly @ j

    # If angle between those vectors is acute, take their sum = m[..., 0, :] + m[..., 1, :],
    # otherwise take the difference = m[..., 0, :] - m[..., 1, :]
    # m is in theory of rank 1 (or 0), so it snaps only when one of the rows is close to 0
    is_acute_sign = _sign_without_zero(
        (m[..., 0, :] * m[..., 1, :]).sum(dim=-1)
    ).detach()

    rowspace = m[..., 0, :] + is_acute_sign[..., None] * m[..., 1, :]
    # rowspace will be near zero for second-order eigenvalues
    # this regularization guarantees abs(rowspace[0]) >= eps in a smooth'ish way
    rowspace += eps * _sign_without_zero(rowspace[..., :1])

    return (
        j
        @ F.normalize(
            rowspace
            @ torch.tensor(
                [[0.0, -1.0], [1.0, 0.0]], dtype=rowspace.dtype, device=rowspace.device
            ),
            dim=-1,
        )[..., None]
    )[..., 0]


def _sign_without_zero(tensor):
    """
    Args:
        tensor: an arbitrary shaped tensor

    Returns:
        Tensor of the same shape as an input, but with 1.0 if tensor > 0.0 and -1.0
         otherwise
    """
    return 2.0 * (tensor > 0.0).to(tensor.dtype) - 1.0


def sym_eig_3x3(
    inputs: torch.Tensor, eigenvectors: bool = True, eps: Optional[float] = None
):
    """
    Compute eigenvalues and (optionally) eigenvectors

    Args:
        inputs: symmetric matrices with shape of (..., 3, 3)
        eigenvectors: whether we should compute only eigenvalues or eigenvectors as well
        eps: the lowest number to avoid nan

    Returns:
        Either a tuple of (eigenvalues, eigenvectors) or eigenvalues only, depending on
         given params. Eigenvalues are of shape (..., 3) and eigenvectors (..., 3, 3)
    """
    eps = eps or torch.finfo(torch.float).eps

    if inputs.shape[-2:] != (3, 3):
        raise ValueError("Only inputs of shape (..., 3, 3) are supported.")

    inputs_diag = inputs.diagonal(dim1=-2, dim2=-1)
    inputs_trace = inputs_diag.sum(-1)
    q = inputs_trace / 3.0

    # Calculate squared sum of elements outside the main diagonal / 2
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    p1 = ((inputs**2).sum(dim=(-1, -2)) - (inputs_diag**2).sum(-1)) / 2
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    p2 = ((inputs_diag - q[..., None]) ** 2).sum(dim=-1) + 2.0 * p1.clamp(eps)

    p = torch.sqrt(p2 / 6.0)
    B = (
        inputs
        - q[..., None, None] * torch.eye(3, dtype=inputs.dtype, device=inputs.device)
    ) / p[..., None, None]

    r = torch.det(B) / 2.0
    # Keep r within (-1.0, 1.0) boundaries with a margin to prevent exploding gradients.
    r = r.clamp(-1.0 + eps, 1.0 - eps)

    phi = torch.acos(r) / 3.0
    eig1 = q + 2 * p * torch.cos(phi)
    eig2 = q + 2 * p * torch.cos(phi + 2 * math.pi / 3)
    eig3 = 3 * q - eig1 - eig2
    # eigenvals[..., i] is the i-th eigenvalue of the input, α0 ≤ α1 ≤ α2.
    eigenvals = torch.stack((eig2, eig3, eig1), dim=-1)

    # Soft dispatch between the degenerate case (diagonal A) and general.
    # diag_soft_cond -> 1.0 when p1 < 6 * eps and diag_soft_cond -> 0.0 otherwise.
    # We use 6 * eps to take into account the error accumulated during the p1 summation
    diag_soft_cond = torch.exp(-((p1 / (6 * eps)) ** 2)).detach()[..., None]

    # Eigenvalues are the ordered elements of main diagonal in the degenerate case
    diag_eigenvals, _ = torch.sort(inputs_diag, dim=-1)
    eigenvals = diag_soft_cond * diag_eigenvals + (1.0 - diag_soft_cond) * eigenvals

    if eigenvectors:
        eigenvecs = _construct_eigenvecs_set(inputs, eigenvals, eps)
    else:
        eigenvecs = None

    return eigenvals, eigenvecs
