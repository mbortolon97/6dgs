#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input

    Args:
        lr_delay_mult: decay multiplier
        lr_delay_steps: number of iteration before starting decay
        lr_final: final learning rate
        lr_init: initial learning rate
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def strip_lowerdiag_2D(L):
    uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty


def strip_symmetric_2D(sym):
    return strip_lowerdiag_2D(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), dtype=r.dtype, device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_rotation_2D(phi):
    if phi.dim() < 1:
        phi = phi.unsqueeze(dim=0)

    s = torch.sin(phi)
    c = torch.cos(phi)

    R = torch.stack(
        (
            torch.stack((c, -s), dim=-1),
            torch.stack((s, c), dim=-1),
        ),
        dim=-2,
    )
    # R[:, 0, 0] = c
    # R[:, 0, 1] = -s
    # R[:, 1, 0] = s
    # R[:, 1, 1] = c

    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_A_mat(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = 1.0 / s[:, 0]
    L[:, 1, 1] = 1.0 / s[:, 1]
    L[:, 2, 2] = 1.0 / s[:, 2]

    L = R @ L @ R.mT
    return L


def cov2D_to_scale_theta(cov2D, K):
    D, R = torch.linalg.eig(cov2D)
    D = torch.real(D)
    R = torch.real(R)
    positive_det = torch.linalg.det(R) > 0
    a_ = torch.sqrt(torch.abs(D[..., 0])) * positive_det + torch.sqrt(
        torch.abs(D[..., 1])
    ) * torch.logical_not(positive_det)
    b_ = torch.sqrt(torch.abs(D[..., 1])) * positive_det + torch.sqrt(
        torch.abs(D[..., 0])
    ) * torch.logical_not(positive_det)
    theta = torch.acos(R[..., 0, 0]) * positive_det + torch.acos(
        R[..., 0, 1]
    ) * torch.logical_not(positive_det)
    K_inv = torch.inverse(K)
    point1 = torch.tensor([0, 0, 1], dtype=a_.dtype, device=a_.device)[None, ..., None]
    point2 = torch.stack(
        (a_, b_, torch.ones((a_.shape[0],), dtype=a_.dtype, device=a_.device)), dim=-1
    )[..., None]
    tmp1 = K_inv @ point1
    tmp2 = K_inv @ point2
    a = tmp2[..., 0, 0] - tmp1[..., 0, 0]
    b = tmp2[..., 1, 0] - tmp1[..., 1, 0]
    return a, b, theta


def build_scaling_rotation_2D(s, r):
    L = torch.zeros((s.shape[0], 2, 2), dtype=s.dtype, device=s.device)
    R = build_rotation_2D(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
