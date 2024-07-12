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
import math

import torch
import numpy as np

from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation_2D,
)
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import (
    strip_symmetric_2D,
    build_scaling_rotation_2D,
    strip_symmetric_2D,
)


def build_covariance2D_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation_2D(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric_2D(actual_covariance)
    return symm


def build_covariance2D_mat_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation_2D(scaling_modifier * scaling, rotation)
    return L @ L.transpose(1, 2)


def is_rotation_matrix(R, threshold=1e-4):
    Rt = R.mT
    shouldBeIdentity = Rt @ R
    I = torch.eye(2, dtype=R.dtype, device=shouldBeIdentity.device)[None]
    n = torch.linalg.norm(I - shouldBeIdentity, dim=(-2, -1))
    return torch.logical_and(n < threshold, torch.linalg.det(R) > 0)


def matrix_to_euler_angle(matrix: torch.Tensor) -> torch.Tensor:
    det_bigger_than_zero = torch.linalg.det(matrix) > 0
    angle_in_radiant = torch.arccos(
        matrix[..., 0, 0]
    ) * det_bigger_than_zero - torch.arccos(matrix[..., 0, 1]) * torch.logical_not(
        det_bigger_than_zero
    )
    return angle_in_radiant


class EllipsisModel(torch.nn.Module):
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance2D_from_scaling_rotation
        self.covariance_mat_activation = build_covariance2D_mat_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.color_activation = torch.sigmoid

    def __init__(self):
        super().__init__()

        self.covariance_mat_activation = None
        self.inverse_opacity_activation = None
        self.opacity_activation = None
        self.covariance_activation = None
        self.scaling_inverse_activation = None
        self.scaling_activation = None
        self.color_activation = None
        self.xy_scheduler_args = None
        self._xy = torch.empty(0)
        self._colors = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._cov2D = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xy_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.model_path = None
        self.setup_functions()

    def capture(self):
        return (
            self._xy,
            self._colors,
            self._scaling,
            self._rotation,
            self._cov2D,
            self._opacity,
            self.max_radii2D,
            self.xy_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xy,
            self._colors,
            self._scaling,
            self._rotation,
            self._cov2D,
            self._opacity,
            self.max_radii2D,
            xy_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xy_gradient_accum = xy_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_rotation_mat(self):
        return build_rotation_2D(self._rotation)

    @property
    def get_xy(self):
        return self._xy

    @property
    def get_colors(self):
        return self.color_activation(self._colors)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def get_covariance_mat(self, scaling_modifier=1):
        return self.covariance_mat_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_components(
        self,
        means_2D: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        theta: torch.Tensor,
        scale: torch.Tensor,
        spatial_lr_scale: float,
    ):
        self.spatial_lr_scale = spatial_lr_scale
        self._xy = nn.Parameter(means_2D.contiguous(), requires_grad=True)
        self._colors = nn.Parameter(colors.contiguous(), requires_grad=True)
        self._opacity = nn.Parameter(opacities.contiguous(), requires_grad=True)
        self._rotation = nn.Parameter(theta.contiguous(), requires_grad=True)
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(scale).contiguous(), requires_grad=True
        )

    def create_from_memory(
        self,
        means_2D: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        cov_2D: torch.Tensor,
        spatial_lr_scale: float,
    ):
        self.spatial_lr_scale = spatial_lr_scale
        self._xy = nn.Parameter(means_2D.contiguous(), requires_grad=True)
        self._colors = nn.Parameter(colors.contiguous(), requires_grad=True)
        self._opacity = nn.Parameter(opacities.contiguous(), requires_grad=True)
        # decompose covariance matrix

        L = torch.linalg.cholesky(cov_2D).mT
        L[..., 1, 0] = L[..., 0, 1]
        # compute the scaling and rotation matrix
        R_mat, scale, _ = torch.linalg.svd(L)
        det_bigger_than_zero = torch.linalg.det(R_mat) > 0
        scale = (
            scale[..., [0, 1]] * det_bigger_than_zero[..., None]
            + scale[..., [1, 0]] * torch.logical_not(det_bigger_than_zero)[..., None]
        )
        print(
            "Valid rotation matrix: ",
            (torch.count_nonzero(is_rotation_matrix(R_mat)) / R_mat.shape[0]),
        )
        self._rotation = nn.Parameter(
            matrix_to_euler_angle(R_mat).contiguous(), requires_grad=True
        )
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(scale).contiguous(), requires_grad=True
        )
        # self._rotation = torch.empty(0)

        self._precomputed_cov2D = False
        # self._cov2D = nn.Parameter(cov_2D.contiguous(), requires_grad=True)

        # from visual import plot_ellipse
        # import matplotlib.pyplot as plt
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # # Plot data on the first subplot (ax1)
        # plot_ellipse(
        #     # self._scaling[0, 1].item(),
        #     # self._scaling[0, 0].item(),
        #     # self._rotation[0].item(),
        #     cov=self.get_covariance_mat().detach()[0].cpu().numpy(),
        #     x_cent=self._xy[0, 0].item(),
        #     y_cent=self._xy[0, 1].item(),
        #     ax=ax1,
        # )
        # plot_ellipse(
        #     cov=cov_2D[0].cpu().numpy(),
        #     x_cent=self._xy[0, 0].item(),
        #     y_cent=self._xy[0, 1].item(),
        #     ax=ax2,
        # )
        # # Adjust the spacing between subplots
        # plt.tight_layout()
        # # Display the plots
        # plt.show()

        self.max_radii2D = torch.zeros((self.get_xy.shape[0]), device="cuda")

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.sqrt(dist2)[..., None].repeat(1, 2)
        rots = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xy = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._colors = nn.Parameter(
            colors.transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(scales).contiguous().requires_grad_(True)
        )
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xy.shape[0]), device="cuda")

    def create_from_random(self, num_ellipsis: int):
        self.spatial_lr_scale = 1.41
        fused_point_cloud = (
            torch.rand((num_ellipsis, 2), dtype=torch.float32, device="cuda") - 0.5
        ) * 2.0
        colors = torch.rand((num_ellipsis, 3), dtype=torch.float32, device="cuda")

        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud),
            0.0000001,
        )
        scales = torch.sqrt(dist2)[..., None].repeat(1, 2)
        rots = torch.zeros((fused_point_cloud.shape[0],), device="cuda")

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xy = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._colors = nn.Parameter(colors.contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(scales).requires_grad_(True)
        )
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xy.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xy_gradient_accum = torch.zeros((self.get_xy.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xy.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xy],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xy",
            },
            {
                "params": [self._colors],
                "lr": training_args.feature_lr,
                "name": "colors",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xy_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xy":
                lr = self.xy_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "nx", "ny", "red", "green", "blue"]
        l.append("opacity")
        for i in range(self._scaling.shape[-1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[-1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xy = self._xy.detach().cpu().numpy()
        normals = np.zeros_like(xy)
        colors_raw = (
            (self.get_colors.detach() * 255.0)
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xy.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xy, normals, colors_raw, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xy = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        colors = np.zeros((xy.shape[0], 3, 1))
        colors[:, 0, 0] = np.asarray(plydata.elements[0]["red"]) / 255.0
        colors[:, 1, 0] = np.asarray(plydata.elements[0]["green"]) / 255.0
        colors[:, 2, 0] = np.asarray(plydata.elements[0]["blue"]) / 255.0

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xy.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xy.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xy = nn.Parameter(
            torch.tensor(xy, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._colors = nn.Parameter(
            inverse_sigmoid(torch.tensor(colors, dtype=torch.float, device="cuda"))
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask]), requires_grad=True
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask], requires_grad=True
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xy = optimizable_tensors["xy"]
        self._colors = optimizable_tensors["colors"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xy_gradient_accum = self.xy_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                exp_avg = stored_state["exp_avg"]
                new_exp_avg = exp_avg[tensors_dict["index"]]
                stored_state["exp_avg"] = torch.cat((exp_avg, new_exp_avg), dim=0)
                exp_avg_sq = stored_state["exp_avg_sq"]
                new_exp_avg_sq = exp_avg_sq[tensors_dict["index"]]
                stored_state["exp_avg_sq"] = torch.cat(
                    (exp_avg_sq, new_exp_avg_sq),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0),
                    requires_grad=True,
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xy,
        new_colors,
        new_opacities,
        new_scaling,
        new_rotation,
        new_index_in_original,
    ):
        d = {
            "xy": new_xy,
            "colors": new_colors,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "index": new_index_in_original,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xy = optimizable_tensors["xy"]
        self._colors = optimizable_tensors["colors"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xy_gradient_accum = torch.zeros((self.get_xy.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xy.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xy.shape[0]), device="cuda")

    def densify_and_split(self, grads: torch.Tensor, grad_threshold: float, N: int = 2):
        n_init_points = self.get_xy.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(n_init_points, dtype=grads.dtype, device=grads.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * 1.41,
        )

        stds = self.get_scaling[selected_pts_mask][..., None, :].expand(-1, N, -1)
        means = torch.zeros((*stds.shape[:-1], 2), dtype=stds.dtype, device=stds.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation_2D(self._rotation[selected_pts_mask])[
            ..., None, :, :
        ].expand(-1, N, -1, -1)
        new_xy = (rots @ samples.unsqueeze(-1)).squeeze(-1) + self.get_xy[
            selected_pts_mask
        ][..., None, :].expand(-1, N, -1)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # points = self.get_xy[selected_pts_mask][..., None, :].expand(-1, N, -1)
        # # Scatter plot for red points
        # ax.scatter(
        #     points.reshape(-1, 2).detach().cpu().numpy()[:, 0],
        #     points.reshape(-1, 2).detach().cpu().numpy()[:, 1],
        #     color="red",
        #     label="Original points",
        # )
        # # Scatter plot for blue points
        # ax.scatter(
        #     new_xy.reshape(-1, 2).detach().cpu().numpy()[:, 0],
        #     new_xy.reshape(-1, 2).detach().cpu().numpy()[:, 1],
        #     color="blue",
        #     label="New points",
        # )
        # for point, new_xy_point in zip(
        #     points.detach().reshape(-1, self._xy.shape[-1]).cpu().numpy(),
        #     new_xy.detach().reshape(-1, self._xy.shape[-1]).cpu().numpy(),
        # ):
        #     ax.annotate(
        #         "",
        #         xy=(new_xy_point[0], new_xy_point[1]),
        #         xytext=(point[0], point[1]),
        #         arrowprops=dict(arrowstyle="->", color="green"),
        #         zorder=5,
        #     )
        # ax.legend()
        # # Display the plot
        # plt.show()

        new_scaling = self.scaling_inverse_activation(
            (
                (self.get_scaling[selected_pts_mask])[..., None, :].expand(-1, N, -1)
                / (0.8 * N)
            )
        )
        new_rotation = (self._rotation[selected_pts_mask])[..., None].expand(-1, N)
        new_colors = (self._colors[selected_pts_mask])[..., None, :].expand(-1, N, -1)
        new_opacity = (self._opacity[selected_pts_mask])[..., None, :].expand(-1, N, -1)
        new_index = (
            torch.arange(
                0,
                selected_pts_mask.shape[0],
                dtype=torch.long,
                device=selected_pts_mask.device,
            )[selected_pts_mask]
        )[..., None].expand(-1, N)

        self.densification_postfix(
            new_xy.reshape(-1, self._xy.shape[-1]),
            new_colors.reshape(-1, self._colors.shape[-1]),
            new_opacity.reshape(-1, self._opacity.shape[-1]),
            new_scaling.reshape(-1, self._scaling.shape[-1]),
            new_rotation.reshape(-1),
            new_index.reshape(-1),
        )

        # breakpoint()
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(),
                    device=selected_pts_mask.device,
                    dtype=selected_pts_mask.dtype,
                ),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(
        self,
        grads,
        grad_threshold,
        remove_background: bool = False,
        white_background: bool = True,
    ):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * 1.41,
        )

        # remove ellipsis with the same color as the background
        if remove_background:
            bg_color = [1, 1, 1] if white_background else [0, 0, 0]
            background = torch.tensor(
                bg_color, dtype=self._colors.dtype, device=self._colors.device
            )
            self.get_colors
            self._colors

        new_xy = self._xy[selected_pts_mask]
        new_colors = self._colors[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_index = torch.arange(
            0,
            selected_pts_mask.shape[0],
            dtype=torch.long,
            device=selected_pts_mask.device,
        )[selected_pts_mask]

        self.densification_postfix(
            new_xy,
            new_colors,
            new_opacities,
            new_scaling,
            new_rotation,
            new_index,
        )

    def densify_and_prune(self, max_grad, min_opacity, max_screen_size):
        grads = self.xy_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad)
        self.densify_and_split(grads, max_grad)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * 1.41
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xy_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
