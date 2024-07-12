import torch

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()

        if weight is not None:
            weight = torch.tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5

        self.n_classes = 2

        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)

    def forward(
        self,
        img_features: torch.Tensor,
        ray_features: torch.Tensor,
        camera_pose: torch.Tensor,
        camera_intrinsic: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        temperature: float = 0.1,
    ):
        with torch.no_grad():
            projection_matrix = camera_intrinsic @ torch.linalg.inv(camera_pose)[:3, :]

            plane_position = torch.tensor(
                [[0.0], [0.0], [1.0e-6], [1.0]],
                dtype=projection_matrix.dtype,
                device=projection_matrix.device,
            )
            world_plane_position = (camera_pose[:3] @ plane_position)[None, :, 0]
            plane_normal = torch.tensor(
                [[0.0], [0.0], [1.0], [1.0]],
                dtype=projection_matrix.dtype,
                device=projection_matrix.device,
            )
            world_plane_normal = (camera_pose[:3] @ plane_normal)[None, :, 0]

            plane_position = torch.tensor(
                [[0.0], [0.0], [0.0], [1.0]],
                dtype=projection_matrix.dtype,
                device=projection_matrix.device,
            )

            # Compute the dot product of the plane normal and ray direction
            # batch version of np.dot(plane_normal, ray_direction)

            denom = torch.bmm(
                world_plane_normal.reshape(
                    world_plane_normal.shape[0], 1, world_plane_normal.shape[-1]
                ).expand(rays_dir.shape[0], -1, -1),
                rays_dir.view(rays_dir.shape[0], rays_dir.shape[-1], 1),
            )[..., 0, 0]

            ray_diff = world_plane_position - rays_ori
            # Compute the ray-plane intersection parameter (t)
            dot_result = torch.bmm(
                world_plane_normal.reshape(
                    world_plane_normal.shape[0], 1, world_plane_normal.shape[-1]
                ).expand(ray_diff.shape[0], -1, -1),
                ray_diff.view(ray_diff.shape[0], ray_diff.shape[-1], 1),
            )[..., 0, 0]
            t = torch.divide(dot_result, denom)

            meeting_point = rays_ori + rays_dir * t[..., None]

            point_project = (
                projection_matrix
                @ torch.cat(
                    (
                        meeting_point,
                        torch.ones(
                            meeting_point.shape[0],
                            1,
                            dtype=meeting_point.dtype,
                            device=meeting_point.device,
                        ),
                    ),
                    dim=-1,
                ).mT
            ).mT

            point_project = torch.divide(
                point_project[..., :2], point_project[..., [-1]]
            )

            # Check if the intersection point is behind the ray origin (no intersection)

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

            is_inside = (
                (point_project[:, 1] >= 0.0)
                & (point_project[:, 1] <= 800)
                & (point_project[:, 0] >= 0.0)
                & (point_project[:, 0] <= 800)
                & (t > 0)
            )

            # from visual import display_select_vector_direction
            # display_select_vector_direction(
            #     rays_ori[is_inside].cpu().numpy(),
            #     (rays_ori[is_inside] + rays_dir[is_inside] * 1.0).cpu().numpy(),
            #     rays_ori.cpu().numpy(),
            #     extrinsic_target_camera=camera_pose.cpu().numpy(),
            # )

        norm_img_features = torch.nn.functional.normalize(img_features, dim=-1)
        norm_ray_features = torch.nn.functional.normalize(ray_features, dim=-1)

        sim = torch.exp(
            torch.einsum("if, jf -> ij", norm_img_features, norm_ray_features)
            / temperature
        )

        # positives
        pos_mask = is_inside[None, :].expand(norm_img_features.shape[0], -1)
        # negatives
        neg_mask = ~pos_mask

        pos = torch.sum(sim * pos_mask, 1)
        neg = torch.sum(sim * neg_mask, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss