import torch

class RecallBasedLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()

        if weight is not None:
            weight = torch.tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5

        self.n_classes = 2

    def forward(
        self,
        pred_score: torch.Tensor,
        camera_pose: torch.Tensor,
        camera_intrinsic: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
    ):
        with torch.no_grad():
            projection_matrix = camera_intrinsic @ torch.linalg.inv(camera_pose)[:3, :]

            plane_position = torch.tensor(
                [[0.0], [0.0], [0.0], [1.0]],
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
                (cam_pixels[:, 1] >= 0.0)
                & (cam_pixels[:, 1] <= 800)
                & (cam_pixels[:, 0] >= 0.0)
                & (cam_pixels[:, 0] <= 800)
                & (t > 0)
            )
            target = is_inside.long()

            # from visual import display_select_vector_direction
            # display_select_vector_direction(
            #     rays_ori[is_inside].cpu().numpy(),
            #     (rays_ori[is_inside] + rays_dir[is_inside] * 0.1).cpu().numpy(),
            #     rays_ori.cpu().numpy(),
            #     extrinsic_target_camera=camera_pose.cpu().numpy(),
            # )

        input = torch.stack((1 - pred_score, pred_score), dim=1)

        # input (batch,n_classes)
        # target (batch)
        pred = input.argmax(dim=1)
        idex = (pred != target).view(-1)

        gt_counter = torch.ones((self.n_classes,)).cuda()
        gt_idx, gt_count = torch.unique(target, return_counts=True)

        # map ignored label to an exisiting one
        gt_counter[gt_idx] = gt_count.float()

        # calculate false negative counts
        fn_counter = torch.ones(self.n_classes).cuda()
        fn = target.view(-1)[idex]
        fn_idx, fn_count = torch.unique(fn, return_counts=True)

        # map ignored label to an exisiting one
        fn_counter[fn_idx] = fn_count.float()

        weight = fn_counter / gt_counter

        CE = torch.nn.functional.cross_entropy(input, target, reduction="none")
        loss = weight[target] * CE
        return loss.mean()
