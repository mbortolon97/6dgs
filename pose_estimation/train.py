from typing import Optional, Tuple, Callable

import numpy as np
import torch

from pose_estimation.distance_based_loss import DistanceBasedScoreLoss
from torch.utils.tensorboard import SummaryWriter

from pose_estimation.identification_module import IdentificationModule
from pose_estimation.test import test_pose_estimation
from scene.scene_structure import SceneInfo
from utils.graphics_utils import fov2focal
from transformers.optimization import Adafactor


def train_id_module(
    ckpt_path,
    device,
    id_module: IdentificationModule,
    rays_generator: Optional[
        Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ],
    scene_info: SceneInfo,
    sequence_id,
    category_id,
    start_iterations: int = 0,
    renewal_every_n_iterations: int = 10,
    display_every_n_iterations: int = 20,
    val_every_n_iterations: int = 20,
    n_iterations: int = 1500,
    gradient_accumulation_steps: int = 32,
    lock_backbone: bool = True,
):
    id_module.train()
    
    if lock_backbone:
        id_module.backbone_wrapper.eval()
        add_parameters = []
    else:
        add_parameters = list(id_module.backbone_wrapper.parameters())
    
    optimizer = Adafactor(
        list(id_module.ray_preprocessor.parameters()) +
        list(id_module.attention.parameters()) +
        list(id_module.camera_direction_prediction_network.parameters()) +
        add_parameters,
    )
    loss_fn = DistanceBasedScoreLoss()
    running_loss = 0.0

    # Initialize the SummaryWriter for TensorBoard
    # Its output will be written to ./runs/
    writer = SummaryWriter()
    writer.add_text("config/ckpt_path", ckpt_path)
    writer.add_text("config/category_id", category_id)
    writer.add_text("config/sequence_id", sequence_id)

    model_up_np = np.mean(
        np.asarray(
            [train_camera.R[:3, 1] for train_camera in scene_info.train_cameras],
            dtype=np.float32,
        ),
        axis=0,
    )
    model_up = torch.from_numpy(model_up_np).to(device=device, non_blocking=True)

    rays_ori, rays_dirs, rays_rgb = None, None, None
    target_scores = None
    for iteration in range(start_iterations, n_iterations):
        if iteration % renewal_every_n_iterations == 0:
            rays_ori, rays_dirs, rays_rgb = rays_generator()
        optimizer.zero_grad()

        # random data extraction

        img_idx = torch.randint(
            0,
            len(scene_info.train_cameras),
            (gradient_accumulation_steps,),
            dtype=torch.long,
            device=device,
        )
        # img_idx = torch.full(
        #     (gradient_accumulation_steps,),
        #     0,
        #     dtype=torch.long,
        #     device=train_dataset.all_rgbs.device,
        # )

        # annealing_1_alpha = 0.5 * (1 + math.cos((iteration / n_iterations) * math.pi))
        # annealing_2_alpha = 0.5 * (
        #     1 + math.sin(((iteration - (n_iterations / 2)) / n_iterations) * math.pi)
        # )
        # writer.add_scalar(
        #     "train/annealing_1_alpha", annealing_1_alpha, global_step=iteration
        # )
        # writer.add_scalar(
        #     "train/annealing_2_alpha", annealing_2_alpha, global_step=iteration
        # )

        accumulation_loss = 0.0
        accumulation_cam_up = 0.0
        accumulation_scores_loss = 0.0
        for gradient_accumulation_step in range(gradient_accumulation_steps):
            gradient_camera_info = scene_info.train_cameras[
                img_idx[gradient_accumulation_step]
            ]
            tensor_image = torch.from_numpy(np.array(gradient_camera_info.image))
            train_img = tensor_image.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            train_img = train_img / 255.0
            if train_img.shape[-1] == 4:
                train_img_mask = train_img[..., -1] > 0.3
                train_img = torch.multiply(train_img[..., :3], train_img[..., -1:]) + (
                    1 - train_img[..., -1:]
                )
            else:
                train_img_mask = torch.ones_like(
                    train_img[..., -1], dtype=torch.bool, device=device
                )

            w2c = torch.eye(4, dtype=torch.float32, device=device)
            w2c[:3, :3] = torch.transpose(
                torch.from_numpy(gradient_camera_info.R), -1, -2
            ).to(device, non_blocking=True)
            w2c[:3, -1] = torch.from_numpy(gradient_camera_info.T).to(
                device, non_blocking=True
            )
            c2w = torch.inverse(w2c)

            target_camera_pose = c2w
            focalX = fov2focal(gradient_camera_info.FovX, gradient_camera_info.width)
            focalY = fov2focal(gradient_camera_info.FovY, gradient_camera_info.height)
            target_camera_intrinsic = torch.tensor(
                [
                    [focalX, 0.0, gradient_camera_info.width / 2],
                    [0.0, focalY, gradient_camera_info.height / 2],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
                device=device,
            )

            # Make predictions for this batch
            scores, attn_map, img_features, camera_up_dir, rays_idx = id_module(
                train_img, train_img_mask, rays_ori, rays_dirs, rays_rgb
            )
            # (
            #     features_img_w_pe_flat,
            #     features_rays,
            #     rays_idx,
            # ) = id_module.compute_features(train_img, rays_ori, -rays_dirs, rays_rgb)

            loss_score, target_scores = loss_fn(
                scores,
                # features_img_w_pe_flat,
                # features_rays,
                target_camera_pose,
                target_camera_intrinsic,
                rays_ori[rays_idx],
                rays_dirs[rays_idx],
                attn_map.shape[-2],
                id_module.backbone_wrapper.backbone_wh,
                model_up=model_up,
            )

            cam_up_similarity = (
                -0.5 * torch.cosine_similarity(model_up, camera_up_dir, dim=-1)
                + 0.5
            )

            combined_loss_score = loss_score + 0.1 * cam_up_similarity
            # combined_loss_score = loss_score

            if combined_loss_score.isnan().any():
                continue

            # Compute the loss and its gradients
            loss = combined_loss_score / gradient_accumulation_steps
            loss.backward()

            accumulation_loss += combined_loss_score.item()
            accumulation_cam_up += cam_up_similarity.item() / gradient_accumulation_steps
            accumulation_scores_loss += loss_score.item() / gradient_accumulation_steps

        # Adjust learning weights
        optimizer.step()

        writer.add_scalar("train/loss", accumulation_loss, global_step=iteration)
        writer.add_scalar("train/cam_up", accumulation_cam_up, global_step=iteration)
        writer.add_scalar("train/loss_score", accumulation_scores_loss, global_step=iteration)

        # Gather data and report
        running_loss += accumulation_loss
        if iteration % display_every_n_iterations == display_every_n_iterations - 1:
            last_loss = running_loss / display_every_n_iterations  # loss per batch
            print(f"[{iteration}] loss: {last_loss}")
            running_loss = 0.0

            # Log the weights of the network for this epoch
            # for name, param in id_module.named_parameters():
            #     combined_name = "weights/" + name
            #     writer.add_histogram(name, param, global_step=iteration)
            #     if param.grad is not None:
            #         writer.add_histogram(
            #             f"{name}_gradients", param.grad, global_step=iteration
            #         )

            # writer.add_histogram(
            #     f"train/target_score", target_scores, global_step=iteration
            # )

        if iteration % val_every_n_iterations == val_every_n_iterations - 1:
            print("Eval on train...")
            (
                _,
                train_avg_translation_error,
                train_avg_angular_error,
                train_avg_score,
                train_recall,
            ) = test_pose_estimation(
                scene_info.train_cameras,
                id_module,
                rays_ori,
                rays_dirs,
                rays_rgb,
                model_up,
                sequence_id=sequence_id,
                category_id=category_id,
                loss_fn=loss_fn,
                # save=True,
                # save=(195 < iteration < 210),
            )

            # if best_model is None or best_angular_score > train_avg_angular_error:
            #     best_model = (
            #         id_module.state_dict(),
            #         optimizer.state_dict(),
            #         running_loss,
            #     )
            #     best_angular_score = train_avg_angular_error

            writer.add_scalar(
                "train/avg_translation_error",
                train_avg_translation_error,
                global_step=iteration,
            )
            writer.add_scalar(
                "train/avg_angular_error",
                train_avg_angular_error,
                global_step=iteration,
            )
            writer.add_scalar(
                "train/avg_loss_score",
                train_avg_score,
                global_step=iteration,
            )
            writer.add_scalar(
                "train/recall",
                train_recall,
                global_step=iteration,
            )

            print("Eval on validation...")
            (
                _,
                val_avg_translation_error,
                val_avg_angular_error,
                val_avg_score,
                val_recall,
            ) = test_pose_estimation(
                scene_info.test_cameras,
                id_module,
                rays_ori,
                rays_dirs,
                rays_rgb,
                model_up,
                sequence_id=sequence_id,
                category_id=category_id,
                loss_fn=loss_fn,
            )

            writer.add_scalar(
                "val/avg_translation_error",
                val_avg_translation_error,
                global_step=iteration,
            )
            writer.add_scalar(
                "val/avg_angular_error",
                val_avg_angular_error,
                global_step=iteration,
            )
            writer.add_scalar(
                "val/avg_loss_score",
                val_avg_score,
                global_step=iteration,
            )
            writer.add_scalar(
                "val/recall",
                val_recall,
                global_step=iteration,
            )

            id_module.train()
            if lock_backbone:
                id_module.backbone_wrapper.eval()

    torch.save(
        {
            "epoch": n_iterations,
            "model_state_dict": id_module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "running_loss": running_loss,
        },
        ckpt_path,
    )
