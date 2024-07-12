from typing import Optional
from pose_estimation.backbone import BackboneWrapper
from pose_estimation.camera_direction_network import CameraDirectionPredictor
from pose_estimation.ray_preprocessor import RayPreprocessor
from pose_estimation.our_multihead_attention import MultiHeadAttention
import torch
from pose_estimation.cam_augmentations import OutputAugmentationTypes, NormalizationReverser, ReversePosEncAugmentation


class IdentificationModule(torch.nn.Module):
    def __init__(self, backbone_type: str = "dino", camera_up_output_augmentation: OutputAugmentationTypes = OutputAugmentationTypes.NONE, target_rays_dirs: Optional[torch.Tensor] = None, augmentation_channels: int = 10):
        super().__init__()

        self.backbone_wrapper = BackboneWrapper(backbone_type=backbone_type)

        self.ray_preprocessor = RayPreprocessor(
            featureC=512, fea_output=self.backbone_wrapper.img_num_features
        )
        ray_fea_size = self.backbone_wrapper.img_num_features
        img_fea_size = self.backbone_wrapper.img_num_features + 14
        # self.attention = PyTorchMultiHeadAttentionWrapper(
        #     ray_fea_size, img_fea_size, img_num_features, 1
        # )

        cam_up_mlp_output_size = 3
        if camera_up_output_augmentation == OutputAugmentationTypes.NORMAL:
            self.camera_up_out_augmentation = NormalizationReverser(target_rays_dirs)
        elif camera_up_output_augmentation == OutputAugmentationTypes.REVERSE_POS_ENC:
            self.camera_up_out_augmentation = ReversePosEncAugmentation(
                augmentation_channels=augmentation_channels
            )
            cam_up_mlp_output_size = (
                cam_up_mlp_output_size + augmentation_channels * cam_up_mlp_output_size
            )
        else:
            self.camera_up_out_augmentation = None
        
        self.camera_direction_prediction_network = CameraDirectionPredictor(
            self.backbone_wrapper.img_num_features,
            self.backbone_wrapper.backbone_wh,
            fea_output=cam_up_mlp_output_size,
        )

        self.attention = MultiHeadAttention(
            ray_fea_size, img_fea_size, self.backbone_wrapper.img_num_features, 1
        )

    def compute_features(
        self,
        img: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        rays_rgb: torch.Tensor,
        rays_to_test: int = -1,
    ):
        features_img_w_pe_flat, _, _ = self.backbone_wrapper(img)

        # features_img_w_pe_flat.register_hook(
        #     lambda grad: print("features_img:", grad.max(), grad.min())
        # )

        #

        # ray random selection
        used_ray_ids = torch.randperm(
            rays_ori.shape[0], device=img.device, dtype=torch.long
        )
        if rays_to_test != -1:
            used_ray_ids = used_ray_ids[:rays_to_test]
        # ray processing
        features_rays = self.ray_preprocessor(
            rays_ori[used_ray_ids], rays_dir[used_ray_ids], rays_rgb[used_ray_ids]
        )

        return features_img_w_pe_flat, features_rays, used_ray_ids

    def run_attention(self, img, mask, rays_ori, rays_dir, rays_rgb):
        features_img_w_pe_flat, features_img_flat, features_img = self.backbone_wrapper(img, mask)
        features_rays = self.ray_preprocessor(rays_ori, rays_dir, rays_rgb)
        attention_map = self.attention(features_img_w_pe_flat, features_rays, mask=None)
        # score = 1.0 - torch.prod(1.0 - attention_map, dim=0)
        score = torch.sum(attention_map, dim=0)

        new_cam_up_dir = self.camera_direction_prediction_network(
            features_img
        )
        if self.camera_up_out_augmentation is not None:
            camera_up_dir = torch.nn.functional.normalize(self.camera_up_out_augmentation(new_cam_up_dir), dim=-1)
        else:
            camera_up_dir = torch.nn.functional.normalize(new_cam_up_dir, dim=-1)

        return score, attention_map, features_img_flat, camera_up_dir

    def forward(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        rays_rgb: torch.Tensor,
        rays_to_test: int = -1,
    ):
        used_ray_ids = torch.randperm(
            rays_ori.shape[0], device=img.device, dtype=torch.long
        )
        if rays_to_test != -1:
            used_ray_ids = used_ray_ids[:rays_to_test]
        scores, attention_map, features_img_w_pe_flat, camera_up_dir = self.run_attention(
            img,
            mask,
            rays_ori[used_ray_ids],
            rays_dir[used_ray_ids],
            rays_rgb[used_ray_ids],
        )
        return scores, attention_map, features_img_w_pe_flat, camera_up_dir, used_ray_ids

    @torch.no_grad()
    def test_image(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        rays_rgb: torch.Tensor,
        rays_to_output: int = 100,
    ):
        scores, attention_map, _, camera_up_dir = self.run_attention(
            img, mask, rays_ori, rays_dir, rays_rgb
        )

        chunk_topk = torch.topk(scores, k=rays_to_output)

        return chunk_topk.indices, chunk_topk.values, scores, camera_up_dir, attention_map
