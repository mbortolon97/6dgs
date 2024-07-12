import torch
from pose_estimation.superpoint import SuperPoint
from typing import Sequence
from torchvision import transforms

def create_backbone(
    type="dino",
    pretrained=False,
    filter_size=4,
    pool_only=True,
    _force_nonfinetuned=False,
    **kwargs,
):
    if type == "dino":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        wh = (16, 16)
        num_features = 384
    else:
        model = SuperPoint()
        wh = (28, 28)
        num_features = 256
    return model, wh, num_features

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

class BackboneWrapper(torch.nn.Module):
    def __init__(self, backbone_type: str = "dino") -> None:
        super().__init__()

        assert backbone_type in ["dino", "superpoint"]

        self.image_preprocessing_net, backbone_wh, img_num_features = create_backbone(
            type=backbone_type, pretrained=True
        )
        self.norm_mean = torch.nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
            requires_grad=False,
        )
        self.norm_std = torch.nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32),
            requires_grad=False,
        )

        resize_size = 256
        crop_size = 224
        interpolation = transforms.InterpolationMode.BICUBIC
        transforms_list = [
            transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
            transforms.CenterCrop(crop_size),
            make_normalize_transform(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            ),
        ]
        self.transformations = transforms.Compose(transforms_list)
        self.mask_transformations = transforms.Compose(
            [
                transforms.Resize(
                    resize_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.CenterCrop(crop_size),
                transforms.Resize(
                    backbone_wh[0],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            ]
        )

        self.backbone_wh = backbone_wh
        self.img_num_features = img_num_features
    
    def forward(self, img, mask):
        # image processing
        permuted_img = img[None].permute(0, 3, 1, 2)  # [B, H, W, 3] => [B, 3, H, W]

        norm_img = self.transformations(permuted_img)
        mask_img = self.mask_transformations(mask[None, None] * 1.0)[0, 0] > 0.1

        img_features = self.image_preprocessing_net.forward_features(norm_img)[
            "x_norm_patchtokens"
        ][0]
        img_features = img_features.reshape(
            self.backbone_wh[0], self.backbone_wh[1], self.img_num_features
        )
        img_features_np_like = img_features
        img_features = img_features.permute(2, 0, 1)
        # img_features = self.image_preprocessing_net(norm_img)[
        #     0
        # ]  # [B, C, H // 2, W // 2]

        position_encoding = self.get_img_position_encoding(
            img_features.shape[-2:], 3, dtype=img.dtype, device=img.device
        )
        features_img_w_pe = torch.cat(
            [img_features, position_encoding.permute(2, 0, 1)], dim=0
        )
        # features_img_w_pe = img_features
        features_img_w_pe = features_img_w_pe.permute(1, 2, 0)
        
        return (
            features_img_w_pe[mask_img].view(-1, features_img_w_pe.shape[-1]),
            img_features_np_like[mask_img].view(-1, img_features_np_like.shape[-1]),
            img_features,
        )
    
    @staticmethod
    def get_img_position_encoding(
        img_features_shape, freqs, dtype=torch.float32, device="cpu"
    ):
        meshgrid_elements = []
        for size in img_features_shape:
            meshgrid_elements.append(
                torch.linspace(-1.0, 1.0, steps=size, dtype=dtype, device=device)
            )
        positions = torch.stack(
            torch.meshgrid(*meshgrid_elements, indexing="ij"), dim=-1
        )
        positions = positions.reshape(-1, positions.shape[-1])
        # start computing the positional encoding itself
        freq_bands = (2 ** torch.arange(freqs).float()).to(
            positions.device, non_blocking=True
        )  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1],)
        )  # (..., DF)
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1)
        # back to the original size
        pts = pts.reshape(*img_features_shape, pts.shape[-1])
        return pts