from enum import Enum
import torch
import torch.nn as nn

class NormalizationReverser(nn.Module):
    def __init__(self, targets: torch.Tensor):
        super().__init__()
        std, mean = torch.std_mean(targets.view(-1, targets.shape[-1]), dim=0)
        self.register_buffer("mean", mean)
        self.register_buffer("std", mean)

    def forward(self, x: torch.Tensor):
        return x * self.std + self.mean

class ReversePosEncAugmentation(nn.Module):
    def __init__(self, augmentation_channels: int = 10):
        super().__init__()

        freq_bands = 2 ** torch.arange(augmentation_channels, dtype=torch.float32)

        self.channel_fraction = 1 / augmentation_channels
        self.register_buffer("channel_coefficients", freq_bands)
        self.augmentation_channels = augmentation_channels

    # def forward(self, x: torch.Tensor):
    #     x_view = x.view(*x.shape[:-1], -1, self.augmentation_channels + 1)
    #     results = (
    #         self.channel_fraction
    #         * torch.sum(torch.arcsin(x_view[..., 1:]) / self.channel_coefficients, dim=-1)
    #         + x_view[..., 0]
    #     )
    #     return results

    # SECOND VERSION OF THE FORWARD
    def forward(self, x: torch.Tensor):
        x_view = x.view(*x.shape[:-1], -1, self.augmentation_channels + 1)
        results = self.channel_fraction * torch.sum(
            (
                torch.arcsin(x_view[..., 1:]) / self.channel_coefficients
                + x_view[..., 0, None]
            ),
            dim=-1,
        )
        return results


class OutputAugmentationTypes(Enum):
    NONE = 1
    NORMAL = 2
    REVERSE_POS_ENC = 3
