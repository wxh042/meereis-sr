import math

import torch
import torch.nn as nn


class ChannelAttentionV2(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttentionV2(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x


class CBAMV2(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.channel_attention = ChannelAttentionV2(in_channels)
        self.spatial_attention = SpatialAttentionV2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))


class ResidualBlockV2(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAMV2(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        res = self.cbam(res)
        return x + res


class ChannelAttentionV4(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ca(x) * x


class SpatialAttentionV4(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sa(torch.cat([avg_out, max_out], dim=1)) * x


class CBAMV4(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sa(torch.cat([avg_out, max_out], dim=1)) * x


class ResidualBlockV4(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            CBAMV4(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class UpsampleBlockV2(nn.Module):
    def __init__(self, in_channels: int, up_scale: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (up_scale ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class UpsampleBlockV4(nn.Module):
    def __init__(self, in_channels: int, up_scale: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (up_scale ** 2), 3, 1, 1),
            nn.PixelShuffle(up_scale),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GeneratorBase(nn.Module):
    residual_block_cls: type[nn.Module]
    upsample_block_cls: type[nn.Module]

    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 64, 9, 1, 4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[self.residual_block_cls(64) for _ in range(6)])
        self.block7 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.upsample = nn.Sequential(*[self.upsample_block_cls(64, 2) for _ in range(int(math.log2(scale_factor)))])
        self.block8 = nn.Conv2d(64, 1, 9, 1, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.block1(x)
        res = self.res_blocks(b1)
        b7 = self.block7(res)
        out = self.block8(self.upsample(b1 + b7))
        return (torch.tanh(out) + 1) / 2


class GeneratorV2(_GeneratorBase):
    residual_block_cls = ResidualBlockV2
    upsample_block_cls = UpsampleBlockV2


class GeneratorV4(_GeneratorBase):
    residual_block_cls = ResidualBlockV4
    upsample_block_cls = UpsampleBlockV4


Generator = GeneratorV2
