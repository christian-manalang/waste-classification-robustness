import os
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torchvision.models as models
import timm

CLASSES: List[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


class ChannelAttention(nn.Module):
    """
    Channel Attention Module of CBAM (Woo et al., 2018).
    Applies shared MLP across average-pooled and max-pooled feature maps.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module of CBAM (Woo et al., 2018).
    Applies 7x7 convolution on concatenated channel-pooled (mean & max) maps.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., ECCV 2018).
    Sequentially combines ChannelAttention and SpatialAttention.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CBAMBottleneck(nn.Module):
    """
    CBAM wrapper around an underlying torchvision Bottleneck block.
    Applies CBAM to out = conv3(out) -> bn3(out) -> cbam(out),
    followed by identity addition and final relu.
    """
    def __init__(self, bottleneck: nn.Module, ratio: int = 16):
        super().__init__()
        self.bottleneck = bottleneck
        c = bottleneck.conv3.out_channels
        self.cbam = CBAM(c, ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.bottleneck.conv1(x)
        out = self.bottleneck.bn1(out)
        out = self.bottleneck.relu(out)

        out = self.bottleneck.conv2(out)
        out = self.bottleneck.bn2(out)
        out = self.bottleneck.relu(out)

        out = self.bottleneck.conv3(out)
        out = self.bottleneck.bn3(out)

        out = self.cbam(out)

        if self.bottleneck.downsample is not None:
            identity = self.bottleneck.downsample(x)

        out += identity
        out = self.bottleneck.relu(out)

        return out


def build_resnet50(num_classes: int = len(CLASSES), pretrained: bool = False) -> nn.Module:
    """Build standard ResNet-50 architecture matching torchvision."""
    if pretrained:
        try:
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception:
            model = models.resnet50(weights=None)
    else:
        model = models.resnet50(weights=None)

    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_cbam_resnet50(num_classes: int = len(CLASSES), pretrained: bool = False) -> nn.Module:
    """Build CBAM-ResNet50 by wrapping each Bottleneck block in layer1-layer4 with CBAMBottleneck."""
    base = models.resnet50(weights=None)
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(base, layer_name)
        wrapped_blocks = nn.Sequential(*[CBAMBottleneck(block) for block in layer])
        setattr(base, layer_name, wrapped_blocks)
    base.fc = nn.Linear(base.fc.in_features, num_classes)
    return base


def build_edgenext_base(num_classes: int = len(CLASSES), pretrained: bool = False) -> nn.Module:
    """Build EdgeNeXt-Base model using timm 'edgenext_base.in21k_ft_in1k'."""
    model = timm.create_model(
        "edgenext_base.in21k_ft_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def load_weights(
    arch: str,
    ckpt_path: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """
    Instantiate model with pretrained=False, load checkpoint state_dict onto device with strict=True,
    and set model to eval mode.
    """
    device = torch.device(device)
    arch_norm = arch.lower().replace("-", "_").replace(" ", "_")

    if "edgenext" in arch_norm:
        model = build_edgenext_base(num_classes=len(CLASSES), pretrained=False)
    elif "cbam" in arch_norm:
        model = build_cbam_resnet50(num_classes=len(CLASSES), pretrained=False)
    elif "resnet" in arch_norm:
        model = build_resnet50(num_classes=len(CLASSES), pretrained=False)
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. Supported options: 'resnet50', 'cbam_resnet50', 'edgenext_base'."
        )

    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file '{ckpt_path}' not found.")

        checkpoint = torch.load(ckpt_path, map_location=device)
        # Unwrap checkpoint if nested
        if isinstance(checkpoint, dict):
            for key in ["state_dict", "model", "model_state_dict", "net"]:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        # Strip 'module.' prefix if saved via DistributedDataParallel
        cleaned_state_dict = {}
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith("module.") else k
            cleaned_state_dict[name] = v

        model.load_state_dict(cleaned_state_dict, strict=True)

    model.to(device)
    model.eval()
    return model
