import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Lightweight, dependency-free Grad-CAM (Gradient-weighted Class Activation Mapping)
    implementation using PyTorch forward and full backward hooks.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = None
        self.backward_handle = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and full backward hooks on the target layer."""
        self.remove_hooks()

        def forward_hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            # Save activation tensors in the forward hook (out.detach())
            self.activations = out.detach()

        def backward_hook(
            module: nn.Module,
            grad_in: Tuple[torch.Tensor, ...],
            grad_out: Tuple[torch.Tensor, ...],
        ) -> None:
            # Save gradient tensors in the full backward hook (grad_out[0].detach())
            if grad_out and grad_out[0] is not None:
                self.gradients = grad_out[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def generate_cam(
        self,
        tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a 224x224 normalized Grad-CAM activation map for the target class.

        Args:
            tensor: Input image tensor of shape (1, 3, H, W) or (3, H, W).
            target_class_idx: Target class index. If None, top predicted class is used.

        Returns:
            np.ndarray: 2D float array of shape (224, 224) with values in [0.0, 1.0].
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Ensure hooks are registered for this generation pass
        if self.forward_handle is None or self.backward_handle is None:
            self._register_hooks()

        try:
            self.model.zero_grad(set_to_none=True)
            self.activations = None
            self.gradients = None

            with torch.enable_grad():
                output = self.model(tensor)
                if target_class_idx is None:
                    target_class_idx = int(output.argmax(dim=1).item())

                self.model.zero_grad(set_to_none=True)
                # Run a backward pass for the target class logit
                score = output[:, int(target_class_idx)]
                score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    "Grad-CAM failed to capture activations or gradients. "
                    "Ensure target_layer participates in the forward/backward computation graph."
                )

            # Compute global average pooled gradients across spatial dimensions as channel importance weights
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

            # Calculate the weighted combination of forward activation maps
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

            # Apply ReLU (torch.clamp(cam, min=0))
            cam = torch.clamp(cam, min=0.0)

            # Normalize to [0.0, 1.0]
            cam_min = cam.min()
            cam_max = cam.max()
            if (cam_max - cam_min) > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            # Resize to 224x224
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            cam_np = cam.squeeze().cpu().numpy()
            cam_np = np.clip(cam_np, 0.0, 1.0)
            return cam_np

        finally:
            # Remove hooks upon completion to prevent memory leaks
            self.remove_hooks()
            self.model.zero_grad(set_to_none=True)


def overlay_cam_on_image(
    rgb_pil: Image.Image,
    cam_map: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay Grad-CAM heatmap onto the original RGB image.

    Args:
        rgb_pil: Original PIL Image (in RGB).
        cam_map: 2D numpy array of shape (224, 224) with values in [0.0, 1.0].
        alpha: Heatmap blend weight between 0.0 (only image) and 1.0 (only heatmap).

    Returns:
        PIL.Image.Image: Blended RGB image.
    """
    if rgb_pil.mode != "RGB":
        rgb_pil = rgb_pil.convert("RGB")

    # Resize original image to match CAM map spatial dimensions (224x224)
    h, w = cam_map.shape[:2]
    img_resized = rgb_pil.resize((w, h), Image.Resampling.BILINEAR)
    original = np.array(img_resized, dtype=np.float32)

    # Apply OpenCV's JET colormap
    cam_uint8 = np.uint8(np.clip(cam_map, 0.0, 1.0) * 255.0)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Blend with original resized RGB image using alpha * heatmap + (1 - alpha) * original
    blended = alpha * heatmap_rgb + (1.0 - alpha) * original
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended)


def get_target_layer(model: nn.Module, arch_key: str) -> nn.Module:
    """
    Resolve the target feature extraction layer for Grad-CAM visualization based on architecture.

    Requirements:
    - For 'resnet50': return model.layer4[-1]
    - For 'cbam_resnet50': return model.layer4[-1].bottleneck.conv3 (or the last block in model.layer4[-1])
    - For 'edgenext_base': return model.stages[-1] (or the final feature extraction stage before global pooling)
    """
    key = arch_key.lower().replace("-", "_").replace(" ", "_")
    if "cbam" in key:
        if hasattr(model, "layer4"):
            last_block = model.layer4[-1]
            if hasattr(last_block, "bottleneck") and hasattr(last_block.bottleneck, "conv3"):
                return last_block.bottleneck.conv3
            return last_block
        raise ValueError(f"Could not resolve target layer for CBAM model: {arch_key}")
    elif "resnet" in key:
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        raise ValueError(f"Could not resolve target layer for ResNet model: {arch_key}")
    elif "edgenext" in key:
        if hasattr(model, "stages"):
            return model.stages[-1]
        raise ValueError(f"Could not resolve target layer for EdgeNeXt model: {arch_key}")
    else:
        raise ValueError(
            f"Unknown architecture key '{arch_key}'. Supported: 'resnet50', 'cbam_resnet50', 'edgenext_base'."
        )
