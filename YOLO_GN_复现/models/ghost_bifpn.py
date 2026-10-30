"""Custom feature fusion blocks used by the YOLO-GN reproduction."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

class BiFPNP4(nn.Module):
    """Fuse P3, P4 and P5 features at P4 resolution with learned weights."""

    def __init__(self, in_channels: Sequence[int], out_channels: int, epsilon: float = 1e-4):
        super().__init__()
        # Import lazily to avoid an import cycle while tasks.py registers this
        # class in Ultralytics' YAML parser.
        from ultralytics.nn.modules.conv import GhostConv

        if len(in_channels) != 3:
            raise ValueError(f"BiFPNP4 expects [P3, P4, P5] channel sizes, got {in_channels}")
        self.projections = nn.ModuleList(
            GhostConv(channels, out_channels, k=1, s=1) for channels in in_channels
        )
        self.fusion_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.epsilon = epsilon
        self.refine = GhostConv(out_channels, out_channels, k=3, s=1)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != 3:
            raise ValueError(f"BiFPNP4 expects [P3, P4, P5] tensors, got {len(features)}")
        target_size = features[1].shape[-2:]
        aligned = []
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(projected, size=target_size, mode="nearest")
            aligned.append(projected)

        weights = F.relu(self.fusion_weights)
        weights = weights / (weights.sum() + self.epsilon)
        fused = sum(weight * feature for weight, feature in zip(weights, aligned))
        return self.refine(fused)
