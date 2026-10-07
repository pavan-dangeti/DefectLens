"""Frozen ImageNet backbone feature extraction (ADR-0005: both backbones shipped)."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

_LAYER_OUT_IDX = {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}


class FeatureExtractor(nn.Module):
    """Extract multi-scale patch embeddings from a frozen TimmFeatureExtractor-equivalent.

    Mirrors anomalib's TimmFeatureExtractor contract: each selected layer is average-pooled
    with kernel 3 / stride 1 / padding 1, later layers are bilinearly upsampled to the first
    layer's spatial size, and the result is concatenated along channels — the identical
    construction described in ADR-0004 and used by both published papers' interpretations.
    """

    def __init__(
        self,
        name: str,
        layers: tuple[str, ...],
        pretrained: bool = True,
        pool: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.name = name
        self.layers = layers
        self.pretrained = pretrained
        self.model = timm.create_model(name, pretrained=pretrained, features_only=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.pooler = nn.AvgPool2d(kernel_size=3, stride=1, padding=1) if pool else nn.Identity()

    def layer_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            feats = self.model(images)
        out: dict[str, torch.Tensor] = {}
        info = cast(list[dict[str, Any]], self.model.feature_info)
        module_names = [fi["module"] for fi in info]
        for layer in self.layers:
            if layer not in module_names:
                raise ValueError(f"{layer} not in {self.name} feature map modules")
            out[layer] = self.pooler(feats[_LAYER_OUT_IDX[layer]])
        return out

    def embeddings(self, images: torch.Tensor, interp: str = "bilinear") -> torch.Tensor:
        """(B, D, H, W) with all layers aligned to the first (largest) layer's resolution."""
        feats = self.layer_features(images)
        first = feats[self.layers[0]]
        merged = first
        align = False if interp == "bilinear" else None
        for layer in self.layers[1:]:
            upsampled = torch.nn.functional.interpolate(
                feats[layer], size=first.shape[-2:], mode=interp, align_corners=align
            )
            merged = torch.cat([merged, upsampled], dim=1)
        return merged

    @staticmethod
    def flatten(embedding: torch.Tensor) -> torch.Tensor:
        """(B, D, H, W) → (B*H*W, D) patch rows."""
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1])

    @property
    def out_dim(self) -> int:
        info = cast(list[dict[str, Any]], self.model.feature_info)
        return sum(info[_LAYER_OUT_IDX[layer]]["num_chs"] for layer in self.layers)
