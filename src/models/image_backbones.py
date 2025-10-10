from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

try:
    import torchvision.models as tvm
except Exception:  # pragma: no cover
    tvm = None


_BACKBONES = {
    "resnet18": (512, lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)),
    "resnet50": (2048, lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)),
    "efficientnet_b0": (1280, lambda: tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)),
}


def get_backbone_names():
    return list(_BACKBONES.keys())


class ImageClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


def _wrap_backbone(net: torch.nn.Module, feat_dim: int) -> torch.nn.Module:
    # Convert to a feature extractor returning a flat feature vector
    # Handles common torchvision models (resnet, efficientnet)
    if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
        # ResNet style
        net.fc = nn.Identity()
        return net
    if hasattr(net, "classifier") and isinstance(net.classifier, nn.Sequential):
        # EfficientNet style: replace final linear layer with identity
        last_idx = None
        for i in reversed(range(len(net.classifier))):
            if isinstance(net.classifier[i], nn.Linear):
                last_idx = i
                break
        if last_idx is not None:
            net.classifier[last_idx] = nn.Identity()
        return net
    # Fallback: assume model returns features already
    return net


def build_model(name: str = "resnet18", num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    if tvm is None:
        raise RuntimeError("torchvision is required for image backbones")
    name = name.lower()
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone {name}. Available: {list(_BACKBONES)}")
    feat_dim, ctor = _BACKBONES[name]
    net = ctor()
    net = _wrap_backbone(net, feat_dim)
    model = ImageClassifier(net, feat_dim=feat_dim, num_classes=num_classes, dropout=dropout)
    return model
