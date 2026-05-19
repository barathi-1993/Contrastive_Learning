"""utils/backbones.py — Backbone encoder registry."""
import torch
import torch.nn as nn
from torchvision import models

class _BackboneWrapper(nn.Module):
    """Strips the final FC layer and exposes out_dim."""
    def __init__(self, encoder: nn.Module, out_dim: int):
        super().__init__()
        self.encoder = encoder
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

def get_backbone(name: str, pretrained: bool = False) -> nn.Module:
    """
    Returns a backbone with the classification head removed.

    Supported names:
      resnet18, resnet50, resnet101
      vit_s  (ViT-Small  patch16, via timm)
      vit_b  (ViT-Base   patch16, via timm)
      vit_l  (ViT-Large  patch16, via timm)
      convnext_tiny, convnext_base (torchvision)
      efficientnet_b0, efficientnet_b3 (torchvision)
    """
    name = name.lower()

    # ── ResNets ───────────────────────────────────────────────────────────────
    if name == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        enc = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        return _BackboneWrapper(enc, 512)

    if name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        enc = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        return _BackboneWrapper(enc, 2048)

    if name == "resnet101":
        m = models.resnet101(weights="IMAGENET1K_V2" if pretrained else None)
        enc = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        return _BackboneWrapper(enc, 2048)

    # ── ConvNeXt ──────────────────────────────────────────────────────────────
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Identity()
        return _BackboneWrapper(m, 768)

    if name == "convnext_base":
        m = models.convnext_base(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Identity()
        return _BackboneWrapper(m, 1024)

    # ── EfficientNet ──────────────────────────────────────────────────────────
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Identity()
        return _BackboneWrapper(m, 1280)

    if name == "efficientnet_b3":
        m = models.efficientnet_b3(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Identity()
        return _BackboneWrapper(m, 1536)

    # ── ViT (via timm) ────────────────────────────────────────────────────────
    if name in ("vit_s", "vit_b", "vit_l"):
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for ViT backbones: pip install timm")
        timm_names = {
            "vit_s": "vit_small_patch16_224",
            "vit_b": "vit_base_patch16_224",
            "vit_l": "vit_large_patch16_224",
        }
        out_dims = {"vit_s": 384, "vit_b": 768, "vit_l": 1024}
        m = timm.create_model(timm_names[name], pretrained=pretrained, num_classes=0)
        return _BackboneWrapper(m, out_dims[name])

    raise ValueError(
        f"Unknown backbone '{name}'. Supported: resnet18, resnet50, resnet101, "
        "convnext_tiny, convnext_base, efficientnet_b0, efficientnet_b3, "
        "vit_s, vit_b, vit_l"
    )
