"""modules/__init__.py — Central model registry."""
from modules.simclr.model      import SimCLR
from modules.byol.model        import BYOL
from modules.dino.model        import DINO
from modules.mocov3.model      import MoCoV3
from modules.barlow_twins.model import BarlowTwins
from modules.vicreg.model      import VICReg
from modules.ibot.model        import iBOT
from modules.swav.model        import SwAV

MODEL_REGISTRY = {
    "simclr":       SimCLR,
    "byol":         BYOL,
    "dino":         DINO,
    "mocov3":       MoCoV3,
    "barlow_twins": BarlowTwins,
    "vicreg":       VICReg,
    "ibot":         iBOT,
    "swav":         SwAV,
}


def get_model(cfg):
    """Instantiate the model specified in cfg.General.model_name."""
    name = str(cfg.General.model_name).lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}"
        )
    from utils.backbones import get_backbone
    backbone = get_backbone(
        cfg.Backbone.name,
        pretrained=cfg.Backbone.get("pretrained", False)
    )
    return MODEL_REGISTRY[name](backbone, cfg)
