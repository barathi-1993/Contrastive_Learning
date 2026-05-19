"""modules/simclr/model.py — SimCLR with NT-Xent + optional triplet loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import nt_xent_loss, triplet_loss

class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

class SimCLR(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        c  = cfg.SimCLR
        self.encoder   = backbone
        self.projector = _ProjectionHead(
            backbone.out_dim, c.projection_hidden_dim, c.projection_dim
        )
        self.temperature      = c.temperature
        self.use_triplet      = c.get("use_triplet_loss", False)
        self.triplet_weight   = c.get("triplet_weight", 1.0)
        self.triplet_margin   = c.get("triplet_margin", 1.0)

    def _encode_pair(self, x1: torch.Tensor, x2: torch.Tensor):
        """Single encoder+projector pass — avoids DataParallel deadlock from dual forward."""
        if x1.shape[0] == x2.shape[0]:
            z = self.projector(self.encoder(torch.cat([x1, x2], dim=0)))
            return z.chunk(2, dim=0)
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2

    def forward(self, views, *args):
        x1, x2 = views[0], views[1]
        z1, z2 = self._encode_pair(x1, x2)
        loss = nt_xent_loss(z1, z2, self.temperature)
        if self.use_triplet:
            loss = loss + self.triplet_weight * triplet_loss(z1, z2, self.triplet_margin)
        return {"loss": loss, "z1": z1, "z2": z2}

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
