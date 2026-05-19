"""modules/byol/model.py — BYOL: Bootstrap Your Own Latent."""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import byol_loss

class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)

class BYOL(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        c   = cfg.BYOL
        dim = backbone.out_dim

        self.online_encoder   = backbone
        self.online_projector = _MLP(dim, c.projection_hidden_dim, c.projection_dim)
        self.online_predictor = _MLP(c.projection_dim, c.prediction_hidden_dim, c.projection_dim)

        # Target network — EMA copy, no gradients
        self.target_encoder   = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            p.requires_grad = False

        self.ema_decay     = c.ema_decay
        self.ema_decay_max = c.get("ema_decay_max", 1.0)
        self._step         = 0
        self._total_steps  = 1  # set by trainer before training

    def _ema(self, tau: float):
        for o, t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            t.data.mul_(tau).add_((1 - tau) * o.data)
        for o, t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            t.data.mul_(tau).add_((1 - tau) * o.data)

    @torch.no_grad()
    def update_target(self):
        # Cosine schedule for EMA decay
        tau = self.ema_decay_max - (self.ema_decay_max - self.ema_decay) * (
            math.cos(math.pi * self._step / self._total_steps) + 1
        ) / 2
        self._ema(tau)
        self._step += 1

    def forward(self, views, *args):
        x1, x2 = views[0], views[1]
        # Online
        op1 = self.online_predictor(self.online_projector(self.online_encoder(x1)))
        op2 = self.online_predictor(self.online_projector(self.online_encoder(x2)))
        # Target (no grad)
        with torch.no_grad():
            tp1 = self.target_projector(self.target_encoder(x1))
            tp2 = self.target_projector(self.target_encoder(x2))
        loss = (byol_loss(op1, tp2) + byol_loss(op2, tp1)) / 2
        return {"loss": loss}

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x)
