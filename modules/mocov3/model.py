"""modules/mocov3/model.py — MoCo v3: Momentum Contrast v3."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import moco_loss

class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(out_dim),
        )
    def forward(self, x): return self.net(x)

class MoCoV3(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        c   = cfg.MoCoV3
        dim = backbone.out_dim

        self.encoder    = backbone
        self.projector  = _MLP(dim, c.projection_hidden_dim, c.projection_dim)
        self.predictor  = _MLP(c.projection_dim, c.prediction_hidden_dim, c.projection_dim)

        self.momentum_encoder    = copy.deepcopy(self.encoder)
        self.momentum_projector  = copy.deepcopy(self.projector)
        for p in list(self.momentum_encoder.parameters()) + list(self.momentum_projector.parameters()):
            p.requires_grad = False

        self.temperature = c.temperature
        self.ema_decay   = c.ema_decay

    @torch.no_grad()
    def update_momentum(self):
        m = self.ema_decay
        for o, t in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            t.data.mul_(m).add_((1 - m) * o.data)
        for o, t in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            t.data.mul_(m).add_((1 - m) * o.data)

    def forward(self, views, *args):
        x1, x2 = views[0], views[1]
        q1 = self.predictor(self.projector(self.encoder(x1)))
        q2 = self.predictor(self.projector(self.encoder(x2)))
        with torch.no_grad():
            k1 = self.momentum_projector(self.momentum_encoder(x1))
            k2 = self.momentum_projector(self.momentum_encoder(x2))
        loss = (moco_loss(q1, k2, self.temperature) + moco_loss(q2, k1, self.temperature)) / 2
        return {"loss": loss}

    @torch.no_grad()
    def get_embedding(self, x): return self.encoder(x)
