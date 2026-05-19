"""modules/barlow_twins/model.py — Barlow Twins."""
import torch
import torch.nn as nn
from utils.losses import barlow_twins_loss

class _Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * 2 + [out_dim]
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1], bias=False)]
            if i < len(dims) - 2:
                layers += [nn.BatchNorm1d(dims[i+1]), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class BarlowTwins(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        c = cfg.BarlowTwins
        self.encoder   = backbone
        self.projector = _Projector(backbone.out_dim, c.projection_hidden_dim, c.projection_dim)
        self.lambda_   = c.lambda_coeff

    def forward(self, views, *args):
        z1 = self.projector(self.encoder(views[0]))
        z2 = self.projector(self.encoder(views[1]))
        return {"loss": barlow_twins_loss(z1, z2, self.lambda_)}

    @torch.no_grad()
    def get_embedding(self, x): return self.encoder(x)
