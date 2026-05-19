"""modules/vicreg/model.py — VICReg: Variance-Invariance-Covariance Regularization."""
import torch
import torch.nn as nn
from utils.losses import vicreg_loss

class _Expander(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)

class VICReg(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        c = cfg.VICReg
        self.encoder  = backbone
        self.expander = _Expander(backbone.out_dim, c.projection_hidden_dim, c.projection_dim)
        self.sim_coeff = c.sim_coeff
        self.std_coeff = c.std_coeff
        self.cov_coeff = c.cov_coeff

    def forward(self, views, *args):
        z1 = self.expander(self.encoder(views[0]))
        z2 = self.expander(self.encoder(views[1]))
        return {"loss": vicreg_loss(z1, z2, self.sim_coeff, self.std_coeff, self.cov_coeff)}

    @torch.no_grad()
    def get_embedding(self, x): return self.encoder(x)
