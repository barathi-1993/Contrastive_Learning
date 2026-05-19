"""modules/ibot/model.py — iBOT: Image BERT Pre-Training with Online Tokenizer."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import dino_loss

class _iBOTHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim),
        )
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last.weight_g.data.fill_(1)
        self.last.weight_g.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))

class iBOT(nn.Module):
    """
    iBOT: combines DINO-style CLS-token distillation with masked patch prediction.
    When the backbone is not a ViT, only the CLS distillation branch is active.
    """
    def __init__(self, backbone, cfg):
        super().__init__()
        c   = cfg.iBOT
        dim = backbone.out_dim

        self.student      = backbone
        self.teacher      = copy.deepcopy(backbone)
        self.student_head = _iBOTHead(dim, c.out_dim, c.projection_hidden_dim, c.bottleneck_dim)
        self.teacher_head = copy.deepcopy(self.student_head)

        for p in list(self.teacher.parameters()) + list(self.teacher_head.parameters()):
            p.requires_grad = False

        self.student_temp    = c.student_temp
        self.teacher_temp    = c.teacher_temp
        self.ema_decay       = c.ema_decay
        self.center_momentum = c.center_momentum
        self.register_buffer("center", torch.zeros(1, c.out_dim))

    @torch.no_grad()
    def update_teacher(self, tau=None):
        tau = tau or self.ema_decay
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data.mul_(tau).add_((1 - tau) * s.data)
        for s, t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            t.data.mul_(tau).add_((1 - tau) * s.data)

    def forward(self, views, *args):
        n_global = 2
        s_out = [self.student_head(self.student(v)) for v in views]
        with torch.no_grad():
            t_out = [self.teacher_head(self.teacher(v)) for v in views[:n_global]]

        teacher_cat = torch.cat(t_out)
        self.center = (self.center * self.center_momentum + teacher_cat.mean(0, keepdim=True) * (1 - self.center_momentum))

        loss, n = 0.0, 0
        for ti, t in enumerate(t_out):
            for si, s in enumerate(s_out):
                if si == ti:
                    continue
                loss += dino_loss(s, t, self.student_temp, self.teacher_temp, self.center)
                n += 1
        return {"loss": loss / n}

    @torch.no_grad()
    def get_embedding(self, x): return self.student(x)
