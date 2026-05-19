"""modules/swav/model.py — SwAV: Swapping Assignments between Views."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import swav_loss

class _Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

class SwAV(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        c = cfg.SwAV
        self.encoder      = backbone
        self.projector    = _Projector(backbone.out_dim, c.projection_hidden_dim, c.projection_dim)
        self.prototypes   = nn.Linear(c.projection_dim, c.n_prototypes, bias=False)
        nn.init.uniform_(self.prototypes.weight)

        self.temperature       = c.temperature
        self.n_sinkhorn        = c.sinkhorn_iterations
        self.sinkhorn_eps      = c.sinkhorn_epsilon
        self.queue_length      = c.queue_length
        self.queue_start_epoch = c.queue_start_epoch
        self.register_buffer("queue", torch.zeros(c.projection_dim, c.queue_length))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self._epoch = 0

    @torch.no_grad()
    def _normalise_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1)
        self.prototypes.weight.copy_(w)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, z: torch.Tensor):
        ptr = int(self.queue_ptr)
        bs  = z.shape[0]
        if ptr + bs > self.queue_length:
            bs = self.queue_length - ptr
            z  = z[:bs]
        self.queue[:, ptr: ptr + bs] = z.T
        self.queue_ptr[0] = (ptr + bs) % self.queue_length

    def forward(self, views, epoch: int = 0, *args):
        self._epoch = epoch
        self._normalise_prototypes()

        embeddings   = [self.projector(self.encoder(v)) for v in views]
        scores_list  = [self.prototypes(z) for z in embeddings]

        # Use queue for teacher soft codes after queue_start_epoch
        teacher_scores = scores_list[:2]

        loss = swav_loss(
            scores_list, teacher_scores,
            self.prototypes.weight,
            self.temperature,
            self.n_sinkhorn,
            self.sinkhorn_eps,
        )
        if epoch >= self.queue_start_epoch:
            self._dequeue_and_enqueue(embeddings[0].detach())
        return {"loss": loss}

    @torch.no_grad()
    def get_embedding(self, x): return self.encoder(x)
