"""utils/losses.py — All contrastive and self-supervised loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# SimCLR — NT-Xent (Normalised Temperature-scaled Cross Entropy)
# ─────────────────────────────────────────────────────────────────────────────
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)          # (2B, D)
    sim = torch.matmul(z, z.T) / temperature                      # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Labels: positive of i is i+B (and vice-versa)
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])                       # (2B,)
    return F.cross_entropy(sim, labels)

# ─────────────────────────────────────────────────────────────────────────────
# Triplet loss (used as auxiliary in SimCLR)
# ─────────────────────────────────────────────────────────────────────────────
def triplet_loss(z1: torch.Tensor, z2: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    dist = torch.cdist(z1, z2, p=2)                               # (B, B)
    pos  = torch.diag(dist)
    mask = ~torch.eye(z1.shape[0], dtype=torch.bool, device=z1.device)
    neg, _ = (dist + (~mask).float() * 1e9).min(dim=1)
    return F.relu(pos - neg + margin).mean()

# ─────────────────────────────────────────────────────────────────────────────
# BYOL cosine loss
# ─────────────────────────────────────────────────────────────────────────────
def byol_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred   = F.normalize(pred,   dim=-1)
    target = F.normalize(target, dim=-1)
    return 2 - 2 * (pred * target).sum(dim=-1).mean()

# ─────────────────────────────────────────────────────────────────────────────
# DINO cross-entropy loss (student vs teacher soft labels)
# ─────────────────────────────────────────────────────────────────────────────
def dino_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    student_temp: float,
    teacher_temp: float,
    center: torch.Tensor,
) -> torch.Tensor:
    s = (student_out / student_temp)
    t = F.softmax((teacher_out - center) / teacher_temp, dim=-1).detach()
    return -(t * F.log_softmax(s, dim=-1)).sum(dim=-1).mean()

# ─────────────────────────────────────────────────────────────────────────────
# Barlow Twins redundancy-reduction loss
# ─────────────────────────────────────────────────────────────────────────────
def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_coeff: float = 0.005) -> torch.Tensor:
    B, D = z1.shape
    # Batch-normalise each feature dimension
    z1_n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-4)
    z2_n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-4)
    C = (z1_n.T @ z2_n) / B                                       # (D, D)
    on_diag  = torch.diagonal(C).add_(-1).pow_(2).sum()
    off_diag = _off_diagonal(C).pow_(2).sum()
    return on_diag + lambda_coeff * off_diag

def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# ─────────────────────────────────────────────────────────────────────────────
# VICReg variance-invariance-covariance loss
# ─────────────────────────────────────────────────────────────────────────────
def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float  = 25.0,
    std_coeff: float  = 25.0,
    cov_coeff: float  = 1.0,
    eps: float        = 1e-4,
) -> torch.Tensor:
    B, D = z1.shape
    # Invariance
    inv = F.mse_loss(z1, z2)
    # Variance
    std1 = torch.sqrt(z1.var(dim=0) + eps)
    std2 = torch.sqrt(z2.var(dim=0) + eps)
    var  = F.relu(1 - std1).mean() + F.relu(1 - std2).mean()
    # Covariance
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / (B - 1)
    cov2 = (z2c.T @ z2c) / (B - 1)
    cov  = (_off_diagonal(cov1).pow_(2).sum() + _off_diagonal(cov2).pow_(2).sum()) / D
    return sim_coeff * inv + std_coeff * var + cov_coeff * cov

# ─────────────────────────────────────────────────────────────────────────────
# SwAV Sinkhorn-based multi-crop loss
# ─────────────────────────────────────────────────────────────────────────────
def sinkhorn(scores: torch.Tensor, n_iter: int = 3, eps: float = 0.05) -> torch.Tensor:
    Q = torch.exp(scores / eps).T
    Q /= Q.sum()
    K, B = Q.shape
    for _ in range(n_iter):
        Q /= (Q.sum(dim=1, keepdim=True) * K)
        Q /= (Q.sum(dim=0, keepdim=True) * B)
    return (Q / Q.sum(dim=0, keepdim=True)).T

def swav_loss(
    student_scores: list,
    teacher_scores: list,
    prototypes: torch.Tensor,
    temperature: float,
    n_sinkhorn: int = 3,
    eps: float = 0.05,
) -> torch.Tensor:
    loss = 0.0
    n    = 0
    for i, t in enumerate(teacher_scores):
        q = sinkhorn(t.detach(), n_iter=n_sinkhorn, eps=eps)
        for j, s in enumerate(student_scores):
            if i == j:
                continue
            p = F.softmax(s / temperature, dim=1)
            loss -= (q * torch.log(p + 1e-8)).sum(dim=1).mean()
            n += 1
    return loss / n

# ─────────────────────────────────────────────────────────────────────────────
# MoCo InfoNCE loss
# ─────────────────────────────────────────────────────────────────────────────
def moco_loss(q: torch.Tensor, k: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Standard InfoNCE / MoCo loss between query q and key k."""
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    logits = torch.mm(q, k.T) / temperature                       # (B, B)
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)
