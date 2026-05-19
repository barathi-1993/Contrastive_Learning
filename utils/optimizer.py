"""utils/optimizer.py — Optimizer and LR-scheduler factory."""
import math
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (CosineAnnealingLR, StepLR, ReduceLROnPlateau, LinearLR, SequentialLR)

def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    t = cfg.Training
    params = model.parameters()

    name = str(t.optimizer).lower()
    if name == "adam":
        return Adam(params, lr=t.learning_rate, weight_decay=t.weight_decay)
    if name == "adamw":
        return AdamW(params, lr=t.learning_rate, weight_decay=t.weight_decay)
    if name == "sgd":
        return SGD(params, lr=t.learning_rate, weight_decay=t.weight_decay,
                   momentum=0.9, nesterov=True)
    if name == "lars":
        try:
            from torch_lars import LARS
            return LARS(params, lr=t.learning_rate, weight_decay=t.weight_decay)
        except ImportError:
            print("⚠️  torch_lars not installed — falling back to SGD with momentum.")
            return SGD(params, lr=t.learning_rate, weight_decay=t.weight_decay,
                       momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    t         = cfg.Training
    n_epochs  = t.epochs
    warmup    = getattr(t, "warmup_epochs", 0)
    min_lr    = getattr(t, "min_lr", 1e-6)
    name      = str(t.scheduler).lower()

    schedulers = []
    milestones = []

    if warmup > 0:
        warmup_sched = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0,
                                total_iters=warmup)
        schedulers.append(warmup_sched)
        milestones.append(warmup)

    main_epochs = n_epochs - warmup
    if name == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=min_lr)
    elif name == "step":
        step_size = getattr(t, "step_size", 30)
        gamma     = getattr(t, "gamma", 0.1)
        main      = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "plateau":
        main      = ReduceLROnPlateau(optimizer, mode="min", patience=10,
                                      factor=0.5, min_lr=min_lr)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    if warmup > 0:
        schedulers.append(main)
        return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
    return main
