"""train.py — Contrastive pretraining entry point.

Usage examples:
  # SimCLR with default config
  python train.py --config configs/simclr.yaml

  # BYOL, override GPU and epochs
  python train.py --config configs/byol.yaml --options General.device=1 Training.epochs=100

  # DINO with ViT backbone
  python train.py --config configs/dino.yaml --options Backbone.name=vit_s Training.batch_size=32
"""
import argparse
import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils.config    import load_config, print_config
from utils.dataset   import ContrastiveDataset, build_contrastive_transform
from utils.logger    import get_logger, make_run_dir, CSVLogger, CheckpointManager
from utils.optimizer import build_optimizer, build_scheduler
from modules         import get_model

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def parse_args():
    p = argparse.ArgumentParser(description="Contrastive Learning Pretrainer")
    p.add_argument("--config",  type=str, required=True, help="Path to YAML config")
    p.add_argument("--options", nargs="+", default=None, help="Override config: Section.key=value ...")
    return p.parse_args()

def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    total_loss = 0.0
    model_name = cfg.General.model_name.lower()
    ga_steps   = cfg.General.gradient_accumulation_steps

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", ncols=100)
    for step, (views, folder, part) in enumerate(pbar):
        views = [v.to(device) for v in views]

        with autocast(enabled=cfg.General.mixed_precision):
            out  = model(views)
            loss = out["loss"] / ga_steps

        if cfg.General.mixed_precision:
            scaler.scale(loss).backward()
            if (step + 1) % ga_steps == 0:
                if cfg.Training.get("clip_grad_norm"):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.Training.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % ga_steps == 0:
                if cfg.Training.get("clip_grad_norm"):
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.Training.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        # EMA update for applicable models
        if hasattr(model, "update_teacher"):
            model.update_teacher()
        elif hasattr(model, "update_momentum"):
            model.update_momentum()
        elif hasattr(model, "update_target"):
            model.update_target()

        total_loss += loss.item() * ga_steps
        pbar.set_postfix(loss=f"{loss.item()*ga_steps:.4f}")

    return total_loss / len(loader)

def main():
    args = parse_args()
    cfg  = load_config(args.config, args.options)
    cfg._yaml_path = args.config

    set_seed(cfg.General.seed)
    logger = get_logger()
    logger.info(f"Model: {cfg.General.model_name}")
    print_config(cfg)

    device_id = cfg.General.device
    device    = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    transform = build_contrastive_transform(cfg)

    # DINO / SwAV: multi-crop
    model_name  = cfg.General.model_name.lower()
    n_views     = 2
    local_tf    = None
    n_local     = 0
    if model_name in ("dino", "ibot"):
        from torchvision import transforms as T
        a = cfg.Augmentation
        local_tf = T.Compose([
            T.RandomResizedCrop(cfg.Dataset.image_size, scale=tuple(a.local_crop_scale)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=list(a.normalize_mean), std=list(a.normalize_std)),
        ])
        n_local = cfg.DINO.get("n_local_crops", 8) if model_name == "dino" else cfg.iBOT.get("n_local_crops", 8)

    dataset = ContrastiveDataset(cfg.Dataset.train_dir, transform, n_views=n_views, local_transform=local_tf, n_local_views=n_local)
    loader  = DataLoader(
        dataset,
        batch_size  = cfg.Training.batch_size,
        shuffle     = True,
        num_workers = cfg.General.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    logger.info(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(cfg).to(device)
    if torch.cuda.device_count() > 1 and device_id >= 0:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")

    # Tell BYOL total steps for EMA cosine schedule
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(base_model, "_total_steps"):
        base_model._total_steps = cfg.Training.epochs * len(loader)

    optimizer  = build_optimizer(model, cfg)
    scheduler  = build_scheduler(optimizer, cfg)
    scaler     = GradScaler(enabled=cfg.General.mixed_precision)

    # ── Logging ───────────────────────────────────────────────────────────────
    run_dir  = make_run_dir(cfg)
    csv_log  = CSVLogger(os.path.join(run_dir, "metrics.csv"), fieldnames=["epoch", "train_loss", "lr"])
    ckpt_mgr = CheckpointManager(run_dir, keep_k=cfg.Logs.keep_best_k)
    logger.info(f"Run directory: {run_dir}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(cfg.Training.epochs):
        loss = train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch)
        lr   = optimizer.param_groups[0]["lr"]

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()

        csv_log.log({"epoch": epoch + 1, "train_loss": f"{loss:.4f}", "lr": f"{lr:.6f}"})
        logger.info(f"Epoch {epoch+1}/{cfg.Training.epochs}  loss={loss:.4f}  lr={lr:.6f}")

        if (epoch + 1) % cfg.Logs.save_every_n_epochs == 0 or epoch == cfg.Training.epochs - 1:
            state = {
                "epoch":      epoch + 1,
                "model":      (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                "optimizer":  optimizer.state_dict(),
                "cfg":        dict(cfg),
            }
            ckpt_mgr.save(state, epoch + 1, loss)

    logger.info("✅ Training complete.")

if __name__ == "__main__":
    main()