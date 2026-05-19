"""utils/logger.py — Console logger, CSV logger, and checkpoint manager."""
import csv
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml

def get_logger(name: str = "cl_framework") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def make_run_dir(cfg) -> str:
    """
    Creates   logs/<timestamp>_<model_name>/
    Returns the path string.
    """
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = cfg.General.get("model_name", "model")
    run   = f"{ts}_{model}"
    path  = Path(cfg.Logs.log_root_dir) / run
    path.mkdir(parents=True, exist_ok=True)

    # Copy the YAML into the run folder for reproducibility
    yaml_src = getattr(cfg, "_yaml_path", None)
    if yaml_src and Path(yaml_src).exists():
        shutil.copy(yaml_src, path / Path(yaml_src).name)

    return str(path)

class CSVLogger:
    """Append one row per epoch to a CSV file."""
    def __init__(self, path: str, fieldnames: list):
        self.path       = path
        self.fieldnames = fieldnames
        if not Path(path).exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames,
                                    extrasaction="ignore")
            writer.writerow(row)

class CheckpointManager:
    """Save / load checkpoints, keeping only the top-k by metric."""

    def __init__(self, run_dir: str, keep_k: int = 3, mode: str = "min"):
        self.ckpt_dir = Path(run_dir) / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.keep_k   = keep_k
        self.mode     = mode
        self._history: list = []   # [(metric_val, path)]

    def save(self, state: dict, epoch: int, metric: float) -> str:
        fname = self.ckpt_dir / f"epoch_{epoch:04d}_metric_{metric:.4f}.pth"
        torch.save(state, fname)
        self._history.append((metric, str(fname)))

        # Sort and prune
        rev = (self.mode == "max")
        self._history.sort(key=lambda x: x[0], reverse=rev)
        while len(self._history) > self.keep_k:
            _, old = self._history.pop()
            if Path(old).exists():
                Path(old).unlink()

        # Always keep a "best" symlink/copy
        best = self._history[0][1]
        best_path = self.ckpt_dir / "best.pth"
        shutil.copy(best, best_path)

        return str(fname)

    def load_best(self, device="cpu") -> dict:
        best = self.ckpt_dir / "best.pth"
        if not best.exists():
            raise FileNotFoundError(f"No checkpoint at {best}")
        return torch.load(best, map_location=device)
