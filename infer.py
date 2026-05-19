"""infer.py — Extract embeddings from unlabelled images and run downstream tasks.

Usage:
  # Clustering only
  python infer.py --config configs/simclr.yaml \
                  --checkpoint logs/.../checkpoints/best.pth \
                  --task clustering \
                  --options Dataset.infer_dir=/path/to/images Downstream.Clustering.n_clusters=9

  # Visualisation only
  python infer.py --config configs/simclr.yaml \
                  --checkpoint logs/.../checkpoints/best.pth \
                  --task visualize \
                  --options Downstream.Visualization.method=umap

  # Clustering + visualise
  python infer.py --config configs/simclr.yaml \
                  --checkpoint logs/.../checkpoints/best.pth \
                  --task cluster_and_visualize
"""
import argparse
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config   import load_config, print_config
from utils.dataset  import EvalDataset, build_eval_transform
from utils.logger   import get_logger
from modules        import get_model
from downstream.tasks import extract_embeddings, run_clustering, run_visualization

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task",       type=str, default="cluster_and_visualize", choices=["embedding", "clustering", "visualize", "cluster_and_visualize"])
    p.add_argument("--options",    nargs="+", default=None)
    return p.parse_args()

def main():
    args   = parse_args()
    cfg    = load_config(args.config, args.options)
    logger = get_logger()
    print_config(cfg)

    device = torch.device(f"cuda:{cfg.General.device}" if torch.cuda.is_available() and cfg.General.device >= 0 else "cpu")
    model = get_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    logger.info(f"Loaded: {args.checkpoint}")

    transform   = build_eval_transform(cfg)
    infer_ds    = EvalDataset(cfg.Dataset.infer_dir, transform)
    infer_loader = DataLoader(infer_ds, batch_size=cfg.Downstream.embed_batch_size, shuffle=False, num_workers=cfg.General.num_workers)
    logger.info(f"Inference dataset: {len(infer_ds)} images")

    feats, labels, paths = extract_embeddings(model, infer_loader, device)
    logger.info(f"Embeddings shape: {feats.shape}")

    # Save raw embeddings
    out_dir = cfg.Downstream.Visualization.output_dir
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), feats)
    with open(os.path.join(out_dir, "image_paths.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path"])
        writer.writerows([[p] for p in paths])
    logger.info(f"Embeddings saved to {out_dir}/embeddings.npy")

    task = args.task
    cluster_labels = None

    if task in ("clustering", "cluster_and_visualize"):
        cluster_labels = run_clustering(feats, cfg)
        # Save cluster assignments
        with open(os.path.join(out_dir, "cluster_assignments.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "cluster"])
            writer.writerows(zip(paths, cluster_labels.tolist()))
        logger.info(f"Cluster assignments saved to {out_dir}/cluster_assignments.csv")

    if task in ("visualize", "cluster_and_visualize"):
        vis_labels = cluster_labels if cluster_labels is not None else labels
        run_visualization(feats, vis_labels, paths, cfg)

    logger.info("✅ Inference complete.")

if __name__ == "__main__":
    main()
