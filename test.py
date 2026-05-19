"""test.py — Evaluate a pretrained encoder on a labelled test set.

Usage:
  python test.py --config configs/simclr.yaml \
                 --checkpoint logs/20240601_120000_simclr/checkpoints/best.pth \
                 --options Dataset.test_dir=/path/to/test
"""
import argparse
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
    p.add_argument("--task",       type=str, default="linear_probe",
                   choices=["linear_probe", "clustering", "visualize"])
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
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    transform  = build_eval_transform(cfg)
    test_ds    = EvalDataset(cfg.Dataset.test_dir, transform)
    test_loader = DataLoader(test_ds, batch_size=cfg.Downstream.embed_batch_size, shuffle=False, num_workers=cfg.General.num_workers)

    feats, labels, paths = extract_embeddings(model, test_loader, device)
    logger.info(f"Embeddings: {feats.shape}")

    task = args.task
    if task == "linear_probe":
        from downstream.tasks import run_classification
        run_classification(model, cfg, device)

    elif task == "clustering":
        cluster_labels = run_clustering(feats, cfg)
        run_visualization(feats, cluster_labels, paths, cfg)

    elif task == "visualize":
        run_visualization(feats, labels, paths, cfg)

if __name__ == "__main__":
    main()
