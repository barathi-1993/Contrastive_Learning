"""downstream/tasks.py — Linear probe, fine-tuning, clustering, visualisation."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset  import EvalDataset, build_eval_transform
from utils.logger   import get_logger

logger = get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_feats, all_labels, all_paths = [], [], []
    for batch in tqdm(loader, desc="Extracting embeddings"):
        imgs, labels, paths, *_ = batch
        imgs = imgs.to(device)
        feats = model.get_embedding(imgs)
        all_feats.append(feats.cpu().numpy())
        all_labels.extend(labels.tolist())
        all_paths.extend(paths)
    return np.vstack(all_feats), np.array(all_labels), all_paths

# ─────────────────────────────────────────────────────────────────────────────
# Linear probe / fine-tuning classifier
# ─────────────────────────────────────────────────────────────────────────────
class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x): return self.head(x)

def run_classification(model, cfg, device):
    """Linear probe (freeze_backbone=True) or full fine-tune."""
    d   = cfg.Downstream
    tra = d.Training
    transform = build_eval_transform(cfg)

    train_ds = EvalDataset(cfg.Dataset.train_dir, transform)
    test_ds  = EvalDataset(cfg.Dataset.test_dir,  transform)

    train_loader = DataLoader(train_ds, batch_size=tra.batch_size, shuffle=True, num_workers=cfg.General.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=tra.batch_size, shuffle=False, num_workers=cfg.General.num_workers)

    # Optionally freeze backbone
    if d.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    backbone_dim = model.encoder.out_dim if hasattr(model, "encoder") else cfg.Backbone.out_dim
    classifier   = LinearClassifier(
        backbone_dim, d.num_classes,
        hidden_dim=d.get("classifier_hidden_dim"),
        dropout=d.get("classifier_dropout", 0.0),
    ).to(device)

    params = classifier.parameters() if d.freeze_backbone else list(model.parameters()) + list(classifier.parameters())
    opt    = torch.optim.AdamW(params, lr=tra.learning_rate, weight_decay=tra.weight_decay)
    crit   = nn.CrossEntropyLoss()

    for epoch in range(tra.epochs):
        model.train() if not d.freeze_backbone else model.eval()
        classifier.train()
        total, correct = 0, 0
        for imgs, labels, *_ in tqdm(train_loader, desc=f"Clf epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            feats  = model.get_embedding(imgs)
            logits = classifier(feats)
            loss   = crit(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        logger.info(f"Epoch {epoch+1}  train_acc={correct/total:.4f}")

    # Evaluation
    model.eval(); classifier.eval()
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for imgs, labels, *_ in test_loader:
            imgs = imgs.to(device)
            preds = classifier(model.get_embedding(imgs)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels_list.extend(labels.tolist())

    from sklearn.metrics import classification_report, accuracy_score
    acc = accuracy_score(all_labels_list, all_preds)
    report = classification_report(all_labels_list, all_preds)
    logger.info(f"\nTest Accuracy: {acc:.4f}\n{report}")
    return acc, report

# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────
def run_clustering(feats: np.ndarray, cfg):
    from sklearn.preprocessing import StandardScaler
    c       = cfg.Downstream.Clustering
    scaler  = StandardScaler()
    X       = scaler.fit_transform(feats)

    # Partial sampling
    if c.partial_ratio and c.partial_ratio < 1.0:
        rng = np.random.default_rng(c.seed)
        idx = rng.choice(len(X), size=int(len(X) * c.partial_ratio), replace=False)
        X   = X[idx]
        logger.info(f"Using {len(X)} / {len(feats)} samples for clustering")

    if c.auto_k:
        _find_optimal_k(X, c.max_k, c.seed)

    algo = str(c.algorithm).lower()
    if algo == "kmeans":
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=c.n_clusters, random_state=c.seed, n_init=10)
    elif algo == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=c.n_clusters)
    elif algo == "dbscan":
        from sklearn.cluster import DBSCAN
        model = DBSCAN()
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")

    labels = model.fit_predict(X)
    logger.info(f"Clustering done: {len(np.unique(labels))} clusters found")
    return labels

def _find_optimal_k(X, max_k, seed):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    inertias = []
    ks = range(2, max_k + 1)
    for k in tqdm(ks, desc="Elbow search"):
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, "bx-")
    plt.xlabel("k"); plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig("elbow_curve.png", dpi=150)
    plt.close()
    logger.info("Elbow curve saved to elbow_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation (t-SNE / UMAP / PCA)
# ─────────────────────────────────────────────────────────────────────────────
def run_visualization(feats: np.ndarray, labels: np.ndarray, image_paths: list, cfg):
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    v = cfg.Downstream.Visualization
    out_dir = Path(v.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = StandardScaler().fit_transform(feats)
    method = str(v.method).lower()

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=v.n_components, perplexity=v.tsne_perplexity, n_iter=v.tsne_n_iter, random_state=42)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=v.n_components, n_neighbors=v.umap_n_neighbors, min_dist=v.umap_min_dist, random_state=42)
        except ImportError:
            raise ImportError("umap-learn required: pip install umap-learn")
    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=v.n_components)
    else:
        raise ValueError(f"Unknown visualisation method: {method}")

    reduced = reducer.fit_transform(X)
    logger.info(f"Dimensionality reduction ({method}) done: {reduced.shape}")

    # Scatter plot
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", alpha=0.7, s=20)
    plt.colorbar(sc, label="Cluster / Class")
    plt.title(f"{method.upper()} Embeddings")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.tight_layout()
    scatter_path = out_dir / f"{method}_scatter.png"
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    logger.info(f"Scatter saved to {scatter_path}")

    # Sample images per cluster
    if v.plot_sample_images:
        for cid in np.unique(labels):
            idx = np.where(labels == cid)[0]
            sample = np.random.choice(idx, size=min(v.n_sample_images_per_cluster, len(idx)), replace=False)
            n = len(sample)
            cols = min(5, n)
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            axes = np.array(axes).reshape(-1)
            for ai, si in enumerate(sample):
                try:
                    img = PILImage.open(image_paths[si]).convert("RGB")
                    axes[ai].imshow(img); axes[ai].axis("off")
                except Exception:
                    axes[ai].axis("off")
            for ai in range(n, len(axes)):
                axes[ai].axis("off")
            plt.suptitle(f"Cluster {cid} — {len(idx)} images", fontsize=12)
            plt.tight_layout()
            plt.savefig(out_dir / f"cluster_{cid}_samples.png", dpi=150)
            plt.close()

    logger.info(f"Visualisation complete. Outputs in {out_dir}")
