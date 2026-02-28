"""
models/train.py — Train AegisLSTM ensemble on PhysioNet Sepsis data.

Usage:
  python -m models.train --data data/sepsis.csv --epochs 50
"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from config.settings import ModelConfig, TrainConfig
from models.model import AegisLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Focal Loss for class imbalance ──────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 3.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = torch.tensor(pos_weight)

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight.to(logits.device),
            reduction="none"
        )
        p_t = torch.exp(-bce)
        loss = ((1 - p_t) ** self.gamma) * bce
        return loss.mean()


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Oversample positives to handle class imbalance."""
    pos = y.sum()
    neg = len(y) - pos
    weights = np.where(y == 1, neg / pos, 1.0)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits, _ = model(X_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_b.cpu().numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # AUC-ROC (manual trapezoidal)
    from sklearn.metrics import roc_auc_score
    try:
        auroc = roc_auc_score(labels, probs)
    except Exception:
        auroc = 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "auroc": auroc}


def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_idx: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str,
    checkpoint_dir: str,
) -> AegisLSTM:

    input_size = X_train.shape[-1]
    model = AegisLSTM(
        input_size=input_size,
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    criterion = FocalLoss(gamma=2.0, pos_weight=5.0)

    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val)

    sampler = make_weighted_sampler(y_train)
    train_loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=train_cfg.batch_size,
        sampler=sampler,
    )
    val_loader = DataLoader(
        TensorDataset(X_v, y_v),
        batch_size=train_cfg.batch_size * 2,
        shuffle=False,
    )

    best_auroc = 0.0
    patience_counter = 0
    best_path = os.path.join(checkpoint_dir, f"model_{model_idx}.pt")

    logger.info(f"[Model {model_idx}] Training on {len(X_train)} samples | input_size={input_size}")

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        metrics = evaluate(model, val_loader, device)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:03d} | loss={total_loss/len(train_loader):.4f} | "
                f"AUROC={metrics['auroc']:.4f} | F1={metrics['f1']:.4f} | "
                f"Precision={metrics['precision']:.4f}"
            )

        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    logger.info(f"[Model {model_idx}] Best AUROC: {best_auroc:.4f} → saved to {best_path}")
    return model


def train_ensemble(
    data_path: str,
    checkpoint_dir: str = "checkpoints",
    seq_len: int = 12,
    n_models: int = 3,
    device: str = "cpu",
):
    """Train the full ensemble of N models with different data splits."""
    from data_pipeline.data_pipeline import prepare_dataset

    os.makedirs(checkpoint_dir, exist_ok=True)

    model_cfg = ModelConfig(sequence_length=seq_len, num_ensemble_models=n_models)
    train_cfg = TrainConfig()

    X, y, feature_cols = prepare_dataset(data_path, seq_len=seq_len, artifact_dir=checkpoint_dir)
    logger.info(f"Full dataset: X={X.shape}, pos_rate={y.mean():.4f}")

    models = []
    for i in range(n_models):
        # Each model sees a slightly different random split (diversity)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=train_cfg.test_split + train_cfg.val_split,
            stratify=y, random_state=42 + i
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test,
            test_size=0.5,
            stratify=y_test,
            random_state=42 + i,
        )
        m = train_single_model(
            X_train, y_train, X_val, y_val,
            model_idx=i,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        models.append(m)

    # Final evaluation on test set
    from models.calibration import calibrate_threshold
    logger.info("\n=== Final Test Evaluation ===")
    all_probs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            logits, _ = m(torch.FloatTensor(X_test).to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    ensemble_probs = np.mean(all_probs, axis=0)
    threshold = calibrate_threshold(ensemble_probs, y_test, target_fdr=train_cfg.target_fdr)
    logger.info(f"Optimal threshold (FDR<5%): {threshold:.3f}")

    import joblib
    joblib.dump(threshold, os.path.join(checkpoint_dir, "threshold.pkl"))
    logger.info("Training complete. Checkpoints saved.")
    return models, threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sepsis.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--n-models", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_ensemble(
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        seq_len=args.seq_len,
        n_models=args.n_models,
        device=args.device,
    )
