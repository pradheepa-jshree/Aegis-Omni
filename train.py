"""
train.py — Train AegisLSTM on real preprocessed sequences.
Saves best checkpoint → artifacts/best_model.pt
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from models.model import build_model

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS    = 40
BATCH     = 64
LR        = 1e-3
ARTIFACTS = "artifacts"


def focal_loss(pred, target, alpha=0.75, gamma=2.0):
    bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
    pt  = torch.where(target == 1, pred, 1 - pred)
    return (alpha * (1 - pt)**gamma * bce).mean()


def loader(X, y, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=BATCH, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    preds, tgts = [], []
    for xb, yb in dl:
        p, _ = model(xb.to(DEVICE))
        preds.append(p.cpu().numpy())
        tgts.append(yb.numpy())
    p, t = np.concatenate(preds), np.concatenate(tgts)
    return roc_auc_score(t, p), average_precision_score(t, p)


def train():
    os.makedirs(ARTIFACTS, exist_ok=True)
    X_tr = np.load(f"{ARTIFACTS}/X_train.npy")
    X_va = np.load(f"{ARTIFACTS}/X_val.npy")
    y_tr = np.load(f"{ARTIFACTS}/y_train.npy")
    y_va = np.load(f"{ARTIFACTS}/y_val.npy")

    tr_dl = loader(X_tr, y_tr, True)
    va_dl = loader(X_va, y_va, False)

    model = build_model(X_tr.shape[2], DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                                  epochs=EPOCHS,
                                                  steps_per_epoch=len(tr_dl))
    best_auprc, no_imp = 0.0, 0
    history = []

    print(f"\n{'Epoch':>5} {'Loss':>9} {'AUROC':>8} {'AUPRC':>8}")
    print("─" * 38)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            p, _ = model(xb)
            loss = focal_loss(p, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            total += loss.item() * len(yb)

        train_loss = total / len(tr_dl.dataset)
        auroc, auprc = evaluate(model, va_dl)
        history.append({"epoch": epoch, "loss": train_loss, "auroc": auroc, "auprc": auprc})

        mark = " ◀ best" if auprc > best_auprc else ""
        print(f"{epoch:>5}  {train_loss:>9.4f}  {auroc:>8.4f}  {auprc:>8.4f}{mark}")

        if auprc > best_auprc:
            best_auprc = auprc
            torch.save(model.state_dict(), f"{ARTIFACTS}/best_model.pt")
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 6:
                print(f"\n[Train] Early stop at epoch {epoch}")
                break

    with open(f"{ARTIFACTS}/train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] Best Val AUPRC: {best_auprc:.4f}")
    print(f"[Train] Checkpoint → {ARTIFACTS}/best_model.pt")
    return model


if __name__ == "__main__":
    train()
