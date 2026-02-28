"""
calibration.py — Isotonic calibration + FDR-constrained threshold search.
Saves → artifacts/calibrator.pkl + artifacts/threshold.json
"""

import json, os, pickle
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

ARTIFACTS  = "artifacts"
DEVICE     = "cpu"
TARGET_FDR = 0.05   # FDR = 1 - Precision


def _load():
    from models.model import build_model
    X_va = np.load(f"{ARTIFACTS}/X_val.npy")
    X_te = np.load(f"{ARTIFACTS}/X_test.npy")
    y_va = np.load(f"{ARTIFACTS}/y_val.npy")
    y_te = np.load(f"{ARTIFACTS}/y_test.npy")
    m    = build_model(X_va.shape[2], DEVICE)
    m.load_state_dict(torch.load(f"{ARTIFACTS}/best_model.pt", map_location=DEVICE))
    m.eval()
    return m, X_va, X_te, y_va, y_te


def infer(model, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            p, _ = model(torch.from_numpy(X[i:i+256]))
            out.append(p.numpy())
    return np.concatenate(out)


def find_threshold(probs, labels, target_fdr=TARGET_FDR):
    best_t, best_rec = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 180):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        if (1 - prec) <= target_fdr and rec > best_rec:
            best_rec, best_t = rec, float(t)
    return best_t


def calibrate():
    model, X_va, X_te, y_va, y_te = _load()

    raw_va = infer(model, X_va)
    raw_te = infer(model, X_te)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_va, y_va)
    cal_va = iso.transform(raw_va)
    cal_te = iso.transform(raw_te)

    threshold = find_threshold(cal_va, y_va)
    preds     = (cal_te >= threshold).astype(int)

    metrics = {
        "threshold": round(threshold, 4),
        "precision": round(float(precision_score(y_te, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_te, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_te, preds, zero_division=0)), 4),
        "auroc":     round(float(roc_auc_score(y_te, cal_te)), 4),
        "fdr":       round(float(1 - precision_score(y_te, preds, zero_division=0)), 4),
    }

    print("[Calibration] Test metrics:")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")

    with open(f"{ARTIFACTS}/calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(f"{ARTIFACTS}/threshold.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Calibration] Saved → {ARTIFACTS}/")
    return threshold, iso


if __name__ == "__main__":
    calibrate()
