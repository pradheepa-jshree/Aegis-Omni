"""
api/main.py — FastAPI backend loading REAL trained artifacts.
POST /predict → runs actual model inference end-to-end.
"""

import json, os, pickle, sys
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTIFACTS = os.getenv("ARTIFACTS_DIR", "artifacts")
DEVICE    = "cpu"


# ── Load all real artifacts at startup ────────────────────────────────────────
def _boot():
    from models.model import build_model
    with open(f"{ARTIFACTS}/scaler.pkl",       "rb") as f: scaler     = pickle.load(f)
    with open(f"{ARTIFACTS}/feature_cols.pkl", "rb") as f: feat_cols  = pickle.load(f)
    with open(f"{ARTIFACTS}/calibrator.pkl",   "rb") as f: calibrator = pickle.load(f)
    with open(f"{ARTIFACTS}/threshold.json")        as f: t_info      = json.load(f)

    model = build_model(len(feat_cols), DEVICE)
    model.load_state_dict(torch.load(f"{ARTIFACTS}/best_model.pt", map_location=DEVICE))
    model.eval()
    return model, scaler, feat_cols, calibrator, t_info


try:
    MODEL, SCALER, FEAT_COLS, CALIBRATOR, T_INFO = _boot()
    THRESHOLD = T_INFO["threshold"]
    print(f"[API] Ready | features={len(FEAT_COLS)} | threshold={THRESHOLD:.4f}")
    READY = True
except Exception as e:
    print(f"[API] Cannot load artifacts: {e}")
    print("[API] Run setup_and_train.py first.")
    READY = False
    THRESHOLD = 0.5


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Aegis-Omni API",
    description="ICU Septic Shock Early Warning — Real LSTM inference",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    patient_id: str = Field(..., example="P001")
    # 12-hour windows for key vitals. Missing features → zero-padded (model handles via training)
    window: Dict[str, List[float]] = Field(..., example={
        "HR":      [96,98,102,105,108,112,115,118,114,110,107,104],
        "MAP":     [72,70,68,65,63,60,58,56,55,54,53,52],
        "Lactate": [1.2,1.4,1.6,1.9,2.1,2.4,2.6,2.8,3.0,3.1,3.2,3.3],
        "SpO2":    [97,97,96,96,95,95,94,94,93,93,92,92],
        "Resp":    [18,18,19,20,22,24,25,26,24,23,22,21],
        "SBP":     [115,112,108,104,100,96,93,90,88,86,85,84],
        "Temp":    [37.0,37.2,37.5,37.8,38.1,38.3,38.4,38.5,38.6,38.5,38.4,38.3],
    })
    vitals_snapshot: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    patient_id:   str
    risk_score:   float
    risk_percent: str
    confidence:   float
    alert_tier:   str
    alert:        bool
    threshold:    float
    top_features: List[str]
    shap_values:  List[float]
    ai_narrative: str
    raw_prob:     float


# ── Inference helpers ─────────────────────────────────────────────────────────

def _build_matrix(window: Dict[str, List[float]], feat_cols: List[str]) -> np.ndarray:
    """Map incoming vitals to the full feature matrix used during training."""
    import pandas as pd

    SEQ = 12
    rows = []
    for t in range(SEQ):
        row = {fc: 0.0 for fc in feat_cols}
        for key, vals in window.items():
            if key in row and t < len(vals):
                row[key] = float(vals[t])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Derive engineered columns that exist in feat_cols
    for col in ["HR","MAP","Lactate","Resp","SBP","Temp"]:
        if col in df.columns:
            if f"{col}_mean4h"  in feat_cols: df[f"{col}_mean4h"]  = df[col].rolling(4,min_periods=1).mean()
            if f"{col}_std4h"   in feat_cols: df[f"{col}_std4h"]   = df[col].rolling(4,min_periods=1).std().fillna(0)
            if f"{col}_slope2h" in feat_cols: df[f"{col}_slope2h"] = df[col].diff(2).fillna(0)

    if "shock_index"       in feat_cols: df["shock_index"]       = (df.get("HR",0) / (df.get("SBP",1).clip(lower=1))).clip(0,5)
    if "lactate_map_ratio" in feat_cols: df["lactate_map_ratio"] = (df.get("Lactate",0) / (df.get("MAP",1).clip(lower=1))).clip(0,10)

    # Align to exact feat_cols order
    mat = np.zeros((SEQ, len(feat_cols)), dtype=np.float32)
    for i, fc in enumerate(feat_cols):
        if fc in df.columns:
            mat[:, i] = df[fc].values.astype(np.float32)

    scaled = SCALER.transform(mat.reshape(-1, len(feat_cols))).reshape(SEQ, len(feat_cols))
    return scaled.astype(np.float32)


def _mc_confidence(x_tensor: torch.Tensor, n: int = 20) -> float:
    MODEL.train()
    samples = []
    with torch.no_grad():
        for _ in range(n):
            p, _ = MODEL(x_tensor)
            samples.append(float(p.item()))
    MODEL.eval()
    return round(max(0.0, min(1.0, 1.0 - float(np.std(samples)) * 4)), 4)


def _top_features_gradient(x_tensor: torch.Tensor, feat_cols: List[str], top_k=8):
    """
    Lightweight gradient-based saliency (no SHAP overhead at inference time).
    Falls back gracefully if anything goes wrong.
    """
    try:
        x = x_tensor.clone().requires_grad_(True)
        MODEL.eval()
        prob, _ = MODEL(x)
        prob.backward()
        importance = x.grad.abs().mean(dim=1).squeeze(0).detach().numpy()  # (F,)
        top_idx = importance.argsort()[::-1][:top_k]
        names = [feat_cols[i] for i in top_idx]
        vals  = [round(float(importance[i]), 5) for i in top_idx]
        return names, vals
    except Exception:
        return feat_cols[:top_k], [0.0]*top_k


def _narrative(risk: float, top_features: List[str], snapshot: dict) -> str:
    """Calls Featherless AI if key is set, else deterministic template."""
    from explainability.explain import generate_ai_narrative
    return generate_ai_narrative(risk, top_features, [0.0]*len(top_features), snapshot)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok" if READY else "no_artifacts",
        "ready":     READY,
        "features":  len(FEAT_COLS) if READY else 0,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not READY:
        raise HTTPException(503, "Artifacts not loaded. Run setup_and_train.py first.")

    # Build + scale feature matrix
    mat      = _build_matrix(req.window, FEAT_COLS)
    x_tensor = torch.from_numpy(mat).unsqueeze(0)  # (1, 12, F)

    # Real model inference
    with torch.no_grad():
        raw_prob_t, _ = MODEL(x_tensor)
    raw_prob = float(raw_prob_t.item())

    # Isotonic calibration
    cal_prob = float(CALIBRATOR.transform([raw_prob])[0])

    # MC Dropout confidence
    confidence = _mc_confidence(x_tensor)

    # Gradient saliency for top features
    top_feats, saliency = _top_features_gradient(x_tensor, FEAT_COLS)

    # Alert tier
    alert  = cal_prob >= THRESHOLD
    if   cal_prob >= 0.80: tier = "CRITICAL"
    elif cal_prob >= 0.65: tier = "WARN"
    else:                  tier = "WATCH"

    # AI narrative (Featherless AI or template)
    snapshot = req.vitals_snapshot or {k: v[-1] for k, v in req.window.items() if v}
    narrative = _narrative(cal_prob, top_feats, snapshot)

    return PredictResponse(
        patient_id   = req.patient_id,
        risk_score   = round(cal_prob, 4),
        risk_percent = f"{cal_prob:.1%}",
        confidence   = confidence,
        alert_tier   = tier,
        alert        = alert,
        threshold    = THRESHOLD,
        top_features = top_feats,
        shap_values  = saliency,
        ai_narrative = narrative,
        raw_prob     = round(raw_prob, 4),
    )


@app.get("/threshold")
def get_threshold():
    return T_INFO if READY else {"error": "not ready"}


@app.get("/features")
def get_features():
    return {"count": len(FEAT_COLS), "features": FEAT_COLS} if READY else {}
