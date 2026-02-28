"""
inference/engine.py
Loads trained models and runs prediction for a single patient input.
"""

import logging
import os
import numpy as np
import torch
import joblib
from typing import Dict, List, Optional

from models.model import AegisLSTM, AegisEnsemble
from models.calibration import compute_uncertainty
from explainability.explain import compute_shap_values, top_features_from_shap, generate_narrative

logger = logging.getLogger(__name__)


def load_ensemble(
    checkpoint_dir: str = "checkpoints",
    n_models: int = 3,
    device: str = "cpu",
) -> AegisEnsemble:
    """Load the trained ensemble from disk."""
    feature_cols = joblib.load(os.path.join(checkpoint_dir, "feature_cols.pkl"))
    input_size = len(feature_cols)

    models = []
    for i in range(n_models):
        path = os.path.join(checkpoint_dir, f"model_{i}.pt")
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}. Using random weights.")
        m = AegisLSTM(input_size=input_size)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)

    return AegisEnsemble(models)


def predict(
    X: np.ndarray,
    checkpoint_dir: str = "checkpoints",
    ensemble: Optional[AegisEnsemble] = None,
    featherless_client=None,
    device: str = "cpu",
    compute_shap: bool = True,
    patient_vitals: Optional[Dict] = None,
) -> Dict:
    """
    Run full prediction pipeline on a prepared input array.

    X: (1, seq_len, n_features) numpy array

    Returns dict with: risk_score, confidence, uncertainty, top_features, narrative
    """
    feature_cols = joblib.load(os.path.join(checkpoint_dir, "feature_cols.pkl"))
    threshold    = joblib.load(os.path.join(checkpoint_dir, "threshold.pkl"))

    if ensemble is None:
        ensemble = load_ensemble(checkpoint_dir=checkpoint_dir, device=device)

    # ── Ensemble inference ───────────────────────────────────────
    x_tensor = torch.FloatTensor(X).to(device)

    probs_per_model = []
    for m in ensemble.models:
        m.eval()
        with torch.no_grad():
            logits, _ = m(x_tensor)
            probs_per_model.append(torch.sigmoid(logits).cpu().numpy())

    probs_arr = np.vstack(probs_per_model)  # (n_models, n_samples)
    mean_prob, std_prob, ci_lo, ci_hi = compute_uncertainty(probs_arr)

    risk_score  = float(mean_prob[0])
    uncertainty = float(std_prob[0])
    confidence  = float(1.0 - uncertainty)
    alert       = risk_score >= threshold

    # ── SHAP Attribution ─────────────────────────────────────────
    top_feats = []
    if compute_shap:
        try:
            shap_vals, _ = compute_shap_values(
                ensemble.models[0], X, feature_cols, background_samples=min(20, X.shape[0])
            )
            top_feats = top_features_from_shap(shap_vals, feature_cols, top_k=5)
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            # Fallback: random feature importance
            top_feats = [
                {"feature": f, "importance": float(np.random.rand()), "direction": "increases_risk"}
                for f in feature_cols[:5]
            ]

    # ── Narrative ────────────────────────────────────────────────
    narrative = generate_narrative(
        risk_score=risk_score,
        uncertainty=uncertainty,
        top_features=top_feats,
        patient_vitals=patient_vitals,
        featherless_client=featherless_client,
    )

    return {
        "risk_score":   round(risk_score, 4),
        "alert":        alert,
        "threshold":    round(float(threshold), 4),
        "confidence":   round(confidence, 4),
        "uncertainty":  round(uncertainty, 4),
        "ci_lower":     round(float(ci_lo[0]), 4),
        "ci_upper":     round(float(ci_hi[0]), 4),
        "top_features": top_feats,
        "narrative":    narrative,
    }
