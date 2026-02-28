"""
explainability/explain.py
SHAP-based feature importance + Featherless AI narrative generation.
"""

import os
import json
import pickle
import numpy as np
import torch
import shap
import requests
from typing import List, Tuple

ARTIFACT_DIR    = "artifacts"
FEATHERLESS_URL = os.getenv("FEATHERLESS_API_URL", "https://api.featherless.ai/v1")
FEATHERLESS_KEY = os.getenv("FEATHERLESS_API_KEY", "")
FEATHERLESS_MODEL = os.getenv("FEATHERLESS_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


# ── SHAP Explainer ────────────────────────────────────────────────────────────

class AegisSHAPExplainer:
    """
    Wraps the AegisLSTM for SHAP DeepExplainer.
    Reshapes sequences to 2D for feature-level attribution.
    """

    def __init__(self, model, background_X: np.ndarray, feature_cols: List[str],
                 device: str = "cpu"):
        self.model       = model
        self.device      = device
        self.feature_cols = feature_cols

        # Use 100 random background samples
        idx = np.random.choice(len(background_X), min(100, len(background_X)), replace=False)
        bg  = torch.from_numpy(background_X[idx]).to(device)

        # Wrapper: model(x) → scalar prob
        def model_fn(x: torch.Tensor) -> torch.Tensor:
            prob, _ = model(x)
            return prob.unsqueeze(-1)

        self.explainer = shap.DeepExplainer(model_fn, bg)

    def explain(self, X_window: np.ndarray, top_k: int = 8) -> Tuple[List[str], List[float]]:
        """
        X_window: (1, seq_len, n_features) numpy array for one patient window.
        Returns top_k feature names and their mean |SHAP| values.
        """
        x_tensor = torch.from_numpy(X_window).to(self.device)
        shap_vals = self.explainer.shap_values(x_tensor)  # (1, seq_len, n_features)

        # Aggregate over time: mean absolute SHAP per feature
        shap_arr  = np.array(shap_vals).squeeze()         # (seq_len, n_features)
        mean_shap = np.abs(shap_arr).mean(axis=0)         # (n_features,)

        top_idx = np.argsort(mean_shap)[::-1][:top_k]
        top_features = [self.feature_cols[i] for i in top_idx]
        top_values   = [float(mean_shap[i]) for i in top_idx]
        return top_features, top_values


def get_shap_explainer(model, X_train: np.ndarray, feature_cols: List[str],
                       device: str = "cpu") -> AegisSHAPExplainer:
    return AegisSHAPExplainer(model, X_train, feature_cols, device)


# ── Featherless AI Narrative Generator ───────────────────────────────────────

SYSTEM_PROMPT = """You are a critical care AI assistant embedded in an ICU early warning system.
Your role is to generate a concise clinical narrative for the attending physician.
Be factual, precise, and use ICU terminology. Maximum 3 sentences.
Always end with a specific, actionable recommendation."""


def generate_ai_narrative(
    risk_score: float,
    top_features: List[str],
    shap_values: List[float],
    vitals_snapshot: dict,
) -> str:
    """
    Calls Featherless AI to generate a clinical narrative.
    Falls back to a template-based narrative if API is unavailable.
    """

    # Build prompt
    feature_lines = "\n".join(
        f"  - {feat}: SHAP={val:.3f}" for feat, val in zip(top_features[:5], shap_values[:5])
    )
    vitals_lines = "\n".join(f"  {k}: {v}" for k, v in vitals_snapshot.items())

    user_prompt = f"""
Patient risk score: {risk_score:.1%} (predicted septic shock within 6 hours)

Top driving features (SHAP attribution):
{feature_lines}

Current vitals snapshot:
{vitals_lines}

Generate a clinical ICU narrative for the attending physician.
"""

    if not FEATHERLESS_KEY:
        return _template_narrative(risk_score, top_features, vitals_snapshot)

    try:
        response = requests.post(
            f"{FEATHERLESS_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {FEATHERLESS_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model": FEATHERLESS_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.3,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"[Featherless] API call failed ({e}), using template fallback.")
        return _template_narrative(risk_score, top_features, vitals_snapshot)


def _template_narrative(risk_score: float, top_features: List[str],
                         vitals_snapshot: dict) -> str:
    """Deterministic fallback narrative when API is unavailable."""
    tier = "HIGH" if risk_score >= 0.65 else "MODERATE" if risk_score >= 0.40 else "LOW"
    top3 = ", ".join(top_features[:3]) if top_features else "compound physiologic changes"

    lactate = vitals_snapshot.get("Lactate", None)
    map_val = vitals_snapshot.get("MAP",     None)

    clinical_note = ""
    if lactate and float(lactate) > 2.0:
        clinical_note += f" Elevated lactate ({lactate} mmol/L) suggests tissue hypoperfusion."
    if map_val and float(map_val) < 65:
        clinical_note += f" MAP below goal ({map_val} mmHg) warrants vasopressor review."

    return (
        f"Aegis-Omni assigns {tier} risk ({risk_score:.1%}) for septic shock within 6 hours. "
        f"Primary contributing factors: {top3}.{clinical_note} "
        f"Recommend immediate bedside assessment, blood cultures, and lactate re-check."
    )
