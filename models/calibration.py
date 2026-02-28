"""
models/calibration.py
Threshold optimization to achieve target False Discovery Rate (FDR < 5%).
Also includes Platt scaling for probability calibration.
"""

import numpy as np
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def calibrate_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_fdr: float = 0.05,
) -> float:
    """
    Find the lowest threshold such that FDR (FP / (TP+FP)) <= target_fdr
    while maximizing recall.

    Returns the optimal threshold value.
    """
    thresholds = np.linspace(0.01, 0.99, 500)
    best_threshold = 0.5
    best_recall = 0.0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        fdr = fp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        if fdr <= target_fdr and recall > best_recall:
            best_recall = recall
            best_threshold = t

    tp = ((probs >= best_threshold) & (labels == 1)).sum()
    fp = ((probs >= best_threshold) & (labels == 0)).sum()
    fn = ((probs < best_threshold) & (labels == 1)).sum()
    fdr = fp / (tp + fp + 1e-8)

    logger.info(
        f"Threshold: {best_threshold:.3f} | FDR: {fdr:.4f} | "
        f"Recall: {best_recall:.4f} | TP={tp}, FP={fp}, FN={fn}"
    )
    return float(best_threshold)


def platt_calibrate(
    probs: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    """
    Fit a Platt scaling model (logistic regression on raw probs).
    Returns the fitted calibrator.
    """
    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(probs.reshape(-1, 1), labels)
    logger.info("Platt calibration fitted.")
    return calibrator


def apply_calibration(
    probs: np.ndarray,
    calibrator: LogisticRegression = None,
) -> np.ndarray:
    """Apply Platt calibration if available, otherwise return raw probs."""
    if calibrator is None:
        return probs
    return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]


def compute_uncertainty(
    probs_per_model: np.ndarray,
) -> tuple:
    """
    probs_per_model: (n_models, n_samples)
    Returns mean_prob, std_prob (epistemic uncertainty), CI_lower, CI_upper
    """
    mean_p = probs_per_model.mean(axis=0)
    std_p  = probs_per_model.std(axis=0)
    ci_lo  = np.clip(mean_p - 2 * std_p, 0, 1)
    ci_hi  = np.clip(mean_p + 2 * std_p, 0, 1)
    return mean_p, std_p, ci_lo, ci_hi
