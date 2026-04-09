"""
Evaluation metrics for glaucoma progression detection.

Metrics:
  - Sensitivity / specificity at fixed specificity thresholds (80%, 85%, 90%)
  - Time-to-detection (sessions needed vs PLR baseline)
  - Mean Absolute Error on progression rate prediction
  - Brier score for event prediction
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float,
) -> Tuple[float, float]:
    """
    Compute sensitivity at a fixed specificity level.

    Args:
        y_true           : [N] binary labels (1=progressed, 0=stable)
        y_score          : [N] predicted progression probability or score
        target_specificity: e.g. 0.80, 0.85, 0.90

    Returns:
        (sensitivity, threshold) at the target specificity
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificities = 1.0 - fpr

    # Find threshold closest to target specificity
    idx = np.argmin(np.abs(specificities - target_specificity))
    return float(tpr[idx]), float(thresholds[idx])


def compute_detection_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificity_levels: List[float] = [0.80, 0.85, 0.90],
) -> Dict[str, Any]:
    """
    Compute sensitivity/specificity at multiple fixed specificity thresholds.

    Returns:
        {
            "auc": float,
            "sensitivity_at_80_spec": float,
            "sensitivity_at_85_spec": float,
            "sensitivity_at_90_spec": float,
        }
    """
    auc = roc_auc_score(y_true, y_score)
    results: Dict[str, Any] = {"auc": float(auc)}

    for spec in specificity_levels:
        sens, thresh = sensitivity_at_specificity(y_true, y_score, spec)
        key = f"sensitivity_at_{int(spec * 100)}_spec"
        results[key] = sens
        results[f"threshold_at_{int(spec * 100)}_spec"] = thresh

    return results


def mean_absolute_error_rates(
    predicted_rates: np.ndarray,
    true_rates: np.ndarray,
) -> Dict[str, float]:
    """
    MAE on per-location progression rate prediction.

    Args:
        predicted_rates : [N, 54] or [54]
        true_rates      : [N, 54] or [54]

    Returns:
        {"mae": float, "mae_per_location": [54]}
    """
    diff = np.abs(predicted_rates - true_rates)
    return {
        "mae": float(diff.mean()),
        "mae_per_location": diff.mean(axis=0).tolist() if diff.ndim == 2 else diff.tolist(),
    }


def brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Brier score for binary event prediction.
    Lower is better (0 = perfect, 1 = worst).

    Args:
        y_true : [N] binary labels
        y_prob : [N] predicted probabilities

    Returns:
        brier score (float)
    """
    return float(np.mean((y_prob - y_true) ** 2))


def time_to_detection(
    patient_session_labels: List[List[int]],
    patient_session_scores: List[List[float]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute average number of sessions needed to detect progression.

    Args:
        patient_session_labels : List of per-patient session-level labels
                                 (1 = progressed at this session, 0 = not yet)
        patient_session_scores : List of per-patient session-level scores
        threshold              : Score threshold for declaring progression

    Returns:
        {
            "mean_sessions_to_detection": float,
            "median_sessions_to_detection": float,
        }
    """
    sessions_needed = []
    for labels, scores in zip(patient_session_labels, patient_session_scores):
        for i, (label, score) in enumerate(zip(labels, scores)):
            if label == 1 and score >= threshold:
                sessions_needed.append(i + 1)
                break

    if not sessions_needed:
        return {"mean_sessions_to_detection": float("nan"),
                "median_sessions_to_detection": float("nan")}

    arr = np.array(sessions_needed)
    return {
        "mean_sessions_to_detection": float(arr.mean()),
        "median_sessions_to_detection": float(np.median(arr)),
    }


def run_ablation_comparison(
    plr_scores: np.ndarray,
    model_scores: np.ndarray,
    model_no_gnn_scores: np.ndarray,
    model_no_time_scores: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    """
    Run ablation study comparing:
      - PLR baseline
      - Full model (GNN + time encoding)
      - Model without GNN prior
      - Model without time encoding

    Returns a comparison table as a dict.
    """
    results = {}
    for name, scores in [
        ("plr", plr_scores),
        ("model_full", model_scores),
        ("model_no_gnn", model_no_gnn_scores),
        ("model_no_time", model_no_time_scores),
    ]:
        try:
            results[name] = compute_detection_metrics(y_true, scores)
        except Exception as e:
            results[name] = {"error": str(e)}
    return results
