"""
Progression Detection.

Method 1: Pointwise Linear Regression (PLR) — industry standard baseline.
Method 2: Spatiotemporal model with anatomical prior (our method).

Both methods output per-location progression rates (dB/year) and a
binary progression flag per patient.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Method 1: Pointwise Linear Regression (PLR)
# ---------------------------------------------------------------------------

def plr_progression_rate(
    sensitivity_series: List[List[float]],
    time_years: List[float],
) -> np.ndarray:
    """
    Compute per-location progression rate via ordinary least squares.

    Args:
        sensitivity_series : List of T sensitivity maps, each [54] dB values.
        time_years         : List of T time points in years from baseline.

    Returns:
        rates : np.ndarray [54]  dB/year (negative = worsening)
    """
    T = len(sensitivity_series)
    assert T == len(time_years), "Mismatch between sessions and time points"
    assert T >= 2, "Need at least 2 sessions for PLR"

    S = np.array(sensitivity_series, dtype=float)  # [T, 54]
    t = np.array(time_years, dtype=float)           # [T]

    t_mean = t.mean()
    s_mean = S.mean(axis=0)                          # [54]

    num = ((t - t_mean)[:, None] * (S - s_mean)).sum(axis=0)  # [54]
    den = ((t - t_mean) ** 2).sum() + 1e-8

    return num / den  # [54] dB/year


def plr_detect_progression(
    sensitivity_series: List[List[float]],
    time_years: List[float],
    threshold_db_per_year: float = -1.0,
    min_locations: int = 3,
) -> Dict[str, Any]:
    """
    PLR-based progression detection.

    A patient is flagged as progressing if >= `min_locations` locations
    have a rate <= `threshold_db_per_year`.

    Returns:
        {
            "rates": [54] dB/year,
            "progressed": bool,
            "progressing_locations": List[int],
            "mean_rate": float,
        }
    """
    rates = plr_progression_rate(sensitivity_series, time_years)
    progressing = np.where(rates <= threshold_db_per_year)[0].tolist()

    return {
        "rates": rates.tolist(),
        "progressed": len(progressing) >= min_locations,
        "progressing_locations": progressing,
        "mean_rate": float(rates.mean()),
        "method": "PLR",
    }


# ---------------------------------------------------------------------------
# Method 2: Spatiotemporal model (our method)
# ---------------------------------------------------------------------------

def model_detect_progression(
    progression_rates: torch.Tensor,
    uncertainty: torch.Tensor,
    sector_labels: torch.Tensor,
    threshold_db_per_year: float = -1.0,
    min_locations: int = 3,
    min_sessions: int = 3,
    uncertainty_suppression_factor: float = 2.0,
    measurement_noise: Optional[torch.Tensor] = None,
    num_sessions: int = 3,
) -> Dict[str, Any]:
    """
    Model-based progression detection with confidence weighting.

    Args:
        progression_rates  : [54]  predicted dB/year
        uncertainty        : [54]  predicted uncertainty SD
        sector_labels      : [54]  RNFB sector per location
        threshold_db_per_year: threshold for flagging a location
        min_locations      : minimum flagged locations to declare progression
        min_sessions       : minimum sessions required (passed in for reporting)
        uncertainty_suppression_factor: suppress if uncertainty > factor * noise
        measurement_noise  : [54]  expected test-retest SD (optional)
        num_sessions       : number of sessions used for prediction

    Returns:
        {
            "rates": [54],
            "uncertainty": [54],
            "progressed": bool,
            "progressing_locations": List[int],
            "suppressed_locations": List[int],
            "mean_rate": float,
            "method": "spatiotemporal",
        }
    """
    rates_np = progression_rates.detach().cpu().numpy()
    unc_np = uncertainty.detach().cpu().numpy()

    # Suppress uncertain locations
    suppressed = []
    if measurement_noise is not None:
        noise_np = measurement_noise.detach().cpu().numpy()
        suppress_mask = unc_np > uncertainty_suppression_factor * noise_np
        suppressed = np.where(suppress_mask)[0].tolist()
    else:
        suppress_mask = np.zeros(54, dtype=bool)

    # Flag progressing locations (not suppressed)
    progressing = [
        i for i in range(54)
        if rates_np[i] <= threshold_db_per_year and not suppress_mask[i]
    ]

    return {
        "rates": rates_np.tolist(),
        "uncertainty": unc_np.tolist(),
        "progressed": len(progressing) >= min_locations and num_sessions >= min_sessions,
        "progressing_locations": progressing,
        "suppressed_locations": suppressed,
        "mean_rate": float(rates_np.mean()),
        "method": "spatiotemporal",
    }


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison_report(
    plr_result: Dict[str, Any],
    model_result: Dict[str, Any],
    patient_id: str,
) -> Dict[str, Any]:
    """
    Generate a side-by-side comparison of PLR vs model-based detection.
    """
    return {
        "patient_id": patient_id,
        "plr": {
            "progressed": plr_result["progressed"],
            "mean_rate_db_per_year": plr_result["mean_rate"],
            "progressing_locations": len(plr_result["progressing_locations"]),
        },
        "model": {
            "progressed": model_result["progressed"],
            "mean_rate_db_per_year": model_result["mean_rate"],
            "progressing_locations": len(model_result["progressing_locations"]),
            "suppressed_locations": len(model_result.get("suppressed_locations", [])),
        },
        "agreement": plr_result["progressed"] == model_result["progressed"],
    }
