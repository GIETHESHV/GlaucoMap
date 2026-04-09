"""
Pointwise event detection based on Visual Field Index (VFI) threshold.

An "event" is defined as VFI dropping below 85% (configurable).
Used alongside rate-based progression for clinical decision support.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def detect_vfi_events(
    vfi_series: List[float],
    time_years: List[float],
    threshold: float = 85.0,
) -> Dict[str, Any]:
    """
    Detect VFI-based progression events.

    Args:
        vfi_series  : List of VFI values (%) over time.
        time_years  : Corresponding time points in years from baseline.
        threshold   : VFI threshold below which an event is declared (default 85%).

    Returns:
        {
            "event_detected": bool,
            "first_event_index": Optional[int],
            "first_event_time_years": Optional[float],
            "vfi_at_event": Optional[float],
            "current_vfi": float,
            "vfi_trend": float,  # dB/year approximation
        }
    """
    assert len(vfi_series) == len(time_years)

    current_vfi = vfi_series[-1] if vfi_series else 100.0
    event_idx = None
    for i, vfi in enumerate(vfi_series):
        if vfi < threshold:
            event_idx = i
            break

    # Simple linear trend
    vfi_trend = 0.0
    if len(vfi_series) >= 2:
        import numpy as np
        t = np.array(time_years)
        v = np.array(vfi_series)
        t_mean = t.mean()
        v_mean = v.mean()
        num = ((t - t_mean) * (v - v_mean)).sum()
        den = ((t - t_mean) ** 2).sum() + 1e-8
        vfi_trend = float(num / den)

    return {
        "event_detected": event_idx is not None,
        "first_event_index": event_idx,
        "first_event_time_years": time_years[event_idx] if event_idx is not None else None,
        "vfi_at_event": vfi_series[event_idx] if event_idx is not None else None,
        "current_vfi": current_vfi,
        "vfi_trend_per_year": vfi_trend,
    }


def generate_progression_report(
    patient_id: str,
    eye: str,
    plr_result: Dict[str, Any],
    model_result: Dict[str, Any],
    alert_result: Dict[str, Any],
    vfi_event_result: Dict[str, Any],
    model_version: str = "1.0.0",
) -> Dict[str, Any]:
    """
    Assemble a full JSON progression report per patient.
    This is the output format fed to the API and clinician dashboard.
    """
    return {
        "patient_id": patient_id,
        "eye": eye,
        "model_version": model_version,
        "alert_level": alert_result["alert_level"],
        "projected_loss_months": alert_result.get("projected_loss_months"),
        "progression": {
            "plr": {
                "progressed": plr_result["progressed"],
                "mean_rate_db_per_year": plr_result["mean_rate"],
                "progressing_locations": plr_result["progressing_locations"],
            },
            "model": {
                "progressed": model_result["progressed"],
                "mean_rate_db_per_year": model_result["mean_rate"],
                "progression_rates": model_result["rates"],
                "uncertainty": model_result.get("uncertainty", []),
                "progressing_locations": model_result["progressing_locations"],
                "suppressed_locations": model_result.get("suppressed_locations", []),
            },
        },
        "vfi_event": vfi_event_result,
        "alert_details": alert_result.get("details", ""),
        "triggered_bundles": alert_result.get("triggered_bundles", []),
    }
