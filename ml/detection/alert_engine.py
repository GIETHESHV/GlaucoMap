"""
Alert Engine.

Generates clinical alerts based on progression detection results.

Alert levels:
  none   — no significant progression
  watch  — borderline, monitor closely
  alert  — significant progression detected
  urgent — rapid progression, imminent vision loss risk
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _months_to_threshold(
    current_sensitivity: float,
    rate_db_per_year: float,
    threshold_fraction: float = 0.10,
) -> Optional[float]:
    """Estimate months until sensitivity drops by threshold_fraction (default 10%)."""
    if rate_db_per_year >= 0:
        return None
    loss_needed = current_sensitivity * threshold_fraction
    years_needed = loss_needed / abs(rate_db_per_year)
    return years_needed * 12.0


class AlertEngine:
    """
    Generates clinical alerts from progression detection results.

    Args:
        sector_labels           : LongTensor [54]
        threshold_db_per_year   : Rate threshold for flagging (default -1.0)
        min_bundle_locations    : Min locations in same bundle (default 3)
        min_sessions            : Min sessions required (default 3)
        uncertainty_suppression : Suppress if uncertainty > factor * noise
        urgent_months_threshold : Months-to-loss threshold for urgent alert
        watch_months_threshold  : Months-to-loss threshold for watch alert
    """

    def __init__(
        self,
        sector_labels: torch.Tensor,
        threshold_db_per_year: float = -1.0,
        min_bundle_locations: int = 3,
        min_sessions: int = 3,
        uncertainty_suppression: float = 2.0,
        urgent_months_threshold: float = 12.0,
        watch_months_threshold: float = 36.0,
    ) -> None:
        self.sector_labels = sector_labels.cpu().numpy()
        self.threshold = threshold_db_per_year
        self.min_bundle_locations = min_bundle_locations
        self.min_sessions = min_sessions
        self.uncertainty_suppression = uncertainty_suppression
        self.urgent_months = urgent_months_threshold
        self.watch_months = watch_months_threshold

    def _check_bundle_coherence(self, progressing_locations: List[int]) -> List[int]:
        """Return list of sector IDs where >= min_bundle_locations are progressing."""
        triggered = []
        for sector in range(6):
            sector_nodes = np.where(self.sector_labels == sector)[0]
            count = sum(1 for loc in progressing_locations if loc in sector_nodes)
            if count >= self.min_bundle_locations:
                triggered.append(sector)
        return triggered

    def _no_alert(self, details: str) -> Dict[str, Any]:
        return {
            "alert_level": "none",
            "triggered_bundles": [],
            "progressing_locations": [],
            "projected_loss_months": None,
            "suppressed_locations": [],
            "details": details,
        }

    def evaluate(
        self,
        progression_rates: np.ndarray,
        uncertainty: np.ndarray,
        baseline_sensitivity: np.ndarray,
        num_sessions: int,
        measurement_noise: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate alert level for a patient.

        Args:
            progression_rates    : [54] dB/year
            uncertainty          : [54] SD in dB/year
            baseline_sensitivity : [54] dB at first session
            num_sessions         : number of sessions used
            measurement_noise    : [54] expected test-retest SD (optional)

        Returns dict with alert_level, triggered_bundles, progressing_locations,
        projected_loss_months, suppressed_locations, details.
        """
        if num_sessions < self.min_sessions:
            return self._no_alert("Insufficient sessions for reliable detection")

        suppress_mask = np.zeros(54, dtype=bool)
        if measurement_noise is not None:
            suppress_mask = uncertainty > self.uncertainty_suppression * measurement_noise

        suppressed = np.where(suppress_mask)[0].tolist()

        progressing = [
            i for i in range(54)
            if progression_rates[i] <= self.threshold and not suppress_mask[i]
        ]

        triggered_bundles = self._check_bundle_coherence(progressing)

        if not triggered_bundles:
            if len(progressing) > 0:
                return {
                    "alert_level": "watch",
                    "triggered_bundles": [],
                    "progressing_locations": progressing,
                    "projected_loss_months": None,
                    "suppressed_locations": suppressed,
                    "details": f"{len(progressing)} locations progressing, no bundle coherence",
                }
            return self._no_alert("No significant progression detected")

        progressing_rates = progression_rates[progressing]
        progressing_sens = baseline_sensitivity[progressing]
        projections = [
            _months_to_threshold(s, r)
            for s, r in zip(progressing_sens, progressing_rates)
            if r < 0
        ]
        projections = [p for p in projections if p is not None]
        min_months = min(projections) if projections else None

        if min_months is not None and min_months <= self.urgent_months:
            level = "urgent"
        else:
            level = "alert"

        return {
            "alert_level": level,
            "triggered_bundles": triggered_bundles,
            "progressing_locations": progressing,
            "projected_loss_months": min_months,
            "suppressed_locations": suppressed,
            "details": (
                f"Bundle-coherent progression in sectors {triggered_bundles}. "
                f"Projected loss in {min_months:.1f} months." if min_months else
                f"Bundle-coherent progression in sectors {triggered_bundles}."
            ),
        }
