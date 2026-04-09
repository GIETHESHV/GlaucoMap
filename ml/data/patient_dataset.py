"""
PyTorch Dataset for glaucoma VF progression modeling.

Each item represents a patient's full longitudinal record:
  - Sequence of test sessions (sorted by date)
  - Per-session sensitivity maps, metadata, reliability flags
  - Progression label (for supervised training)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PatientDataset(Dataset):
    """
    Dataset of patient longitudinal VF records.

    Each item:
        patient_id      : str
        eye             : str ("L" or "R")
        session_index   : int (0-based, sorted by date)
        sensitivity_map : Tensor [54]  (dB values)
        age             : float
        md_score        : float
        psd_score       : float
        test_duration   : float (seconds)
        reliability_flags: Dict  {fp, fn, fl rates}
        time_deltas     : Tensor [T]  (weeks between sessions, 0 for first)
        progressed      : int (1 = progressed, 0 = stable) — optional label

    Args:
        records: List of patient dicts, each containing a "sessions" list.
                 Sessions must be pre-sorted by test_date.
        min_sessions: Minimum number of sessions to include a patient.
        max_sessions: Truncate sequences longer than this.
        age_correct: Apply age-correction to sensitivity values.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        min_sessions: int = 3,
        max_sessions: int = 20,
        age_correct: bool = True,
    ) -> None:
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        self.age_correct = age_correct
        self.patients = self._build(records)

    def _build(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and structure patient records."""
        from ml.data.dataset_loader import age_correct_sensitivity

        patients = []
        for rec in records:
            sessions = rec.get("sessions", [])
            if len(sessions) < self.min_sessions:
                continue
            sessions = sessions[: self.max_sessions]

            # Compute time deltas in weeks
            dates = [self._parse_date(s["test_date"]) for s in sessions]
            time_deltas = [0.0]
            for i in range(1, len(dates)):
                delta_days = (dates[i] - dates[0]).days if dates[i] and dates[0] else 0
                time_deltas.append(max(delta_days / 7.0, 0.0))

            session_tensors = []
            for i, s in enumerate(sessions):
                sens = s["sensitivity_map"]
                if self.age_correct:
                    sens = age_correct_sensitivity(sens, s.get("age", 50.0))
                session_tensors.append({
                    "sensitivity_map": torch.tensor(sens, dtype=torch.float),
                    "age": float(s.get("age", 50.0)),
                    "md_score": float(s.get("md_score", 0.0)),
                    "psd_score": float(s.get("psd_score", 0.0)),
                    "test_duration": float(s.get("duration_seconds", 0.0)),
                    "reliability_flags": {
                        "fp": float(s.get("false_positive_rate", 0.0)),
                        "fn": float(s.get("false_negative_rate", 0.0)),
                        "fl": float(s.get("fixation_loss_rate", 0.0)),
                    },
                    "session_index": i,
                })

            patients.append({
                "patient_id": rec.get("patient_id", ""),
                "eye": rec.get("eye", "R"),
                "sessions": session_tensors,
                "time_deltas": torch.tensor(time_deltas, dtype=torch.float),
                "progressed": int(rec.get("progressed", 0)),
                "baseline_md": float(sessions[0].get("md_score", 0.0)),
            })
        return patients

    @staticmethod
    def _parse_date(date_str: str) -> Optional[Any]:
        """Parse ISO date string, return None on failure."""
        try:
            from datetime import date
            return date.fromisoformat(str(date_str)[:10])
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.patients[idx]

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate: pad sequences to the same length within a batch.
        Returns:
            sensitivity_maps : [B, T_max, 54]
            time_deltas      : [B, T_max]
            session_lengths  : [B]  (actual number of sessions per patient)
            md_scores        : [B, T_max]
            progressed       : [B]
        """
        max_t = max(len(p["sessions"]) for p in batch)
        B = len(batch)

        sensitivity_maps = torch.zeros(B, max_t, 54)
        time_deltas = torch.zeros(B, max_t)
        md_scores = torch.zeros(B, max_t)
        session_lengths = torch.zeros(B, dtype=torch.long)
        progressed = torch.zeros(B, dtype=torch.long)

        for b, patient in enumerate(batch):
            T = len(patient["sessions"])
            session_lengths[b] = T
            time_deltas[b, :T] = patient["time_deltas"]
            progressed[b] = patient["progressed"]
            for t, sess in enumerate(patient["sessions"]):
                sensitivity_maps[b, t] = sess["sensitivity_map"]
                md_scores[b, t] = sess["md_score"]

        return {
            "sensitivity_maps": sensitivity_maps,
            "time_deltas": time_deltas,
            "session_lengths": session_lengths,
            "md_scores": md_scores,
            "progressed": progressed,
            "patient_ids": [p["patient_id"] for p in batch],
        }
