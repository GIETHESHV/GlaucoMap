"""
Stratified train/val/test split utility.

Stratifies by:
  - Progression status (progressed / stable)
  - Age group (< 50, 50-65, > 65)
  - Baseline severity (MD score quartiles)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def _stratify_key(patient: Dict[str, Any]) -> str:
    """Build a stratification key from progression, age group, and severity."""
    progressed = int(patient.get("progressed", 0))
    age = float(patient.get("sessions", [{}])[0].get("age", 50.0))
    md = float(patient.get("baseline_md", 0.0))

    age_group = 0 if age < 50 else (1 if age <= 65 else 2)
    severity = 0 if md >= -6 else (1 if md >= -12 else 2)

    return f"{progressed}_{age_group}_{severity}"


def split_patients(
    patients: List[Dict[str, Any]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split patient list into train / val / test sets.

    Args:
        patients: List of patient dicts (from PatientDataset).
        train_frac: Fraction for training set.
        val_frac: Fraction for validation set (remainder goes to test).
        seed: Random seed for reproducibility.

    Returns:
        (train_patients, val_patients, test_patients)
    """
    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 0, "train_frac + val_frac must be < 1.0"

    labels = [_stratify_key(p) for p in patients]

    # First split: train vs (val + test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(val_frac + test_frac), random_state=seed
    )
    train_idx, valtest_idx = next(sss1.split(patients, labels))

    valtest_patients = [patients[i] for i in valtest_idx]
    valtest_labels = [labels[i] for i in valtest_idx]

    # Second split: val vs test
    val_ratio = val_frac / (val_frac + test_frac)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1.0 - val_ratio), random_state=seed
    )
    val_idx, test_idx = next(sss2.split(valtest_patients, valtest_labels))

    train = [patients[i] for i in train_idx]
    val = [valtest_patients[i] for i in val_idx]
    test = [valtest_patients[i] for i in test_idx]

    logger.info(
        "Split: %d train / %d val / %d test (total %d)",
        len(train), len(val), len(test), len(patients),
    )
    return train, val, test
