"""
Dataset ingestion pipeline supporting multiple clinical VF data formats:
  - Rotterdam Optic Nerve Study
  - UK Biobank (UKBB) visual field CSV
  - Moorfields (DICOM metadata + perimetry export)
  - Custom hospital CSV (template: data/templates/hospital_export_template.csv)

All formats are normalized to a common internal record schema.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NUM_VF_LOCATIONS = 54
SENSITIVITY_COLS = [f"s{i}" for i in range(1, NUM_VF_LOCATIONS + 1)]

# Common internal record schema
# {
#   patient_id: str,
#   eye: str,           # "L" or "R"
#   test_date: str,     # ISO 8601
#   age: float,
#   md_score: float,
#   psd_score: float,
#   vfi: float,
#   duration_seconds: float,
#   false_positive_rate: float,
#   false_negative_rate: float,
#   fixation_loss_rate: float,
#   sensitivity_map: List[float],  # 54 values in dB
# }


def _validate_sensitivity(values: List[float]) -> List[float]:
    """Clip sensitivity values to valid dB range [0, 40]."""
    return [float(np.clip(v, 0.0, 40.0)) for v in values]


def _reliability_ok(
    fp: float,
    fn: float,
    fl: float,
    max_fp: float = 0.15,
    max_fn: float = 0.20,
    max_fl: float = 0.20,
) -> bool:
    """Return True if reliability indices are within acceptable limits."""
    return fp <= max_fp and fn <= max_fn and fl <= max_fl


# ---------------------------------------------------------------------------
# Rotterdam Optic Nerve Study loader
# ---------------------------------------------------------------------------

def load_rotterdam(path: str) -> List[Dict[str, Any]]:
    """
    Load Rotterdam Optic Nerve Study CSV.

    Expected columns: PatientID, Eye, ExamDate, Age, MD, PSD, VFI,
    Duration, FP_Rate, FN_Rate, FL_Rate, S1..S54
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {
        "patientid": "patient_id", "eye": "eye", "examdate": "test_date",
        "age": "age", "md": "md_score", "psd": "psd_score", "vfi": "vfi",
        "duration": "duration_seconds", "fp_rate": "false_positive_rate",
        "fn_rate": "false_negative_rate", "fl_rate": "fixation_loss_rate",
    }
    df = df.rename(columns=col_map)

    records = []
    for _, row in df.iterrows():
        sens = [row.get(f"s{i}", np.nan) for i in range(1, 55)]
        if any(np.isnan(v) for v in sens):
            sens = _impute_missing(sens)
        records.append({
            "patient_id": str(row["patient_id"]),
            "eye": str(row["eye"]).upper(),
            "test_date": str(row["test_date"]),
            "age": float(row.get("age", 50.0)),
            "md_score": float(row.get("md_score", 0.0)),
            "psd_score": float(row.get("psd_score", 0.0)),
            "vfi": float(row.get("vfi", 100.0)),
            "duration_seconds": float(row.get("duration_seconds", 0.0)),
            "false_positive_rate": float(row.get("false_positive_rate", 0.0)),
            "false_negative_rate": float(row.get("false_negative_rate", 0.0)),
            "fixation_loss_rate": float(row.get("fixation_loss_rate", 0.0)),
            "sensitivity_map": _validate_sensitivity(sens),
        })
    logger.info("Loaded %d records from Rotterdam dataset: %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# UKBB visual field CSV loader
# ---------------------------------------------------------------------------

def load_ukbb(path: str) -> List[Dict[str, Any]]:
    """
    Load UK Biobank visual field CSV.

    Expected columns: eid, eye_side, date_of_assessment, age_at_recruitment,
    mean_deviation, pattern_sd, vfi, test_duration,
    false_pos_errors, false_neg_errors, fixation_losses,
    threshold_1..threshold_54
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    records = []
    for _, row in df.iterrows():
        sens = [row.get(f"threshold_{i}", np.nan) for i in range(1, 55)]
        if any(np.isnan(v) for v in sens):
            sens = _impute_missing(sens)
        records.append({
            "patient_id": str(row.get("eid", "")),
            "eye": "L" if str(row.get("eye_side", "")).lower() in ("left", "l") else "R",
            "test_date": str(row.get("date_of_assessment", "")),
            "age": float(row.get("age_at_recruitment", 50.0)),
            "md_score": float(row.get("mean_deviation", 0.0)),
            "psd_score": float(row.get("pattern_sd", 0.0)),
            "vfi": float(row.get("vfi", 100.0)),
            "duration_seconds": float(row.get("test_duration", 0.0)),
            "false_positive_rate": float(row.get("false_pos_errors", 0.0)),
            "false_negative_rate": float(row.get("false_neg_errors", 0.0)),
            "fixation_loss_rate": float(row.get("fixation_losses", 0.0)),
            "sensitivity_map": _validate_sensitivity(sens),
        })
    logger.info("Loaded %d records from UKBB dataset: %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Moorfields loader (DICOM metadata + perimetry CSV export)
# ---------------------------------------------------------------------------

def load_moorfields(dicom_meta_path: str, perimetry_csv_path: str) -> List[Dict[str, Any]]:
    """
    Load Moorfields format: DICOM metadata CSV + perimetry export CSV.
    Joins on PatientID + ExamDate.
    """
    try:
        meta = pd.read_csv(dicom_meta_path)
        perim = pd.read_csv(perimetry_csv_path)
        meta.columns = [c.strip().lower() for c in meta.columns]
        perim.columns = [c.strip().lower() for c in perim.columns]
        df = pd.merge(meta, perim, on=["patientid", "examdate"], how="inner")
    except Exception as exc:
        logger.error("Failed to load Moorfields data: %s", exc)
        return []

    records = []
    for _, row in df.iterrows():
        sens = [row.get(f"s{i}", np.nan) for i in range(1, 55)]
        if any(np.isnan(v) for v in sens):
            sens = _impute_missing(sens)
        records.append({
            "patient_id": str(row.get("patientid", "")),
            "eye": str(row.get("eye", "R")).upper(),
            "test_date": str(row.get("examdate", "")),
            "age": float(row.get("age", 50.0)),
            "md_score": float(row.get("md", 0.0)),
            "psd_score": float(row.get("psd", 0.0)),
            "vfi": float(row.get("vfi", 100.0)),
            "duration_seconds": float(row.get("duration", 0.0)),
            "false_positive_rate": float(row.get("fp_rate", 0.0)),
            "false_negative_rate": float(row.get("fn_rate", 0.0)),
            "fixation_loss_rate": float(row.get("fl_rate", 0.0)),
            "sensitivity_map": _validate_sensitivity(sens),
        })
    logger.info("Loaded %d records from Moorfields dataset", len(records))
    return records


# ---------------------------------------------------------------------------
# Custom hospital CSV loader
# ---------------------------------------------------------------------------

def load_hospital_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load custom hospital CSV export.
    Template: data/templates/hospital_export_template.csv
    """
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]

    records = []
    for _, row in df.iterrows():
        sens = [row.get(f"s{i}", np.nan) for i in range(1, 55)]
        if any(np.isnan(v) for v in sens):
            sens = _impute_missing(sens)
        records.append({
            "patient_id": str(row.get("patient_id", "")),
            "eye": str(row.get("eye", "R")).upper(),
            "test_date": str(row.get("test_date", "")),
            "age": float(row.get("age", 50.0)),
            "md_score": float(row.get("md_score", 0.0)),
            "psd_score": float(row.get("psd_score", 0.0)),
            "vfi": float(row.get("vfi", 100.0)),
            "duration_seconds": float(row.get("duration_seconds", 0.0)),
            "false_positive_rate": float(row.get("false_positive_rate", 0.0)),
            "false_negative_rate": float(row.get("false_negative_rate", 0.0)),
            "fixation_loss_rate": float(row.get("fixation_loss_rate", 0.0)),
            "sensitivity_map": _validate_sensitivity(sens),
        })
    logger.info("Loaded %d records from hospital CSV: %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Missing value imputation (spatial kriging approximation)
# ---------------------------------------------------------------------------

def _impute_missing(
    sensitivity: List[float],
    vf_locations: Optional[List[tuple]] = None,
) -> List[float]:
    """
    Impute missing sensitivity values using inverse-distance weighted
    interpolation over the VF grid (kriging approximation).

    Falls back to column mean if all neighbors are also missing.
    """
    from ml.data.anatomical_graph import VF_LOCATIONS as _VF_LOCS

    if vf_locations is None:
        vf_locations = [(x, y) for x, y, _ in _VF_LOCS]

    values = np.array(sensitivity, dtype=float)
    missing_idx = np.where(np.isnan(values))[0]

    if len(missing_idx) == 0:
        return values.tolist()

    known_idx = np.where(~np.isnan(values))[0]
    if len(known_idx) == 0:
        return np.full(NUM_VF_LOCATIONS, 20.0).tolist()  # fallback to 20 dB

    coords = np.array(vf_locations)
    for idx in missing_idx:
        dists = np.sqrt(np.sum((coords[known_idx] - coords[idx]) ** 2, axis=1))
        dists = np.maximum(dists, 1e-6)
        weights = 1.0 / dists
        values[idx] = np.average(values[known_idx], weights=weights)

    return values.tolist()


# ---------------------------------------------------------------------------
# Reliability filtering
# ---------------------------------------------------------------------------

def filter_reliable(
    records: List[Dict[str, Any]],
    max_fp: float = 0.15,
    max_fn: float = 0.20,
    max_fl: float = 0.20,
) -> List[Dict[str, Any]]:
    """Filter out unreliable test sessions based on reliability indices."""
    before = len(records)
    reliable = [
        r for r in records
        if _reliability_ok(
            r["false_positive_rate"],
            r["false_negative_rate"],
            r["fixation_loss_rate"],
            max_fp, max_fn, max_fl,
        )
    ]
    logger.info("Reliability filter: %d → %d records", before, len(reliable))
    return reliable


# ---------------------------------------------------------------------------
# Age-correction of sensitivity values
# ---------------------------------------------------------------------------

def age_correct_sensitivity(
    sensitivity_map: List[float],
    age: float,
    reference_age: float = 50.0,
) -> List[float]:
    """
    Apply age-correction to sensitivity values.
    Correction: +0.07 dB per year below reference, -0.07 dB per year above.
    (Heijl et al. normative approximation)
    """
    correction = 0.07 * (reference_age - age)
    corrected = [np.clip(s + correction, 0.0, 40.0) for s in sensitivity_map]
    return corrected
