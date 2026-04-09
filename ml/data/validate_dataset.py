"""
Dataset validation script.

Checks:
  - Sensitivity array length (must be 54)
  - Sensitivity value range (0–40 dB)
  - Reliability index ranges
  - Minimum sessions per patient
  - Date ordering within patient sessions
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def validate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a list of session records.

    Returns a report dict with:
        total, valid, errors (list of {record_index, field, message})
    """
    errors = []

    for i, rec in enumerate(records):
        sens = rec.get("sensitivity_map", [])
        if len(sens) != 54:
            errors.append({"index": i, "field": "sensitivity_map",
                           "message": f"Expected 54 values, got {len(sens)}"})

        for j, v in enumerate(sens):
            if not (0.0 <= v <= 40.0):
                errors.append({"index": i, "field": f"s{j+1}",
                               "message": f"Value {v} out of range [0, 40]"})

        for field, limit in [
            ("false_positive_rate", 1.0),
            ("false_negative_rate", 1.0),
            ("fixation_loss_rate", 1.0),
        ]:
            val = rec.get(field, 0.0)
            if not (0.0 <= val <= limit):
                errors.append({"index": i, "field": field,
                               "message": f"Value {val} out of range [0, {limit}]"})

        eye = rec.get("eye", "")
        if eye not in ("L", "R"):
            errors.append({"index": i, "field": "eye",
                           "message": f"Invalid eye value: '{eye}'"})

    report = {
        "total": len(records),
        "valid": len(records) - len({e["index"] for e in errors}),
        "error_count": len(errors),
        "errors": errors,
    }
    if errors:
        logger.warning("Validation found %d issues in %d records", len(errors), len(records))
    else:
        logger.info("All %d records passed validation", len(records))
    return report


def validate_patient_sessions(patients: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate patient-level records (grouped sessions).
    Checks date ordering and minimum session count.
    """
    errors = []
    for i, patient in enumerate(patients):
        sessions = patient.get("sessions", [])
        if len(sessions) < 1:
            errors.append({"patient_index": i, "message": "No sessions found"})
            continue

        dates = []
        for s in sessions:
            try:
                from datetime import date
                dates.append(date.fromisoformat(str(s.get("test_date", ""))[:10]))
            except Exception:
                dates.append(None)

        valid_dates = [d for d in dates if d is not None]
        if valid_dates != sorted(valid_dates):
            errors.append({"patient_index": i, "message": "Sessions not in chronological order"})

    return {
        "total_patients": len(patients),
        "error_count": len(errors),
        "errors": errors,
    }


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <records.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    report = validate_records(data)
    print(json.dumps(report, indent=2))
