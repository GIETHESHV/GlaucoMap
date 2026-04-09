"""
Pydantic schemas for ML serving input/output.
Shared with Person B (FastAPI backend).
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class VFSession(BaseModel):
    """A single visual field test session."""
    session_id: str
    test_date: str = Field(..., description="ISO 8601 date string")
    eye: str = Field(..., pattern="^[LR]$")
    age: float = Field(..., ge=0, le=120)
    sensitivity_map: List[float] = Field(..., min_length=54, max_length=54)
    md_score: float = Field(default=0.0)
    psd_score: float = Field(default=0.0)
    vfi: float = Field(default=100.0, ge=0, le=100)
    duration_seconds: float = Field(default=0.0, ge=0)
    false_positive_rate: float = Field(default=0.0, ge=0, le=1)
    false_negative_rate: float = Field(default=0.0, ge=0, le=1)
    fixation_loss_rate: float = Field(default=0.0, ge=0, le=1)

    @field_validator("sensitivity_map")
    @classmethod
    def validate_sensitivity(cls, v: List[float]) -> List[float]:
        for val in v:
            if not (0.0 <= val <= 40.0):
                raise ValueError(f"Sensitivity value {val} out of range [0, 40]")
        return v


class PredictionRequest(BaseModel):
    """Input to the predictor: a patient's full session history."""
    patient_id: str
    eye: str = Field(..., pattern="^[LR]$")
    sessions: List[VFSession] = Field(..., min_length=1)


class LocationProgression(BaseModel):
    """Per-location progression result."""
    location_index: int
    progression_rate_db_per_year: float
    uncertainty_sd: float
    sector_id: int
    sector_name: str


class ProgressionResult(BaseModel):
    """Full progression prediction output."""
    patient_id: str
    eye: str
    model_version: str
    alert_level: str = Field(..., pattern="^(none|watch|alert|urgent)$")
    projected_loss_months: Optional[float] = None
    mean_progression_rate: float
    locations: List[LocationProgression]
    progressing_location_indices: List[int]
    suppressed_location_indices: List[int]
    triggered_bundle_sectors: List[int]
    plr_progressed: bool
    model_progressed: bool
    vfi_event_detected: bool
    current_vfi: float
    details: str
