"""
Clean inference interface for the GlaucomaProgressionModel.

Input  : List of VF test sessions (JSON / PredictionRequest)
Output : ProgressionResult (progression rates, uncertainty, alert, timeline)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ml.serving.schema import PredictionRequest, ProgressionResult, LocationProgression
from ml.data.anatomical_graph import build_rnfb_graph, SECTORS
from ml.detection.progression_detector import plr_detect_progression, model_detect_progression
from ml.detection.alert_engine import AlertEngine
from ml.detection.event_analysis import detect_vfi_events, generate_progression_report

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"


class GlaucomaPredictor:
    """
    Inference wrapper for the full GlaucomaProgressionModel.

    Args:
        model_path  : Path to trained model checkpoint (.pt) or TorchScript (.torchscript.pt).
        config_path : Path to base_config.yaml.
        device      : "cpu" or "cuda".
    """

    def __init__(
        self,
        model_path: str = "ml/checkpoints/best_model.pt",
        config_path: str = "ml/config/base_config.yaml",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.graph = build_rnfb_graph()
        self.sector_labels = self.graph.sector_labels

        self.alert_engine = AlertEngine(sector_labels=self.sector_labels)

        self.model = self._load_model(model_path, config_path)
        self.model.eval()
        logger.info("GlaucomaPredictor ready (device=%s)", device)

    def _load_model(self, model_path: str, config_path: str) -> torch.nn.Module:
        """Load model from checkpoint or TorchScript."""
        if not Path(model_path).exists():
            logger.warning("Model checkpoint not found at %s, using untrained model", model_path)
            from ml.models.full_pipeline import GlaucomaProgressionModel
            return GlaucomaProgressionModel().to(self.device)

        if model_path.endswith(".torchscript.pt"):
            model = torch.jit.load(model_path, map_location=self.device)
        else:
            import yaml
            from ml.models.full_pipeline import GlaucomaProgressionModel
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            model = GlaucomaProgressionModel(
                graph_path=cfg["paths"]["graph"],
                gnn_hidden_dim=cfg["model"]["gnn"]["hidden_dim"],
                gnn_output_dim=cfg["model"]["gnn"]["output_dim"],
                gru_hidden_dim=cfg["model"]["spatiotemporal"]["hidden_dim"],
                gru_layers=cfg["model"]["spatiotemporal"]["gru_layers"],
                time_encoding_dim=cfg["model"]["spatiotemporal"]["time_encoding_dim"],
                dropout=0.0,
            )
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])

        return model.to(self.device)

    def predict(self, request: PredictionRequest) -> ProgressionResult:
        """
        Run full progression analysis for a patient.

        Args:
            request : PredictionRequest with patient_id, eye, sessions list.

        Returns:
            ProgressionResult with rates, uncertainty, alert, timeline.
        """
        t0 = time.perf_counter()

        sessions = sorted(request.sessions, key=lambda s: s.test_date)
        T = len(sessions)

        # Build tensors
        sens_list = [s.sensitivity_map for s in sessions]
        sens_tensor = torch.tensor(sens_list, dtype=torch.float).unsqueeze(0).to(self.device)  # [1, T, 54]

        # Time deltas in weeks
        from datetime import date
        base_date = date.fromisoformat(sessions[0].test_date[:10])
        time_deltas = []
        time_years = []
        for s in sessions:
            d = date.fromisoformat(s.test_date[:10])
            weeks = max((d - base_date).days / 7.0, 0.0)
            time_deltas.append(weeks)
            time_years.append(weeks / 52.0)

        time_tensor = torch.tensor([time_deltas], dtype=torch.float).to(self.device)  # [1, T]
        lengths_tensor = torch.tensor([T], dtype=torch.long).to(self.device)

        # Model inference
        with torch.no_grad():
            result = self.model.predict_with_noise_calibration(
                sens_tensor, time_tensor, lengths_tensor
            )

        rates = result["progression_rates"][0].cpu().numpy()       # [54]
        uncertainty = result["uncertainty"][0].cpu().numpy()       # [54]
        noise = result["measurement_noise"][0].cpu().numpy()       # [54]
        baseline_sens = np.array(sens_list[0])

        # PLR detection
        plr_result = plr_detect_progression(sens_list, time_years)

        # Model-based detection
        model_result = model_detect_progression(
            torch.tensor(rates),
            torch.tensor(uncertainty),
            self.sector_labels,
            measurement_noise=torch.tensor(noise),
            num_sessions=T,
        )

        # Alert engine
        alert_result = self.alert_engine.evaluate(
            rates, uncertainty, baseline_sens,
            num_sessions=T, measurement_noise=noise,
        )

        # VFI event detection
        vfi_series = [s.vfi for s in sessions]
        vfi_result = detect_vfi_events(vfi_series, time_years)

        # Build location-level results
        location_results = []
        for i in range(54):
            sector_id = int(self.sector_labels[i].item())
            location_results.append(LocationProgression(
                location_index=i,
                progression_rate_db_per_year=float(rates[i]),
                uncertainty_sd=float(uncertainty[i]),
                sector_id=sector_id,
                sector_name=SECTORS[sector_id],
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Inference completed in %.1f ms for patient %s", elapsed_ms, request.patient_id)
        if elapsed_ms > 200:
            logger.warning("Inference time %.1f ms exceeds 200ms target", elapsed_ms)

        return ProgressionResult(
            patient_id=request.patient_id,
            eye=request.eye,
            model_version=MODEL_VERSION,
            alert_level=alert_result["alert_level"],
            projected_loss_months=alert_result.get("projected_loss_months"),
            mean_progression_rate=float(rates.mean()),
            locations=location_results,
            progressing_location_indices=model_result["progressing_locations"],
            suppressed_location_indices=model_result.get("suppressed_locations", []),
            triggered_bundle_sectors=alert_result.get("triggered_bundles", []),
            plr_progressed=plr_result["progressed"],
            model_progressed=model_result["progressed"],
            vfi_event_detected=vfi_result["event_detected"],
            current_vfi=vfi_result["current_vfi"],
            details=alert_result.get("details", ""),
        )

    def predict_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience wrapper accepting raw dict input, returns dict output."""
        request = PredictionRequest(**data)
        result = self.predict(request)
        return result.model_dump()
