"""
Full end-to-end model pipeline: GNN + Spatiotemporal.

Combines:
  1. RNFB anatomical graph (loaded once)
  2. AnatomicalGNN (per-session embedding)
  3. SpatiotemporalModel (progression rate prediction)
  4. MeasurementNoiseModel (uncertainty calibration)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ml.data.anatomical_graph import load_graph
from ml.models.anatomical_gnn import AnatomicalGNN
from ml.models.spatiotemporal_model import SpatiotemporalModel
from ml.models.measurement_noise_model import MeasurementNoiseModel


class GlaucomaProgressionModel(nn.Module):
    """
    Full pipeline: sensitivity maps → GNN embeddings → GRU → progression rates.

    Args:
        graph_path      : Path to serialized RNFB graph.
        gnn_hidden_dim  : GNN hidden dimension.
        gnn_output_dim  : GNN output embedding dimension.
        gru_hidden_dim  : GRU hidden dimension.
        gru_layers      : Number of GRU layers.
        time_encoding_dim: Time encoding dimension.
        dropout         : Dropout probability.
    """

    def __init__(
        self,
        graph_path: str = "ml/data/rnfb_graph.pt",
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 64,
        gru_hidden_dim: int = 128,
        gru_layers: int = 2,
        time_encoding_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Load anatomical graph
        if Path(graph_path).exists():
            self.graph = load_graph(graph_path)
        else:
            from ml.data.anatomical_graph import build_rnfb_graph
            self.graph = build_rnfb_graph()

        self.register_buffer("edge_index", self.graph.edge_index)
        self.register_buffer("edge_attr", self.graph.edge_attr)
        self.register_buffer("sector_labels", self.graph.sector_labels)

        # GNN: sensitivity [54] → embedding [54, gnn_output_dim]
        # Input: sensitivity + graph node features (10) = 11 total
        # But we'll just use sensitivity (1) + session index (1) = 2 for simplicity
        # Adjust to match actual usage
        self.gnn = AnatomicalGNN(
            node_feature_dim=1,  # just sensitivity per location
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout=dropout,
        )

        # Spatiotemporal model
        self.spatiotemporal = SpatiotemporalModel(
            embed_dim=gnn_output_dim,
            hidden_dim=gru_hidden_dim,
            gru_layers=gru_layers,
            time_encoding_dim=time_encoding_dim,
            num_locations=54,
            dropout=dropout,
        )

        # Measurement noise model (fixed parameters)
        self.noise_model = MeasurementNoiseModel()

    def forward(
        self,
        sensitivity_maps: torch.Tensor,
        time_deltas: torch.Tensor,
        session_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sensitivity_maps : [B, T, 54]  sensitivity values in dB
            time_deltas      : [B, T]      weeks since first session
            session_lengths  : [B]         actual number of sessions per patient

        Returns:
            progression_rates : [B, 54]  predicted dB/year
            uncertainty       : [B, 54]  predicted uncertainty (SD in dB/year)
        """
        B, T, N = sensitivity_maps.shape
        assert N == 54

        # Apply GNN to each session independently
        embeddings = []
        for t in range(T):
            sens_t = sensitivity_maps[:, t, :].unsqueeze(-1)  # [B, 54, 1]
            # Process each batch item through GNN
            batch_embeds = []
            for b in range(B):
                x = sens_t[b]  # [54, 1]
                embed = self.gnn(x, self.edge_index, self.edge_attr)  # [54, embed_dim]
                batch_embeds.append(embed)
            embeddings.append(torch.stack(batch_embeds, dim=0))  # [B, 54, embed_dim]

        embeddings = torch.stack(embeddings, dim=1)  # [B, T, 54, embed_dim]

        # Spatiotemporal model
        progression_rates, log_variance = self.spatiotemporal(
            embeddings, time_deltas, session_lengths
        )

        # Convert log-variance to SD
        uncertainty = log_variance.exp().sqrt()

        return progression_rates, uncertainty

    def predict_with_noise_calibration(
        self,
        sensitivity_maps: torch.Tensor,
        time_deltas: torch.Tensor,
        session_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict progression rates with noise-calibrated uncertainty.

        Returns:
            {
                "progression_rates": [B, 54],
                "uncertainty": [B, 54],
                "measurement_noise": [B, 54],  # expected test-retest SD
                "signal_to_noise": [B, 54],    # |rate| / total_uncertainty
            }
        """
        rates, uncertainty = self.forward(sensitivity_maps, time_deltas, session_lengths)

        # Compute measurement noise from baseline sensitivity
        baseline_sens = sensitivity_maps[:, 0, :]  # [B, 54]
        measurement_noise = self.noise_model(baseline_sens)

        # Total uncertainty = sqrt(model_uncertainty^2 + measurement_noise^2)
        total_uncertainty = torch.sqrt(uncertainty**2 + measurement_noise**2)

        signal_to_noise = torch.abs(rates) / (total_uncertainty + 1e-6)

        return {
            "progression_rates": rates,
            "uncertainty": uncertainty,
            "measurement_noise": measurement_noise,
            "total_uncertainty": total_uncertainty,
            "signal_to_noise": signal_to_noise,
        }
