"""
Spatiotemporal Progression Model.

Takes a sequence of GNN embeddings across T test sessions per patient
and predicts per-location progression rate (dB/year) + uncertainty.

Architecture:
  Input  : GNN embeddings [T, 54, embed_dim] + time deltas [T]
  Time   : Sinusoidal time-delta encoding [T, time_encoding_dim]
  GRU    : Over session embeddings (handles irregular time intervals)
  Output : [54, 2]  — mean progression rate + log-variance per location
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEncoding(nn.Module):
    """
    Encodes time deltas (weeks between sessions) as sinusoidal features.
    Handles irregular time intervals naturally.

    Args:
        encoding_dim: Dimension of the time encoding vector (must be even).
    """

    def __init__(self, encoding_dim: int = 32) -> None:
        super().__init__()
        assert encoding_dim % 2 == 0, "encoding_dim must be even"
        self.encoding_dim = encoding_dim
        # Precompute frequency bands
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, encoding_dim, 2).float() / encoding_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_deltas : [T] or [B, T]  weeks since first session

        Returns:
            encoding : [T, encoding_dim] or [B, T, encoding_dim]
        """
        t = time_deltas.unsqueeze(-1)  # [..., 1]
        args = t * self.freqs  # [..., encoding_dim/2]
        encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return encoding


class SpatiotemporalModel(nn.Module):
    """
    GRU-based spatiotemporal model over session embeddings.

    Args:
        embed_dim       : GNN output embedding dimension per location.
        hidden_dim      : GRU hidden state dimension.
        gru_layers      : Number of GRU layers.
        time_encoding_dim: Sinusoidal time encoding dimension.
        num_locations   : Number of VF test locations (54).
        dropout         : Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        time_encoding_dim: int = 32,
        num_locations: int = 54,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_locations = num_locations
        self.embed_dim = embed_dim

        self.time_encoder = SinusoidalTimeEncoding(time_encoding_dim)

        # Project [embed_dim + time_encoding_dim] → hidden_dim per location
        gru_input_dim = embed_dim + time_encoding_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Output head: hidden_dim → 2 (mean + log_var) per location
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # [mean, log_var]
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        time_deltas: torch.Tensor,
        session_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings      : [B, T, 54, embed_dim]  GNN embeddings per session
            time_deltas     : [B, T]                  weeks since first session
            session_lengths : [B]                     actual number of sessions

        Returns:
            progression_rates : [B, 54]  predicted dB/year (mean)
            log_variance      : [B, 54]  log-variance (uncertainty)
        """
        B, T, N, D = embeddings.shape
        assert N == self.num_locations

        # Time encoding: [B, T, time_encoding_dim]
        time_enc = self.time_encoder(time_deltas)  # [B, T, time_dim]

        # Expand time encoding to all locations: [B, T, N, time_dim]
        time_enc_expanded = time_enc.unsqueeze(2).expand(B, T, N, -1)

        # Concatenate embeddings + time: [B, T, N, embed_dim + time_dim]
        x = torch.cat([embeddings, time_enc_expanded], dim=-1)

        # Reshape for GRU: process each location independently
        # [B*N, T, embed_dim + time_dim]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)

        # Pack padded sequences for efficiency
        lengths_expanded = session_lengths.unsqueeze(1).expand(B, N).reshape(B * N)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_expanded.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Take the last valid hidden state per sequence
        # out: [B*N, T, hidden_dim]
        idx = (lengths_expanded - 1).clamp(min=0).long()
        last_hidden = out[torch.arange(B * N), idx]  # [B*N, hidden_dim]

        last_hidden = self.dropout(last_hidden)

        # Output: [B*N, 2]
        output = self.output_head(last_hidden)

        # Reshape: [B, N, 2]
        output = output.reshape(B, N, 2)

        progression_rates = output[:, :, 0]  # [B, 54]
        log_variance = output[:, :, 1]       # [B, 54]

        return progression_rates, log_variance
