"""
Anatomical loss functions for the GNN prior.

1. Bundle coherence loss: penalize if two nodes in the same RNFB bundle
   deviate by > 3 dB in predicted progression rate.

2. Asymmetry prior: nasal step patterns should have sharp gradients
   (nasal sector vs adjacent sectors).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BundleCoherenceLoss(nn.Module):
    """
    Penalizes predictions where nodes in the same RNFB bundle deviate
    by more than `threshold_db` dB/year in progression rate.

    Loss = mean(max(0, |rate_i - rate_j| - threshold) ^ 2)
           for all (i, j) pairs in the same bundle.

    Args:
        sector_labels : LongTensor [54]  sector assignment per node
        threshold_db  : float  tolerance before penalty kicks in (default 3.0)
        weight        : float  loss weight
    """

    def __init__(
        self,
        sector_labels: torch.Tensor,
        threshold_db: float = 3.0,
        weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("sector_labels", sector_labels)
        self.threshold_db = threshold_db
        self.weight = weight

        # Precompute same-sector pairs
        pairs = []
        for s in range(6):
            nodes = (sector_labels == s).nonzero(as_tuple=True)[0].tolist()
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pairs.append((nodes[i], nodes[j]))
        if pairs:
            idx = torch.tensor(pairs, dtype=torch.long)
            self.register_buffer("pair_i", idx[:, 0])
            self.register_buffer("pair_j", idx[:, 1])
        else:
            self.register_buffer("pair_i", torch.zeros(0, dtype=torch.long))
            self.register_buffer("pair_j", torch.zeros(0, dtype=torch.long))

    def forward(self, progression_rates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            progression_rates : [54] or [B, 54]  predicted dB/year per location

        Returns:
            scalar loss
        """
        if self.pair_i.numel() == 0:
            return torch.tensor(0.0, device=progression_rates.device)

        if progression_rates.dim() == 1:
            rates_i = progression_rates[self.pair_i]
            rates_j = progression_rates[self.pair_j]
        else:
            rates_i = progression_rates[:, self.pair_i]
            rates_j = progression_rates[:, self.pair_j]

        deviation = torch.abs(rates_i - rates_j)
        penalty = torch.clamp(deviation - self.threshold_db, min=0.0) ** 2
        return self.weight * penalty.mean()


class AsymmetryPriorLoss(nn.Module):
    """
    Encourages sharp gradients at the nasal step boundary.
    Nasal sector (2) should differ significantly from adjacent sectors (1, 3)
    when a nasal step pattern is present.

    Loss = -mean(|rate_nasal - rate_adjacent|) * step_indicator
    (negative because we want to maximize the gradient at the step)

    In practice used as a regularizer to prevent over-smoothing at the
    nasal horizontal meridian.

    Args:
        sector_labels : LongTensor [54]
        weight        : float
    """

    def __init__(self, sector_labels: torch.Tensor, weight: float = 0.05) -> None:
        super().__init__()
        self.register_buffer("sector_labels", sector_labels)
        self.weight = weight

        nasal_nodes = (sector_labels == 2).nonzero(as_tuple=True)[0]
        adjacent_nodes = (
            (sector_labels == 1) | (sector_labels == 3)
        ).nonzero(as_tuple=True)[0]
        self.register_buffer("nasal_nodes", nasal_nodes)
        self.register_buffer("adjacent_nodes", adjacent_nodes)

    def forward(self, progression_rates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            progression_rates : [54] or [B, 54]

        Returns:
            scalar loss (encourages gradient at nasal step)
        """
        if self.nasal_nodes.numel() == 0 or self.adjacent_nodes.numel() == 0:
            return torch.tensor(0.0, device=progression_rates.device)

        if progression_rates.dim() == 1:
            nasal_mean = progression_rates[self.nasal_nodes].mean()
            adj_mean = progression_rates[self.adjacent_nodes].mean()
        else:
            nasal_mean = progression_rates[:, self.nasal_nodes].mean(dim=1)
            adj_mean = progression_rates[:, self.adjacent_nodes].mean(dim=1)

        # Penalize if nasal and adjacent are too similar (over-smoothed)
        gradient = torch.abs(nasal_mean - adj_mean)
        loss = self.weight * torch.clamp(1.0 - gradient, min=0.0).mean()
        return loss
