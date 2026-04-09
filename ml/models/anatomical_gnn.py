"""
GNN Anatomical Prior Layer.

3-layer Graph Attention Network (GAT) using the RNFB anatomical graph.
Produces anatomically-smoothed latent embeddings per VF test session.

Architecture:
  Input  : [54, node_feature_dim]  (sensitivity + spatial encoding)
  Layer 1: GATConv(node_feature_dim → hidden_dim, heads=4)
  Layer 2: GATConv(hidden_dim*4 → hidden_dim, heads=4)
  Layer 3: GATConv(hidden_dim*4 → output_dim, heads=1)
  Skip   : Linear projection of input added to final output
  Output : [54, output_dim]  anatomically-smoothed embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class AnatomicalGNN(nn.Module):
    """
    Graph Attention Network with anatomical edge weights as attention bias.

    Args:
        node_feature_dim: Input node feature dimension (default 54 + 10 = 64
                          when sensitivity is concatenated with graph features).
        hidden_dim: Hidden layer dimension.
        output_dim: Output embedding dimension.
        heads: Number of attention heads.
        dropout: Dropout probability.
        skip_connections: Whether to add skip connection from input to output.
    """

    def __init__(
        self,
        node_feature_dim: int = 55,  # 54 sensitivity + 1 session index
        hidden_dim: int = 64,
        output_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
        skip_connections: bool = True,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.skip_connections = skip_connections

        self.conv1 = GATConv(
            node_feature_dim, hidden_dim, heads=heads,
            dropout=dropout, edge_dim=1, concat=True,
        )
        self.conv2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=heads,
            dropout=dropout, edge_dim=1, concat=True,
        )
        self.conv3 = GATConv(
            hidden_dim * heads, output_dim, heads=1,
            dropout=dropout, edge_dim=1, concat=False,
        )

        if skip_connections:
            self.skip_proj = nn.Linear(node_feature_dim, output_dim)

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim * heads)
        self.norm3 = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x          : [N, node_feature_dim]  node features
            edge_index : [2, E]                 edge connectivity
            edge_attr  : [E, 1]                 anatomical edge weights

        Returns:
            out : [N, output_dim]  anatomically-smoothed embeddings
        """
        residual = x

        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, edge_index, edge_attr=edge_attr)
        h = self.norm3(h)

        if self.skip_connections:
            h = h + self.skip_proj(residual)

        return h  # [N, output_dim]
