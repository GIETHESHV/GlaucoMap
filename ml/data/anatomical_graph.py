"""
Retinal Nerve Fiber Bundle (RNFB) Graph Construction.

Encodes the Garway-Heath map: 54 standard Humphrey 24-2 VF test locations
mapped to 6 RNFB sectors. Builds a PyTorch Geometric graph where:
  - Nodes = 54 VF test locations
  - Edges = anatomical connectivity (same bundle = strong, adjacent = weak)
  - Node features: eccentricity, angular position, age-normalized expected sensitivity
  - Edge weights: anatomical co-vulnerability score

References:
  Garway-Heath et al. (2000). Ophthalmology 107(10):1809-1815.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Sector definitions (Garway-Heath map)
# ---------------------------------------------------------------------------

SECTORS = {
    0: "superotemporal",
    1: "superonasal",
    2: "nasal",
    3: "inferonasal",
    4: "inferotemporal",
    5: "temporal",
}

# Adjacency between sectors (undirected)
SECTOR_ADJACENCY: List[Tuple[int, int]] = [
    (0, 1), (0, 5),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]

# ---------------------------------------------------------------------------
# 54-point Humphrey 24-2 grid (x, y in degrees visual angle)
# Standard layout: x = horizontal (-24 to +24), y = vertical (-24 to +24)
# Blind spot locations (15°, ±3°) are included as placeholder nodes.
# Sector assignment follows Garway-Heath (2000).
# ---------------------------------------------------------------------------

# fmt: off
# (x_deg, y_deg, sector_id)
VF_LOCATIONS: List[Tuple[float, float, int]] = [
    # Row y=+21
    (-9,  21, 1), (-3,  21, 1), (3,  21, 0), (9,  21, 0),
    # Row y=+15
    (-21, 15, 1), (-15, 15, 1), (-9, 15, 1), (-3, 15, 1),
    (3,  15, 0), (9,  15, 0), (15, 15, 0), (21, 15, 0),
    # Row y=+9
    (-21,  9, 2), (-15,  9, 1), (-9,  9, 1), (-3,  9, 1),
    (3,   9, 0), (9,   9, 0), (15,  9, 0), (21,  9, 0),
    # Row y=+3
    (-21,  3, 2), (-15,  3, 2), (-9,  3, 2), (-3,  3, 1),
    (3,   3, 0), (9,   3, 0), (15,  3, 5), (21,  3, 5),
    # Row y=-3
    (-21, -3, 2), (-15, -3, 2), (-9, -3, 2), (-3, -3, 3),
    (3,  -3, 4), (9,  -3, 4), (15, -3, 5), (21, -3, 5),
    # Row y=-9
    (-21, -9, 3), (-15, -9, 3), (-9, -9, 3), (-3, -9, 3),
    (3,  -9, 4), (9,  -9, 4), (15, -9, 4), (21, -9, 4),
    # Row y=-15
    (-21,-15, 3), (-15,-15, 3), (-9,-15, 3), (-3,-15, 3),
    (3, -15, 4), (9, -15, 4), (15,-15, 4), (21,-15, 4),
    # Row y=-21
    (-9, -21, 3), (-3, -21, 3), (3, -21, 4), (9, -21, 4),
]
# fmt: on

assert len(VF_LOCATIONS) == 54, f"Expected 54 VF locations, got {len(VF_LOCATIONS)}"


def _eccentricity(x: float, y: float) -> float:
    """Eccentricity in degrees from fixation."""
    return math.sqrt(x**2 + y**2)


def _angular_position(x: float, y: float) -> float:
    """Angular position in radians (atan2)."""
    return math.atan2(y, x)


def _expected_sensitivity(eccentricity: float, age: float = 50.0) -> float:
    """
    Age-normalized expected sensitivity (dB) at a given eccentricity.
    Approximation: sensitivity decreases ~0.1 dB/degree from fovea,
    and ~0.07 dB/year with age (Heijl et al. normative data approximation).
    """
    base = max(35.0 - 0.1 * eccentricity, 10.0)
    age_correction = 0.07 * (age - 40.0)
    return max(base - age_correction, 5.0)


def build_rnfb_graph(age: float = 50.0) -> Data:
    """
    Build the RNFB anatomical graph as a PyTorch Geometric Data object.

    Node features (per location):
        [eccentricity, angular_position_sin, angular_position_cos,
         expected_sensitivity, sector_one_hot x6]  → 10 features total

    Edge weights:
        1.0 = same RNFB sector (strong co-vulnerability)
        0.5 = adjacent sector (weak co-vulnerability)

    Returns:
        torch_geometric.data.Data with x, edge_index, edge_attr, sector_labels
    """
    n = len(VF_LOCATIONS)

    # --- Node features ---
    node_features = []
    sector_labels = []
    for x, y, sector in VF_LOCATIONS:
        ecc = _eccentricity(x, y)
        ang = _angular_position(x, y)
        exp_sens = _expected_sensitivity(ecc, age)
        sector_oh = [0.0] * 6
        sector_oh[sector] = 1.0
        node_features.append([ecc, math.sin(ang), math.cos(ang), exp_sens] + sector_oh)
        sector_labels.append(sector)

    x = torch.tensor(node_features, dtype=torch.float)  # [54, 10]
    sector_labels_t = torch.tensor(sector_labels, dtype=torch.long)  # [54]

    # --- Edges ---
    # Group nodes by sector
    sector_to_nodes: Dict[int, List[int]] = {s: [] for s in range(6)}
    for i, (_, _, sector) in enumerate(VF_LOCATIONS):
        sector_to_nodes[sector].append(i)

    # Build adjacency set for sectors
    adjacent_sectors = set()
    for s1, s2 in SECTOR_ADJACENCY:
        adjacent_sectors.add((s1, s2))
        adjacent_sectors.add((s2, s1))

    edge_src, edge_dst, edge_weights = [], [], []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            si = VF_LOCATIONS[i][2]
            sj = VF_LOCATIONS[j][2]
            if si == sj:
                edge_src.append(i)
                edge_dst.append(j)
                edge_weights.append(1.0)
            elif (si, sj) in adjacent_sectors:
                edge_src.append(i)
                edge_dst.append(j)
                edge_weights.append(0.5)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)  # [2, E]
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)  # [E, 1]

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        sector_labels=sector_labels_t,
        num_nodes=n,
    )
    return graph


def save_graph(path: str = "ml/data/rnfb_graph.pt", age: float = 50.0) -> Data:
    """Build and serialize the RNFB graph to disk."""
    graph = build_rnfb_graph(age=age)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)
    print(f"Saved RNFB graph → {path}")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    return graph


def load_graph(path: str = "ml/data/rnfb_graph.pt") -> Data:
    """Load serialized RNFB graph from disk."""
    return torch.load(path, weights_only=False)


def visualize_graph(graph: Data, save_path: str = "docs/figures/rnfb_graph_visualization.png") -> None:
    """
    Visualize the RNFB graph overlaid on the 24-2 VF grid.
    Nodes colored by sector; edges drawn with alpha proportional to weight.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    sector_colors = {
        0: "#e74c3c",  # superotemporal — red
        1: "#3498db",  # superonasal — blue
        2: "#2ecc71",  # nasal — green
        3: "#9b59b6",  # inferonasal — purple
        4: "#f39c12",  # inferotemporal — orange
        5: "#1abc9c",  # temporal — teal
    }

    coords = [(x, y) for x, y, _ in VF_LOCATIONS]
    sectors = [s for _, _, s in VF_LOCATIONS]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    colors = [sector_colors[s] for s in sectors]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw edges
    edge_index = graph.edge_index.numpy()
    edge_attr = graph.edge_attr.numpy().flatten()
    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k], edge_index[1, k]
        w = edge_attr[k]
        ax.plot(
            [xs[i], xs[j]], [ys[i], ys[j]],
            color="gray", alpha=w * 0.15, linewidth=0.5, zorder=1,
        )

    # Draw nodes
    ax.scatter(xs, ys, c=colors, s=120, zorder=2, edgecolors="black", linewidths=0.5)
    for i, (xi, yi) in enumerate(coords):
        ax.text(xi, yi + 0.8, str(i + 1), ha="center", va="bottom", fontsize=5)

    # Legend
    patches = [mpatches.Patch(color=c, label=SECTORS[s]) for s, c in sector_colors.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    ax.set_title("RNFB Anatomical Graph — Garway-Heath Map (54-point 24-2 grid)", fontsize=12)
    ax.set_xlabel("Horizontal (degrees)")
    ax.set_ylabel("Vertical (degrees)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # standard VF convention: temporal on left

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization → {save_path}")


if __name__ == "__main__":
    graph = save_graph()
    visualize_graph(graph)
