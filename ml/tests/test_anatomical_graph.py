"""Unit tests for the RNFB anatomical graph construction."""

import pytest
import torch
from ml.data.anatomical_graph import (
    VF_LOCATIONS,
    SECTOR_ADJACENCY,
    build_rnfb_graph,
    SECTORS,
)


@pytest.fixture
def graph():
    return build_rnfb_graph()


def test_node_count(graph):
    """Graph must have exactly 54 nodes (standard 24-2 grid)."""
    assert graph.num_nodes == 54
    assert graph.x.shape[0] == 54


def test_node_feature_dim(graph):
    """Each node has 10 features: ecc, sin, cos, exp_sens, 6x sector one-hot."""
    assert graph.x.shape[1] == 10


def test_sector_labels_range(graph):
    """All sector labels must be in [0, 5]."""
    assert graph.sector_labels.min().item() >= 0
    assert graph.sector_labels.max().item() <= 5


def test_sector_assignment_count():
    """All 54 locations must be assigned to a sector."""
    assert len(VF_LOCATIONS) == 54
    for x, y, s in VF_LOCATIONS:
        assert s in SECTORS, f"Location ({x},{y}) has invalid sector {s}"


def test_edge_symmetry(graph):
    """Graph must be undirected: for every (i→j) edge there must be a (j→i) edge."""
    edge_set = set(
        zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist())
    )
    for i, j in edge_set:
        assert (j, i) in edge_set, f"Missing reverse edge ({j}→{i}) for ({i}→{j})"


def test_no_self_loops(graph):
    """No self-loop edges."""
    src, dst = graph.edge_index
    assert (src == dst).sum().item() == 0


def test_edge_weights_valid(graph):
    """Edge weights must be either 0.5 (adjacent sector) or 1.0 (same sector)."""
    weights = graph.edge_attr.flatten().tolist()
    for w in weights:
        assert w in (0.5, 1.0), f"Unexpected edge weight: {w}"


def test_same_sector_edges_weight_one(graph):
    """Edges between nodes in the same sector must have weight 1.0."""
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr.flatten()
    sector_labels = graph.sector_labels
    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k].item(), edge_index[1, k].item()
        if sector_labels[i] == sector_labels[j]:
            assert edge_attr[k].item() == 1.0


def test_eccentricity_positive(graph):
    """Eccentricity (first node feature) must be positive for all non-foveal locations."""
    eccentricities = graph.x[:, 0]
    assert (eccentricities >= 0).all()


def test_expected_sensitivity_range(graph):
    """Expected sensitivity (4th node feature) must be in [5, 40] dB."""
    sens = graph.x[:, 3]
    assert (sens >= 5.0).all()
    assert (sens <= 40.0).all()
