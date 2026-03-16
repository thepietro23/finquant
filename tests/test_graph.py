"""Phase 4 Tests: Graph Construction — edges, correlation, PyG Data objects.

Tests:
  T4.1: Sector edges count matches expected pairs
  T4.2: Supply chain edges exist and are bidirectional
  T4.3: Correlation edges respect threshold (only |corr| > 0.6)
  T4.4: Full graph has all 3 edge types
  T4.5: PyG Data object has correct shape (num_nodes, features)
  T4.6: No self-loops in any edge type

Edge Cases:
  E4.1: Zero correlation (identity matrix) → only static edges
  E4.2: Perfect correlation (all 1s) → maximum edges
  E4.3: Single stock graph → 0 edges
  E4.4: Graph stats computation
"""

import os
import sys

import numpy as np
import torch
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.graph.builder import (
    build_sector_edges,
    build_supply_chain_edges,
    build_correlation_edges_fast,
    build_static_graph,
    build_full_graph,
    get_graph_stats,
    EDGE_SECTOR,
    EDGE_SUPPLY_CHAIN,
    EDGE_CORRELATION,
)
from src.data.stocks import (
    get_all_tickers,
    get_sector_pairs,
    get_supply_chain_pairs,
    get_ticker_to_index,
)


# ===========================
# Unit Tests
# ===========================

class TestSectorEdges:
    """T4.1: Sector edge construction."""

    def test_sector_edge_count(self):
        """Sector edges = 2 * number of sector pairs (bidirectional)."""
        ticker_to_idx = get_ticker_to_index()
        edge_index = build_sector_edges(ticker_to_idx)

        pairs = get_sector_pairs()
        # Filter pairs where both tickers exist in index
        valid_pairs = [(a, b) for a, b in pairs
                       if a in ticker_to_idx and b in ticker_to_idx]

        # Each pair → 2 directed edges (a→b, b→a)
        expected = len(valid_pairs) * 2
        assert edge_index.shape[1] == expected, \
            f'Expected {expected} edges, got {edge_index.shape[1]}'

    def test_sector_edges_bidirectional(self):
        """Every sector edge has a reverse edge."""
        ticker_to_idx = get_ticker_to_index()
        edge_index = build_sector_edges(ticker_to_idx)

        # For each edge (i, j), check (j, i) exists
        edge_set = set()
        for k in range(edge_index.shape[1]):
            edge_set.add((edge_index[0, k].item(), edge_index[1, k].item()))

        for i, j in list(edge_set):
            assert (j, i) in edge_set, f'Edge ({i},{j}) has no reverse'

    def test_sector_edges_no_self_loops(self):
        """T4.6: No self-loops in sector edges."""
        edge_index = build_sector_edges()
        for k in range(edge_index.shape[1]):
            assert edge_index[0, k] != edge_index[1, k], 'Self-loop found'


class TestSupplyChainEdges:
    """T4.2: Supply chain edge construction."""

    def test_supply_chain_exists(self):
        """Supply chain edges > 0."""
        edge_index = build_supply_chain_edges()
        assert edge_index.shape[1] > 0, 'No supply chain edges'

    def test_supply_chain_bidirectional(self):
        """Supply chain edges are bidirectional (info flows both ways)."""
        edge_index = build_supply_chain_edges()
        edge_set = set()
        for k in range(edge_index.shape[1]):
            edge_set.add((edge_index[0, k].item(), edge_index[1, k].item()))

        for i, j in list(edge_set):
            assert (j, i) in edge_set, f'Supply edge ({i},{j}) has no reverse'

    def test_supply_chain_no_self_loops(self):
        """No self-loops in supply chain edges."""
        edge_index = build_supply_chain_edges()
        for k in range(edge_index.shape[1]):
            assert edge_index[0, k] != edge_index[1, k], 'Self-loop found'


class TestCorrelationEdges:
    """T4.3: Dynamic correlation edges."""

    def test_threshold_respected(self):
        """Only edges with |corr| > threshold are included."""
        n = 10
        # Create a correlation matrix with known values
        corr = np.eye(n)
        corr[0, 1] = corr[1, 0] = 0.8  # Above threshold
        corr[2, 3] = corr[3, 2] = 0.3  # Below threshold
        corr[4, 5] = corr[5, 4] = -0.7  # Negative but |corr| > 0.6

        edge_index = build_correlation_edges_fast(corr, threshold=0.6)

        # Should have edges for (0,1) and (4,5) — both above threshold
        # Each pair → 2 directed edges
        assert edge_index.shape[1] == 4, \
            f'Expected 4 edges (2 pairs × 2 directions), got {edge_index.shape[1]}'

    def test_no_self_loops(self):
        """T4.6: Correlation edges don't include self-loops."""
        corr = np.ones((5, 5))  # All perfectly correlated
        edge_index = build_correlation_edges_fast(corr, threshold=0.6)

        for k in range(edge_index.shape[1]):
            assert edge_index[0, k] != edge_index[1, k], 'Self-loop in correlation edges'

    def test_correlation_bidirectional(self):
        """Correlation edges are undirected (bidirectional)."""
        corr = np.eye(5)
        corr[0, 1] = corr[1, 0] = 0.9

        edge_index = build_correlation_edges_fast(corr, threshold=0.6)
        edge_set = set()
        for k in range(edge_index.shape[1]):
            edge_set.add((edge_index[0, k].item(), edge_index[1, k].item()))

        for i, j in list(edge_set):
            assert (j, i) in edge_set


class TestStaticGraph:
    """Combined static graph (sector + supply chain)."""

    def test_static_has_both_types(self):
        """Static graph contains both sector and supply chain edges."""
        edge_index, edge_type = build_static_graph()

        has_sector = (edge_type == EDGE_SECTOR).any()
        has_supply = (edge_type == EDGE_SUPPLY_CHAIN).any()

        assert has_sector, 'No sector edges in static graph'
        assert has_supply, 'No supply chain edges in static graph'

    def test_edge_type_length_matches(self):
        """edge_type length matches edge_index columns."""
        edge_index, edge_type = build_static_graph()
        assert edge_index.shape[1] == len(edge_type)


class TestFullGraph:
    """T4.4-T4.5: Full graph with all edge types."""

    def test_full_graph_shape(self):
        """T4.5: PyG Data has correct num_nodes and feature dimensions."""
        ticker_to_idx = get_ticker_to_index()
        n_stocks = len(ticker_to_idx)
        n_features = 21

        node_features = torch.randn(n_stocks, n_features)
        data = build_full_graph(node_features, ticker_to_idx=ticker_to_idx)

        assert data.num_nodes == n_stocks
        assert data.x.shape == (n_stocks, n_features)

    def test_full_graph_with_correlation(self):
        """T4.4: Full graph has all 3 edge types when correlation provided."""
        ticker_to_idx = get_ticker_to_index()
        n_stocks = len(ticker_to_idx)

        node_features = torch.randn(n_stocks, 21)
        # Create synthetic correlation matrix with some high correlations
        corr = np.eye(n_stocks)
        for i in range(0, n_stocks - 1, 2):
            corr[i, i + 1] = corr[i + 1, i] = 0.8

        data = build_full_graph(node_features, corr_matrix=corr,
                                threshold=0.6, ticker_to_idx=ticker_to_idx)

        assert data.num_edges > 0
        assert hasattr(data, 'edge_type')
        # Should have all 3 types
        assert (data.edge_type == EDGE_SECTOR).any()
        assert (data.edge_type == EDGE_SUPPLY_CHAIN).any()
        assert (data.edge_type == EDGE_CORRELATION).any()

    def test_numpy_features_accepted(self):
        """Node features as numpy array are auto-converted to tensor."""
        ticker_to_idx = get_ticker_to_index()
        n_stocks = len(ticker_to_idx)

        node_features = np.random.randn(n_stocks, 21).astype(np.float32)
        data = build_full_graph(node_features, ticker_to_idx=ticker_to_idx)

        assert isinstance(data.x, torch.Tensor)
        assert data.x.dtype == torch.float32


class TestGraphStats:
    """T4.7: Graph statistics utility."""

    def test_stats_keys(self):
        """get_graph_stats returns expected keys."""
        ticker_to_idx = get_ticker_to_index()
        node_features = torch.randn(len(ticker_to_idx), 21)
        data = build_full_graph(node_features, ticker_to_idx=ticker_to_idx)

        stats = get_graph_stats(data)
        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'density' in stats
        assert 'sector_edges' in stats
        assert 'supply_chain_edges' in stats

    def test_density_range(self):
        """Graph density is between 0 and 1."""
        ticker_to_idx = get_ticker_to_index()
        node_features = torch.randn(len(ticker_to_idx), 21)
        data = build_full_graph(node_features, ticker_to_idx=ticker_to_idx)

        stats = get_graph_stats(data)
        assert 0 <= stats['density'] <= 1


# ===========================
# Edge Cases
# ===========================

class TestEdgeCases:
    """Edge case handling."""

    def test_zero_correlation(self):
        """E4.1: Identity correlation matrix → 0 correlation edges."""
        corr = np.eye(10)  # No correlations between different stocks
        edge_index = build_correlation_edges_fast(corr, threshold=0.6)
        assert edge_index.shape[1] == 0, 'Should have 0 correlation edges for identity matrix'

    def test_perfect_correlation(self):
        """E4.2: All-ones correlation → maximum edges."""
        n = 5
        corr = np.ones((n, n))
        edge_index = build_correlation_edges_fast(corr, threshold=0.6)

        # C(5,2) = 10 pairs × 2 directions = 20 edges
        expected = n * (n - 1)  # All pairs, both directions
        assert edge_index.shape[1] == expected, \
            f'Expected {expected}, got {edge_index.shape[1]}'

    def test_single_stock(self):
        """E4.3: Single stock → graph with 0 edges."""
        node_features = torch.randn(1, 21)
        ticker_to_idx = {'SINGLE.NS': 0}

        # Can't use build_full_graph with custom ticker_to_idx directly
        # because sector/supply chain use global registry
        # But we can test correlation edges with single stock
        corr = np.eye(1)
        edge_index = build_correlation_edges_fast(corr, threshold=0.6)
        assert edge_index.shape[1] == 0

    def test_negative_correlation_edges(self):
        """Negative correlations beyond threshold also create edges."""
        corr = np.eye(5)
        corr[0, 1] = corr[1, 0] = -0.8  # Strong negative correlation

        edge_index = build_correlation_edges_fast(corr, threshold=0.6)
        # Should have 2 edges (0→1, 1→0)
        assert edge_index.shape[1] == 2
