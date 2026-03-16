"""Phase 4: Graph Construction for T-GAT.

Builds multi-relational stock graphs with 3 edge types:
  1. Sector edges (static): Same sector stocks connected
  2. Supply chain edges (static): Business relationships
  3. Correlation edges (dynamic): Rolling price correlation > threshold

Output: PyTorch Geometric Data objects ready for T-GAT.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.stocks import (
    get_all_tickers,
    get_sector_pairs,
    get_supply_chain_pairs,
    get_ticker_to_index,
)
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('graph')

# Edge type constants
EDGE_SECTOR = 0
EDGE_SUPPLY_CHAIN = 1
EDGE_CORRELATION = 2


# ---------------------------------------------------------------------------
# Static edges (don't change over time)
# ---------------------------------------------------------------------------

def build_sector_edges(ticker_to_idx=None):
    """Build sector edges: undirected edges between stocks in same sector.

    Returns:
        tuple: (edge_index tensor [2, num_edges], num_edges)
    """
    if ticker_to_idx is None:
        ticker_to_idx = get_ticker_to_index()

    pairs = get_sector_pairs()
    sources, targets = [], []

    for a, b in pairs:
        if a in ticker_to_idx and b in ticker_to_idx:
            i, j = ticker_to_idx[a], ticker_to_idx[b]
            # Undirected: add both directions
            sources.extend([i, j])
            targets.extend([j, i])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    logger.info(f'Sector edges: {len(pairs)} pairs → {edge_index.shape[1]} directed edges')
    return edge_index


def build_supply_chain_edges(ticker_to_idx=None):
    """Build supply chain edges: directed edges from supplier to consumer.

    Also adds reverse edges (information flows both ways).

    Returns:
        tuple: (edge_index tensor [2, num_edges], num_edges)
    """
    if ticker_to_idx is None:
        ticker_to_idx = get_ticker_to_index()

    pairs = get_supply_chain_pairs()
    sources, targets = [], []

    for a, b in pairs:
        if a in ticker_to_idx and b in ticker_to_idx:
            i, j = ticker_to_idx[a], ticker_to_idx[b]
            # Both directions (information flows both ways)
            sources.extend([i, j])
            targets.extend([j, i])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    logger.info(f'Supply chain edges: {len(pairs)} pairs → {edge_index.shape[1]} directed edges')
    return edge_index


# ---------------------------------------------------------------------------
# Dynamic correlation edges (change every day)
# ---------------------------------------------------------------------------

def compute_correlation_matrix(close_prices, window=60):
    """Compute rolling pairwise correlation matrix.

    Args:
        close_prices: DataFrame with stocks as columns, dates as index.
        window: Rolling window size in trading days.

    Returns:
        dict: {date: correlation_matrix (n_stocks x n_stocks numpy array)}
    """
    returns = close_prices.pct_change().dropna()

    corr_by_date = {}
    dates = returns.index[window:]  # Need `window` days of history

    for i, date in enumerate(dates):
        # Rolling window: [date - window, date]
        window_returns = returns.iloc[i:i + window]
        corr = window_returns.corr().values
        # Replace NaN with 0 (stocks with no variance in window)
        corr = np.nan_to_num(corr, nan=0.0)
        corr_by_date[date] = corr

    logger.info(f'Correlation matrices: {len(corr_by_date)} dates, window={window}')
    return corr_by_date


def build_correlation_edges(corr_matrix, threshold=0.6, ticker_to_idx=None):
    """Build correlation edges from a single correlation matrix.

    Edges between stocks with |correlation| > threshold.
    Self-loops excluded.

    Args:
        corr_matrix: numpy array (n_stocks x n_stocks).
        threshold: Minimum absolute correlation for an edge.
        ticker_to_idx: Optional ticker-to-index mapping (for size reference).

    Returns:
        edge_index: tensor [2, num_edges]
    """
    n = corr_matrix.shape[0]
    sources, targets = [], []

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > threshold:
                # Undirected
                sources.extend([i, j])
                targets.extend([j, i])

    if not sources:
        # Return empty edge index with correct shape
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor([sources, targets], dtype=torch.long)


def build_correlation_edges_fast(corr_matrix, threshold=0.6):
    """Vectorized version of correlation edge building (much faster).

    Args:
        corr_matrix: numpy array (n_stocks x n_stocks).
        threshold: Minimum absolute correlation.

    Returns:
        edge_index: tensor [2, num_edges]
    """
    n = corr_matrix.shape[0]
    # Upper triangle only (avoid self-loops and duplicates)
    mask = np.triu(np.abs(corr_matrix) > threshold, k=1)
    sources, targets = np.where(mask)

    if len(sources) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    # Make undirected
    all_src = np.concatenate([sources, targets])
    all_tgt = np.concatenate([targets, sources])

    return torch.tensor(np.stack([all_src, all_tgt]), dtype=torch.long)


# ---------------------------------------------------------------------------
# Combine all edges into a single graph
# ---------------------------------------------------------------------------

def build_static_graph(ticker_to_idx=None):
    """Build the static part of the graph (sector + supply chain edges).

    Returns:
        tuple: (edge_index, edge_type) — combined static edges with type labels
    """
    if ticker_to_idx is None:
        ticker_to_idx = get_ticker_to_index()

    sector_edges = build_sector_edges(ticker_to_idx)
    supply_edges = build_supply_chain_edges(ticker_to_idx)

    # Combine edges
    edge_index = torch.cat([sector_edges, supply_edges], dim=1)

    # Edge type labels
    sector_types = torch.full((sector_edges.shape[1],), EDGE_SECTOR, dtype=torch.long)
    supply_types = torch.full((supply_edges.shape[1],), EDGE_SUPPLY_CHAIN, dtype=torch.long)
    edge_type = torch.cat([sector_types, supply_types])

    # Remove duplicate edges (same pair can appear in both sector and supply chain)
    edge_index, edge_type = _deduplicate_edges(edge_index, edge_type)

    logger.info(f'Static graph: {edge_index.shape[1]} edges '
                f'(sector={int((edge_type == EDGE_SECTOR).sum())}, '
                f'supply={int((edge_type == EDGE_SUPPLY_CHAIN).sum())})')

    return edge_index, edge_type


def _deduplicate_edges(edge_index, edge_type):
    """Remove duplicate edges, keeping the first occurrence's type."""
    # Encode each edge as a single number for fast dedup
    n = edge_index.max().item() + 1
    edge_codes = edge_index[0] * n + edge_index[1]

    _, unique_idx = torch.unique(edge_codes, return_inverse=True)
    # Keep first occurrence of each unique edge
    seen = set()
    keep = []
    for i in range(len(edge_codes)):
        code = edge_codes[i].item()
        if code not in seen:
            seen.add(code)
            keep.append(i)

    keep = torch.tensor(keep, dtype=torch.long)
    return edge_index[:, keep], edge_type[keep]


def build_full_graph(node_features, corr_matrix=None, threshold=None,
                     ticker_to_idx=None):
    """Build complete graph with all 3 edge types for a single timestep.

    Args:
        node_features: tensor (n_stocks, n_features) — features for this timestep.
        corr_matrix: numpy array (n_stocks x n_stocks). If None, only static edges.
        threshold: Correlation threshold. Defaults to config value.
        ticker_to_idx: Optional ticker-to-index mapping.

    Returns:
        torch_geometric.data.Data object with:
          - x: node features
          - edge_index: all edges combined
          - edge_type: edge type labels (0=sector, 1=supply, 2=corr)
    """
    cfg = get_config('gnn')
    if threshold is None:
        threshold = cfg.get('correlation_threshold', 0.6)
    if ticker_to_idx is None:
        ticker_to_idx = get_ticker_to_index()

    # Static edges
    static_edges, static_types = build_static_graph(ticker_to_idx)

    # Dynamic correlation edges
    if corr_matrix is not None:
        corr_edges = build_correlation_edges_fast(corr_matrix, threshold)
        corr_types = torch.full((corr_edges.shape[1],), EDGE_CORRELATION, dtype=torch.long)

        edge_index = torch.cat([static_edges, corr_edges], dim=1)
        edge_type = torch.cat([static_types, corr_types])
    else:
        edge_index = static_edges
        edge_type = static_types

    # Ensure node_features is a tensor
    if isinstance(node_features, np.ndarray):
        node_features = torch.from_numpy(node_features).float()

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_type=edge_type,
    )

    return data


# ---------------------------------------------------------------------------
# Build graph sequence (one graph per trading day)
# ---------------------------------------------------------------------------

def build_graph_sequence(feature_tensor, close_prices_df, tickers,
                         corr_window=None, corr_threshold=None):
    """Build a sequence of PyG Data objects — one per trading day.

    This is the main entry point for Phase 4. Takes the feature tensor from
    Phase 2 and builds a graph for each timestep with dynamic correlation edges.

    Args:
        feature_tensor: numpy array (n_stocks, n_timesteps, n_features)
        close_prices_df: DataFrame of close prices (dates x tickers)
        tickers: list of ticker strings matching tensor axis 0 order
        corr_window: Rolling correlation window (default from config)
        corr_threshold: Correlation threshold (default from config)

    Returns:
        list of Data objects, one per timestep
    """
    cfg = get_config('gnn')
    if corr_window is None:
        corr_window = cfg.get('correlation_window', 60)
    if corr_threshold is None:
        corr_threshold = cfg.get('correlation_threshold', 0.6)

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    n_stocks, n_time, n_feat = feature_tensor.shape

    # Compute all correlation matrices
    # Align close_prices columns with ticker order
    aligned_prices = close_prices_df[
        [t for t in tickers if t in close_prices_df.columns]
    ]
    corr_matrices = compute_correlation_matrix(aligned_prices, window=corr_window)
    corr_dates = sorted(corr_matrices.keys())

    # Build static graph once (reused every timestep)
    static_edges, static_types = build_static_graph(ticker_to_idx)

    graphs = []
    for t in range(n_time):
        node_feat = torch.from_numpy(feature_tensor[:, t, :]).float()

        # Find closest correlation matrix for this timestep
        if t < len(corr_dates):
            corr_mat = corr_matrices[corr_dates[min(t, len(corr_dates) - 1)]]
            corr_edges = build_correlation_edges_fast(corr_mat, corr_threshold)
            corr_types = torch.full((corr_edges.shape[1],), EDGE_CORRELATION, dtype=torch.long)

            edge_index = torch.cat([static_edges, corr_edges], dim=1)
            edge_type = torch.cat([static_types, corr_types])
        else:
            edge_index = static_edges
            edge_type = static_types

        data = Data(x=node_feat, edge_index=edge_index, edge_type=edge_type)
        graphs.append(data)

    logger.info(f'Graph sequence: {len(graphs)} timesteps, '
                f'avg edges/graph: {np.mean([g.num_edges for g in graphs]):.0f}')

    return graphs


def get_graph_stats(data):
    """Get summary statistics for a PyG Data object.

    Returns:
        dict with node count, edge counts by type, density, etc.
    """
    n_nodes = data.num_nodes
    n_edges = data.num_edges

    stats = {
        'num_nodes': n_nodes,
        'num_edges': n_edges,
        'density': n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0,
    }

    if hasattr(data, 'edge_type') and data.edge_type is not None:
        stats['sector_edges'] = int((data.edge_type == EDGE_SECTOR).sum())
        stats['supply_chain_edges'] = int((data.edge_type == EDGE_SUPPLY_CHAIN).sum())
        stats['correlation_edges'] = int((data.edge_type == EDGE_CORRELATION).sum())

    return stats
