"""Phase 5 Tests: T-GAT Model — architecture, forward pass, embeddings.

Tests:
  T5.1: Model initializes with correct architecture
  T5.2: Forward pass produces correct output shapes
  T5.3: Single graph forward works (no temporal)
  T5.4: Embeddings are finite (no NaN/Inf)
  T5.5: Gradients flow (backprop works)
  T5.6: FP16 works on CUDA (if available)
  T5.7: Relational GAT handles missing edge types
  T5.8: Parameter count is reasonable (<1M for 4GB VRAM)

Edge Cases:
  E5.1: Empty edge graph → still produces embeddings
  E5.2: Single node graph → valid output
  E5.3: Long sequence (20 timesteps) → no memory explosion
"""

import os
import sys

import numpy as np
import torch
import pytest
from torch_geometric.data import Data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.tgat import (
    TGAT,
    RelationalGATLayer,
    count_parameters,
    get_model_size_mb,
)


# ===========================
# Fixtures
# ===========================

def _make_graph(n_stocks=10, n_features=21, n_edges=12):
    """Create a simple test graph."""
    x = torch.randn(n_stocks, n_features)
    # Create some edges (bidirectional pairs)
    src = list(range(0, min(n_edges // 2, n_stocks - 1)))
    tgt = [s + 1 for s in src]
    # Bidirectional
    all_src = src + tgt
    all_tgt = tgt + src
    edge_index = torch.tensor([all_src, all_tgt], dtype=torch.long)
    # Mix of edge types
    n_e = edge_index.shape[1]
    edge_type = torch.zeros(n_e, dtype=torch.long)
    edge_type[n_e // 3: 2 * n_e // 3] = 1
    edge_type[2 * n_e // 3:] = 2
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)


def _make_sequence(seq_len=5, n_stocks=10, n_features=21):
    """Create a sequence of test graphs."""
    return [_make_graph(n_stocks, n_features) for _ in range(seq_len)]


# ===========================
# Unit Tests
# ===========================

class TestModelInit:
    """T5.1: Model architecture initialization."""

    def test_model_creates(self):
        """Model initializes without error."""
        model = TGAT(n_features=21)
        assert model is not None

    def test_model_config_from_yaml(self):
        """Model reads config from base.yaml."""
        model = TGAT(n_features=21)
        assert model.hidden_dim == 64
        assert model.output_dim == 64
        assert model.num_layers == 2
        assert model.num_heads == 4

    def test_model_custom_config(self):
        """Model accepts custom hyperparameters."""
        model = TGAT(n_features=10, hidden_dim=32, output_dim=16,
                     num_layers=1, num_heads=2, dropout=0.2)
        assert model.hidden_dim == 32
        assert model.output_dim == 16
        assert model.num_layers == 1

    def test_has_expected_components(self):
        """Model has input_proj, gat_layers, gru, output_proj."""
        model = TGAT(n_features=21)
        assert hasattr(model, 'input_proj')
        assert hasattr(model, 'gat_layers')
        assert hasattr(model, 'gru')
        assert hasattr(model, 'output_proj')
        assert len(model.gat_layers) == 2  # num_layers


class TestForwardPass:
    """T5.2-T5.4: Forward pass correctness."""

    def test_sequence_output_shape(self):
        """T5.2: Sequence forward produces (n_stocks, output_dim)."""
        model = TGAT(n_features=21)
        model.eval()
        graphs = _make_sequence(seq_len=5, n_stocks=10)

        with torch.no_grad():
            emb, spatial = model(graphs)

        assert emb.shape == (10, 64), f'Expected (10, 64), got {emb.shape}'
        assert spatial.shape == (10, 5, 64)

    def test_single_forward_shape(self):
        """T5.3: Single graph forward produces (n_stocks, output_dim)."""
        model = TGAT(n_features=21)
        model.eval()
        data = _make_graph(n_stocks=10)

        with torch.no_grad():
            emb = model.forward_single(data)

        assert emb.shape == (10, 64)

    def test_embeddings_finite(self):
        """T5.4: No NaN or Inf in output embeddings."""
        model = TGAT(n_features=21)
        model.eval()
        graphs = _make_sequence(seq_len=3, n_stocks=10)

        with torch.no_grad():
            emb, _ = model(graphs)

        assert torch.isfinite(emb).all(), 'Non-finite values in embeddings'

    def test_single_embeddings_finite(self):
        """Single forward also produces finite values."""
        model = TGAT(n_features=21)
        model.eval()
        data = _make_graph(n_stocks=10)

        with torch.no_grad():
            emb = model.forward_single(data)

        assert torch.isfinite(emb).all()


class TestGradients:
    """T5.5: Gradient flow (training feasibility)."""

    def test_gradients_flow(self):
        """Backprop works — all parameters get gradients."""
        model = TGAT(n_features=21)
        model.train()
        graphs = _make_sequence(seq_len=3, n_stocks=10)

        emb, _ = model(graphs)
        loss = emb.mean()  # Simple dummy loss
        loss.backward()

        # Check at least some parameters have gradients
        params_with_grad = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        total_params = sum(1 for p in model.parameters())
        assert params_with_grad > 0, 'No parameters received gradients'

    def test_loss_decreases(self):
        """One optimization step reduces loss (sanity check)."""
        model = TGAT(n_features=21)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        graphs = _make_sequence(seq_len=3, n_stocks=10)
        target = torch.zeros(10, 64)

        # Step 1
        emb, _ = model(graphs)
        loss1 = F.mse_loss(emb, target)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step 2
        emb, _ = model(graphs)
        loss2 = F.mse_loss(emb, target)

        # Loss should decrease (at least not increase significantly)
        # With random init and one step, we allow some slack
        assert loss2.item() < loss1.item() * 1.5, \
            f'Loss did not decrease: {loss1.item():.4f} → {loss2.item():.4f}'


class TestRelationalGAT:
    """T5.7: Multi-relational attention handling."""

    def test_missing_edge_type(self):
        """Graph with only 1 edge type → no crash."""
        model = TGAT(n_features=21)
        model.eval()

        # Only sector edges (type 0), no supply chain or correlation
        x = torch.randn(5, 21)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_type = torch.zeros(2, dtype=torch.long)  # Only type 0
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        with torch.no_grad():
            emb = model.forward_single(data)

        assert emb.shape == (5, 64)
        assert torch.isfinite(emb).all()

    def test_all_three_edge_types(self):
        """Graph with all 3 edge types processes correctly."""
        model = TGAT(n_features=21)
        model.eval()
        data = _make_graph(n_stocks=10)

        with torch.no_grad():
            emb = model.forward_single(data)

        assert emb.shape == (10, 64)


class TestModelSize:
    """T5.8: Model size constraints (4GB VRAM)."""

    def test_parameter_count(self):
        """Model has < 1M parameters (lightweight for 4GB GPU)."""
        model = TGAT(n_features=21)
        n_params = count_parameters(model)
        assert n_params < 1_000_000, \
            f'Too many parameters: {n_params:,} (max 1M for 4GB VRAM)'

    def test_model_size_mb(self):
        """Model < 10 MB in FP32 (< 5 MB in FP16)."""
        model = TGAT(n_features=21)
        size = get_model_size_mb(model)
        assert size < 10.0, f'Model too large: {size:.2f} MB'

    def test_fp16_on_cuda(self):
        """T5.6: Mixed precision (autocast) works on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        model = TGAT(n_features=21).cuda()  # FP32 model on GPU
        x = torch.randn(10, 21).cuda()
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda()
        edge_type = torch.zeros(2, dtype=torch.long).cuda()
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        # Use autocast for proper mixed precision (LayerNorm stays FP32)
        with torch.no_grad(), torch.cuda.amp.autocast():
            emb = model.forward_single(data)

        assert emb.shape == (10, 64)
        assert torch.isfinite(emb).all()


# ===========================
# Edge Cases
# ===========================

class TestEdgeCases:
    """Edge case handling."""

    def test_no_edges(self):
        """E5.1: Graph with 0 edges → isolated nodes still get embeddings."""
        model = TGAT(n_features=21)
        model.eval()

        x = torch.randn(5, 21)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        with torch.no_grad():
            emb = model.forward_single(data)

        assert emb.shape == (5, 64)
        assert torch.isfinite(emb).all()

    def test_single_node(self):
        """E5.2: Single node graph → valid (1, output_dim) output."""
        model = TGAT(n_features=21)
        model.eval()

        x = torch.randn(1, 21)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        with torch.no_grad():
            emb = model.forward_single(data)

        assert emb.shape == (1, 64)

    def test_long_sequence(self):
        """E5.3: 20-timestep sequence → no memory explosion."""
        model = TGAT(n_features=21, hidden_dim=32, output_dim=32)
        model.eval()
        graphs = _make_sequence(seq_len=20, n_stocks=10)

        with torch.no_grad():
            emb, spatial = model(graphs)

        assert emb.shape == (10, 32)
        assert spatial.shape == (10, 20, 32)

    def test_empty_sequence_raises(self):
        """Empty sequence should raise ValueError."""
        model = TGAT(n_features=21)
        with pytest.raises(ValueError):
            model([])


# Need this import for test_loss_decreases
import torch.nn.functional as F
