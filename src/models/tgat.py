"""Phase 5: Temporal Graph Attention Network (T-GAT).

Architecture:
  1. Input projection: (n_features) → (hidden_dim)
  2. Multi-relational GAT layers: graph attention with 3 edge types
  3. GRU temporal encoder: sequence of graph snapshots → temporal embeddings
  4. Output: (n_stocks, output_dim) per timestep

Edge types: 0=sector, 1=supply_chain, 2=correlation
Each edge type gets its own attention parameters via edge-type embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('tgat')


class RelationalGATLayer(nn.Module):
    """Multi-relational GAT layer.

    Separate GAT convolution per edge type, then aggregate.
    This lets the model learn different attention patterns for
    sector vs supply chain vs correlation relationships.
    """

    def __init__(self, in_dim, out_dim, num_heads=4, num_relations=3,
                 dropout=0.1, concat=True):
        super().__init__()
        self.num_relations = num_relations
        self.concat = concat

        # One GATConv per edge type
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=in_dim,
                out_channels=out_dim // num_heads if concat else out_dim,
                heads=num_heads,
                dropout=dropout,
                concat=concat,
            )
            for _ in range(num_relations)
        ])

        # Learnable importance weight per edge type
        self.relation_weights = nn.Parameter(torch.ones(num_relations))

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x: node features (n_nodes, in_dim)
            edge_index: all edges (2, n_edges)
            edge_type: edge type labels (n_edges,)

        Returns:
            Updated node features (n_nodes, out_dim)
        """
        # Softmax over relation weights for stable aggregation
        weights = F.softmax(self.relation_weights, dim=0)

        out = torch.zeros(x.size(0), self.gat_layers[0].out_channels *
                          (self.gat_layers[0].heads if self.concat else 1),
                          device=x.device)

        for rel_id in range(self.num_relations):
            # Filter edges for this relation type
            mask = edge_type == rel_id
            if mask.sum() == 0:
                continue

            rel_edges = edge_index[:, mask]
            rel_out = self.gat_layers[rel_id](x, rel_edges)
            out = out + weights[rel_id] * rel_out

        return out


class TGAT(nn.Module):
    """Temporal Graph Attention Network.

    Combines:
    - Multi-relational GAT for spatial (graph) processing
    - GRU for temporal (time-series) processing

    Input:  sequence of PyG Data objects (one per timestep)
    Output: stock embeddings (n_stocks, output_dim)
    """

    def __init__(self, n_features=21, hidden_dim=None, output_dim=None,
                 num_layers=None, num_heads=None, dropout=None,
                 num_relations=3):
        super().__init__()

        cfg = get_config('gnn')
        self.hidden_dim = hidden_dim or cfg.get('hidden_dim', 64)
        self.output_dim = output_dim or cfg.get('output_dim', 64)
        self.num_layers = num_layers or cfg.get('num_layers', 2)
        self.num_heads = num_heads or cfg.get('num_heads', 4)
        self.dropout = dropout or cfg.get('dropout', 0.1)
        self.num_relations = num_relations

        # 1. Input projection: n_features → hidden_dim
        self.input_proj = nn.Linear(n_features, self.hidden_dim)

        # 2. Stacked relational GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_dim
            # After first layer with concat, dim = hidden_dim (heads * out_per_head)
            out_dim = self.hidden_dim
            self.gat_layers.append(
                RelationalGATLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_heads=self.num_heads,
                    num_relations=num_relations,
                    dropout=self.dropout,
                    concat=True,
                )
            )

        # Layer norms for stable training
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(self.num_layers)
        ])

        # 3. GRU temporal encoder
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 4. Output projection → final embedding
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

        # Dropout
        self.drop = nn.Dropout(self.dropout)

        logger.info(f'T-GAT initialized: features={n_features}, '
                    f'hidden={self.hidden_dim}, output={self.output_dim}, '
                    f'layers={self.num_layers}, heads={self.num_heads}')

    def encode_graph(self, data):
        """Process a single graph snapshot through GAT layers.

        Args:
            data: PyG Data with x, edge_index, edge_type

        Returns:
            Node embeddings (n_nodes, hidden_dim)
        """
        x = self.input_proj(data.x)
        x = F.elu(x)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x
            x = gat(x, data.edge_index, data.edge_type)
            x = norm(x)
            x = F.elu(x)
            x = self.drop(x)
            # Residual connection
            x = x + residual

        return x

    def forward(self, graph_sequence):
        """Process a sequence of graph snapshots.

        Args:
            graph_sequence: list of PyG Data objects (length = seq_len)
                Each Data has: x (n_stocks, n_features), edge_index, edge_type

        Returns:
            embeddings: (n_stocks, output_dim) — final temporal embeddings
            all_spatial: (n_stocks, seq_len, hidden_dim) — all spatial embeddings
        """
        if not graph_sequence:
            raise ValueError("Empty graph sequence")

        n_stocks = graph_sequence[0].x.size(0)

        # Encode each timestep through GAT
        spatial_embeddings = []
        for data in graph_sequence:
            h = self.encode_graph(data)  # (n_stocks, hidden_dim)
            spatial_embeddings.append(h)

        # Stack: (n_stocks, seq_len, hidden_dim)
        spatial_stack = torch.stack(spatial_embeddings, dim=1)

        # GRU over time: (n_stocks, seq_len, hidden_dim) → (n_stocks, hidden_dim)
        gru_out, _ = self.gru(spatial_stack)  # (n_stocks, seq_len, hidden_dim)
        # Take last timestep
        temporal_out = gru_out[:, -1, :]  # (n_stocks, hidden_dim)

        # Output projection
        embeddings = self.output_proj(temporal_out)  # (n_stocks, output_dim)

        return embeddings, spatial_stack

    def forward_single(self, data):
        """Process a single graph (no temporal component).

        Useful for inference on a single day.

        Args:
            data: PyG Data with x, edge_index, edge_type

        Returns:
            embeddings: (n_stocks, output_dim)
        """
        h = self.encode_graph(data)
        return self.output_proj(h)

    def get_attention_weights(self, data, layer_idx=0, relation_idx=0):
        """Extract attention weights from a specific GAT layer.

        Useful for interpretability — which neighbors matter most?

        Args:
            data: PyG Data object
            layer_idx: which GAT layer (0-indexed)
            relation_idx: which edge type

        Returns:
            attention_weights: (n_edges, num_heads)
        """
        x = self.input_proj(data.x)
        x = F.elu(x)

        # Forward through layers up to target
        for i in range(layer_idx):
            gat = self.gat_layers[i]
            norm = self.layer_norms[i]
            residual = x
            x = gat(x, data.edge_index, data.edge_type)
            x = norm(x)
            x = F.elu(x)
            x = x + residual

        # Get attention from target layer
        target_gat = self.gat_layers[layer_idx]
        mask = data.edge_type == relation_idx
        rel_edges = data.edge_index[:, mask]

        if rel_edges.size(1) == 0:
            return None

        # GATConv returns attention when return_attention_weights=True
        _, (edge_idx, alpha) = target_gat.gat_layers[relation_idx](
            x, rel_edges, return_attention_weights=True
        )
        return alpha


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024
