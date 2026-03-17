"""Phase 11: Federated Learning Client.

Each client represents a simulated institutional investor with a
subset of NIFTY 50 stocks (sector-based non-IID split).

Clients:
  0: Banking + Finance (10 stocks)
  1: IT + Telecom (6 stocks)
  2: Pharma + FMCG (8 stocks)
  3: Energy + Auto + Metals + Infra + Others (~23 stocks)
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('fl_client')


# Sector-based client assignment (non-IID split)
CLIENT_SECTORS = {
    0: ['Banking', 'Finance'],           # ~10 stocks
    1: ['IT', 'Telecom'],                # ~6 stocks
    2: ['Pharma', 'FMCG'],              # ~8 stocks
    3: ['Energy', 'Auto', 'Metals', 'Infra', 'Others'],  # ~23 stocks
}


def get_client_sectors(client_id):
    """Return sector names for a client."""
    if client_id not in CLIENT_SECTORS:
        raise ValueError(f"Invalid client_id: {client_id}. Must be 0-3.")
    return CLIENT_SECTORS[client_id]


def get_client_tickers(client_id):
    """Return stock tickers assigned to a client."""
    from src.data.stocks import NIFTY50
    sectors = get_client_sectors(client_id)
    tickers = []
    for sector in sectors:
        if sector in NIFTY50:
            tickers.extend(NIFTY50[sector])
    return tickers


class FLClient:
    """Federated Learning Client.

    Holds local data and model. Trains locally and shares only
    model weights with the server — data never leaves the client.

    Args:
        client_id: 0-3 (sector-based assignment)
        model: nn.Module (same architecture as global model)
        train_data: (X, y) tuple — local training data
        val_data: optional (X, y) — local validation data
        lr: learning rate
        device: 'cpu' or 'cuda'
    """

    def __init__(self, client_id, model, train_data, val_data=None,
                 lr=1e-3, device='cpu'):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.device = device
        self.lr = lr

        # Store data as tensors
        X_train, y_train = train_data
        self.X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        self.data_size = len(X_train)

        if val_data is not None:
            X_val, y_val = val_data
            self.X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
            self.y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
        else:
            self.X_val = None
            self.y_val = None

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        sectors = CLIENT_SECTORS.get(client_id, ['Unknown'])
        logger.info(f'Client {client_id} ({"/".join(sectors)}): '
                    f'{self.data_size} samples')

    def set_weights(self, state_dict):
        """Load global model weights."""
        self.model.load_state_dict(state_dict)
        # Re-create optimizer for new weights
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def get_weights(self):
        """Return current model weights."""
        return copy.deepcopy(self.model.state_dict())

    def train_local(self, epochs=5, global_weights=None, proximal_mu=0.0):
        """Train model on local data for specified epochs.

        Args:
            epochs: number of local training epochs
            global_weights: state_dict for FedProx proximal term
            proximal_mu: FedProx proximal weight (0 = pure FedAvg)

        Returns:
            average training loss over all epochs
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(self.X_train)
            loss = F.mse_loss(pred, self.y_train)

            # FedProx: add proximal term to prevent drift from global model
            if proximal_mu > 0 and global_weights is not None:
                prox_loss = 0.0
                for name, param in self.model.named_parameters():
                    if name in global_weights:
                        global_param = global_weights[name].to(self.device)
                        prox_loss += ((param - global_param) ** 2).sum()
                loss = loss + (proximal_mu / 2) * prox_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self):
        """Evaluate model on local validation data (or train if no val)."""
        self.model.eval()
        X = self.X_val if self.X_val is not None else self.X_train
        y = self.y_val if self.y_val is not None else self.y_train

        with torch.no_grad():
            pred = self.model(X)
            loss = F.mse_loss(pred, y)
        return loss.item()


def create_fl_clients(global_model, client_data, val_data=None,
                      lr=1e-3, device='cpu'):
    """Factory function to create multiple FL clients.

    Args:
        global_model: nn.Module (shared architecture)
        client_data: dict {client_id: (X_train, y_train)}
        val_data: dict {client_id: (X_val, y_val)} or None
        lr: learning rate
        device: 'cpu' or 'cuda'

    Returns:
        list of FLClient objects
    """
    clients = []
    for cid in sorted(client_data.keys()):
        v = val_data.get(cid) if val_data else None
        client = FLClient(
            client_id=cid,
            model=global_model,
            train_data=client_data[cid],
            val_data=v,
            lr=lr,
            device=device,
        )
        clients.append(client)

    logger.info(f'Created {len(clients)} FL clients')
    return clients
