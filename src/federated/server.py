"""Phase 11: Federated Learning Server.

Implements FedAvg and FedProx aggregation strategies.
Server coordinates training across multiple clients without
accessing their raw data — only model weight updates are shared.

FedAvg:  weighted average of client model updates
FedProx: FedAvg + proximal term to prevent client drift
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('fl_server')


@dataclass
class RoundResult:
    """Result of one FL communication round."""
    round_num: int
    avg_train_loss: float
    avg_val_loss: float = 0.0
    client_losses: list = field(default_factory=list)
    client_val_losses: list = field(default_factory=list)


@dataclass
class FLResult:
    """Full federated learning result."""
    rounds: list = field(default_factory=list)        # list of RoundResult
    final_global_loss: float = float('inf')
    best_round: int = 0
    strategy: str = 'FedAvg'
    num_clients: int = 0
    total_rounds: int = 0


class FLServer:
    """Federated Learning Server.

    Coordinates training across multiple clients using FedAvg or FedProx.
    Server never sees client data — only receives model weight updates.

    Args:
        global_model: nn.Module — the shared model architecture
        strategy: 'FedAvg' or 'FedProx'
        fedprox_mu: proximal term weight (only for FedProx)
        device: 'cpu' or 'cuda'
    """

    def __init__(self, global_model, strategy=None, fedprox_mu=None, device='cpu'):
        cfg = get_config('fl')

        self.strategy = strategy or cfg.get('strategy', 'FedAvg')
        self.fedprox_mu = fedprox_mu if fedprox_mu is not None else cfg.get('fedprox_mu', 0.01)
        self.device = device

        # Global model (shared architecture)
        self.global_model = copy.deepcopy(global_model).to(device)
        self.best_model_state = None
        self.best_val_loss = float('inf')

        logger.info(f'FLServer initialized: strategy={self.strategy}, '
                    f'fedprox_mu={self.fedprox_mu}')

    def get_global_weights(self):
        """Return a copy of the global model's state dict."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_weights, client_sizes):
        """Aggregate client model updates into global model.

        FedAvg: weighted average by dataset size.
        FedProx: same aggregation (proximal term is applied client-side).

        Args:
            client_weights: list of state_dicts from each client
            client_sizes: list of dataset sizes per client (for weighting)
        """
        if not client_weights:
            raise ValueError("No client weights to aggregate")

        total_size = sum(client_sizes)
        weights_factors = [s / total_size for s in client_sizes]

        # Weighted average of all parameters
        avg_state = {}
        for key in client_weights[0]:
            avg_state[key] = sum(
                w * client_weights[i][key].float()
                for i, w in enumerate(weights_factors)
            )

        self.global_model.load_state_dict(avg_state)
        return avg_state

    def run_fl(self, clients, n_rounds=None, local_epochs=None,
               val_data=None):
        """Run full federated learning loop.

        Args:
            clients: list of FLClient objects
            n_rounds: communication rounds (default from config)
            local_epochs: local training epochs per round (default from config)
            val_data: optional (X, y) tuple for global validation

        Returns:
            FLResult with training history
        """
        cfg = get_config('fl')
        n_rounds = n_rounds or cfg.get('rounds', 50)
        local_epochs = local_epochs or cfg.get('local_epochs', 5)

        result = FLResult(
            strategy=self.strategy,
            num_clients=len(clients),
            total_rounds=n_rounds,
        )

        logger.info(f'Starting FL: {n_rounds} rounds, {len(clients)} clients, '
                    f'{local_epochs} local epochs, strategy={self.strategy}')

        for round_num in range(n_rounds):
            # 1. Broadcast global model to all clients
            global_weights = self.get_global_weights()

            # 2. Each client trains locally
            client_weights_list = []
            client_sizes = []
            client_losses = []

            for i, client in enumerate(clients):
                # Set client model to global weights
                client.set_weights(global_weights)

                # Train locally
                if self.strategy == 'FedProx':
                    loss = client.train_local(
                        epochs=local_epochs,
                        global_weights=global_weights,
                        proximal_mu=self.fedprox_mu,
                    )
                else:
                    loss = client.train_local(epochs=local_epochs)

                # Collect updated weights
                client_weights_list.append(client.get_weights())
                client_sizes.append(client.data_size)
                client_losses.append(loss)

            # 3. Aggregate client updates
            self.aggregate(client_weights_list, client_sizes)

            # 4. Optional global validation
            val_loss = 0.0
            client_val_losses = []
            if val_data is not None:
                val_loss = self._evaluate(val_data)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.get_global_weights()
                    result.best_round = round_num

            # Per-client validation
            for client in clients:
                client.set_weights(self.get_global_weights())
                client_val_losses.append(client.evaluate())

            avg_loss = np.mean(client_losses)
            avg_val = np.mean(client_val_losses) if client_val_losses else 0.0

            round_result = RoundResult(
                round_num=round_num,
                avg_train_loss=avg_loss,
                avg_val_loss=avg_val,
                client_losses=client_losses,
                client_val_losses=client_val_losses,
            )
            result.rounds.append(round_result)

            if (round_num + 1) % max(1, n_rounds // 5) == 0:
                logger.info(f'Round {round_num+1}/{n_rounds}: '
                            f'avg_loss={avg_loss:.4f}, avg_val={avg_val:.4f}')

        result.final_global_loss = result.rounds[-1].avg_train_loss
        logger.info(f'FL complete: final_loss={result.final_global_loss:.4f}, '
                    f'best_round={result.best_round}')

        return result

    def _evaluate(self, val_data):
        """Evaluate global model on validation data."""
        X, y = val_data
        self.global_model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
            pred = self.global_model(X_t)
            loss = nn.functional.mse_loss(pred, y_t)
        return loss.item()

    def get_convergence_curve(self):
        """Return (round_nums, avg_losses) for plotting."""
        return (
            [r.round_num for r in self.result.rounds] if hasattr(self, 'result') else [],
            [r.avg_train_loss for r in self.result.rounds] if hasattr(self, 'result') else [],
        )
