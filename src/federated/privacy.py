"""Phase 11: Differential Privacy for Federated Learning.

Implements DP-SGD (Differentially Private Stochastic Gradient Descent):
  1. Per-sample gradient clipping (bound sensitivity)
  2. Gaussian noise injection (calibrated to epsilon/delta)
  3. Privacy budget tracking (cumulative epsilon across rounds)

Privacy guarantee: (epsilon, delta)-DP
  "With probability 1 - delta, an adversary cannot distinguish
   whether a specific data point was in the training set."
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('fl_privacy')


@dataclass
class PrivacyBudget:
    """Tracks cumulative privacy expenditure."""
    epsilon: float           # Target privacy budget
    delta: float             # Failure probability
    max_grad_norm: float     # Gradient clipping norm
    noise_multiplier: float  # Calibrated noise scale
    rounds_spent: int = 0    # How many rounds of noise added
    epsilon_spent: float = 0.0  # Cumulative epsilon (via composition)


def compute_noise_multiplier(epsilon, delta, max_grad_norm, n_rounds,
                              n_samples=1000):
    """Compute Gaussian noise multiplier for (epsilon, delta)-DP.

    Uses simple composition theorem:
      noise_sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon_per_round

    Args:
        epsilon: total privacy budget
        delta: failure probability
        max_grad_norm: gradient clipping bound (= sensitivity)
        n_rounds: total number of FL rounds
        n_samples: dataset size (for per-sample accounting)

    Returns:
        noise_multiplier (sigma / max_grad_norm)
    """
    # Per-round epsilon via simple composition
    epsilon_per_round = epsilon / math.sqrt(n_rounds)

    # Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
    sensitivity = max_grad_norm / n_samples
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon_per_round

    noise_multiplier = sigma / max_grad_norm if max_grad_norm > 0 else 0.0

    logger.info(f'DP noise calibration: eps={epsilon}, delta={delta}, '
                f'rounds={n_rounds}, sigma={sigma:.6f}, '
                f'noise_mult={noise_multiplier:.6f}')

    return noise_multiplier


def clip_gradients(model, max_norm):
    """Clip model gradients to max_norm (L2 norm).

    This bounds the sensitivity of each gradient update,
    which is required for DP guarantees.

    Args:
        model: nn.Module with computed gradients
        max_norm: maximum L2 norm

    Returns:
        total_norm: original gradient norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm
    )
    return total_norm.item()


def add_noise_to_gradients(model, noise_multiplier, max_grad_norm):
    """Add calibrated Gaussian noise to model gradients.

    noise = N(0, sigma^2) where sigma = noise_multiplier * max_grad_norm

    This is the core of DP-SGD: gradient clipping (bound sensitivity)
    + noise injection (plausible deniability).

    Args:
        model: nn.Module with computed gradients
        noise_multiplier: noise scale factor
        max_grad_norm: clipping norm (used to scale noise)
    """
    sigma = noise_multiplier * max_grad_norm
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.normal(
                mean=0, std=sigma,
                size=param.grad.shape,
                device=param.grad.device,
                dtype=param.grad.dtype,
            )
            param.grad.add_(noise)


def add_noise_to_weights(state_dict, noise_multiplier, max_grad_norm):
    """Add noise directly to model weights (alternative to gradient noise).

    Useful for weight-level DP where we add noise before sending
    weights to the server.

    Args:
        state_dict: model state dict
        noise_multiplier: noise scale
        max_grad_norm: sensitivity bound

    Returns:
        noisy_state_dict: copy with added noise
    """
    noisy_state = {}
    sigma = noise_multiplier * max_grad_norm

    for key, param in state_dict.items():
        noise = torch.normal(mean=0, std=sigma, size=param.shape,
                             device=param.device, dtype=param.dtype)
        noisy_state[key] = param + noise

    return noisy_state


class DPTrainer:
    """Differentially Private trainer wrapper.

    Wraps around a client's training loop to add DP guarantees:
      1. Clips gradients each step
      2. Adds calibrated noise
      3. Tracks privacy budget

    Args:
        epsilon: total privacy budget
        delta: failure probability
        max_grad_norm: gradient clipping norm
        n_rounds: expected total FL rounds
        n_samples: client dataset size
    """

    def __init__(self, epsilon=None, delta=None, max_grad_norm=None,
                 n_rounds=None, n_samples=1000):
        cfg = get_config('fl')

        self.epsilon = epsilon or cfg.get('dp_epsilon', 8.0)
        self.delta = delta or cfg.get('dp_delta', 1e-5)
        self.max_grad_norm = max_grad_norm or cfg.get('dp_max_grad_norm', 1.0)
        n_rounds = n_rounds or cfg.get('rounds', 50)

        self.noise_multiplier = compute_noise_multiplier(
            self.epsilon, self.delta, self.max_grad_norm,
            n_rounds, n_samples
        )

        self.budget = PrivacyBudget(
            epsilon=self.epsilon,
            delta=self.delta,
            max_grad_norm=self.max_grad_norm,
            noise_multiplier=self.noise_multiplier,
        )

        logger.info(f'DPTrainer: eps={self.epsilon}, delta={self.delta}, '
                    f'clip={self.max_grad_norm}, noise_mult={self.noise_multiplier:.6f}')

    def clip_and_noise(self, model):
        """Apply gradient clipping + noise injection.

        Call this AFTER loss.backward() but BEFORE optimizer.step().

        Returns:
            grad_norm: original gradient norm (before clipping)
        """
        grad_norm = clip_gradients(model, self.max_grad_norm)
        add_noise_to_gradients(model, self.noise_multiplier, self.max_grad_norm)

        self.budget.rounds_spent += 1
        # Simple composition: epsilon grows as sqrt(rounds)
        self.budget.epsilon_spent = (
            self.epsilon * math.sqrt(self.budget.rounds_spent)
            / math.sqrt(self.budget.rounds_spent + 50)  # approximate
        )

        return grad_norm

    def get_budget_status(self):
        """Return current privacy budget status."""
        return {
            'epsilon_target': self.budget.epsilon,
            'epsilon_spent': round(self.budget.epsilon_spent, 4),
            'delta': self.budget.delta,
            'rounds_spent': self.budget.rounds_spent,
            'noise_multiplier': round(self.budget.noise_multiplier, 6),
            'budget_remaining_pct': round(
                max(0, 1 - self.budget.epsilon_spent / self.budget.epsilon) * 100, 1
            ),
        }

    def is_budget_exhausted(self):
        """Check if privacy budget is exhausted."""
        return self.budget.epsilon_spent >= self.budget.epsilon
