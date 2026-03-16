"""Phase 7: Deep RL Agents — PPO (primary) + SAC (comparison).

Wraps Stable-Baselines3 algorithms with:
  - Custom policy network sizes for 4GB VRAM
  - Training with callbacks (logging, early stopping)
  - Model saving/loading
  - Evaluation and comparison utilities
"""

import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.metrics import sharpe_ratio, max_drawdown

logger = get_logger('rl_agent')


# ---------------------------------------------------------------------------
# Custom callback for logging portfolio metrics
# ---------------------------------------------------------------------------

class PortfolioMetricsCallback(BaseCallback):
    """Log portfolio-specific metrics during training."""

    def __init__(self, eval_env, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_sharpe = -np.inf
        self.metrics_history = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            metrics = evaluate_agent(self.model, self.eval_env, n_episodes=3)
            self.metrics_history.append(metrics)

            if metrics['mean_sharpe'] > self.best_sharpe:
                self.best_sharpe = metrics['mean_sharpe']

            if self.verbose > 0:
                logger.info(
                    f'Step {self.n_calls}: '
                    f'return={metrics["mean_return"]:.2%}, '
                    f'sharpe={metrics["mean_sharpe"]:.2f}, '
                    f'max_dd={metrics["mean_max_dd"]:.2%}'
                )
        return True


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------

def create_ppo_agent(env, device='auto', **kwargs):
    """Create PPO agent with config-driven hyperparameters.

    Args:
        env: Gymnasium environment (or VecEnv)
        device: 'auto', 'cpu', or 'cuda'
        **kwargs: Override any PPO parameter

    Returns:
        PPO model
    """
    cfg = get_config('rl')

    # Default policy network: small for 4GB VRAM
    policy_kwargs = kwargs.pop('policy_kwargs', {
        'net_arch': dict(pi=[128, 64], vf=[128, 64]),
    })

    params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': cfg.get('lr', 0.0003),
        'n_steps': cfg.get('n_steps', 2048),
        'batch_size': cfg.get('batch_size', 64),
        'n_epochs': cfg.get('n_epochs', 10),
        'gamma': cfg.get('gamma', 0.99),
        'clip_range': cfg.get('clip_range', 0.2),
        'policy_kwargs': policy_kwargs,
        'device': device,
        'verbose': 0,
    }
    params.update(kwargs)

    model = PPO(**params)
    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f'PPO agent created: {n_params:,} policy parameters, '
                f'device={model.device}')
    return model


def create_sac_agent(env, device='auto', **kwargs):
    """Create SAC agent with config-driven hyperparameters.

    Args:
        env: Gymnasium environment (or VecEnv)
        device: 'auto', 'cpu', or 'cuda'
        **kwargs: Override any SAC parameter

    Returns:
        SAC model
    """
    cfg_rl = get_config('rl')
    cfg_sac = cfg_rl.get('sac', {})

    policy_kwargs = kwargs.pop('policy_kwargs', {
        'net_arch': dict(pi=[128, 64], qf=[128, 64]),
    })

    params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': cfg_sac.get('lr', 0.0003),
        'buffer_size': cfg_sac.get('buffer_size', 100000),
        'batch_size': cfg_sac.get('batch_size', 256),
        'tau': cfg_sac.get('tau', 0.005),
        'gamma': cfg_rl.get('gamma', 0.99),
        'ent_coef': cfg_sac.get('ent_coef', 'auto'),
        'policy_kwargs': policy_kwargs,
        'device': device,
        'verbose': 0,
    }
    params.update(kwargs)

    model = SAC(**params)
    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f'SAC agent created: {n_params:,} policy parameters, '
                f'device={model.device}')
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_agent(model, total_timesteps=None, eval_env=None,
                eval_freq=5000, save_path=None, callbacks=None):
    """Train an RL agent.

    Args:
        model: SB3 model (PPO or SAC)
        total_timesteps: Training steps (default from config)
        eval_env: Optional evaluation environment
        eval_freq: Evaluation frequency in steps
        save_path: Path to save best model
        callbacks: Additional callbacks

    Returns:
        model: Trained model
        metrics: Training metrics history
    """
    cfg = get_config('rl')
    if total_timesteps is None:
        total_timesteps = cfg.get('total_timesteps', 500000)

    callback_list = []

    # Portfolio metrics callback
    if eval_env is not None:
        portfolio_cb = PortfolioMetricsCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,
            verbose=1,
        )
        callback_list.append(portfolio_cb)

    # SB3 eval callback for best model saving
    if eval_env is not None and save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.dirname(save_path),
            eval_freq=eval_freq,
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        )
        callback_list.append(eval_cb)

    if callbacks:
        callback_list.extend(callbacks)

    combined_cb = CallbackList(callback_list) if callback_list else None

    logger.info(f'Training {type(model).__name__} for {total_timesteps:,} steps')
    model.learn(total_timesteps=total_timesteps, callback=combined_cb)
    logger.info('Training complete')

    # Save final model
    if save_path is not None:
        model.save(save_path)
        logger.info(f'Model saved to {save_path}')

    metrics = portfolio_cb.metrics_history if eval_env is not None else []
    return model, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(model, env, n_episodes=5, deterministic=True):
    """Evaluate agent performance over multiple episodes.

    Args:
        model: Trained SB3 model
        env: Evaluation environment
        n_episodes: Number of episodes to average
        deterministic: Use deterministic policy

    Returns:
        dict with mean metrics across episodes
    """
    all_returns = []
    all_sharpes = []
    all_max_dds = []
    all_steps = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_returns = []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if 'portfolio_return' in info:
                episode_returns.append(info['portfolio_return'])

        summary = env.get_portfolio_summary()
        all_returns.append(summary['total_return'])
        all_sharpes.append(summary['sharpe'])
        all_max_dds.append(summary['max_drawdown'])
        all_steps.append(summary['n_steps'])

    return {
        'mean_return': np.mean(all_returns),
        'std_return': np.std(all_returns),
        'mean_sharpe': np.mean(all_sharpes),
        'mean_max_dd': np.mean(all_max_dds),
        'mean_steps': np.mean(all_steps),
        'n_episodes': n_episodes,
    }


def compare_agents(ppo_model, sac_model, env, n_episodes=10):
    """Compare PPO vs SAC performance.

    Args:
        ppo_model: Trained PPO model
        sac_model: Trained SAC model
        env: Evaluation environment
        n_episodes: Episodes per agent

    Returns:
        dict with comparison results
    """
    ppo_metrics = evaluate_agent(ppo_model, env, n_episodes)
    sac_metrics = evaluate_agent(sac_model, env, n_episodes)

    logger.info(
        f'PPO: return={ppo_metrics["mean_return"]:.2%}, '
        f'sharpe={ppo_metrics["mean_sharpe"]:.2f}'
    )
    logger.info(
        f'SAC: return={sac_metrics["mean_return"]:.2%}, '
        f'sharpe={sac_metrics["mean_sharpe"]:.2f}'
    )

    return {
        'ppo': ppo_metrics,
        'sac': sac_metrics,
        'winner': 'PPO' if ppo_metrics['mean_sharpe'] > sac_metrics['mean_sharpe'] else 'SAC',
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_agent(model, path):
    """Save model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logger.info(f'Agent saved to {path}')


def load_agent(path, env=None, algorithm='PPO'):
    """Load model from disk.

    Args:
        path: Path to saved model (.zip)
        env: Optional environment for continued training
        algorithm: 'PPO' or 'SAC'

    Returns:
        Loaded model
    """
    cls = PPO if algorithm.upper() == 'PPO' else SAC
    model = cls.load(path, env=env)
    logger.info(f'{algorithm} agent loaded from {path}')
    return model
