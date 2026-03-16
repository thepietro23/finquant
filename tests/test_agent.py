"""Phase 7 Tests: Deep RL Agents — PPO + SAC.

Tests:
  T7.1: PPO agent creates with correct config
  T7.2: SAC agent creates with correct config
  T7.3: PPO trains without error (short run)
  T7.4: SAC trains without error (short run)
  T7.5: Evaluation returns expected metrics
  T7.6: Model save and load works
  T7.7: Agent predicts valid actions
  T7.8: Custom hyperparameters accepted

Edge Cases:
  E7.1: Single stock environment
  E7.2: Very short training (10 steps)
  E7.3: Compare agents function
  E7.4: Training with eval callback
"""

import os
import sys
import tempfile

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.rl.environment import PortfolioEnv
from src.rl.agent import (
    create_ppo_agent,
    create_sac_agent,
    train_agent,
    evaluate_agent,
    compare_agents,
    save_agent,
    load_agent,
)


# ===========================
# Fixtures
# ===========================

def _make_env(n_stocks=5, n_time=300, episode_length=50):
    """Create test environment."""
    np.random.seed(42)
    features = np.random.randn(n_stocks, n_time, 21).astype(np.float32)
    prices = 100 * np.cumprod(
        1 + np.random.randn(n_stocks, n_time) * 0.01, axis=1
    ).astype(np.float32)
    return PortfolioEnv(features, prices, episode_length=episode_length)


# ===========================
# Unit Tests
# ===========================

class TestPPOAgent:
    """T7.1, T7.3: PPO agent creation and training."""

    def test_ppo_creates(self):
        """T7.1: PPO agent initializes."""
        env = _make_env()
        model = create_ppo_agent(env, device='cpu')
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'learn')

    def test_ppo_trains(self):
        """T7.3: PPO trains without error."""
        env = _make_env()
        model = create_ppo_agent(env, device='cpu')
        model.learn(total_timesteps=200)
        # If we get here without error, training works

    def test_ppo_predicts(self):
        """T7.7: PPO outputs valid actions."""
        env = _make_env()
        model = create_ppo_agent(env, device='cpu')
        model.learn(total_timesteps=200)

        obs, _ = env.reset(seed=42)
        action, _ = model.predict(obs, deterministic=True)

        assert action.shape == (env.n_stocks,)
        assert env.action_space.contains(action)


class TestSACAgent:
    """T7.2, T7.4: SAC agent creation and training."""

    def test_sac_creates(self):
        """T7.2: SAC agent initializes."""
        env = _make_env()
        model = create_sac_agent(env, device='cpu')
        assert model is not None

    def test_sac_trains(self):
        """T7.4: SAC trains without error."""
        env = _make_env()
        model = create_sac_agent(env, device='cpu',
                                 buffer_size=500, batch_size=32,
                                 learning_starts=50)
        model.learn(total_timesteps=200)

    def test_sac_predicts(self):
        """SAC outputs valid actions."""
        env = _make_env()
        model = create_sac_agent(env, device='cpu',
                                 buffer_size=500, batch_size=32,
                                 learning_starts=50)
        model.learn(total_timesteps=200)

        obs, _ = env.reset(seed=42)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (env.n_stocks,)


class TestEvaluation:
    """T7.5: Evaluation metrics."""

    def test_evaluate_returns_metrics(self):
        """T7.5: evaluate_agent returns expected keys."""
        env = _make_env()
        model = create_ppo_agent(env, device='cpu')
        model.learn(total_timesteps=200)

        metrics = evaluate_agent(model, env, n_episodes=2)

        assert 'mean_return' in metrics
        assert 'mean_sharpe' in metrics
        assert 'mean_max_dd' in metrics
        assert 'mean_steps' in metrics
        assert metrics['n_episodes'] == 2

    def test_evaluate_returns_finite(self):
        """Evaluation metrics are finite numbers."""
        env = _make_env()
        model = create_ppo_agent(env, device='cpu')
        model.learn(total_timesteps=200)

        metrics = evaluate_agent(model, env, n_episodes=2)
        assert np.isfinite(metrics['mean_return'])
        assert np.isfinite(metrics['mean_sharpe'])


class TestSaveLoad:
    """T7.6: Model persistence."""

    def test_save_and_load_ppo(self):
        """T7.6: PPO save → load → predict same action."""
        env = _make_env()
        model = create_ppo_agent(env, device='cpu')
        model.learn(total_timesteps=200)

        obs, _ = env.reset(seed=42)
        action_before, _ = model.predict(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_ppo')
            save_agent(model, path)

            loaded = load_agent(path, env=env, algorithm='PPO')
            action_after, _ = loaded.predict(obs, deterministic=True)

        np.testing.assert_array_almost_equal(action_before, action_after)

    def test_save_and_load_sac(self):
        """SAC save → load → predict same action."""
        env = _make_env()
        model = create_sac_agent(env, device='cpu',
                                 buffer_size=500, batch_size=32,
                                 learning_starts=50)
        model.learn(total_timesteps=200)

        obs, _ = env.reset(seed=42)
        action_before, _ = model.predict(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_sac')
            save_agent(model, path)

            loaded = load_agent(path, env=env, algorithm='SAC')
            action_after, _ = loaded.predict(obs, deterministic=True)

        np.testing.assert_array_almost_equal(action_before, action_after)


class TestCustomConfig:
    """T7.8: Custom hyperparameters."""

    def test_custom_ppo(self):
        """PPO accepts custom learning rate."""
        env = _make_env()
        model = create_ppo_agent(env, device='cpu', learning_rate=0.001)
        assert model.learning_rate == 0.001

    def test_custom_policy_arch(self):
        """Custom policy network architecture."""
        env = _make_env()
        model = create_ppo_agent(
            env, device='cpu',
            policy_kwargs={'net_arch': dict(pi=[64, 32], vf=[64, 32])}
        )
        model.learn(total_timesteps=100)
        # No error = architecture accepted


# ===========================
# Edge Cases
# ===========================

class TestEdgeCases:
    """Edge case handling."""

    def test_single_stock(self):
        """E7.1: Agent works with 1-stock environment."""
        env = _make_env(n_stocks=1)
        model = create_ppo_agent(env, device='cpu')
        model.learn(total_timesteps=200)

        obs, _ = env.reset(seed=42)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1,)

    def test_very_short_training(self):
        """E7.2: 10 steps of training doesn't crash."""
        env = _make_env()
        model = create_ppo_agent(env, device='cpu', n_steps=64)
        model.learn(total_timesteps=64)

    def test_compare_agents(self):
        """E7.3: Compare function returns winner."""
        env = _make_env()
        ppo = create_ppo_agent(env, device='cpu')
        ppo.learn(total_timesteps=200)

        sac = create_sac_agent(env, device='cpu',
                               buffer_size=500, batch_size=32,
                               learning_starts=50)
        sac.learn(total_timesteps=200)

        result = compare_agents(ppo, sac, env, n_episodes=2)
        assert 'ppo' in result
        assert 'sac' in result
        assert result['winner'] in ('PPO', 'SAC')

    def test_train_with_eval(self):
        """E7.4: Training with evaluation callback."""
        env = _make_env()
        eval_env = _make_env()
        model = create_ppo_agent(env, device='cpu', n_steps=64)

        model, metrics = train_agent(
            model,
            total_timesteps=200,
            eval_env=eval_env,
            eval_freq=100,
        )
        # metrics may be empty with short training, but no crash
        assert isinstance(metrics, list)
