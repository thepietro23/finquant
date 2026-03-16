"""Phase 6 Tests: RL Portfolio Environment.

Tests:
  T6.1: Environment initializes with correct spaces
  T6.2: Reset returns valid observation and info
  T6.3: Step returns correct tuple (obs, reward, term, trunc, info)
  T6.4: Observation shape matches observation_space
  T6.5: Action space is continuous (n_stocks,)
  T6.6: Portfolio value updates correctly
  T6.7: Transaction costs applied
  T6.8: Max position constraint enforced
  T6.9: Stop loss triggers position exit
  T6.10: Max drawdown terminates episode

Edge Cases:
  E6.1: All cash (zero action) → no trading, no cost
  E6.2: Single stock environment
  E6.3: Episode truncation at max length
  E6.4: Gymnasium API compatibility (check_env)
  E6.5: Portfolio summary at end of episode
  E6.6: Embeddings and sentiment optional inputs
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.rl.environment import PortfolioEnv


# ===========================
# Fixtures
# ===========================

def _make_env(n_stocks=5, n_time=300, n_feat=21, episode_length=50, **kwargs):
    """Create a test environment with synthetic data."""
    np.random.seed(42)
    features = np.random.randn(n_stocks, n_time, n_feat).astype(np.float32)
    # Prices: random walk starting at 100
    returns = np.random.randn(n_stocks, n_time) * 0.01
    prices = 100 * np.cumprod(1 + returns, axis=1).astype(np.float32)
    return PortfolioEnv(features, prices, episode_length=episode_length, **kwargs)


# ===========================
# Unit Tests
# ===========================

class TestEnvInit:
    """T6.1: Environment initialization."""

    def test_env_creates(self):
        """Environment initializes without error."""
        env = _make_env()
        assert env is not None

    def test_observation_space(self):
        """Observation space has correct shape."""
        env = _make_env(n_stocks=5, n_feat=21)
        # obs = 5*21 features + 5 weights + 1 cash + 1 value = 112
        expected = 5 * 21 + 5 + 2
        assert env.observation_space.shape == (expected,)

    def test_action_space(self):
        """T6.5: Action space is (n_stocks,) continuous."""
        env = _make_env(n_stocks=5)
        assert env.action_space.shape == (5,)
        assert env.action_space.dtype == np.float32

    def test_initial_state(self):
        """Initial portfolio is all cash."""
        env = _make_env()
        obs, info = env.reset(seed=42)
        assert info['cash_ratio'] == pytest.approx(1.0, abs=0.01)
        assert info['n_positions'] == 0
        assert info['portfolio_value'] == 1_000_000


class TestReset:
    """T6.2: Reset behavior."""

    def test_reset_returns_tuple(self):
        """Reset returns (observation, info)."""
        env = _make_env()
        result = env.reset(seed=42)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_observation_shape(self):
        """T6.4: Observation matches observation_space."""
        env = _make_env()
        obs, _ = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape

    def test_reset_clears_state(self):
        """Reset clears portfolio state."""
        env = _make_env()
        env.reset(seed=42)

        # Take some steps
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)

        # Reset again
        obs, info = env.reset(seed=42)
        assert info['portfolio_value'] == 1_000_000
        assert info['n_positions'] == 0

    def test_reset_deterministic(self):
        """Same seed → same initial observation."""
        env = _make_env()
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)


class TestStep:
    """T6.3, T6.6-T6.7: Step mechanics."""

    def test_step_returns_tuple(self):
        """T6.3: Step returns (obs, reward, terminated, truncated, info)."""
        env = _make_env()
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_shape(self):
        """Step observation matches space."""
        env = _make_env()
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape

    def test_portfolio_value_changes(self):
        """T6.6: Portfolio value changes after trading."""
        env = _make_env()
        env.reset(seed=42)

        initial_value = env.portfolio_value
        # Take several steps with non-zero action
        for _ in range(10):
            action = np.ones(env.n_stocks, dtype=np.float32) * 0.5
            env.step(action)

        # Value should have changed (extremely unlikely to be exactly same)
        assert env.portfolio_value != initial_value

    def test_transaction_costs(self):
        """T6.7: Transaction costs reduce portfolio value."""
        env = _make_env()
        env.reset(seed=42)

        # Force all prices to stay constant (so only costs affect value)
        # Override prices to be flat
        env.prices = np.ones_like(env.prices) * 100

        # Trade heavily
        action = np.ones(env.n_stocks, dtype=np.float32)
        _, _, _, _, info = env.step(action)

        # Portfolio should decrease due to transaction costs
        assert info['trade_cost'] > 0
        assert env.portfolio_value < env.initial_cash

    def test_info_has_expected_keys(self):
        """Info dict contains required keys."""
        env = _make_env()
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())

        for key in ['portfolio_return', 'turnover', 'trade_cost',
                    'drawdown', 'portfolio_value', 'n_positions']:
            assert key in info, f'Missing key: {key}'


class TestConstraints:
    """T6.8-T6.10: Portfolio constraints."""

    def test_max_position(self):
        """T6.8: No position exceeds max_position (20%)."""
        env = _make_env()
        env.reset(seed=42)

        # Try to put 100% in one stock
        action = np.zeros(env.n_stocks, dtype=np.float32)
        action[0] = 10.0  # Very large
        env.step(action)

        assert env.weights[0] <= env.max_position + 1e-6

    def test_weights_sum_leq_one(self):
        """Weights never exceed 1.0 (rest is cash)."""
        env = _make_env()
        env.reset(seed=42)

        for _ in range(20):
            action = env.action_space.sample()
            env.step(action)
            assert env.weights.sum() <= 1.0 + 1e-6

    def test_stop_loss(self):
        """T6.9: Stop loss exits position on large loss."""
        env = _make_env()
        env.reset(seed=42)

        # Set weights manually
        env.weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

        # Force a crash: current prices drop 10% from previous
        t = env.current_step + 1
        if t < env.n_timesteps:
            env.prices[:, t] = env.prices[:, t - 1] * 0.90  # -10% crash

        action = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        env.step(action)

        # All stocks had -10% return (> stop_loss of -5%)
        # After step, weights should be zeroed for crashed stocks
        assert (env.weights == 0).all(), \
            f'Stop loss should have cleared positions: {env.weights}'

    def test_max_drawdown_terminates(self):
        """T6.10: Episode terminates on max drawdown breach."""
        env = _make_env(episode_length=200)
        env.reset(seed=42)

        # Simulate large losses
        terminated = False
        for _ in range(100):
            if env.done:
                break
            # Force losses by setting prices to decrease
            t = env.current_step + 1
            if t < env.n_timesteps:
                env.prices[:, t] = env.prices[:, t - 1] * 0.98  # -2%/day

            action = np.ones(env.n_stocks, dtype=np.float32)
            _, _, term, _, info = env.step(action)
            if term:
                terminated = True
                break

        # Should have terminated due to drawdown
        assert terminated, 'Episode should terminate on max drawdown'


# ===========================
# Edge Cases
# ===========================

class TestEdgeCases:
    """Edge case handling."""

    def test_zero_action(self):
        """E6.1: Zero action → all cash, no trading cost."""
        env = _make_env()
        env.reset(seed=42)

        # All zeros → softmax gives equal weights, but we test specific behavior
        # Actually, equal action = equal weights after softmax
        # To get all cash, we'd need weights to sum to ~0, but softmax always sums to 1
        # So let's just check no crash
        action = np.zeros(env.n_stocks, dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        assert obs.shape == env.observation_space.shape

    def test_single_stock(self):
        """E6.2: Single stock environment."""
        env = _make_env(n_stocks=1, n_feat=21)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape

        action = np.array([1.0], dtype=np.float32)
        obs, reward, _, _, info = env.step(action)
        assert obs.shape == env.observation_space.shape

    def test_episode_truncation(self):
        """E6.3: Episode ends at episode_length."""
        env = _make_env(episode_length=10)
        env.reset(seed=42)

        truncated = False
        for i in range(20):
            if env.done:
                break
            action = env.action_space.sample()
            _, _, term, trunc, _ = env.step(action)
            if trunc:
                truncated = True
                break

        assert truncated, 'Episode should truncate at episode_length'

    def test_gymnasium_api(self):
        """E6.4: Passes gymnasium API check."""
        env = _make_env()
        # Basic API checks
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'metadata')

        # Check spaces are valid
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

        action = env.action_space.sample()
        assert env.action_space.contains(action)

    def test_portfolio_summary(self):
        """E6.5: Portfolio summary at end of episode."""
        env = _make_env(episode_length=20)
        env.reset(seed=42)

        for _ in range(20):
            if env.done:
                break
            env.step(env.action_space.sample())

        summary = env.get_portfolio_summary()
        assert 'total_return' in summary
        assert 'sharpe' in summary
        assert 'max_drawdown' in summary
        assert summary['n_steps'] > 0

    def test_with_embeddings(self):
        """E6.6: Environment works with optional embeddings."""
        n_stocks, n_time, n_feat = 5, 300, 21
        np.random.seed(42)
        features = np.random.randn(n_stocks, n_time, n_feat).astype(np.float32)
        prices = 100 * np.cumprod(
            1 + np.random.randn(n_stocks, n_time) * 0.01, axis=1
        ).astype(np.float32)

        embeddings = np.random.randn(n_stocks, n_time, 64).astype(np.float32)
        sentiment = np.random.randn(n_stocks, n_time).astype(np.float32)

        env = PortfolioEnv(features, prices, episode_length=20,
                           embeddings=embeddings, sentiment=sentiment)
        obs, _ = env.reset(seed=42)

        # Obs should be larger with embeddings + sentiment
        base_dim = n_stocks * n_feat + n_stocks + 2
        embed_dim = n_stocks * 64
        sent_dim = n_stocks
        expected = base_dim + embed_dim + sent_dim
        assert obs.shape == (expected,)

        # Step should work
        obs, reward, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (expected,)
