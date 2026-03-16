"""Phase 6: Portfolio Management RL Environment.

Gymnasium-compatible environment for portfolio optimization.

Observation: stock features + current portfolio state
Action: target portfolio weights
Reward: risk-adjusted return (Sharpe-based)

Constraints:
  - Max 20% per stock
  - Stop loss: -5% per stock
  - Max drawdown circuit breaker: -15%
  - Transaction costs: 0.1% + 0.05% slippage
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('rl_env')


class PortfolioEnv(gym.Env):
    """Portfolio management environment.

    State: (n_stocks * n_features) + n_stocks (current weights) + 2 (cash, portfolio_value)
    Action: n_stocks continuous values → softmax → target weights
    Reward: risk-adjusted daily return with penalties

    Args:
        feature_tensor: numpy array (n_stocks, n_timesteps, n_features)
        price_tensor: numpy array (n_stocks, n_timesteps) — close prices
        initial_cash: starting portfolio value
        episode_length: max steps per episode (default: 252 = 1 year)
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, feature_tensor, price_tensor, initial_cash=1_000_000,
                 episode_length=None, embeddings=None, sentiment=None):
        super().__init__()

        cfg_rl = get_config('rl')
        cfg_data = get_config('data')

        self.features = feature_tensor  # (n_stocks, n_timesteps, n_features)
        self.prices = price_tensor      # (n_stocks, n_timesteps)
        self.n_stocks = feature_tensor.shape[0]
        self.n_timesteps = feature_tensor.shape[1]
        self.n_features = feature_tensor.shape[2]
        self.initial_cash = initial_cash

        # Optional: T-GAT embeddings and sentiment scores
        self.embeddings = embeddings    # (n_stocks, n_timesteps, embed_dim) or None
        self.sentiment = sentiment      # (n_stocks, n_timesteps) or None

        # Config
        self.episode_length = episode_length or cfg_rl.get('episode_length', 252)
        self.max_position = cfg_rl.get('max_position', 0.20)
        self.stop_loss = cfg_rl.get('stop_loss', -0.05)
        self.max_drawdown = cfg_rl.get('max_drawdown', -0.15)
        self.transaction_cost = cfg_data.get('transaction_cost', 0.001)
        self.slippage = cfg_data.get('slippage', 0.0005)
        self.trading_days = cfg_data.get('trading_days_per_year', 248)

        # Reward weights
        reward_cfg = cfg_rl.get('reward', {})
        self.sharpe_weight = reward_cfg.get('sharpe_weight', 1.0)
        self.drawdown_penalty = reward_cfg.get('drawdown_penalty', 0.1)
        self.turnover_penalty = reward_cfg.get('turnover_penalty', 0.01)

        # Observation dimension
        self.obs_feature_dim = self.n_stocks * self.n_features
        self.obs_portfolio_dim = self.n_stocks + 2  # weights + cash_ratio + norm_value
        embed_dim = 0
        if self.embeddings is not None:
            embed_dim = self.embeddings.shape[2] * self.n_stocks
        sentiment_dim = self.n_stocks if self.sentiment is not None else 0
        self.obs_dim = (self.obs_feature_dim + self.obs_portfolio_dim +
                        embed_dim + sentiment_dim)

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_stocks,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # State variables (initialized in reset)
        self.current_step = 0
        self.start_step = 0
        self.weights = np.zeros(self.n_stocks, dtype=np.float32)
        self.cash = initial_cash
        self.portfolio_value = initial_cash
        self.peak_value = initial_cash
        self.returns_history = []
        self.done = False

        logger.info(f'PortfolioEnv: {self.n_stocks} stocks, '
                    f'{self.n_timesteps} timesteps, '
                    f'obs_dim={self.obs_dim}, '
                    f'episode_length={self.episode_length}')

    def reset(self, seed=None, options=None):
        """Reset environment to initial state.

        Returns:
            observation, info dict
        """
        super().reset(seed=seed)

        # Random start within available data (leave room for episode)
        max_start = self.n_timesteps - self.episode_length - 1
        if max_start <= 0:
            self.start_step = 0
        else:
            self.start_step = self.np_random.integers(0, max_start)

        self.current_step = self.start_step
        self.weights = np.zeros(self.n_stocks, dtype=np.float32)
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        self.returns_history = []
        self.done = False

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one trading day.

        Args:
            action: raw action (n_stocks,) — will be converted to weights

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        # 1. Convert action to target weights
        target_weights = self._action_to_weights(action)

        # 2. Apply position limits
        target_weights = np.clip(target_weights, 0, self.max_position)
        # Re-normalize to sum ≤ 1
        weight_sum = target_weights.sum()
        if weight_sum > 1.0:
            target_weights = target_weights / weight_sum

        # 3. Calculate turnover (trading cost)
        turnover = np.abs(target_weights - self.weights).sum()
        trade_cost = turnover * (self.transaction_cost + self.slippage)

        # 4. Update weights
        old_weights = self.weights.copy()
        self.weights = target_weights

        # 5. Move to next day and calculate returns
        self.current_step += 1
        t = self.current_step

        if t >= self.n_timesteps:
            # End of data
            self.done = True
            obs = self._get_observation()
            return obs, 0.0, True, False, self._get_info()

        # Daily stock returns
        prev_prices = self.prices[:, t - 1]
        curr_prices = self.prices[:, t]

        # Avoid division by zero
        safe_prev = np.where(prev_prices > 0, prev_prices, 1.0)
        stock_returns = (curr_prices - prev_prices) / safe_prev

        # 6. Portfolio return
        portfolio_return = np.dot(self.weights, stock_returns)
        # Cash portion earns nothing (simplification)
        cash_weight = 1.0 - self.weights.sum()
        # Net return after costs
        net_return = portfolio_return - trade_cost

        # 7. Update portfolio value
        self.portfolio_value *= (1 + net_return)
        self.returns_history.append(net_return)

        # 8. Update peak for drawdown
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # 9. Check stop conditions
        terminated = False
        truncated = False

        # Max drawdown circuit breaker
        current_drawdown = (self.portfolio_value - self.peak_value) / self.peak_value
        if current_drawdown < self.max_drawdown:
            terminated = True
            self.done = True

        # Episode length reached
        steps_done = self.current_step - self.start_step
        if steps_done >= self.episode_length:
            truncated = True
            self.done = True

        # 10. Apply stop loss per stock
        for i in range(self.n_stocks):
            if self.weights[i] > 0 and stock_returns[i] < self.stop_loss:
                self.weights[i] = 0  # Force exit

        # 11. Calculate reward
        reward = self._calculate_reward(
            net_return, turnover, current_drawdown
        )

        obs = self._get_observation()
        info = self._get_info()
        info['portfolio_return'] = net_return
        info['turnover'] = turnover
        info['trade_cost'] = trade_cost
        info['drawdown'] = current_drawdown

        return obs, reward, terminated, truncated, info

    def _action_to_weights(self, action):
        """Convert raw action to portfolio weights using softmax."""
        # Shift to positive, then normalize
        exp_action = np.exp(action - np.max(action))  # Numerical stability
        weights = exp_action / exp_action.sum()
        return weights.astype(np.float32)

    def _get_observation(self):
        """Build observation vector."""
        t = min(self.current_step, self.n_timesteps - 1)

        # Stock features: (n_stocks, n_features) → flattened
        features_flat = self.features[:, t, :].flatten()

        # Portfolio state: current weights + cash ratio + normalized value
        cash_ratio = np.array([1.0 - self.weights.sum()], dtype=np.float32)
        norm_value = np.array(
            [self.portfolio_value / self.initial_cash], dtype=np.float32
        )
        portfolio_state = np.concatenate([
            self.weights, cash_ratio, norm_value
        ])

        parts = [features_flat, portfolio_state]

        # Optional: T-GAT embeddings
        if self.embeddings is not None:
            emb_flat = self.embeddings[:, t, :].flatten()
            parts.append(emb_flat)

        # Optional: sentiment
        if self.sentiment is not None:
            sent = self.sentiment[:, t]
            parts.append(sent)

        obs = np.concatenate(parts).astype(np.float32)
        return obs

    def _calculate_reward(self, daily_return, turnover, drawdown):
        """Compute reward: Sharpe-based with penalties.

        reward = sharpe_component - drawdown_penalty - turnover_penalty
        """
        # Sharpe component: risk-adjusted return
        # Use rolling window for Sharpe estimation
        if len(self.returns_history) >= 20:
            recent = np.array(self.returns_history[-20:])
            mean_r = recent.mean()
            std_r = recent.std() + 1e-8
            sharpe_component = (mean_r / std_r) * np.sqrt(self.trading_days)
        else:
            # Not enough history — use raw return
            sharpe_component = daily_return * 100  # Scale up

        # Drawdown penalty (only when in drawdown)
        dd_penalty = abs(min(drawdown, 0)) * self.drawdown_penalty

        # Turnover penalty
        to_penalty = turnover * self.turnover_penalty

        reward = (self.sharpe_weight * sharpe_component
                  - dd_penalty - to_penalty)

        return float(reward)

    def _get_info(self):
        """Return info dict for logging."""
        steps_done = self.current_step - self.start_step
        return {
            'step': steps_done,
            'portfolio_value': self.portfolio_value,
            'cash_ratio': 1.0 - self.weights.sum(),
            'peak_value': self.peak_value,
            'n_positions': int((self.weights > 0.01).sum()),
            'total_return': (self.portfolio_value / self.initial_cash) - 1,
        }

    def get_portfolio_summary(self):
        """Get end-of-episode portfolio summary."""
        returns = np.array(self.returns_history)
        if len(returns) == 0:
            return {'total_return': 0, 'sharpe': 0, 'max_drawdown': 0}

        total_return = (self.portfolio_value / self.initial_cash) - 1
        mean_r = returns.mean()
        std_r = returns.std() + 1e-8
        sharpe = (mean_r / std_r) * np.sqrt(self.trading_days)

        # Max drawdown from returns
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - peak) / peak
        max_dd = drawdowns.min()

        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'avg_daily_return': mean_r,
            'daily_volatility': std_r,
            'n_steps': len(returns),
            'final_value': self.portfolio_value,
        }
