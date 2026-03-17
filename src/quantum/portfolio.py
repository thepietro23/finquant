"""Phase 12: Quantum Portfolio Optimization.

Connects QAOA to real portfolio problem:
  1. Prepare candidate assets (top-N by Sharpe from NIFTY 50)
  2. QAOA selects which K assets to hold (binary decision)
  3. Classical Markowitz computes optimal weights for selected subset
  4. Compare with classical brute-force (exact solution for N<=12)
  5. Scaling benchmark across problem sizes
"""

import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from src.quantum.qaoa import run_qaoa, QAOAResult
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('quantum_portfolio')


@dataclass
class PortfolioResult:
    """Comparison of quantum vs classical portfolio optimization."""
    quantum_assets: list             # asset indices from QAOA
    quantum_weights: np.ndarray      # optimal weights for QAOA-selected subset
    quantum_return: float            # expected annualized return
    quantum_risk: float              # portfolio std dev (annualized)
    quantum_sharpe: float
    classical_assets: list           # asset indices from brute-force
    classical_weights: np.ndarray
    classical_return: float
    classical_risk: float
    classical_sharpe: float
    qaoa_result: QAOAResult
    metadata: dict = field(default_factory=dict)


@dataclass
class ScalingResult:
    """Result of one scaling benchmark point."""
    n_assets: int
    k_select: int
    qaoa_sharpe: float
    classical_sharpe: float
    qaoa_time_sec: float
    classical_time_sec: float
    qaoa_assets: list
    classical_assets: list


def prepare_portfolio_data(returns, n_assets=None):
    """Prepare mean returns and covariance from historical data.

    If more than n_assets stocks, selects top-N by individual Sharpe ratio.

    Args:
        returns: (n_days, n_stocks) daily returns array
        n_assets: max candidate pool size (default from config)

    Returns:
        mu: (n_assets,) mean daily returns
        sigma: (n_assets, n_assets) covariance matrix
        selected_indices: indices of selected stocks in original array
    """
    cfg = get_config('quantum')
    n_assets = n_assets or cfg.get('num_assets', 8)

    returns = np.asarray(returns, dtype=np.float64)
    n_days, n_stocks = returns.shape

    if n_stocks <= n_assets:
        mu = returns.mean(axis=0)
        sigma = np.cov(returns, rowvar=False)
        if sigma.ndim == 0:
            sigma = np.array([[sigma]])
        return mu, sigma, list(range(n_stocks))

    # Select top-N by Sharpe ratio
    mu_all = returns.mean(axis=0)
    std_all = returns.std(axis=0)
    std_all = np.where(std_all < 1e-10, 1e-10, std_all)

    cfg_data = get_config('data')
    rf_daily = cfg_data.get('risk_free_rate', 0.07) / cfg_data.get('trading_days_per_year', 248)
    sharpes = (mu_all - rf_daily) / std_all

    top_idx = np.argsort(sharpes)[-n_assets:][::-1]
    top_idx = np.sort(top_idx)

    mu = returns[:, top_idx].mean(axis=0)
    sigma = np.cov(returns[:, top_idx], rowvar=False)
    if sigma.ndim == 0:
        sigma = np.array([[sigma]])

    logger.info(f'Selected top {n_assets} assets by Sharpe from {n_stocks}')
    return mu, sigma, top_idx.tolist()


def compute_markowitz_weights(mu, sigma, risk_free_rate=0.07,
                               trading_days=248, max_weight=0.20):
    """Compute optimal Markowitz weights for given assets.

    Maximize Sharpe ratio: (w^T mu_ann - rf) / sqrt(w^T Sigma_ann w)
    Subject to: sum(w) = 1, 0 <= w_i <= max_weight

    Args:
        mu: (K,) mean daily returns of selected assets
        sigma: (K, K) covariance matrix
        risk_free_rate: annual risk-free rate
        trading_days: annualization factor
        max_weight: max per-asset weight

    Returns:
        weights: (K,) optimal portfolio weights
        port_return: annualized return
        port_risk: annualized std dev
        sharpe: Sharpe ratio
    """
    n = len(mu)

    if n == 1:
        return (np.array([1.0]),
                float(mu[0] * trading_days),
                float(np.sqrt(sigma[0, 0] * trading_days)),
                float((mu[0] * trading_days - risk_free_rate) /
                      max(np.sqrt(sigma[0, 0] * trading_days), 1e-10)))

    mu_ann = mu * trading_days
    sigma_ann = sigma * trading_days

    def neg_sharpe(w):
        port_ret = w @ mu_ann
        port_var = w @ sigma_ann @ w
        port_std = np.sqrt(max(port_var, 1e-12))
        return -(port_ret - risk_free_rate) / port_std

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n

    result = scipy_minimize(
        neg_sharpe, w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500},
    )

    w = result.x
    w = np.clip(w, 0, max_weight)
    w = w / w.sum()

    port_ret = float(w @ mu_ann)
    port_risk = float(np.sqrt(w @ sigma_ann @ w))
    sharpe = float((port_ret - risk_free_rate) / max(port_risk, 1e-10))

    return w, port_ret, port_risk, sharpe


def classical_optimal_subset(mu, sigma, k_assets, risk_free_rate=0.07,
                              trading_days=248):
    """Find the best K-asset subset by brute-force enumeration.

    For N<=12, C(N, K) is at most 924 — very fast.

    Args:
        mu: (N,) mean daily returns
        sigma: (N, N) covariance matrix
        k_assets: number of assets to select

    Returns:
        best_indices: list of K asset indices
        best_weights: (K,) optimal weights
        best_sharpe: Sharpe ratio of best subset
    """
    n = len(mu)
    k_assets = min(k_assets, n)

    if k_assets == n:
        w, ret, risk, sharpe = compute_markowitz_weights(
            mu, sigma, risk_free_rate, trading_days)
        return list(range(n)), w, sharpe

    best_sharpe = -np.inf
    best_indices = None
    best_weights = None

    for combo in combinations(range(n), k_assets):
        idx = list(combo)
        mu_sub = mu[idx]
        sigma_sub = sigma[np.ix_(idx, idx)]

        w, ret, risk, sharpe = compute_markowitz_weights(
            mu_sub, sigma_sub, risk_free_rate, trading_days)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_indices = idx
            best_weights = w

    if best_indices is None:
        best_indices = list(range(k_assets))
        best_weights = np.ones(k_assets) / k_assets
        best_sharpe = 0.0

    logger.info(f'Classical brute-force: best={best_indices}, '
                f'Sharpe={best_sharpe:.4f}, '
                f'combos={len(list(combinations(range(n), k_assets)))}')

    return best_indices, best_weights, best_sharpe


def quantum_portfolio_optimize(returns, n_assets=None, k_select=None,
                                risk_aversion=0.5, qaoa_layers=None,
                                shots=None, seed=42):
    """Full quantum portfolio optimization pipeline.

    1. Prepare data (top-N candidates)
    2. QAOA selects K assets
    3. Markowitz weights for QAOA selection
    4. Classical brute-force for comparison
    5. Return PortfolioResult

    Args:
        returns: (n_days, n_stocks) daily returns
        n_assets: candidate pool size
        k_select: assets to select (default: n_assets//2)
        risk_aversion: QAOA objective parameter
        qaoa_layers: QAOA depth
        shots: measurement shots
        seed: random seed

    Returns:
        PortfolioResult
    """
    cfg = get_config('quantum')
    cfg_data = get_config('data')
    n_assets = n_assets or cfg.get('num_assets', 8)
    qaoa_layers = qaoa_layers or cfg.get('qaoa_layers', 3)
    shots = shots or cfg.get('shots', 1024)
    risk_free_rate = cfg_data.get('risk_free_rate', 0.07)
    trading_days = cfg_data.get('trading_days_per_year', 248)

    # 1. Prepare data
    mu, sigma, asset_indices = prepare_portfolio_data(returns, n_assets)
    n = len(mu)
    k_select = k_select or n // 2

    logger.info(f'Quantum portfolio: {n} candidates, select {k_select}')

    # 2. QAOA
    qaoa_result = run_qaoa(
        mu, sigma, risk_aversion=risk_aversion,
        k_assets=k_select, n_layers=qaoa_layers,
        shots=shots, seed=seed,
    )

    # Handle case where QAOA selects wrong number of assets
    q_assets = qaoa_result.selected_assets
    if len(q_assets) == 0:
        q_assets = list(range(k_select))
    elif len(q_assets) > k_select:
        q_assets = q_assets[:k_select]

    # 3. Markowitz weights for QAOA selection
    mu_q = mu[q_assets]
    sigma_q = sigma[np.ix_(q_assets, q_assets)]
    q_weights, q_ret, q_risk, q_sharpe = compute_markowitz_weights(
        mu_q, sigma_q, risk_free_rate, trading_days)

    # 4. Classical benchmark
    c_assets, c_weights, c_sharpe = classical_optimal_subset(
        mu, sigma, k_select, risk_free_rate, trading_days)
    mu_c = mu[c_assets]
    sigma_c = sigma[np.ix_(c_assets, c_assets)]
    _, c_ret, c_risk, _ = compute_markowitz_weights(
        mu_c, sigma_c, risk_free_rate, trading_days)

    result = PortfolioResult(
        quantum_assets=[asset_indices[i] for i in q_assets],
        quantum_weights=q_weights,
        quantum_return=q_ret,
        quantum_risk=q_risk,
        quantum_sharpe=q_sharpe,
        classical_assets=[asset_indices[i] for i in c_assets],
        classical_weights=c_weights,
        classical_return=c_ret,
        classical_risk=c_risk,
        classical_sharpe=c_sharpe,
        qaoa_result=qaoa_result,
    )

    logger.info(f'Quantum Sharpe={q_sharpe:.4f} vs Classical Sharpe={c_sharpe:.4f}')
    return result


def run_scaling_benchmark(returns, benchmark_sizes=None, k_ratio=0.5,
                           qaoa_layers=None, shots=None, seed=42):
    """Run QAOA at multiple problem sizes for thesis scaling study.

    Args:
        returns: (n_days, n_stocks) daily returns
        benchmark_sizes: list of N values to test (default from config)
        k_ratio: fraction of N to select (default 0.5)
        qaoa_layers: QAOA depth
        shots: measurement shots
        seed: random seed

    Returns:
        list of ScalingResult
    """
    cfg = get_config('quantum')
    cfg_data = get_config('data')
    benchmark_sizes = benchmark_sizes or cfg.get('benchmark_sizes', [4, 6, 8])
    qaoa_layers = qaoa_layers or cfg.get('qaoa_layers', 3)
    shots = shots or cfg.get('shots', 1024)
    risk_free_rate = cfg_data.get('risk_free_rate', 0.07)
    trading_days = cfg_data.get('trading_days_per_year', 248)

    returns = np.asarray(returns, dtype=np.float64)
    n_stocks = returns.shape[1]

    results = []
    for n in benchmark_sizes:
        if n > n_stocks:
            logger.warning(f'Skipping n={n} (only {n_stocks} stocks)')
            continue

        k = max(1, int(n * k_ratio))
        mu, sigma, indices = prepare_portfolio_data(returns, n)

        # QAOA
        t0 = time.time()
        qaoa_result = run_qaoa(
            mu, sigma, k_assets=k, n_layers=qaoa_layers,
            shots=shots, seed=seed, maxiter=100,
        )
        qaoa_time = time.time() - t0

        q_assets = qaoa_result.selected_assets
        if len(q_assets) == 0:
            q_assets = list(range(k))
        elif len(q_assets) > k:
            q_assets = q_assets[:k]

        mu_q = mu[q_assets]
        sigma_q = sigma[np.ix_(q_assets, q_assets)]
        _, _, _, q_sharpe = compute_markowitz_weights(
            mu_q, sigma_q, risk_free_rate, trading_days)

        # Classical
        t0 = time.time()
        c_assets, _, c_sharpe = classical_optimal_subset(
            mu, sigma, k, risk_free_rate, trading_days)
        classical_time = time.time() - t0

        results.append(ScalingResult(
            n_assets=n, k_select=k,
            qaoa_sharpe=q_sharpe, classical_sharpe=c_sharpe,
            qaoa_time_sec=qaoa_time, classical_time_sec=classical_time,
            qaoa_assets=q_assets, classical_assets=c_assets,
        ))

        logger.info(f'Scaling n={n}: QAOA={q_sharpe:.3f} ({qaoa_time:.1f}s) '
                     f'vs Classical={c_sharpe:.3f} ({classical_time:.3f}s)')

    return results
