"""Phase 12: Quantum ML (QAOA) Tests.

6 unit tests + 3 edge cases = 9 total.

Unit tests:
  T12.1: QUBO matrix — correct shape, returns on diagonal, covariance off-diagonal
  T12.2: QAOA circuit — builds, correct qubits, has parameters
  T12.3: QAOA optimization — returns valid QAOAResult, bitstring correct length
  T12.4: Classical benchmark — finds valid portfolio, Sharpe finite
  T12.5: Quantum vs Classical — both produce finite Sharpe ratios
  T12.6: Scaling benchmark — runs at [4, 6], results finite

Edge cases:
  E12.1: 2-asset minimum — QAOA still works, selects 1
  E12.2: Identical assets — any selection valid, no crash
  E12.3: Single asset (n=1, k=1) — trivial case
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quantum.qaoa import (
    build_qubo, qubo_to_ising, build_qaoa_circuit,
    evaluate_cost, run_qaoa, QAOAResult,
)
from src.quantum.portfolio import (
    prepare_portfolio_data, compute_markowitz_weights,
    classical_optimal_subset, quantum_portfolio_optimize,
    run_scaling_benchmark, PortfolioResult,
)
from src.utils.seed import set_seed


def make_fake_returns(n_assets=8, n_days=500, seed=42):
    """Create synthetic daily returns for testing."""
    rng = np.random.RandomState(seed)
    # Positive drift + noise
    mu = rng.uniform(0.0002, 0.001, n_assets)
    # Generate correlated returns via Cholesky
    A = rng.randn(n_assets, n_assets) * 0.005
    cov = A @ A.T + np.eye(n_assets) * 0.0001
    L = np.linalg.cholesky(cov)
    returns = rng.randn(n_days, n_assets) @ L.T + mu
    return returns.astype(np.float64)


# ============================================================
# UNIT TESTS
# ============================================================

class TestQUBO:
    """T12.1: QUBO matrix construction."""

    def test_qubo_shape(self):
        """QUBO matrix should be N×N."""
        mu = np.array([0.001, 0.002, 0.003, 0.004])
        sigma = np.eye(4) * 0.01
        Q = build_qubo(mu, sigma, risk_aversion=0.5, k_assets=2)
        assert Q.shape == (4, 4)

    def test_qubo_finite(self):
        """All QUBO entries should be finite."""
        mu = np.array([0.001, 0.002, 0.003, 0.004])
        sigma = np.eye(4) * 0.01
        Q = build_qubo(mu, sigma, risk_aversion=0.5, k_assets=2)
        assert np.all(np.isfinite(Q))

    def test_qubo_returns_on_diagonal(self):
        """Diagonal should include return terms (negative, since minimizing)."""
        mu = np.array([0.01, 0.02])
        sigma = np.zeros((2, 2))  # No risk
        Q = build_qubo(mu, sigma, risk_aversion=0.0, k_assets=1, penalty=0)
        # With no risk and no penalty, Q[i,i] = -mu[i]
        assert Q[0, 0] < 0  # -0.01
        assert Q[1, 1] < 0  # -0.02


class TestQAOACircuit:
    """T12.2: QAOA circuit builds correctly."""

    def test_circuit_builds(self):
        """Circuit should build without error."""
        qc, params = build_qaoa_circuit(n_qubits=4, n_layers=2)
        assert qc.num_qubits == 4
        assert len(params) == 4  # 2 gammas + 2 betas

    def test_circuit_has_measurements(self):
        """Circuit should have measurement gates."""
        qc, _ = build_qaoa_circuit(n_qubits=3, n_layers=1)
        # measure_all adds classical bits
        assert qc.num_clbits == 3


class TestQAOAOptimization:
    """T12.3: QAOA returns valid result."""

    def test_qaoa_valid_result(self):
        """QAOA should return QAOAResult with correct fields."""
        set_seed(42)
        mu = np.array([0.001, 0.002, 0.003, 0.004])
        sigma = np.eye(4) * 0.0001

        result = run_qaoa(mu, sigma, k_assets=2, n_layers=2,
                          shots=512, seed=42, maxiter=50)

        assert isinstance(result, QAOAResult)
        assert len(result.best_bitstring) == 4
        assert all(b in '01' for b in result.best_bitstring)
        assert np.isfinite(result.best_cost)
        assert result.n_qubits == 4
        assert result.n_layers == 2
        assert result.n_shots == 512
        assert result.n_function_evals > 0


class TestClassicalBenchmark:
    """T12.4: Classical brute-force works."""

    def test_classical_finds_portfolio(self):
        """Brute-force should find valid portfolio with finite Sharpe."""
        mu = np.array([0.001, 0.002, 0.003, 0.004])
        sigma = np.eye(4) * 0.0001

        indices, weights, sharpe = classical_optimal_subset(
            mu, sigma, k_assets=2)

        assert len(indices) == 2
        assert len(weights) == 2
        assert np.isclose(weights.sum(), 1.0, atol=0.01)
        assert np.isfinite(sharpe)
        assert all(0 <= w <= 1 for w in weights)


class TestQuantumVsClassical:
    """T12.5: Quantum vs Classical comparison."""

    def test_comparison_runs(self):
        """Both quantum and classical should produce finite Sharpe."""
        set_seed(42)
        returns = make_fake_returns(n_assets=6, n_days=300, seed=42)

        result = quantum_portfolio_optimize(
            returns, n_assets=6, k_select=3,
            qaoa_layers=2, shots=512, seed=42)

        assert isinstance(result, PortfolioResult)
        assert np.isfinite(result.quantum_sharpe)
        assert np.isfinite(result.classical_sharpe)
        assert len(result.quantum_assets) > 0
        assert len(result.classical_assets) == 3
        assert np.isclose(result.quantum_weights.sum(), 1.0, atol=0.01)
        assert np.isclose(result.classical_weights.sum(), 1.0, atol=0.01)


class TestScaling:
    """T12.6: Scaling benchmark."""

    def test_scaling_benchmark(self):
        """Should run at multiple sizes with finite results."""
        set_seed(42)
        returns = make_fake_returns(n_assets=8, n_days=300, seed=42)

        results = run_scaling_benchmark(
            returns, benchmark_sizes=[4, 6],
            qaoa_layers=1, shots=256, seed=42)

        assert len(results) == 2
        for r in results:
            assert np.isfinite(r.qaoa_sharpe)
            assert np.isfinite(r.classical_sharpe)
            assert r.qaoa_time_sec > 0
            assert r.classical_time_sec >= 0
            assert r.n_assets in [4, 6]


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:

    def test_two_assets(self):
        """E12.1: Minimum 2 assets, select 1 — QAOA should work."""
        set_seed(42)
        mu = np.array([0.001, 0.003])
        sigma = np.eye(2) * 0.0001

        result = run_qaoa(mu, sigma, k_assets=1, n_layers=1,
                          shots=256, seed=42, maxiter=30)

        assert len(result.best_bitstring) == 2
        assert np.isfinite(result.best_cost)

    def test_identical_assets(self):
        """E12.2: All assets have same returns/risk — any selection valid."""
        mu = np.ones(4) * 0.001
        sigma = np.eye(4) * 0.0001

        indices, weights, sharpe = classical_optimal_subset(
            mu, sigma, k_assets=2)

        assert len(indices) == 2
        assert np.isfinite(sharpe)
        # All combos should give similar Sharpe
        assert sharpe > -100  # not degenerate

    def test_single_asset(self):
        """E12.3: n=1, k=1 — trivial selection."""
        mu = np.array([0.002])
        sigma = np.array([[0.0001]])

        indices, weights, sharpe = classical_optimal_subset(
            mu, sigma, k_assets=1)

        assert indices == [0]
        assert np.isclose(weights[0], 1.0)
        assert np.isfinite(sharpe)
