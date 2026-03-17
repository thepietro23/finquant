"""Phase 12: QAOA (Quantum Approximate Optimization Algorithm).

Solves the discrete portfolio selection problem:
  "Given N candidate assets, select exactly K to maximize
   risk-adjusted return (Sharpe ratio)."

Pipeline:
  1. Build QUBO matrix from returns + covariance + cardinality constraint
  2. Convert QUBO to Ising Hamiltonian (Z/ZZ terms)
  3. Build parameterized QAOA circuit (cost + mixer unitaries)
  4. Classical optimizer (COBYLA) tunes circuit parameters
  5. Measure → decode best bitstring → selected assets
"""

import math
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('quantum_qaoa')


@dataclass
class QAOAResult:
    """Result of a QAOA optimization run."""
    best_bitstring: str              # e.g., "10110010"
    best_cost: float                 # QUBO objective value (lower = better)
    selected_assets: list            # indices of selected assets
    optimal_params: np.ndarray       # optimized [gamma..., beta...] angles
    all_counts: dict                 # bitstring -> count from shots
    n_qubits: int
    n_layers: int
    n_shots: int
    optimization_converged: bool
    n_function_evals: int = 0
    metadata: dict = field(default_factory=dict)


def build_qubo(mu, sigma, risk_aversion=0.5, k_assets=None, penalty=None):
    """Build QUBO matrix for portfolio selection.

    Objective: maximize  mu^T x - lambda * x^T Sigma x
    Subject to: sum(x) = K  (select exactly K assets)

    Converted to minimization QUBO:
      Q_ij encodes: -return_terms + risk_terms + penalty*(cardinality constraint)

    Args:
        mu: (N,) expected returns per asset
        sigma: (N, N) covariance matrix
        risk_aversion: lambda — tradeoff between return and risk
        k_assets: number of assets to select (default: N//2)
        penalty: constraint penalty weight (default: auto)

    Returns:
        Q: (N, N) QUBO matrix (minimize x^T Q x)
    """
    n = len(mu)
    k_assets = k_assets or n // 2

    # Auto-tune penalty: should dominate the objective terms
    if penalty is None:
        penalty = max(2.0 * np.max(np.abs(mu)) * n, 1.0)

    Q = np.zeros((n, n))

    # Return terms (diagonal): -mu_i (we minimize, so negate returns)
    for i in range(n):
        Q[i, i] += -mu[i]

    # Risk terms (off-diagonal): +lambda * sigma_ij
    for i in range(n):
        for j in range(n):
            Q[i, j] += risk_aversion * sigma[i, j]

    # Cardinality constraint: penalty * (sum(x_i) - K)^2
    # Expand: penalty * (sum_i x_i^2 + 2*sum_{i<j} x_i*x_j - 2K*sum_i x_i + K^2)
    # Since x_i is binary (0/1): x_i^2 = x_i (goes to diagonal)
    for i in range(n):
        Q[i, i] += penalty * (1 - 2 * k_assets)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += penalty * 2
            Q[j, i] += penalty * 2
    # K^2 constant — doesn't affect optimization, dropped

    logger.info(f'QUBO built: {n}×{n}, k={k_assets}, '
                f'risk_aversion={risk_aversion}, penalty={penalty:.4f}')
    return Q


def qubo_to_ising(Q):
    """Convert QUBO (binary 0/1) to Ising (spin +1/-1).

    Substitution: x_i = (1 - z_i) / 2  where z_i ∈ {+1, -1}

    Returns:
        J: (N, N) coupling matrix (ZZ interactions)
        h: (N,) field vector (Z rotations)
        offset: constant energy offset
    """
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)
    offset = 0.0

    for i in range(n):
        offset += Q[i, i] / 4
        h[i] += -Q[i, i] / 4

    for i in range(n):
        for j in range(i + 1, n):
            q_ij = Q[i, j] + Q[j, i]
            J[i, j] = q_ij / 4
            h[i] += -q_ij / 4
            h[j] += -q_ij / 4
            offset += q_ij / 4

    return J, h, offset


def build_qaoa_circuit(n_qubits, n_layers):
    """Build a parameterized QAOA circuit.

    Structure per layer p:
      1. Cost unitary:  exp(-i * gamma_p * C)
         - ZZ gates for J couplings
         - RZ gates for h fields
      2. Mixer unitary: exp(-i * beta_p * B)
         - RX gates on all qubits

    Parameters: [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]

    Args:
        n_qubits: number of qubits (= number of candidate assets)
        n_layers: QAOA depth (p)

    Returns:
        circuit: QuantumCircuit with 2*n_layers parameters
        param_names: list of parameter names for binding
    """
    from qiskit.circuit import Parameter

    qc = QuantumCircuit(n_qubits)

    # Initial state: uniform superposition
    for i in range(n_qubits):
        qc.h(i)

    # Create named parameters
    gammas = [Parameter(f'gamma_{p}') for p in range(n_layers)]
    betas = [Parameter(f'beta_{p}') for p in range(n_layers)]

    for p in range(n_layers):
        # Cost unitary (placeholder — actual J/h applied at evaluation time)
        # We encode J and h into RZZ and RZ gates
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.rzz(gammas[p], i, j)
            qc.rz(gammas[p], i)

        # Mixer unitary
        for i in range(n_qubits):
            qc.rx(2 * betas[p], i)

    # Measurement
    qc.measure_all()

    return qc, gammas + betas


def evaluate_cost(bitstring, Q):
    """Compute QUBO cost for a given bitstring.

    Args:
        bitstring: str of '0'/'1', length N
        Q: (N, N) QUBO matrix

    Returns:
        cost: x^T Q x (scalar)
    """
    x = np.array([int(b) for b in bitstring], dtype=float)
    return float(x @ Q @ x)


def _build_cost_circuit(n_qubits, n_layers, J, h):
    """Build QAOA circuit with J/h baked into gate angles.

    Instead of generic RZZ/RZ, we scale by actual J[i,j] and h[i].
    This way, gamma just scales the overall cost Hamiltonian strength.
    """
    from qiskit.circuit import Parameter

    qc = QuantumCircuit(n_qubits)

    # Uniform superposition
    for i in range(n_qubits):
        qc.h(i)

    gammas = [Parameter(f'gamma_{p}') for p in range(n_layers)]
    betas = [Parameter(f'beta_{p}') for p in range(n_layers)]

    for p in range(n_layers):
        # Cost unitary: ZZ interactions + Z fields
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if abs(J[i, j]) > 1e-10:
                    qc.rzz(2 * gammas[p] * J[i, j], i, j)
            if abs(h[i]) > 1e-10:
                qc.rz(2 * gammas[p] * h[i], i)

        # Mixer unitary: X rotations
        for i in range(n_qubits):
            qc.rx(2 * betas[p], i)

    qc.measure_all()
    return qc, gammas + betas


def run_qaoa(mu, sigma, risk_aversion=0.5, k_assets=None,
             n_layers=None, shots=None, optimizer_name=None,
             seed=42, maxiter=200):
    """Run QAOA for portfolio asset selection.

    Full pipeline: QUBO → Ising → circuit → optimize → decode.

    Args:
        mu: (N,) expected returns
        sigma: (N, N) covariance matrix
        risk_aversion: return vs risk tradeoff
        k_assets: assets to select (default: N//2)
        n_layers: QAOA depth (default from config)
        shots: measurement shots (default from config)
        optimizer_name: classical optimizer (default from config)
        seed: random seed
        maxiter: max optimizer iterations

    Returns:
        QAOAResult
    """
    cfg = get_config('quantum')
    n_layers = n_layers or cfg.get('qaoa_layers', 3)
    shots = shots or cfg.get('shots', 1024)
    optimizer_name = optimizer_name or cfg.get('optimizer', 'COBYLA')

    n = len(mu)
    k_assets = k_assets or n // 2

    logger.info(f'QAOA: n={n}, k={k_assets}, layers={n_layers}, '
                f'shots={shots}, optimizer={optimizer_name}')

    # 1. Build QUBO
    Q = build_qubo(mu, sigma, risk_aversion, k_assets)

    # 2. Convert to Ising
    J, h, offset = qubo_to_ising(Q)

    # 3. Build parameterized circuit
    circuit, params = _build_cost_circuit(n, n_layers, J, h)

    # 4. Setup simulator
    simulator = AerSimulator(method='automatic', seed_simulator=seed)

    # 5. Objective function for classical optimizer
    n_evals = [0]

    def objective(param_values):
        n_evals[0] += 1
        # Bind parameters
        param_dict = dict(zip(params, param_values))
        bound = circuit.assign_parameters(param_dict)

        # Run simulation
        result = simulator.run(bound, shots=shots).result()
        counts = result.get_counts()

        # Compute expected cost
        total_cost = 0.0
        for bitstring, count in counts.items():
            # Qiskit returns bitstrings in reverse order
            bs = bitstring.replace(' ', '')[::-1]
            cost = evaluate_cost(bs, Q)
            total_cost += cost * count

        return total_cost / shots

    # 6. Initial parameters (random)
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(0, 2 * np.pi, 2 * n_layers)

    # 7. Optimize
    opt_result = minimize(
        objective, x0,
        method=optimizer_name,
        options={'maxiter': maxiter, 'rhobeg': 0.5},
    )

    # 8. Final evaluation with optimized parameters
    param_dict = dict(zip(params, opt_result.x))
    bound = circuit.assign_parameters(param_dict)
    final_result = simulator.run(bound, shots=shots).result()
    final_counts = final_result.get_counts()

    # 9. Find best bitstring
    best_bs = None
    best_cost = float('inf')
    for bitstring, count in final_counts.items():
        bs = bitstring.replace(' ', '')[::-1]
        cost = evaluate_cost(bs, Q)
        if cost < best_cost:
            best_cost = cost
            best_bs = bs

    selected = [i for i, b in enumerate(best_bs) if b == '1']

    qaoa_result = QAOAResult(
        best_bitstring=best_bs,
        best_cost=best_cost,
        selected_assets=selected,
        optimal_params=opt_result.x,
        all_counts=final_counts,
        n_qubits=n,
        n_layers=n_layers,
        n_shots=shots,
        optimization_converged=opt_result.success,
        n_function_evals=n_evals[0],
    )

    logger.info(f'QAOA done: best_cost={best_cost:.4f}, '
                f'selected={selected}, evals={n_evals[0]}')

    return qaoa_result
