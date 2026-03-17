/** FastAPI backend client — all endpoints from Phase 13 */

const BASE = '/api';

async function fetchJSON<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

// --- Types ---

export interface StockInfo { ticker: string; sector: string }
export interface StockListResponse { count: number; stocks: StockInfo[] }

export interface HealthResponse {
  status: string; version: string; project: string; phases_complete: number;
}

export interface ConfigResponse {
  seed: number; device: string; fp16: boolean;
  data: Record<string, unknown>; rl: Record<string, unknown>;
  quantum: Record<string, unknown>; fl: Record<string, unknown>;
}

export interface SentimentResponse {
  text: string; score: number; positive: number;
  negative: number; neutral: number; label: string;
}

export interface BatchSentimentResponse {
  count: number; results: SentimentResponse[];
}

export interface ScenarioResult {
  scenario: string; mean_return: string;
  var_95: string; cvar_95: string; survival_rate: string;
}

export interface StressTestResponse {
  n_stocks: number; n_simulations: number; scenarios: ScenarioResult[];
}

export interface QAOAResponse {
  quantum_assets: number[]; quantum_sharpe: number; quantum_weights: number[];
  classical_assets: number[]; classical_sharpe: number; classical_weights: number[];
  n_qubits: number; best_bitstring: string; n_function_evals: number;
}

export interface MetricsResponse {
  sharpe_ratio: number; sortino_ratio: number; annualized_return: number;
  annualized_volatility: number; max_drawdown: number; n_days: number;
}

// --- API Calls ---

export const api = {
  health: () => fetchJSON<HealthResponse>('/health'),
  config: () => fetchJSON<ConfigResponse>('/config'),
  stocks: () => fetchJSON<StockListResponse>('/stocks'),

  sentiment: (text: string) =>
    fetchJSON<SentimentResponse>('/sentiment', {
      method: 'POST', body: JSON.stringify({ text }),
    }),

  sentimentBatch: (texts: string[]) =>
    fetchJSON<BatchSentimentResponse>('/sentiment/batch', {
      method: 'POST', body: JSON.stringify({ texts }),
    }),

  stressTest: (n_stocks = 10, n_simulations = 1000) =>
    fetchJSON<StressTestResponse>('/stress-test', {
      method: 'POST', body: JSON.stringify({ n_stocks, n_simulations }),
    }),

  qaoa: (n_assets = 6, k_select = 3, qaoa_layers = 2, shots = 512, risk_aversion = 0.5) =>
    fetchJSON<QAOAResponse>('/qaoa', {
      method: 'POST',
      body: JSON.stringify({ n_assets, k_select, qaoa_layers, shots, risk_aversion }),
    }),

  metrics: (returns: number[]) =>
    fetchJSON<MetricsResponse>('/metrics', {
      method: 'POST', body: JSON.stringify({ returns }),
    }),
};
