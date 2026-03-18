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

export interface PortfolioHolding {
  ticker: string; sector: string; weight: number;
  daily_return: number; cumulative_return: number;
}

export interface PerformancePoint {
  date: string; portfolio: number; nifty: number;
}

export interface PortfolioSummaryResponse {
  portfolio_value: number;
  sharpe_ratio: number; sortino_ratio: number;
  annualized_return: number; annualized_volatility: number;
  max_drawdown: number;
  n_stocks: number; n_days: number;
  date_start: string; date_end: string;
  holdings: PortfolioHolding[];
  performance: PerformancePoint[];
  sector_weights: Record<string, number>;
}

// --- Stock Detail ---
export interface StockPricePoint { date: string; price: number }
export interface StockDetailResponse {
  ticker: string; sector: string;
  current_price: number; prev_close: number;
  daily_change: number; daily_change_pct: number;
  high_52w: number; low_52w: number;
  cumulative_return_1y: number; annualized_volatility: number;
  sharpe_ratio: number; max_drawdown: number; weight: number;
  price_history: StockPricePoint[];
}

// --- RL Agent ---
export interface RLRewardPoint { episode: number; ppo_reward: number; sac_reward: number }
export interface RLStockWeight { ticker: string; sector: string; ppo_weight: number; sac_weight: number }
export interface RLSummaryResponse {
  ppo_episodes: number; sac_episodes: number;
  ppo_avg_reward: number; sac_avg_reward: number;
  ppo_sharpe: number; sac_sharpe: number;
  ppo_max_drawdown: number; sac_max_drawdown: number;
  reward_curve: RLRewardPoint[];
  weights: RLStockWeight[];
  constraints: Record<string, number>;
}

// --- NAS Lab ---
export interface AlphaPoint { epoch: number; linear: number; conv1d: number; attention: number; skip: number; zero: number }
export interface NASCompareItem { metric: string; nas_value: number; handcraft_value: number }
export interface NASLabResponse {
  search_epochs: number; best_op: string;
  nas_sharpe: number; improvement_pct: number;
  best_architecture: string[];
  alpha_convergence: AlphaPoint[];
  comparison: NASCompareItem[];
}

// --- Federated Learning ---
export interface FLRoundPoint {
  round: number; fedprox_loss: number; fedavg_loss: number;
  client_0_loss: number; client_1_loss: number; client_2_loss: number; client_3_loss: number;
}
export interface FLClientInfo { client_id: number; name: string; sectors: string[]; n_stocks: number }
export interface FLFairnessItem { client: string; with_fl: number; without_fl: number }
export interface FLSummaryResponse {
  n_rounds: number; n_clients: number; strategy: string;
  privacy_epsilon: number; privacy_delta: number; global_sharpe: number;
  clients: FLClientInfo[];
  convergence: FLRoundPoint[];
  fairness: FLFairnessItem[];
}

// --- News Sentiment ---
export interface NewsItem {
  headline: string; source: string; published: string;
  ticker: string; sector: string;
  score: number; positive: number; negative: number; neutral: number; label: string;
}
export interface SectorSentimentItem {
  sector: string; avg_score: number; n_headlines: number;
  positive_pct: number; negative_pct: number;
}
export interface SentimentPortfolioHolding {
  ticker: string; sector: string; base_weight: number;
  sentiment_score: number; adjusted_weight: number; weight_change: number;
}
export interface NewsSentimentResponse {
  n_headlines: number; avg_score: number; market_mood: string;
  news: NewsItem[];
  sector_sentiment: SectorSentimentItem[];
  portfolio_impact: SentimentPortfolioHolding[];
  score_distribution: Record<string, number>;
}

// --- GNN Summary ---
export interface GNNNode {
  ticker: string; sector: string; degree: number;
  weight: number; daily_return: number;
}
export interface GNNEdge {
  source: string; target: string; type: string; weight: number;
}
export interface TopConnection {
  stock_a: string; stock_b: string; correlation: number; type: string;
}
export interface SectorConnectivity {
  sector_a: string; sector_b: string; n_edges: number; avg_weight: number;
}
export interface GNNSummaryResponse {
  n_nodes: number; n_edges: number;
  sector_edges: number; supply_chain_edges: number; correlation_edges: number;
  density: number; avg_degree: number;
  nodes: GNNNode[];
  edges: GNNEdge[];
  attention_matrix: number[][];
  attention_tickers: string[];
  top_connections: TopConnection[];
  sector_connectivity: SectorConnectivity[];
  degree_distribution: Record<string, number>;
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

  portfolioSummary: () =>
    fetchJSON<PortfolioSummaryResponse>('/portfolio-summary'),

  stockDetail: (ticker: string) =>
    fetchJSON<StockDetailResponse>(`/stock/${encodeURIComponent(ticker)}`),

  rlSummary: () =>
    fetchJSON<RLSummaryResponse>('/rl-summary'),

  nasSummary: () =>
    fetchJSON<NASLabResponse>('/nas-summary'),

  flSummary: () =>
    fetchJSON<FLSummaryResponse>('/fl-summary'),

  gnnSummary: () =>
    fetchJSON<GNNSummaryResponse>('/gnn-summary'),

  newsSentiment: () =>
    fetchJSON<NewsSentimentResponse>('/news-sentiment'),
};
