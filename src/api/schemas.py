"""Phase 13: Pydantic schemas for API request/response models.

Every endpoint has typed request and response schemas.
React dashboard (Phase 14) will consume these.
"""

from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# HEALTH / CONFIG
# ============================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "4.0.0"
    project: str = "FINQUANT-NEXUS v4"
    phases_complete: int = 13


class ConfigResponse(BaseModel):
    seed: int
    device: str
    fp16: bool
    data: dict
    rl: dict
    quantum: dict
    fl: dict


# ============================================================
# STOCKS
# ============================================================

class StockInfo(BaseModel):
    ticker: str
    sector: str


class StockListResponse(BaseModel):
    count: int
    stocks: list[StockInfo]


# ============================================================
# FEATURES
# ============================================================

class FeatureResponse(BaseModel):
    ticker: str
    n_days: int
    n_features: int
    feature_names: list[str]
    latest: dict  # last day's features


# ============================================================
# SENTIMENT
# ============================================================

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class SentimentResponse(BaseModel):
    text: str
    score: float = Field(..., ge=-1.0, le=1.0)
    positive: float
    negative: float
    neutral: float
    label: str


class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)


class BatchSentimentResponse(BaseModel):
    count: int
    results: list[SentimentResponse]


# ============================================================
# STRESS TEST
# ============================================================

class StressTestRequest(BaseModel):
    n_stocks: int = Field(default=10, ge=2, le=47)
    n_simulations: int = Field(default=1000, ge=100, le=50000)
    scenarios: list[str] = Field(
        default=["normal", "crash_2008", "crash_covid", "flash_crash"]
    )


class ScenarioResult(BaseModel):
    scenario: str
    mean_return: str
    var_95: str
    cvar_95: str
    survival_rate: str


class StressTestResponse(BaseModel):
    n_stocks: int
    n_simulations: int
    scenarios: list[ScenarioResult]


# ============================================================
# QUANTUM / QAOA
# ============================================================

class QAOARequest(BaseModel):
    n_assets: int = Field(default=6, ge=2, le=12)
    k_select: int = Field(default=3, ge=1, le=12)
    qaoa_layers: int = Field(default=2, ge=1, le=5)
    shots: int = Field(default=512, ge=64, le=4096)
    risk_aversion: float = Field(default=0.5, ge=0.0, le=5.0)


class QAOAResponse(BaseModel):
    quantum_assets: list[int]
    quantum_sharpe: float
    quantum_weights: list[float]
    classical_assets: list[int]
    classical_sharpe: float
    classical_weights: list[float]
    n_qubits: int
    best_bitstring: str
    n_function_evals: int


# ============================================================
# METRICS
# ============================================================

class MetricsRequest(BaseModel):
    returns: list[float] = Field(..., min_length=10)


class MetricsResponse(BaseModel):
    sharpe_ratio: float
    sortino_ratio: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    n_days: int


# ============================================================
# PORTFOLIO SUMMARY (real data from CSVs)
# ============================================================

class PortfolioHolding(BaseModel):
    ticker: str
    sector: str
    weight: float
    daily_return: float
    cumulative_return: float

class PerformancePoint(BaseModel):
    date: str
    portfolio: float
    nifty: float

class PortfolioSummaryResponse(BaseModel):
    portfolio_value: float
    sharpe_ratio: float
    sortino_ratio: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    n_stocks: int
    n_days: int
    date_start: str
    date_end: str
    holdings: list[PortfolioHolding]
    performance: list[PerformancePoint]
    sector_weights: dict[str, float]


# ============================================================
# RL AGENT SUMMARY (computed from real stock data)
# ============================================================

class RLRewardPoint(BaseModel):
    episode: int
    ppo_reward: float
    sac_reward: float

class RLStockWeight(BaseModel):
    ticker: str
    sector: str
    ppo_weight: float
    sac_weight: float

class RLSummaryResponse(BaseModel):
    ppo_episodes: int
    sac_episodes: int
    ppo_avg_reward: float
    sac_avg_reward: float
    ppo_sharpe: float
    sac_sharpe: float
    ppo_max_drawdown: float
    sac_max_drawdown: float
    reward_curve: list[RLRewardPoint]
    weights: list[RLStockWeight]
    constraints: dict


# ============================================================
# NAS LAB SUMMARY (computed from real stock data)
# ============================================================

class AlphaPoint(BaseModel):
    epoch: int
    linear: float
    conv1d: float
    attention: float
    skip: float
    zero: float

class NASCompareItem(BaseModel):
    metric: str
    nas_value: float
    handcraft_value: float

class NASLabResponse(BaseModel):
    search_epochs: int
    best_op: str
    nas_sharpe: float
    improvement_pct: float
    best_architecture: list[str]
    alpha_convergence: list[AlphaPoint]
    comparison: list[NASCompareItem]


# ============================================================
# FEDERATED LEARNING SUMMARY (computed from real stock data)
# ============================================================

class FLRoundPoint(BaseModel):
    round: int
    fedprox_loss: float
    fedavg_loss: float
    client_0_loss: float
    client_1_loss: float
    client_2_loss: float
    client_3_loss: float

class FLClientInfo(BaseModel):
    client_id: int
    name: str
    sectors: list[str]
    n_stocks: int

class FLFairnessItem(BaseModel):
    client: str
    with_fl: float
    without_fl: float

# ============================================================
# INDIVIDUAL STOCK DETAIL (real price data)
# ============================================================

class StockPricePoint(BaseModel):
    date: str
    price: float

class StockDetailResponse(BaseModel):
    ticker: str
    sector: str
    current_price: float
    prev_close: float
    daily_change: float
    daily_change_pct: float
    high_52w: float
    low_52w: float
    cumulative_return_1y: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    weight: float
    price_history: list[StockPricePoint]


class FLSummaryResponse(BaseModel):
    n_rounds: int
    n_clients: int
    strategy: str
    privacy_epsilon: float
    privacy_delta: float
    global_sharpe: float
    clients: list[FLClientInfo]
    convergence: list[FLRoundPoint]
    fairness: list[FLFairnessItem]


# ============================================================
# GNN SUMMARY (real graph data from stock correlations)
# ============================================================

# ============================================================
# NEWS SENTIMENT (real-time Google News + FinBERT)
# ============================================================

class NewsItem(BaseModel):
    headline: str
    source: str = ''
    published: str = ''
    ticker: str = ''
    sector: str = ''
    score: float = 0.0
    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 0.0
    label: str = 'neutral'

class SectorSentiment(BaseModel):
    sector: str
    avg_score: float
    n_headlines: int
    positive_pct: float
    negative_pct: float

class SentimentPortfolioHolding(BaseModel):
    ticker: str
    sector: str
    base_weight: float       # equal-weight %
    sentiment_score: float   # avg sentiment for this stock
    adjusted_weight: float   # sentiment-adjusted weight %
    weight_change: float     # change from base weight

class NewsSentimentResponse(BaseModel):
    n_headlines: int
    avg_score: float
    market_mood: str         # 'Bullish' / 'Bearish' / 'Neutral'
    news: list[NewsItem]
    sector_sentiment: list[SectorSentiment]
    portfolio_impact: list[SentimentPortfolioHolding]
    score_distribution: dict[str, int]  # bucket -> count


class GNNNode(BaseModel):
    ticker: str
    sector: str
    degree: int
    weight: float          # portfolio weight %
    daily_return: float    # latest daily return %

class GNNEdge(BaseModel):
    source: str
    target: str
    type: str              # 'sector' | 'supply' | 'correlation'
    weight: float          # correlation strength (1.0 for static edges)

class TopConnection(BaseModel):
    stock_a: str
    stock_b: str
    correlation: float
    type: str

class SectorConnectivity(BaseModel):
    sector_a: str
    sector_b: str
    n_edges: int
    avg_weight: float

class GNNSummaryResponse(BaseModel):
    n_nodes: int
    n_edges: int
    sector_edges: int
    supply_chain_edges: int
    correlation_edges: int
    density: float
    avg_degree: float
    nodes: list[GNNNode]
    edges: list[GNNEdge]
    attention_matrix: list[list[float]]   # correlation-derived, top-15 stocks
    attention_tickers: list[str]          # tickers for attention matrix rows/cols
    top_connections: list[TopConnection]
    sector_connectivity: list[SectorConnectivity]
    degree_distribution: dict[int, int]   # degree -> count
