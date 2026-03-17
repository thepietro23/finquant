# FINQUANT-NEXUS v4 — Phase-wise Progress Tracker

> **Last Updated:** 2026-03-17
> **Current Phase:** Phase 14 (Dashboard) — ✅ DONE
> **Overall:** Phase 0-14 = 232/232 tests GREEN + Dashboard 10 pages built

---

## Phase Overview (Quick Reference)

| Phase | Name | Status | Days | Key Deliverable |
|-------|------|--------|------|-----------------|
| 0 | Global Setup | ✅ DONE | D0 | Config, seed, logger, metrics |
| 1 | Data Pipeline | ✅ DONE | D1-D2 | 45 stocks + index downloaded, quality verified |
| 2 | Feature Engineering | ✅ DONE | D3-D4 | 21 technical indicators + z-score normalization |
| 3 | FinBERT Sentiment | ✅ DONE | D5-D6 | FinBERT + news fetcher + sentiment matrix |
| 4 | Graph Construction | ✅ DONE | D6-D7 | Correlation + sector + supply chain edges |
| 5 | T-GAT Model | ✅ DONE | D8-D10 | Temporal Graph Attention Network |
| 6 | RL Environment | ✅ DONE | D10-D12 | Gym env for portfolio management |
| 7 | Deep RL Agent | ✅ DONE | D12-D17 | PPO + SAC training |
| 8-9 | TimeGAN + Stress | ✅ DONE | D18-D24 | Synthetic data + stress testing |
| 10 | NAS/DARTS | ✅ DONE | D25-D30 | DARTS T-GAT search + RL policy grid search |
| 11 | Federated Learning | ✅ DONE | D31-D37 | FedAvg/FedProx + DP-SGD, 4 sector clients |
| 12 | Quantum ML | ✅ DONE | D38-D42 | QAOA portfolio selection + classical benchmark |
| 13 | API + Docker | ✅ DONE | D43-D46 | FastAPI + Docker + 15 tests |
| 14 | Dashboard | ✅ DONE | D46-D49 | React + Tailwind + 10 pages + Graph Viz |
| 15 | Thesis + Demo | NOT STARTED | D50-D56 | Final thesis document |

---

## PHASE 0: Global Setup — DONE

### Kya Banaya (What)
| File | Purpose | Lines |
|------|---------|-------|
| `configs/base.yaml` | All hyperparameters in one place | 160 |
| `src/utils/config.py` | YAML loader with caching | 33 |
| `src/utils/seed.py` | Reproducibility (Python + NumPy + PyTorch + CUDA) | 16 |
| `src/utils/logger.py` | File + console logging with timestamps | 37 |
| `src/utils/metrics.py` | 7 financial metrics (Sharpe, MaxDD, Sortino, Calmar, etc.) | 68 |
| `requirements.txt` | All 70+ dependencies locked | 70 |
| `.gitignore` | Ignore data, models, venv, .env | 52 |
| `__init__.py` (12) | Python package markers for all modules | - |

### Kyu Banaya (Why / Reasoning)
1. **Config centralization**: Agar hyperparameters code mein hardcoded hain toh:
   - 10 files mein dhundhna padega kahan kya likha hai
   - Ek jagah change kiya, dusri jagah bhool gaya = silent bug
   - YAML mein rakhne se W&B mein log kar sakte, git history se track kar sakte
2. **Seed fixing**: Bina seed ke har run alag result dega. Thesis mein "reproducible" claim karna hai toh seed zaruri. `set_seed(42)` ek call mein Python random, NumPy, PyTorch, CUDA sab fix.
3. **Logger**: `print()` production mein kaam nahi karta. Logger timestamp deta hai, file mein save karta hai, module name dikhata hai. Debugging 10x easier.
4. **Metrics**: Yeh 7 metrics EVERYWHERE use honge — RL reward function mein, backtesting mein, baseline comparison mein, thesis results mein. Ek baar sahi likho, baar baar use karo.
5. **India-specific constants**: Risk-free rate 7% (US mein 4%), 248 trading days/year (US mein 252). Galat constant = galat Sharpe ratio = thesis mein wrong results.

### Tests: 18/18 PASSING
- Config loading (3), Seed reproducibility (2), Logger creation (3), Financial metrics (5), Project structure (2), Edge cases (3)

### Git Commit
```
7fb98a0 — Phase 0: Project scaffolding complete (2026-03-07)
```

---

## PHASE 1: Data Pipeline — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/data/stocks.py` | NIFTY 50 registry: 47 stocks, 10 sectors, 27 supply chain edges | 95 | ✅ |
| `src/data/download.py` | yfinance downloader with retry + backoff + SSL fix | 123 | ✅ |
| `src/data/quality.py` | 7 quality checks + cleaning function | 136 | ✅ |
| `tests/test_data.py` | 12 tests for data pipeline | 123 | ✅ 12/12 PASS |

### Data Download Results
- **45 stocks** successfully downloaded (2015-2025) as individual CSVs
- **NIFTY50_INDEX.csv** — benchmark index data
- **all_close_prices.csv** — combined Adj Close for correlation analysis
- **SSL Issue Resolved**: College/corporate proxy was injecting self-signed certificates. Fixed with `curl_cffi` SSL verification bypass.

### Kyu Banaya (Why / Reasoning)

**1. Stock Registry (`stocks.py`)**
- **Sector mapping**: GNN ke liye zaruri. Same sector stocks = sector edges in graph.
- **Supply chain edges (27)**: Real world mein TATASTEEL→MARUTI (steel for cars) jaise relationships GNN mein capture.
- **47 stocks**: NIFTY 50 composition changes hoti hai. Quality check se kuch filter honge.

**2. Download Module (`download.py`)**
- **Retry + exponential backoff**: yfinance free API, rate limits handle karna zaruri.
- **Adj Close**: Stock splits/dividends automatically adjusted. RELIANCE 2020 split correctly handled.
- **Per-stock + combined CSV**: Individual analysis + correlation matrix dono ke liye.

**3. Quality Checker (`quality.py`)**
- **7 automated checks**: NaN, duplicates, negative prices, extreme returns, volume, date order, minimum days.
- **Forward fill (ffill)**: NSE holidays pe last known price carry forward — standard finance practice.

### Tests: 12/12 PASSING ✅
- Stock registry (4): stock count 45+, sector mapping, sector pairs, supply chain
- Data download (4): CSV files exist 40+, columns correct, NIFTY index 1000+ rows, combined 40+ cols
- Data quality (4): quality checks pass, clean removes NaN, no duplicates, date range 2015-2025

---

## PHASE 2: Feature Engineering — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/data/features.py` | 21 technical indicators + z-score normalization + tensor builder | ~280 | ✅ |
| `tests/test_features.py` | 18 tests (14 unit + 4 edge cases) | ~280 | ✅ 18/18 PASS |

### 21 Features Generated (per stock, per day)
| Category | Features | Count |
|----------|----------|-------|
| Trend | RSI, MACD, MACD Signal, MACD Histogram | 4 |
| Bollinger | BB Upper, BB Mid, BB Lower | 3 |
| Moving Avg | SMA 20, SMA 50, EMA 12, EMA 26 | 4 |
| Volatility | ATR, Volatility 20d, Volatility 60d | 3 |
| Stochastic | Stoch %K, Stoch %D | 2 |
| Volume | Volume SMA, Volume Ratio | 2 |
| Returns | Return 1d, 5d, 20d | 3 |

### Key Decisions
1. **Pure pandas/numpy calculations** — `pandas_ta` aur `ta-lib` dono Python 3.11 pe install issues the. Manual implementation = zero external dependency, full control.
2. **Per-stock rolling z-score** — Har stock ka apna mean/std (252-day window). Cross-sectional nahi kiya kyunki har stock ka scale alag hai.
3. **NaN rows DROPPED** — Rolling windows ke warm-up period (~252 days) ka data hata diya. Downstream mein zero NaN issues.
4. **Clip [-5, +5]** — Extreme z-scores clip kiye to prevent outlier domination in neural networks.
5. **3D tensor output** — `build_feature_tensor()` returns `(n_stocks, n_timesteps, n_features)` shape — directly usable for T-GAT/RL.
6. **Look-ahead bias tested** — Test verifies z-score at time t is identical whether computed on full data or truncated data.

### Kyu Banaya (Why / Reasoning)
1. **Raw OHLCV se model nahi seekhta** — Close price ek number hai, usse trend/momentum/volatility ka pata nahi chalta. Indicators yeh "derived signals" provide karte hain.
2. **RSI (Relative Strength Index)** — Overbought (>70) / oversold (<30) detect karta hai. RL agent ko "stock overpriced" signal milta hai.
3. **MACD** — Trend reversal indicator. MACD line signal line cross kare = buy/sell signal.
4. **Bollinger Bands** — Volatility bands. Price upper band touch kare = potentially sell, lower = potentially buy.
5. **Rolling z-score kyu?** — Static normalization (fit on training set) mein problem: val/test ke statistics alag hote hain. Rolling = adaptive, time-aware, no leakage.
6. **NaN drop kyu?** — Agar NaN chhod dete toh T-GAT mein NaN propagate hota, loss NaN hota, training fail hoti. Clean input = clean training.

### Tests: 18/18 PASSING ✅
- Technical indicators (3): all 21 columns present, count check, real data verification
- Normalization (2): z-score clipped [-5,+5], no look-ahead bias
- Full pipeline (4): no NaN output, all features present, rows reduced, real RELIANCE data
- Feature tensor (3): correct 3D shape, no NaN, float32 dtype
- Edge cases (4): short history, zero volume, constant price, single stock
- Feature columns (2): match config, get_feature_columns returns copy

---

## PHASE 3: FinBERT Sentiment — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/sentiment/finbert.py` | FinBERT model loading, single/batch prediction, daily aggregation, decay series, sentiment matrix | ~300 | ✅ |
| `src/sentiment/news_fetcher.py` | Google News RSS fetcher, ticker-to-company mapping, SQLite cache | ~200 | ✅ |
| `tests/test_sentiment.py` | 19 tests (15 unit + 4 edge cases) | ~280 | ✅ 19/19 PASS |

### Key Decisions
1. **Local model loading** — ProsusAI/finbert (417MB) downloaded manually to `data/finbert_local/` because HuggingFace downloads fail behind college proxy (SSL interception). Code auto-detects local vs hub.
2. **torch.load patch** — torch 2.5.1 `weights_only=True` breaks with .bin files. Patched for compatibility.
3. **FP16 on GPU** — FinBERT loads in half precision on CUDA (~200MB VRAM instead of ~400MB). CPU fallback for testing.
4. **Sentiment decay (0.95)** — Days without news: previous sentiment * 0.95. After ~60 days without news, sentiment decays to near-zero.
5. **Score = P(positive) - P(negative)** — Range [-1, +1]. Simple, interpretable.
6. **SQLite cache** — Avoid re-computing sentiment for same headlines. Persistent across runs.

### Kyu Banaya (Why / Reasoning)
1. **Market sirf numbers se nahi chalta** — "RBI raises interest rates" headline se banking stocks girengi. Yeh info OHLCV data mein nahi hai.
2. **FinBERT kyu?** — General BERT financial text samajhta nahi. "Bearish" ka matlab finance mein alag hai. FinBERT financial corpus pe trained hai.
3. **Google News RSS kyu?** — Free, no API key. Real-time headlines. Limitation: only recent ~100 results, not historical archive.
4. **Decay kyu?** — Agar Monday ko news aayi "+ve", Tuesday ko koi news nahi, toh Tuesday ka sentiment Monday jaisa hi hona chahiye (thoda kam). Bina decay ke gaps mein 0 hoga = misleading.
5. **Batch prediction kyu?** — 50 stocks × 15 headlines = 750 predictions. One-by-one slow hai. Batching = GPU parallelism.

### Tests: 19/19 PASSING ✅
- Model loading (2): loads successfully on CPU, has 3 output labels
- Prediction accuracy (4): positive text → +ve score, negative → -ve, neutral → ~0, all in [-1,+1]
- Batch (2): correct count, matches individual predictions
- Aggregation (1): multiple headlines per day averaged correctly
- Decay series (2): fills gaps with decay, new headline resets decay
- Matrix (1): correct (n_stocks, n_timesteps) shape, float32
- Edge cases (4): empty text → neutral, short text → neutral, decay → 0 over time, single headline
- News fetcher (3): company name lookup, unknown ticker fallback, SQLite DB init

---

## PHASE 4: Graph Construction — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/graph/builder.py` | Multi-relational graph builder: 3 edge types + PyG Data objects | ~360 | ✅ |
| `tests/test_graph.py` | 20 tests (16 unit + 4 edge cases) | ~290 | ✅ 20/20 PASS |

### Graph Structure
| Edge Type | Constant | Source | Count | Nature |
|-----------|----------|--------|-------|--------|
| Sector edges | `EDGE_SECTOR=0` | Same sector stocks connected | ~160 directed | Static (never changes) |
| Supply chain | `EDGE_SUPPLY_CHAIN=1` | Business relationships (TATASTEEL→MARUTI) | ~54 directed | Static (never changes) |
| Correlation | `EDGE_CORRELATION=2` | Rolling |corr| > 0.6 threshold | Variable | Dynamic (changes daily) |

### Key Functions
| Function | Input | Output | Kya Karta Hai |
|----------|-------|--------|---------------|
| `build_sector_edges()` | ticker_to_idx | edge_index [2, N] | Same sector stocks connect (bidirectional) |
| `build_supply_chain_edges()` | ticker_to_idx | edge_index [2, N] | Business relationship edges (bidirectional) |
| `build_correlation_edges_fast()` | corr_matrix, threshold | edge_index [2, N] | Vectorized: |corr| > threshold pairs (no self-loops) |
| `build_static_graph()` | ticker_to_idx | (edge_index, edge_type) | Sector + supply chain combined, deduplicated |
| `build_full_graph()` | node_features, corr_matrix | PyG Data object | All 3 edge types → ready for T-GAT |
| `build_graph_sequence()` | feature_tensor, close_prices | list[Data] | One graph per trading day with dynamic corr edges |
| `get_graph_stats()` | PyG Data | dict | Node count, edge counts by type, density |

### Key Decisions
1. **Vectorized correlation edges** — `np.triu` + `np.where` instead of double for-loop. O(n²) but with NumPy C-level speed.
2. **Bidirectional all edges** — Sector, supply chain, and correlation edges all added in both directions (a→b, b→a). GNN message passing works better with undirected graphs.
3. **Edge deduplication** — Same stock pair can be in both sector AND supply chain. `_deduplicate_edges()` keeps first occurrence to avoid double-counting.
4. **Graph sequence builder** — One PyG Data object per trading day. Static edges computed once, reused. Only correlation edges recomputed daily.
5. **Empty edge handling** — Single stock or zero correlation → returns `torch.zeros((2, 0))` shape, not error.

### Kyu Banaya (Why / Reasoning)
1. **GNN ko adjacency matrix chahiye** — T-GAT ko batana padta hai kaunse stocks connected hain. Random connections galat honge — domain knowledge based edges correct hain.
2. **3 edge types kyu?** — Different relationships capture karte hain: sector = similar industry, supply chain = business dependency, correlation = statistical co-movement. Multi-relational GNN in teeno ko alag alag process kar sakta hai.
3. **Dynamic correlation kyu?** — 2020 COVID mein sab stocks correlated the (panic selling). Normal times mein IT aur Pharma uncorrelated. Static correlation misleading hogi — rolling window (60 days) se current market regime capture hota hai.
4. **PyG Data object kyu?** — PyTorch Geometric ka standard format. T-GAT, GCN, GAT sab isse directly accept karte hain. Reinventing the wheel ki zarurat nahi.
5. **Threshold 0.6 kyu?** — Too low (0.3) = too many edges = noise. Too high (0.9) = too few edges = information loss. 0.6 = moderate, literature standard for financial correlation networks.

### Tests: 20/20 PASSING ✅
- Sector edges (3): correct count, bidirectional, no self-loops
- Supply chain edges (3): exist, bidirectional, no self-loops
- Correlation edges (3): threshold respected, no self-loops, bidirectional
- Static graph (2): both edge types present, type length matches
- Full graph (3): correct shape, all 3 types with correlation, numpy auto-convert
- Graph stats (2): expected keys, density [0,1]
- Edge cases (4): zero correlation, perfect correlation, single stock, negative correlation

### Git Commit
```
Phase 4: Graph construction — 3 edge types + PyG Data (2026-03-16)
```

---

## PHASE 5: T-GAT Model — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/models/tgat.py` | T-GAT: Multi-relational GAT + GRU temporal encoder | ~230 | ✅ |
| `tests/test_tgat.py` | 19 tests (15 unit + 4 edge cases) | ~280 | ✅ 19/19 PASS |

### Architecture
```
Input (n_stocks, 21 features)
  → Linear Projection (21 → 64)
  → RelationalGATLayer × 2 (3 edge types, 4 heads each, residual + LayerNorm)
  → GRU Temporal Encoder (sequence of graph snapshots → temporal embedding)
  → Output Projection (64 → 64)
Output: (n_stocks, 64) stock embeddings
```

### Key Decisions
1. **Multi-relational GAT** — Separate GATConv per edge type (sector/supply/correlation). Each learns different attention patterns. Weighted aggregation with learnable relation importance.
2. **Residual connections** — Skip connections around each GAT layer. Prevents gradient vanishing in deeper networks. LayerNorm for stable training.
3. **GRU (not LSTM)** — GRU has fewer parameters than LSTM (2 gates vs 3), similar performance. Better for 4GB VRAM constraint.
4. **Mixed precision via autocast** — LayerNorm doesn't support pure FP16. Using `torch.cuda.amp.autocast()` for proper mixed precision.
5. **56K parameters, 0.22 MB** — Lightweight model. FP16 = 0.11 MB. Plenty of room on 4GB VRAM.

### Tests: 19/19 PASSING ✅
- Model init (4): creates, config from YAML, custom config, expected components
- Forward pass (4): sequence shape, single shape, finite embeddings ×2
- Gradients (2): flow check, loss decrease
- Relational GAT (2): missing edge type, all 3 types
- Model size (3): <1M params, <10MB, FP16 on CUDA
- Edge cases (4): no edges, single node, 20-step sequence, empty raises error

---

## PHASE 6: RL Environment — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/rl/environment.py` | Gymnasium-compatible portfolio management environment | ~250 | ✅ |
| `tests/test_env.py` | 23 tests (17 unit + 6 edge cases) | ~270 | ✅ 23/23 PASS |

### Environment Design
| Component | Detail |
|-----------|--------|
| **Observation** | stock features (n×21) + portfolio weights (n) + cash_ratio + norm_value + optional: embeddings + sentiment |
| **Action** | continuous (n_stocks,) → softmax → target portfolio weights |
| **Reward** | Sharpe-based + drawdown penalty + turnover penalty |
| **Constraints** | Max 20%/stock, -5% stop loss, -15% max drawdown circuit breaker |
| **Costs** | Transaction 0.1% + slippage 0.05% per turnover unit |

### Key Decisions
1. **Gymnasium API** — Standard RL interface. Compatible with Stable-Baselines3, CleanRL, RLlib.
2. **Softmax action** — Raw actions → softmax → portfolio weights. Ensures valid weights (positive, sum-to-1). Numerically stable.
3. **Random start** — Each episode starts at random date within training data. Prevents overfitting to specific start dates.
4. **Sharpe-based reward** — Rolling 20-day Sharpe ratio. Risk-adjusted returns, not just raw returns.
5. **Circuit breaker** — -15% drawdown terminates episode. Prevents catastrophic losses during training.
6. **Stop loss** — Per-stock -5% daily loss → forced exit. Realistic risk management.

### Tests: 23/23 PASSING ✅
- Init (4): creates, obs_space, action_space, initial all-cash
- Reset (4): returns tuple, obs shape, clears state, deterministic with seed
- Step (5): returns 5-tuple, obs shape, value changes, costs applied, info keys
- Constraints (4): max position, weights ≤ 1, stop loss, max drawdown terminates
- Edge cases (6): zero action, single stock, truncation, gym API, summary, embeddings+sentiment

---

## PHASE 7: Deep RL Agent — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/rl/agent.py` | PPO + SAC agents, training, evaluation, comparison | ~250 | ✅ |
| `tests/test_agent.py` | 16 tests (12 unit + 4 edge cases) | ~230 | ✅ 16/16 PASS |

### Key Components
| Function | Kya Karta Hai |
|----------|---------------|
| `create_ppo_agent()` | PPO with config-driven hyperparams, MLP policy [128, 64] |
| `create_sac_agent()` | SAC with replay buffer, auto entropy |
| `train_agent()` | Training with eval callback + portfolio metrics logging |
| `evaluate_agent()` | Multi-episode evaluation → mean return, Sharpe, max DD |
| `compare_agents()` | PPO vs SAC head-to-head comparison |
| `save_agent() / load_agent()` | Model persistence (.zip format) |

### Key Decisions
1. **Stable-Baselines3** — Production-quality RL implementations. Tested, documented, maintained. No need to implement PPO from scratch.
2. **Small policy network** [128, 64] — Only ~46K params for PPO. Lightweight for 4GB VRAM. Larger networks don't help with 47 stocks.
3. **PPO primary, SAC comparison** — PPO is stable and good for continuous actions. SAC is more sample-efficient. Compare both in thesis.
4. **PortfolioMetricsCallback** — Custom callback logs Sharpe/return/drawdown during training, not just loss.

### Tests: 16/16 PASSING ✅
- PPO (3): creates, trains, predicts valid actions
- SAC (3): creates, trains, predicts valid actions
- Evaluation (2): returns expected keys, finite metrics
- Save/Load (2): PPO and SAC save → load → same predictions
- Custom config (2): custom LR, custom architecture
- Edge cases (4): single stock, 10-step training, compare, eval callback

---

## PHASE 8-9: TimeGAN + Stress Testing — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/gan/timegan.py` | TimeGAN: 5 GRU-based components, 3-phase training | ~360 | ✅ |
| `src/gan/stress.py` | VaR, CVaR, Monte Carlo, 4 crash scenarios | ~320 | ✅ |
| `tests/test_gan.py` | 25 tests (18 unit + 7 edge cases) | ~340 | ✅ 25/25 PASS |

### TimeGAN Architecture
```
TimeGAN = Autoencoder + Adversarial + Temporal Dynamics

5 Components (All GRU-based):
  1. Embedder:      Real data → Latent space (sigmoid output)
  2. Recovery:      Latent space → Data space (reconstruction)
  3. Generator:     Random noise → Fake latent sequences (sigmoid output)
  4. Discriminator: Real vs Fake classifier (logits)
  5. Supervisor:    Latent → Next-step latent (temporal dynamics)

3 Training Phases:
  Phase 1 (40%): Autoencoder (Embedder + Recovery) — learn latent representation
  Phase 2 (20%): Supervisor — learn temporal dynamics in latent space
  Phase 3 (40%): Joint adversarial — Generator + Discriminator + moment matching
```

### Stress Testing Framework
| Component | Kya Karta Hai |
|-----------|---------------|
| `compute_var()` | Value at Risk — "95% confidence se worst loss kitni hogi?" |
| `compute_cvar()` | Conditional VaR — "Worst 5% cases mein average loss" |
| `monte_carlo_simulation()` | Cholesky decomposition + random paths → portfolio returns distribution |
| `simulate_crash_scenario()` | Stressed covariance + shocked returns → survival rate tracking |
| `run_all_stress_tests()` | 4 scenarios: normal, 2008 crash, COVID, flash crash |
| `stress_test_summary()` | Formatted results dict with percentages |

### 4 Crash Scenarios
| Scenario | Daily Shock Mean | Shock Std | Duration | Correlation Boost |
|----------|-----------------|-----------|----------|-------------------|
| Normal | 0.0% | 1.0% | 252 days | 0% |
| 2008 Crisis | -0.3% | 3.5% | 120 days | +30% |
| COVID March 2020 | -0.5% | 5.0% | 30 days | +40% |
| Flash Crash | -2.0% | 8.0% | 5 days | +50% |

### Key Decisions
1. **GRU, not LSTM** — Fewer parameters (2 gates vs 3). Same performance for our scale. VRAM friendly.
2. **Sigmoid activation in latent space** — Bounds outputs [0,1], stabilizes GAN training (no exploding values).
3. **3-phase training** — First learn good representations, then temporal dynamics, then adversarial. Joint from scratch = unstable.
4. **Moment matching loss** — Generator penalized if mean/std differs from real data. Stabilizes GAN beyond just adversarial signal.
5. **Cholesky decomposition for Monte Carlo** — Generates correlated random returns from covariance matrix. Standard quantitative finance technique.
6. **Correlation boost in crisis** — In crashes, all stocks become correlated (panic selling). Boosting off-diagonal correlations simulates this.
7. **Survival rate** — % of simulations where drawdown stays above -15% threshold. Direct risk metric for portfolio.

### Kyu Banaya (Why / Reasoning)
1. **TimeGAN kyu?** — Real market data limited hai (10 years = ~2500 trading days). Synthetic data augmentation se RL agent ko zyada training data milta hai. Regular GAN temporal patterns nahi samajhta — TimeGAN specifically time series ke liye designed hai.
2. **Stress testing kyu?** — "Acha model banaya, but 2008 jaisi crash aaye toh?" VaR/CVaR standard risk metrics hain jo banks mein mandatory hain. Monte Carlo future possibilities explore karta hai. Crash scenarios historical extreme events simulate karte hain.
3. **CVaR > VaR kyu?** — VaR says "95% chance loss se zyada nahi hogi X". But worst 5% mein kitna lose karenge? CVaR (Expected Shortfall) average worst-case batata hai. More conservative, regulators prefer it.

### Tests: 25/25 PASSING ✅
- TimeGAN init (3): creates, components exist, stats
- Training (2): 2D input, 3D pre-windowed input
- Generation (3): correct shape, finite values, reasonable statistics
- Data prep (1): sliding window correctness
- VaR (3): 95% VaR value, CVaR ≤ VaR, 99% worse than 95%
- Monte Carlo (2): returns StressResult, VaR values present
- Crash scenarios (4): all scenarios run, 2008 worse than normal, survival rate [0,1], summary format
- Edge cases (7): single feature, short training, generate-before-train, equal weights, concentrated portfolio, unknown scenario, zero variance

### Git Commit
```
Phase 8-9: TimeGAN + Stress Testing (2026-03-16)
```

---

## PHASE 10: NAS / DARTS — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/nas/search_space.py` | 5 candidate ops (linear/conv1d/attention/skip/none), MixedOp, SearchSpace config | ~170 | ✅ |
| `src/nas/darts.py` | TGATSupernet, DARTSSearcher (bilevel optimization), RL policy grid search, PDF report | ~430 | ✅ |
| `tests/test_nas.py` | 18 tests (14 unit + 4 edge cases) | ~300 | ✅ 18/18 PASS |

### Architecture
```
DARTS for T-GAT:
  TGATSupernet: input_proj → MixedOp×N (each blends 5 ops) → GRU → output_proj
  Bilevel optimization: alpha (arch) on val, W (weights) on train
  After search: extract top-3 architectures → retrain from scratch

RL Policy Grid Search:
  5 candidates: [64,32], [128,64], [256,128], [128,128,64], [64,64]
  Train PPO with each → evaluate Sharpe → rank by performance
```

### Key Decisions
1. **DARTS on T-GAT only** — Full DARTS wrapping SB3 PPO is hacky. T-GAT DARTS is the real thesis contribution. RL uses simple grid search.
2. **5 operations** — linear (standard), conv1d (feature mixing), attention (self-attention), skip (residual), none (prune path).
3. **Bilevel optimization** — Outer loop optimizes architecture weights (alpha) on validation, inner loop optimizes model weights on training. Standard DARTS approach.
4. **Top-3 extraction** — Best (argmax alpha) + 2 variants (swap one layer to 2nd-best op). Retraining from scratch removes weight sharing bias.
5. **Soft comparison** — NAS vs hand-designed test logs comparison, warns if <5% improvement. Unit tests shouldn't hard-fail on training outcomes.
6. **PDF report** — matplotlib + PdfPages generates convergence plots + alpha heatmap + architecture descriptions.

### Tests: 18/18 PASSING ✅
- Supernet (4): param count <50MB, single forward, sequence forward, arch/weight param separation
- Alpha convergence (1): entropy decreases during 30-epoch search
- Architecture extraction (1): top-3 extracted with valid ops
- NAS comparison (1): valid convergence info with finite loss
- Report (1): PDF generated with 3 pages
- Reproducibility (1): same seed = same architecture
- RL policy search (2): 5 candidates exist, grid search returns ranked results
- Edge cases (3): tiny search space, single layer, skip dominance
- Search space (4): all ops instantiate, MixedOp blends, config loads, unknown op raises

### Git Commit
```
Phase 10: NAS/DARTS — architecture search for T-GAT + RL policy (18/18 tests)
```

---

## PHASE 11: Federated Learning — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/federated/server.py` | FLServer: FedAvg + FedProx aggregation, training loop | ~180 | ✅ |
| `src/federated/client.py` | FLClient: 4 sector-wise clients, local training | ~160 | ✅ |
| `src/federated/privacy.py` | DP-SGD: gradient clipping, noise injection, budget tracking | ~190 | ✅ |
| `tests/test_fl.py` | 17 tests (13 unit + 4 edge cases) | ~300 | ✅ 17/17 PASS |

### Architecture
```
FL System:
  4 Clients (sector-wise non-IID split):
    Client 0: Banking + Finance (~10 stocks)
    Client 1: IT + Telecom (~6 stocks)
    Client 2: Pharma + FMCG (~8 stocks)
    Client 3: Energy + Auto + Metals + Infra + Others (~23 stocks)

  Server aggregation:
    FedAvg:  weighted_avg(client_weights, by=data_size)
    FedProx: FedAvg + proximal_term(mu=0.01) on client side

  Differential Privacy (DP-SGD):
    1. Clip gradients to max_norm=1.0
    2. Add calibrated Gaussian noise (epsilon=8, delta=1e-5)
    3. Track cumulative privacy budget per round
```

### Key Decisions
1. **From scratch, no Flower** — More thesis-friendly, demonstrates understanding of FL algorithms.
2. **Sector-wise non-IID** — Realistic: hedge funds specialize in sectors. More challenging for FL than random split.
3. **FedProx over FedAvg** — FedProx adds proximal term preventing client drift. Better for non-IID.
4. **DP-SGD** — Gradient clipping + noise. epsilon=8 is usable (model still learns), delta=1e-5 standard.
5. **Privacy budget tracking** — Cumulative epsilon via composition theorem. Alerts when budget exhausted.

### Tests: 17/17 PASSING ✅
- Client init (4): 4 clients create, sector mapping, tickers, invalid ID
- Convergence (1): loss decreases over 20 FL rounds
- FedAvg vs FedProx (1): both produce finite losses
- Federated vs individual (1): FL model evaluated against solo clients
- DP noise (1): epsilon=8 training still converges
- Privacy budget (2): tracking increments, noise multiplier positive
- Client fairness (1): all clients benefit from FL
- Aggregation (2): weighted average correct, size-weighted correct
- Edge cases (4): Byzantine client, tiny client, small epsilon, single client

---

## PHASE 12: Quantum ML (QAOA) — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/quantum/qaoa.py` | QAOA: QUBO build, Ising convert, circuit build, COBYLA optimize | ~250 | ✅ |
| `src/quantum/portfolio.py` | Portfolio encoding, Markowitz weights, classical brute-force, scaling study | ~230 | ✅ |
| `tests/test_quantum.py` | 12 tests (9 unit + 3 edge cases) | ~200 | ✅ 12/12 PASS |

### Architecture
```
QAOA Portfolio Selection:
  1. 47 NIFTY stocks → top 8 by Sharpe → candidate pool
  2. QUBO matrix: -returns + risk_aversion × covariance + penalty × (sum=K)²
  3. QUBO → Ising (Z/ZZ terms): x = (1-z)/2 substitution
  4. QAOA circuit: H gates → [cost_unitary + mixer_unitary] × p layers → measure
  5. COBYLA optimizer tunes gamma/beta angles (200 iterations)
  6. Best bitstring → selected K assets → Markowitz optimal weights
  7. Compare with classical brute-force (exact for N≤12)
```

### Key Decisions
1. **Qiskit 2.3 + AerSimulator** — Most mature quantum framework. Shot-based simulation matches real hardware.
2. **Binary selection + classical weights** — QAOA selects WHICH assets (binary), Markowitz computes HOW MUCH (continuous). Standard hybrid approach.
3. **COBYLA optimizer** — Gradient-free, handles noisy shot-based objectives. Matches config.
4. **N≤12 qubit ceiling** — Classical simulation is O(2^N). 12 qubits = 4096 states, feasible.
5. **Penalty-based cardinality constraint** — penalty × (sum(x) - K)² forces exactly K assets selected.
6. **Brute-force classical baseline** — C(12,6) = 924 combos, exact optimal. Fair comparison for thesis.

### Tests: 12/12 PASSING ✅
- QUBO (3): correct shape, finite entries, returns on diagonal
- Circuit (2): builds correctly, has measurement gates
- Optimization (1): QAOA returns valid result with correct bitstring
- Classical (1): brute-force finds portfolio with finite Sharpe
- Quantum vs Classical (1): both produce finite Sharpe ratios
- Scaling (1): benchmark at [4, 6] sizes completes
- Edge cases (3): 2-asset minimum, identical assets, single asset

---

## PHASE 13: API + Docker — ✅ DONE

### Kya Banaya (What)
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/api/schemas.py` | Pydantic v2 request/response models for all endpoints | ~150 | ✅ |
| `src/api/main.py` | FastAPI app: 8 REST endpoints + CORS middleware | ~260 | ✅ |
| `Dockerfile` | Multi-stage Docker image, CPU-only PyTorch for serving | ~30 | ✅ |
| `docker-compose.yml` | API + PostgreSQL services with health checks | ~45 | ✅ |
| `tests/test_api.py` | 15 tests (10 unit + 5 edge cases) | ~200 | ✅ 15/15 PASS |

### API Endpoints
| Method | Path | Kya Karta Hai |
|--------|------|---------------|
| GET | `/api/health` | Health check — status, version, project name |
| GET | `/api/config` | Non-sensitive config (seed, device, fp16) |
| GET | `/api/stocks` | NIFTY 50 stock list with sectors |
| POST | `/api/sentiment` | Single text → FinBERT sentiment score |
| POST | `/api/sentiment/batch` | Multiple texts → batch sentiment |
| POST | `/api/stress-test` | Monte Carlo stress testing with crash scenarios |
| POST | `/api/qaoa` | QAOA quantum portfolio optimization |
| POST | `/api/metrics` | Sharpe, Sortino, drawdown from daily returns |

### Key Decisions
1. **FastAPI over Flask** — Async support, auto Swagger docs, Pydantic validation built-in. Modern Python API standard.
2. **Lazy imports** — Heavy modules (FinBERT, Qiskit, stress) imported inside endpoint functions. Keeps startup < 1s.
3. **CORS for React** — `localhost:3000` allowed for Phase 14 dashboard development.
4. **CPU-only Docker** — API serving doesn't need GPU. Smaller image, runs anywhere.
5. **PostgreSQL** — Production-grade DB for storing results, predictions, user data.
6. **Health check in Dockerfile** — Docker auto-restarts unhealthy containers.
7. **Pydantic v2 validation** — Request validation with min/max constraints catches bad input before processing.

### Kyu Banaya (Why / Reasoning)
1. **API kyu?** — Model train karne se koi product nahi banta. API expose karo toh koi bhi consume kar sakta hai — React dashboard, mobile app, external users.
2. **Docker kyu?** — "Mere machine pe chalta hai" se "kisi bhi machine pe chalta hai". Reproducible deployment. Thesis examiner ke system pe bhi chalega.
3. **Pydantic schemas kyu?** — Bina validation ke agar koi n_stocks=-5 bheje toh server crash ho jayega. Pydantic automatically 422 error de deta hai with helpful message.
4. **Swagger auto-docs kyu?** — `/docs` pe jaake koi bhi endpoint try kar sakta hai without Postman. Thesis demo mein interactive.

### Tests: 15/15 PASSING ✅
- Health + Config (2): correct status/version, config keys present
- Stocks (1): returns tickers with sectors
- Sentiment (3): positive score > 0, negative < 0, batch processes multiple
- Stress test (1): returns scenarios with VaR/CVaR
- QAOA (1): quantum + classical results, correct bitstring length
- Metrics (1): Sharpe, Sortino, drawdown computed from returns
- CORS (1): headers present for localhost:3000
- Edge cases (5): empty text 422, long text 422, empty batch 422, invalid n_stocks 422, too few returns 422

---

## PHASE 14: React Dashboard — ✅ DONE

### Kya Banaya (What)
| File/Dir | Purpose | Status |
|----------|---------|--------|
| `dashboard/` | Full React dashboard app (Vite + TypeScript) | ✅ |
| `src/components/layout/` | Sidebar, Header, DashboardLayout | ✅ |
| `src/components/ui/` | Card, MetricCard, Badge, PageHeader | ✅ |
| `src/components/charts/` | SparkLine, PerformanceChart, SectorDonut | ✅ |
| `src/lib/` | API client, formatters (INR), animation presets | ✅ |
| `src/pages/` (10 pages) | All dashboard views | ✅ |

### 10 Dashboard Pages
| Page | Route | Kya Dikhata Hai |
|------|-------|-----------------|
| Overview | `/` | Portfolio value, Sharpe, Max DD, performance chart, sector donut, top holdings |
| Portfolio | `/portfolio` | Detailed metrics from API, sector weights, full stock list |
| GNN Insights | `/gnn` | Stock network SVG, attention heatmap, edge type legend |
| RL Agent | `/rl` | PPO vs SAC reward curves, portfolio weights bar chart |
| Stress Testing | `/stress` | Monte Carlo paths, VaR/CVaR gauges, scenario table (live API) |
| NAS Lab | `/nas` | Architecture diagram, alpha convergence, NAS vs hand-designed |
| Federated | `/fl` | FL convergence curves, client cards, fairness comparison |
| Quantum | `/quantum` | QAOA circuit SVG, Sharpe comparison, weight bars (live API) |
| Sentiment | `/sentiment` | Real-time FinBERT analysis, batch headlines, sector sentiment (live API) |
| **Graph Viz** | `/graph` | **Interactive force-directed stock network — node size=RL weight, edges=sector/supply/correlation** |

### Tech Stack
- React 18 + TypeScript + Vite
- Tailwind CSS (v2 Light & Warm Theme — terracotta #C15F3C primary)
- Framer Motion (spring animations, scroll reveals)
- Recharts (area, bar, line, pie charts)
- Lucide React (icons)
- react-countup (animated numbers)
- Custom SVG force-directed graph (Graph Viz page)
- API proxy: Vite dev server → FastAPI backend (:8000)

### Key Decisions
1. **React (NOT Next.js)** — No SSR needed for a dashboard. Simpler, lighter.
2. **v2 Light Warm Theme** — White/cream bg, terracotta primary (#C15F3C), professional + warm.
3. **Framer Motion** — Spring physics animations. Cards fade-slide on scroll, numbers count up.
4. **Live API integration** — Sentiment, Stress Test, QAOA, Metrics pages call real FastAPI endpoints.
5. **Custom force graph** — No Three.js dependency. Pure SVG + force simulation for stock network.
6. **Indian number format** — ₹ symbol, Lakh/Crore system for all monetary values.

---

## PHASE 15: Upcoming

| Phase | Key Challenge | Reasoning |
|-------|--------------|-----------|
| 15: Thesis | Final document + demo | Everything compiled into thesis format. |

---

## Test Score Tracker

| Phase | Unit Tests | Edge Cases | Integration | Status |
|-------|-----------|------------|-------------|--------|
| 0 | 18/18 | - | - | ✅ PASS |
| 1 | 12/12 | ✓ handled | - | ✅ PASS |
| 2 | 14/14 | 4/4 | - | ✅ PASS |
| 3 | 15/15 | 4/4 | - | ✅ PASS |
| 4 | 16/16 | 4/4 | Integration #1 | ✅ PASS |
| 5 | 15/15 | 4/4 | - | ✅ PASS |
| 6 | 17/17 | 6/6 | - | ✅ PASS |
| 7 | 12/12 | 4/4 | - | ✅ PASS |
| 8-9 | 18/18 | 7/7 | Integration #2 | ✅ PASS |
| 10 | 14/14 | 4/4 | - | ✅ PASS |
| 11 | 13/13 | 4/4 | - | ✅ PASS |
| 12 | 9/9 | 3/3 | - | ✅ PASS |
| 13 | 10/10 | 5/5 | Integration #3 | ✅ PASS |
| 14 | - | - | - | - |
| **Total** | **183/183** | **49/49** | **3/3** | **232/232** |

---

## Rules (Kabhi Mat Bhoolna)
1. **No phase starts until previous phase tests 100% green + git committed**
2. **FP16 mandatory** for all neural networks (4GB VRAM constraint)
3. **Train: 2015-2021, Val: 2022-2023, Test: 2024-2025** (test set use ONCE at the end)
4. **India constants**: Risk-free 7%, Transaction cost 0.1%, Slippage 0.05%, 248 trading days/year
5. **Git**: Commit after every phase. Clean history.

---

> This file will be updated after every phase completion with actual results, test scores, and lessons learned.
