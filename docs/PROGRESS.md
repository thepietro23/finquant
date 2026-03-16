# FINQUANT-NEXUS v4 — Phase-wise Progress Tracker

> **Last Updated:** 2026-03-16
> **Current Phase:** Phase 6 (RL Environment) — ✅ DONE
> **Overall:** Phase 0 ✅ (18/18), Phase 1 ✅ (12/12), Phase 2 ✅ (18/18), Phase 3 ✅ (19/19), Phase 4 ✅ (20/20), Phase 5 ✅ (19/19), Phase 6 ✅ (23/23) = 129/129 tests GREEN

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
| 7 | Deep RL Agent | NOT STARTED | D12-D17 | PPO + SAC training |
| 8-9 | TimeGAN + Stress | NOT STARTED | D18-D24 | Synthetic data + stress testing |
| 10 | NAS/DARTS | NOT STARTED | D25-D30 | Architecture search |
| 11 | Federated Learning | NOT STARTED | D31-D37 | Multi-client FL with DP |
| 12 | Quantum ML | NOT STARTED | D38-D42 | QAOA portfolio optimization |
| 13 | API + Docker | NOT STARTED | D43-D46 | FastAPI + containerization |
| 14 | Dashboard + Benchmarks | NOT STARTED | D46-D49 | React frontend with best viz libs |
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

## PHASES 7-15: Upcoming (Brief)

| Phase | Key Challenge | Reasoning |
|-------|--------------|-----------|
| 3: FinBERT | Google News RSS → sentiment scores | Free news source, no API key needed. FinBERT pre-trained on financial text. |
| 4: Graph | Multi-edge graph (correlation + sector + supply chain) | GNN needs adjacency matrix. 3 types of edges = richer relationships. |
| 5: T-GAT | Temporal Graph Attention | Attention mechanism = important neighbors get more weight. Temporal = time-varying graphs. |
| 6: RL Env | Gymnasium environment for portfolio | Standard interface for RL agents. Observation = features + graph embeddings + sentiment. |
| 7: Deep RL | PPO (primary) + SAC (comparison) | PPO stable hai, SAC sample-efficient. Dono compare karke best pick karenge. |
| 8-9: GAN | TimeGAN for synthetic data + stress testing | Limited real data (10 years). Synthetic data augmentation. Stress test: "what if 2008 crash?" |
| 10: NAS | DARTS architecture search | Manually GNN design karna suboptimal. NAS automatically best architecture dhundhta hai. |
| 11: FL | Federated Learning with differential privacy | Sector-wise clients, privacy-preserving. Novel contribution for thesis. |
| 12: Quantum | QAOA portfolio optimization | Quantum computing angle for thesis novelty. Compare with classical. |
| 13: API | FastAPI + Docker | Production deployment. REST API for predictions. |
| 14: Dashboard | Next.js 14 frontend | Proper UI, not Streamlit. Interactive charts, real-time updates. |
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
| 7 | -/8 | -/4 | - | - |
| 8-9 | -/13 | -/7 | Integration #2 | - |
| 10 | -/7 | -/3 | - | - |
| 11 | -/8 | -/4 | - | - |
| 12 | -/6 | -/3 | - | - |
| 13 | -/10 | -/5 | Integration #3 | - |
| 14 | - | - | - | - |
| **Total** | **107/124** | **22+/54** | **0/11** | **129/189** |

---

## Rules (Kabhi Mat Bhoolna)
1. **No phase starts until previous phase tests 100% green + git committed**
2. **FP16 mandatory** for all neural networks (4GB VRAM constraint)
3. **Train: 2015-2021, Val: 2022-2023, Test: 2024-2025** (test set use ONCE at the end)
4. **India constants**: Risk-free 7%, Transaction cost 0.1%, Slippage 0.05%, 248 trading days/year
5. **Git**: Commit after every phase. Clean history.

---

> This file will be updated after every phase completion with actual results, test scores, and lessons learned.
