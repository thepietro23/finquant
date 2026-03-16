# FINQUANT-NEXUS v4 — Practical Testing Guide (Hands-on)

> **Purpose:** Yeh guide tere liye hai — manually cheezein run karna, output dekhna, samajhna ki kya ho raha hai.
> Har phase ke liye: Python REPL commands, FastAPI endpoints (jab ready hoon), aur expected outputs.
>
> **How to use:** Terminal kholo, venv activate karo, aur copy-paste karte jao.

---

## Setup (Ek Baar Karo)

```bash
cd d:/Personal/Clg/finquant/fqn1
source venv/Scripts/activate    # Windows Git Bash
# ya
.\venv\Scripts\activate         # Windows PowerShell
```

---

## Phase 0: Global Setup — Manual Testing

### 1. Config System Test
```python
python -c "
from src.utils.config import get_config

# Pura config dekho
cfg = get_config()
print('Sections:', list(cfg.keys()))

# Specific section
rl = get_config('rl')
print('RL algorithm:', rl['algorithm'])
print('RL learning rate:', rl['lr'])
print('RL total timesteps:', rl['total_timesteps'])

# Data config
data = get_config('data')
print('Risk-free rate:', data['risk_free_rate'])
print('Trading days/year:', data['trading_days_per_year'])
"
```
**Expected Output:**
```
Sections: ['seed', 'device', 'fp16', 'data', 'features', 'sentiment', 'gnn', 'rl', 'gan', 'stress', 'nas', 'fl', 'quantum', 'api', 'logging']
RL algorithm: PPO
RL learning rate: 0.0003
RL total timesteps: 500000
Risk-free rate: 0.07
Trading days/year: 248
```

### 2. Seed Reproducibility Test
```python
python -c "
from src.utils.seed import set_seed
import numpy as np
import torch

# Run 1
set_seed(42)
a = np.random.rand(3)
b = torch.randn(3)
print('Run 1 — NumPy:', a)
print('Run 1 — Torch:', b)

# Run 2 (same seed = SAME output)
set_seed(42)
c = np.random.rand(3)
d = torch.randn(3)
print('Run 2 — NumPy:', c)
print('Run 2 — Torch:', d)

print('Match?', np.allclose(a, c) and torch.allclose(b, d))
"
```
**Expected:** Dono runs ka output EXACTLY same hona chahiye. `Match? True`

### 3. Logger Test
```python
python -c "
from src.utils.logger import get_logger

log = get_logger('test')
log.info('Hello from FINQUANT!')
log.warning('Yeh ek warning hai')
log.error('Yeh ek error hai')
print('Check logs/ folder for log file')
"
```
**Expected:** Console pe colored log messages + `logs/` mein file created.

### 4. Metrics Test
```python
python -c "
import numpy as np
from src.utils.metrics import sharpe_ratio, max_drawdown, sortino_ratio, annualized_return

# Simulate daily returns: 0.1% average daily return with 1% volatility
np.random.seed(42)
returns = np.random.normal(0.001, 0.01, 248)  # 1 year

print('--- Simulated Portfolio (1 Year) ---')
print(f'Sharpe Ratio: {sharpe_ratio(returns):.2f}')
print(f'Sortino Ratio: {sortino_ratio(returns):.2f}')
print(f'Annualized Return: {annualized_return(returns):.1%}')

portfolio_values = 100 * np.cumprod(1 + returns)
print(f'Max Drawdown: {max_drawdown(portfolio_values):.1%}')
print(f'Final Value: Rs {portfolio_values[-1]:.2f} (started at Rs 100)')
"
```
**Expected:** Sharpe ~1.5-2.5, Positive annualized return, Negative max drawdown.

---

## Phase 1: Data Pipeline — Manual Testing

### 1. Stock Registry
```python
python -c "
from src.data.stocks import get_all_tickers, get_sector, get_sector_pairs, get_supply_chain_pairs

tickers = get_all_tickers()
print(f'Total stocks: {len(tickers)}')
print(f'First 5: {tickers[:5]}')
print()

# Sector lookup
for t in ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'MARUTI.NS']:
    print(f'{t} -> {get_sector(t)}')
print()

# Graph edges preview
pairs = get_sector_pairs()
sc = get_supply_chain_pairs()
print(f'Sector edges: {len(pairs)}')
print(f'Supply chain edges: {len(sc)}')
print(f'Sample supply chain: {sc[:3]}')
"
```

### 2. Data Quality Check
```python
python -c "
from src.data.quality import DataQualityChecker
import pandas as pd

qc = DataQualityChecker()

# Check RELIANCE
df = pd.read_csv('data/RELIANCE_NS.csv', index_col=0, parse_dates=True)
print(f'RELIANCE raw: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}')
print(f'NaN count: {df.isnull().sum().sum()}')

passed = qc.check_stock(df, 'RELIANCE.NS')
print(f'Quality check: {\"PASS\" if passed else \"FAIL\"}')

# Clean it
clean = qc.clean_stock(df)
print(f'After cleaning: {len(clean)} rows, NaN: {clean.isnull().sum().sum()}')
"
```

### 3. Look at Raw Data
```python
python -c "
import pandas as pd

# Dekho RELIANCE ka raw data kaisa dikhta hai
df = pd.read_csv('data/RELIANCE_NS.csv', index_col=0, parse_dates=True)
print('=== RELIANCE Raw Data (Last 5 Days) ===')
print(df.tail())
print()
print('=== Stats ===')
print(df.describe().round(2))
"
```

---

## Phase 2: Feature Engineering — Manual Testing

### 1. Single Stock Features (RELIANCE)
```python
python -c "
import pandas as pd
from src.data.quality import DataQualityChecker
from src.data.features import compute_technical_indicators, FEATURE_COLUMNS

# Load and clean
df = pd.read_csv('data/RELIANCE_NS.csv', index_col=0, parse_dates=True)
qc = DataQualityChecker()
clean = qc.clean_stock(df)
print(f'Clean data: {len(clean)} rows')

# Compute indicators (WITHOUT normalization — raw values dekhne ke liye)
featured = compute_technical_indicators(clean)
print(f'After indicators: {len(featured)} rows, {len(featured.columns)} columns')
print()

# Dekho kuch features kaise dikhte hain
print('=== RELIANCE Features (Last 5 Days) ===')
cols_to_show = ['Close', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr', 'return_1d', 'volatility_20d']
print(featured[cols_to_show].tail().round(2).to_string())
print()

# Feature stats
print('=== Feature Statistics ===')
print(featured[FEATURE_COLUMNS].describe().round(2).to_string())
"
```
**Expected:** RSI between 0-100, MACD around 0, returns small decimals, volatility positive.

### 2. Full Pipeline (Single Stock)
```python
python -c "
import pandas as pd
from src.data.quality import DataQualityChecker
from src.data.features import engineer_stock_features, FEATURE_COLUMNS

# Load, clean, engineer
df = pd.read_csv('data/RELIANCE_NS.csv', index_col=0, parse_dates=True)
qc = DataQualityChecker()
clean = qc.clean_stock(df)
result = engineer_stock_features(clean)

print(f'Input: {len(clean)} rows')
print(f'Output: {len(result)} rows (dropped {len(clean) - len(result)} NaN warm-up rows)')
print(f'Features: {len(FEATURE_COLUMNS)}')
print(f'NaN count: {result[FEATURE_COLUMNS].isna().sum().sum()} (should be 0)')
print()

# Z-score normalized values — should be roughly [-5, +5]
print('=== Normalized Feature Ranges ===')
for col in FEATURE_COLUMNS:
    mn, mx = result[col].min(), result[col].max()
    print(f'  {col:20s}: [{mn:+.2f}, {mx:+.2f}]')
"
```
**Expected:** Sab features [-5, +5] ke andar. Zero NaN.

### 3. Feature Tensor (Multiple Stocks)
```python
python -c "
import pandas as pd
import numpy as np
from src.data.quality import DataQualityChecker
from src.data.features import engineer_stock_features, build_feature_tensor, FEATURE_COLUMNS

qc = DataQualityChecker()
features_dict = {}

# 5 stocks test karo
for ticker in ['RELIANCE_NS', 'TCS_NS', 'HDFCBANK_NS', 'INFY_NS', 'ICICIBANK_NS']:
    df = pd.read_csv(f'data/{ticker}.csv', index_col=0, parse_dates=True)
    clean = qc.clean_stock(df)
    feat = engineer_stock_features(clean)
    features_dict[ticker] = feat
    print(f'{ticker}: {len(feat)} rows')

# Build tensor
tensor, dates, tickers = build_feature_tensor(features_dict)
print(f'\n=== Feature Tensor ===')
print(f'Shape: {tensor.shape}')
print(f'  Stocks: {tensor.shape[0]}')
print(f'  Timesteps: {tensor.shape[1]}')
print(f'  Features: {tensor.shape[2]}')
print(f'Date range: {dates[0].date()} to {dates[-1].date()}')
print(f'Dtype: {tensor.dtype}')
print(f'NaN count: {np.isnan(tensor).sum()}')
print(f'Memory: {tensor.nbytes / 1024 / 1024:.1f} MB')
"
```
**Expected:** Shape like `(5, ~2000, 21)`, dtype `float32`, zero NaN.

### 4. Visualize Features (Optional — agar matplotlib installed hai)
```python
python -c "
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # No GUI needed
import matplotlib.pyplot as plt
from src.data.quality import DataQualityChecker
from src.data.features import compute_technical_indicators

# Load RELIANCE (raw indicators, not normalized)
df = pd.read_csv('data/RELIANCE_NS.csv', index_col=0, parse_dates=True)
qc = DataQualityChecker()
clean = qc.clean_stock(df)
featured = compute_technical_indicators(clean)

# Last 1 year only
recent = featured.last('365D')

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# 1. Price + Bollinger Bands
axes[0].plot(recent.index, recent['Close'], label='Close', color='black')
axes[0].plot(recent.index, recent['bb_upper'], '--', label='BB Upper', color='red', alpha=0.5)
axes[0].plot(recent.index, recent['bb_lower'], '--', label='BB Lower', color='green', alpha=0.5)
axes[0].fill_between(recent.index, recent['bb_lower'],
 recent['bb_upper'], alpha=0.1)
axes[0].set_title('RELIANCE — Price + Bollinger Bands')
axes[0].legend()

# 2. RSI
axes[1].plot(recent.index, recent['rsi'], color='purple')
axes[1].axhline(70, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(30, color='green', linestyle='--', alpha=0.5)
axes[1].set_title('RSI (Overbought >70, Oversold <30)')

# 3. MACD
axes[2].plot(recent.index, recent['macd'], label='MACD', color='blue')
axes[2].plot(recent.index, recent['macd_signal'], label='Signal', color='orange')
axes[2].bar(recent.index, recent['macd_hist'], label='Histogram', alpha=0.3)
axes[2].set_title('MACD')
axes[2].legend()

# 4. Volume Ratio
axes[3].bar(recent.index, recent['volume_ratio'], alpha=0.5, color='gray')
axes[3].axhline(1.0, color='black', linestyle='--')
axes[3].set_title('Volume Ratio (>1 = above average)')

plt.tight_layout()
plt.savefig('data/reliance_features_preview.png', dpi=100)
print('Chart saved to data/reliance_features_preview.png')
print('Open it to see RELIANCE ke Bollinger Bands, RSI, MACD, aur Volume!')
"
```

---

## Phase 3: FinBERT Sentiment — Manual Testing

### 1. Single Headline Sentiment
```python
python -c "
from src.sentiment.finbert import predict_sentiment

# Positive news
result = predict_sentiment('Reliance Q3 profit beats estimates, revenue up 25%')
print(f'Positive news: score={result[\"score\"]:.3f}')
print(f'  P(pos)={result[\"positive\"]:.3f}, P(neg)={result[\"negative\"]:.3f}, P(neu)={result[\"neutral\"]:.3f}')
print()

# Negative news
result = predict_sentiment('Stock crashes 15% after company reports massive losses and debt default')
print(f'Negative news: score={result[\"score\"]:.3f}')
print(f'  P(pos)={result[\"positive\"]:.3f}, P(neg)={result[\"negative\"]:.3f}, P(neu)={result[\"neutral\"]:.3f}')
print()

# Neutral news
result = predict_sentiment('The company held its annual general meeting on Monday')
print(f'Neutral news: score={result[\"score\"]:.3f}')
print(f'  P(pos)={result[\"positive\"]:.3f}, P(neg)={result[\"negative\"]:.3f}, P(neu)={result[\"neutral\"]:.3f}')
print()

# Try your own!
result = predict_sentiment('RBI raises repo rate by 25 basis points')
print(f'RBI rate hike: score={result[\"score\"]:.3f}')
"
```
**Expected:** Positive news → score ~0.9, Negative → ~-0.9, Neutral → ~0.0

### 2. Batch Prediction (Multiple Headlines)
```python
python -c "
from src.sentiment.finbert import predict_batch

headlines = [
    'TCS wins \$2 billion deal from major US bank',
    'Infosys faces \$50 million penalty for data breach',
    'HDFC Bank to open 500 new branches next year',
    'Market crashes on global recession fears',
    'Adani stock rebounds after short seller report dismissed',
]

results = predict_batch(headlines, batch_size=16)
print('=== Batch Sentiment Scores ===')
for headline, r in zip(headlines, results):
    emoji = '+' if r['score'] > 0.1 else ('-' if r['score'] < -0.1 else '~')
    print(f'  [{emoji}] {r[\"score\"]:+.3f}  {headline[:60]}')
"
```

### 3. Sentiment Decay Demo
```python
python -c "
import pandas as pd
from src.sentiment.finbert import build_sentiment_series

dates = pd.bdate_range('2024-01-01', periods=20)
# Only 2 days have news
daily = {
    '2024-01-01': {'avg_score': 0.8, 'num_headlines': 3},
    '2024-01-10': {'avg_score': -0.5, 'num_headlines': 2},
}
series = build_sentiment_series(daily, dates, decay_factor=0.95)

print('=== Sentiment Decay Over 20 Days ===')
for date, val in zip(dates, series):
    bar = '#' * int(abs(val) * 30)
    sign = '+' if val >= 0 else '-'
    marker = ' <-- NEWS' if str(date.date()) in daily else ''
    print(f'  {date.date()} [{sign}] {val:+.3f} {bar}{marker}')
"
```
**Expected:** Score decays from 0.8 towards 0, resets to -0.5 on Jan 10, then decays again.

### 4. News Fetcher (Live — Needs Internet)
```python
python -c "
from src.sentiment.news_fetcher import fetch_stock_news, get_company_name

# Check company name mapping
for t in ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']:
    print(f'{t} -> {get_company_name(t)}')
print()

# Fetch live news (needs internet)
print('Fetching RELIANCE news...')
headlines = fetch_stock_news('RELIANCE.NS', max_results=5)
for h in headlines:
    date = h['published'].strftime('%Y-%m-%d') if h['published'] else 'unknown'
    print(f'  [{date}] {h[\"title\"][:80]}')
"
```

### 5. Full Pipeline: Headlines → Sentiment Scores
```python
python -c "
from src.sentiment.news_fetcher import fetch_stock_news
from src.sentiment.finbert import predict_batch

# Fetch + Score for RELIANCE
headlines = fetch_stock_news('RELIANCE.NS', max_results=10)
texts = [h['title'] for h in headlines]
scores = predict_batch(texts)

print('=== RELIANCE.NS Live Sentiment ===')
for h, s in zip(headlines, scores):
    emoji = '+' if s['score'] > 0.1 else ('-' if s['score'] < -0.1 else '~')
    date = h['published'].strftime('%m-%d') if h['published'] else '??'
    print(f'  [{emoji}] {s[\"score\"]:+.3f}  [{date}] {h[\"title\"][:65]}')

import numpy as np
avg = np.mean([s['score'] for s in scores])
print(f'\nAverage sentiment: {avg:+.3f}')
"
```

---

## Phase 4: Graph Construction — Manual Testing

### 1. Sector Edges
```python
python -c "
from src.graph.builder import build_sector_edges, EDGE_SECTOR
from src.data.stocks import get_ticker_to_index, get_sector_pairs

ticker_to_idx = get_ticker_to_index()
edge_index = build_sector_edges(ticker_to_idx)

pairs = get_sector_pairs()
valid = [(a,b) for a,b in pairs if a in ticker_to_idx and b in ticker_to_idx]

print('=== Sector Edges ===')
print(f'Sector pairs: {len(valid)}')
print(f'Directed edges: {edge_index.shape[1]} (should be {len(valid)*2})')
print(f'Shape: {edge_index.shape}')
print(f'Sample edges (first 5):')
for k in range(min(5, edge_index.shape[1])):
    i, j = edge_index[0,k].item(), edge_index[1,k].item()
    print(f'  Node {i} → Node {j}')
"
```

### 2. Supply Chain Edges
```python
python -c "
from src.graph.builder import build_supply_chain_edges
from src.data.stocks import get_supply_chain_pairs, get_ticker_to_index

edge_index = build_supply_chain_edges()
pairs = get_supply_chain_pairs()
ticker_to_idx = get_ticker_to_index()

print('=== Supply Chain Edges ===')
print(f'Defined pairs: {len(pairs)}')
print(f'Directed edges: {edge_index.shape[1]}')

# Show actual stock names
idx_to_ticker = {v:k for k,v in ticker_to_idx.items()}
print(f'Sample edges:')
for k in range(min(6, edge_index.shape[1])):
    i, j = edge_index[0,k].item(), edge_index[1,k].item()
    print(f'  {idx_to_ticker.get(i, i)} → {idx_to_ticker.get(j, j)}')
"
```

### 3. Correlation Edges (Synthetic Data)
```python
python -c "
import numpy as np
from src.graph.builder import build_correlation_edges_fast

# Create known correlation matrix
n = 10
corr = np.eye(n)
corr[0,1] = corr[1,0] = 0.85   # High positive
corr[2,3] = corr[3,2] = 0.40   # Below threshold
corr[4,5] = corr[5,4] = -0.75  # High negative

edge_index = build_correlation_edges_fast(corr, threshold=0.6)
print('=== Correlation Edges (Synthetic) ===')
print(f'Matrix size: {n}x{n}')
print(f'Edges found: {edge_index.shape[1]}')
print(f'Expected: 4 (pairs 0-1 and 4-5, each bidirectional)')
print()

for k in range(edge_index.shape[1]):
    i, j = edge_index[0,k].item(), edge_index[1,k].item()
    print(f'  Node {i} → Node {j} (corr={corr[i,j]:.2f})')
"
```

### 4. Full Graph (All 3 Edge Types)
```python
python -c "
import numpy as np
import torch
from src.graph.builder import build_full_graph, get_graph_stats, EDGE_SECTOR, EDGE_SUPPLY_CHAIN, EDGE_CORRELATION
from src.data.stocks import get_ticker_to_index

ticker_to_idx = get_ticker_to_index()
n = len(ticker_to_idx)

# Fake features + correlation matrix
features = torch.randn(n, 21)
corr = np.eye(n)
for i in range(0, n-1, 2):
    corr[i, i+1] = corr[i+1, i] = 0.8

data = build_full_graph(features, corr_matrix=corr, threshold=0.6, ticker_to_idx=ticker_to_idx)
stats = get_graph_stats(data)

print('=== Full Graph ===')
print(f'Nodes: {stats[\"num_nodes\"]}')
print(f'Total edges: {stats[\"num_edges\"]}')
print(f'Density: {stats[\"density\"]:.4f}')
print(f'Sector edges: {stats[\"sector_edges\"]}')
print(f'Supply chain edges: {stats[\"supply_chain_edges\"]}')
print(f'Correlation edges: {stats[\"correlation_edges\"]}')
print(f'Feature shape: {data.x.shape}')
print(f'Edge index shape: {data.edge_index.shape}')
"
```
**Expected:** ~200+ total edges, all 3 types present, density ~0.05-0.15.

### 5. Static Graph (Without Correlation)
```python
python -c "
from src.graph.builder import build_static_graph, EDGE_SECTOR, EDGE_SUPPLY_CHAIN

edge_index, edge_type = build_static_graph()
print('=== Static Graph (Sector + Supply Chain) ===')
print(f'Total edges: {edge_index.shape[1]}')
print(f'Sector: {int((edge_type == EDGE_SECTOR).sum())}')
print(f'Supply chain: {int((edge_type == EDGE_SUPPLY_CHAIN).sum())}')
print(f'Edge type tensor: {edge_type.shape}')
"
```

---

## Running Tests (Automated — Har Phase Ke Baad)

### Run All Tests
```bash
cd d:/Personal/Clg/finquant/fqn1
source venv/Scripts/activate
python -m pytest tests/ -v
```

### Run Specific Phase
```bash
python -m pytest tests/test_phase0.py -v    # Phase 0 only
python -m pytest tests/test_data.py -v      # Phase 1 only
python -m pytest tests/test_features.py -v  # Phase 2 only
python -m pytest tests/test_sentiment.py -v # Phase 3 only
python -m pytest tests/test_graph.py -v     # Phase 4 only
```

### Run Single Test
```bash
python -m pytest tests/test_features.py::TestNormalization::test_zscore_clipped -v
```

### Run with Print Output (debugging)
```bash
python -m pytest tests/test_features.py -v -s  # -s shows print() output
```

---

## FastAPI Endpoints (Phase 13 mein aayenge)

> Abhi Phase 13 dur hai, but jab ready hoga toh yeh endpoints milenge:

```bash
# Server start
uvicorn src.api.main:app --reload

# Health check
curl http://localhost:8000/health

# Get stock features
curl http://localhost:8000/api/features/RELIANCE.NS

# Get sentiment score
curl http://localhost:8000/api/sentiment/RELIANCE.NS

# Get portfolio recommendation
curl http://localhost:8000/api/portfolio/recommend

# Run backtest
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2024-01-01", "end_date": "2024-12-31"}'
```

---

## Quick Reference: Project Structure

```
fqn1/
├── configs/
│   └── base.yaml           # Sab hyperparameters
├── src/
│   ├── utils/
│   │   ├── config.py        # Config loader
│   │   ├── seed.py          # Reproducibility
│   │   ├── logger.py        # Logging
│   │   └── metrics.py       # Financial metrics (Sharpe, MaxDD, etc.)
│   ├── data/
│   │   ├── stocks.py        # NIFTY 50 registry
│   │   ├── download.py      # yfinance downloader
│   │   ├── quality.py       # Data quality checker
│   │   └── features.py      # 21 indicators + z-score normalization
│   ├── sentiment/
│   │   ├── finbert.py       # FinBERT sentiment scoring
│   │   └── news_fetcher.py  # Google News RSS + SQLite cache
│   ├── graph/
│   │   └── builder.py       # 3 edge types + PyG Data objects
│   ├── rl/                  # Phase 6-7: RL environment + agents
│   ├── gan/                 # Phase 8-9: TimeGAN
│   ├── nas/                 # Phase 10: DARTS
│   ├── federated/           # Phase 11: FL
│   ├── quantum/             # Phase 12: Quantum ML
│   └── api/                 # Phase 13: FastAPI
├── tests/
│   ├── test_phase0.py       # 18 tests
│   ├── test_data.py         # 12 tests
│   ├── test_features.py     # 18 tests
│   ├── test_sentiment.py    # 19 tests
│   └── test_graph.py        # 20 tests
├── data/                    # Raw CSVs (gitignored)
│   └── features/            # Feature CSVs + pickle (gitignored)
├── models/                  # Saved models (gitignored)
├── logs/                    # Log files (gitignored)
└── docs/
    ├── PROGRESS.md          # Phase tracker
    ├── PHASE_0_1_EXPLAINED.md  # Theory + reasoning
    └── PRACTICAL_GUIDE.md   # THIS FILE — hands-on testing
```

---

## Debugging Tips

```python
# Agar koi import error aaye
python -c "import sys; sys.path.insert(0, '.'); from src.data.features import FEATURE_COLUMNS; print(FEATURE_COLUMNS)"

# Agar data nahi mil raha
python -c "import os; print(os.listdir('data/')[:10])"

# Agar CUDA check karna hai
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Memory check
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB') if torch.cuda.is_available() else print('No GPU')"
```
