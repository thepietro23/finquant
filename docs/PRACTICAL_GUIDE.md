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

## Phase 5: T-GAT Model — Manual Testing

### 1. Model Creation + Architecture
```python
python -c "
from src.models.tgat import TGAT, count_parameters, get_model_size_mb

model = TGAT(n_features=21)
print('=== T-GAT Architecture ===')
print(model)
print()
print(f'Parameters: {count_parameters(model):,}')
print(f'Model size: {get_model_size_mb(model):.2f} MB (FP32)')
print(f'FP16 size:  {get_model_size_mb(model)/2:.2f} MB')
"
```
**Expected:** ~56K parameters, 0.22 MB, 2 GAT layers, 4 heads each.

### 2. Single Graph Forward Pass
```python
python -c "
import torch
from torch_geometric.data import Data
from src.models.tgat import TGAT

model = TGAT(n_features=21)
model.eval()

# Fake graph: 10 stocks, 21 features, some edges
x = torch.randn(10, 21)
edge_index = torch.tensor([[0,1,2,3,4,5,1,0,3,2,5,4],
                            [1,0,3,2,5,4,0,1,2,3,4,5]], dtype=torch.long)
edge_type = torch.tensor([0,0,0,0,1,1,0,0,0,0,1,1], dtype=torch.long)
data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

with torch.no_grad():
    emb = model.forward_single(data)

print(f'Input: {x.shape} (10 stocks, 21 features)')
print(f'Output: {emb.shape} (10 stocks, 64 embedding dims)')
print(f'Embedding sample (stock 0): {emb[0, :5].tolist()}')
print(f'All finite: {torch.isfinite(emb).all()}')
"
```

### 3. Graph Sequence (Temporal)
```python
python -c "
import torch
from torch_geometric.data import Data
from src.models.tgat import TGAT

model = TGAT(n_features=21)
model.eval()

# 5 timesteps, 10 stocks
graphs = []
for t in range(5):
    x = torch.randn(10, 21)
    ei = torch.tensor([[0,1,2,3],[1,0,3,2]], dtype=torch.long)
    et = torch.tensor([0,0,1,1], dtype=torch.long)
    graphs.append(Data(x=x, edge_index=ei, edge_type=et))

with torch.no_grad():
    emb, spatial = model(graphs)

print(f'Sequence length: {len(graphs)} timesteps')
print(f'Final embedding: {emb.shape} (stocks × output_dim)')
print(f'All spatial: {spatial.shape} (stocks × timesteps × hidden_dim)')
print(f'GRU captured temporal info: {emb.shape[1]} dims per stock')
"
```

### 4. GPU + Mixed Precision Test
```python
python -c "
import torch
from torch_geometric.data import Data
from src.models.tgat import TGAT

if not torch.cuda.is_available():
    print('No CUDA — skipping GPU test')
else:
    model = TGAT(n_features=21).cuda()
    x = torch.randn(10, 21).cuda()
    ei = torch.tensor([[0,1],[1,0]], dtype=torch.long).cuda()
    et = torch.zeros(2, dtype=torch.long).cuda()
    data = Data(x=x, edge_index=ei, edge_type=et)

    # Mixed precision
    with torch.no_grad(), torch.amp.autocast('cuda'):
        emb = model.forward_single(data)

    print(f'GPU output: {emb.shape}, dtype: {emb.dtype}')
    print(f'GPU memory used: {torch.cuda.memory_allocated()/1024/1024:.1f} MB')
    print(f'GPU memory reserved: {torch.cuda.memory_reserved()/1024/1024:.1f} MB')
"
```

### 5. Training Sanity Check (1 step)
```python
python -c "
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from src.models.tgat import TGAT

model = TGAT(n_features=21)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Fake data
graphs = []
for t in range(3):
    x = torch.randn(10, 21)
    ei = torch.tensor([[0,1,2,3],[1,0,3,2]], dtype=torch.long)
    et = torch.tensor([0,0,1,1], dtype=torch.long)
    graphs.append(Data(x=x, edge_index=ei, edge_type=et))

target = torch.zeros(10, 64)

# Step 1
emb, _ = model(graphs)
loss1 = F.mse_loss(emb, target)
loss1.backward()
optimizer.step()
optimizer.zero_grad()

# Step 2
emb, _ = model(graphs)
loss2 = F.mse_loss(emb, target)

print(f'Loss before: {loss1.item():.4f}')
print(f'Loss after:  {loss2.item():.4f}')
print(f'Decreased:   {loss2.item() < loss1.item()}')
print('Training works!' if loss2.item() < loss1.item() else 'Check model')
"
```

---

## Phase 6: RL Environment — Manual Testing

### 1. Create and Explore Environment
```python
python -c "
import numpy as np
from src.rl.environment import PortfolioEnv

# Synthetic data: 10 stocks, 500 days
n_stocks, n_time, n_feat = 10, 500, 21
np.random.seed(42)
features = np.random.randn(n_stocks, n_time, n_feat).astype(np.float32)
prices = 100 * np.cumprod(1 + np.random.randn(n_stocks, n_time) * 0.01, axis=1).astype(np.float32)

env = PortfolioEnv(features, prices, episode_length=100)
print('=== RL Environment ===')
print(f'Observation space: {env.observation_space.shape}')
print(f'Action space: {env.action_space.shape}')
print(f'Stocks: {env.n_stocks}')
print(f'Episode length: {env.episode_length}')
print(f'Max position: {env.max_position}')
print(f'Stop loss: {env.stop_loss}')
print(f'Max drawdown: {env.max_drawdown}')
"
```

### 2. Run a Full Episode
```python
python -c "
import numpy as np
from src.rl.environment import PortfolioEnv

np.random.seed(42)
n = 10
features = np.random.randn(n, 500, 21).astype(np.float32)
prices = 100 * np.cumprod(1 + np.random.randn(n, 500) * 0.01, axis=1).astype(np.float32)

env = PortfolioEnv(features, prices, episode_length=50)
obs, info = env.reset(seed=42)

total_reward = 0
for step in range(50):
    if env.done:
        break
    action = env.action_space.sample()  # Random agent
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward

summary = env.get_portfolio_summary()
print('=== Episode Complete ===')
print(f'Steps: {summary[\"n_steps\"]}')
print(f'Total return: {summary[\"total_return\"]:.2%}')
print(f'Sharpe ratio: {summary[\"sharpe\"]:.2f}')
print(f'Max drawdown: {summary[\"max_drawdown\"]:.2%}')
print(f'Final value: Rs {summary[\"final_value\"]:,.0f}')
print(f'Total reward: {total_reward:.2f}')
print(f'Positions at end: {info[\"n_positions\"]}')
"
```

### 3. Constraint Demo (Stop Loss + Drawdown)
```python
python -c "
import numpy as np
from src.rl.environment import PortfolioEnv

np.random.seed(42)
n = 5
features = np.random.randn(n, 300, 21).astype(np.float32)
# Create crashing prices (-3% every day)
prices = 100 * np.cumprod(1 + np.full((n, 300), -0.03), axis=1).astype(np.float32)

env = PortfolioEnv(features, prices, episode_length=100)
env.reset(seed=42)

for step in range(50):
    if env.done:
        print(f'Episode ended at step {step}!')
        break
    action = np.ones(n, dtype=np.float32)  # Fully invested
    obs, reward, term, trunc, info = env.step(action)
    if step % 5 == 0:
        print(f'Step {step}: value={info[\"portfolio_value\"]:,.0f}, dd={info[\"drawdown\"]:.1%}')

if env.done:
    print(f'Terminated: drawdown breached -15% limit')
    print(f'Final value: Rs {env.portfolio_value:,.0f}')
"
```

---

## Phase 7: Deep RL Agent — Manual Testing

### 1. Create PPO Agent
```python
python -c "
import numpy as np
from src.rl.environment import PortfolioEnv
from src.rl.agent import create_ppo_agent

np.random.seed(42)
n = 5
features = np.random.randn(n, 300, 21).astype(np.float32)
prices = 100 * np.cumprod(1 + np.random.randn(n, 300) * 0.01, axis=1).astype(np.float32)
env = PortfolioEnv(features, prices, episode_length=50)

model = create_ppo_agent(env, device='cpu')
print(f'PPO params: {sum(p.numel() for p in model.policy.parameters()):,}')
print(f'Device: {model.device}')
print(f'Policy: {model.policy}')
"
```

### 2. Train + Evaluate PPO (Quick)
```python
python -c "
import numpy as np
from src.rl.environment import PortfolioEnv
from src.rl.agent import create_ppo_agent, evaluate_agent

np.random.seed(42)
n = 5
features = np.random.randn(n, 300, 21).astype(np.float32)
prices = 100 * np.cumprod(1 + np.random.randn(n, 300) * 0.01, axis=1).astype(np.float32)
env = PortfolioEnv(features, prices, episode_length=50)

model = create_ppo_agent(env, device='cpu')
print('Training PPO (1000 steps)...')
model.learn(total_timesteps=1000)
print('Training done!')

metrics = evaluate_agent(model, env, n_episodes=3)
print(f'=== PPO Evaluation (3 episodes) ===')
print(f'Mean return: {metrics[\"mean_return\"]:.2%}')
print(f'Sharpe ratio: {metrics[\"mean_sharpe\"]:.2f}')
print(f'Max drawdown: {metrics[\"mean_max_dd\"]:.2%}')
print(f'Avg steps/episode: {metrics[\"mean_steps\"]:.0f}')
"
```

### 3. PPO vs SAC Comparison
```python
python -c "
import numpy as np
from src.rl.environment import PortfolioEnv
from src.rl.agent import create_ppo_agent, create_sac_agent, compare_agents

np.random.seed(42)
n = 5
features = np.random.randn(n, 300, 21).astype(np.float32)
prices = 100 * np.cumprod(1 + np.random.randn(n, 300) * 0.01, axis=1).astype(np.float32)
env = PortfolioEnv(features, prices, episode_length=50)

print('Training PPO (1000 steps)...')
ppo = create_ppo_agent(env, device='cpu')
ppo.learn(total_timesteps=1000)

print('Training SAC (1000 steps)...')
sac = create_sac_agent(env, device='cpu', buffer_size=1000, batch_size=32, learning_starts=100)
sac.learn(total_timesteps=1000)

print()
result = compare_agents(ppo, sac, env, n_episodes=3)
print(f'=== PPO vs SAC ===')
print(f'PPO — return: {result[\"ppo\"][\"mean_return\"]:.2%}, sharpe: {result[\"ppo\"][\"mean_sharpe\"]:.2f}')
print(f'SAC — return: {result[\"sac\"][\"mean_return\"]:.2%}, sharpe: {result[\"sac\"][\"mean_sharpe\"]:.2f}')
print(f'Winner: {result[\"winner\"]}')
"
```

### 4. Save and Load Model
```python
python -c "
import numpy as np
from src.rl.environment import PortfolioEnv
from src.rl.agent import create_ppo_agent, save_agent, load_agent

np.random.seed(42)
n = 5
features = np.random.randn(n, 300, 21).astype(np.float32)
prices = 100 * np.cumprod(1 + np.random.randn(n, 300) * 0.01, axis=1).astype(np.float32)
env = PortfolioEnv(features, prices, episode_length=50)

model = create_ppo_agent(env, device='cpu')
model.learn(total_timesteps=500)

save_agent(model, 'models/test_ppo')
print('Model saved!')

loaded = load_agent('models/test_ppo', env=env, algorithm='PPO')
print('Model loaded!')

obs, _ = env.reset(seed=42)
action, _ = loaded.predict(obs, deterministic=True)
print(f'Action: {action}')
print(f'Weights sum: {action.sum():.4f}')
"
```

---

## Phase 8: TimeGAN — Manual Testing

### 1. Create and Inspect
```python
python -c "
from src.gan.timegan import TimeGAN
gan = TimeGAN(input_dim=5, hidden_dim=24, latent_dim=12, n_layers=1)
print(f'Components: embedder, recovery, generator, discriminator, supervisor')
print(f'Total params: {sum(p.numel() for p in gan.parameters()):,}')
print(f'Trained: {gan.trained}')
"
```

### 2. Train on Synthetic Data
```python
python -c "
import numpy as np
from src.gan.timegan import TimeGAN
np.random.seed(42)
data = np.cumsum(np.random.randn(500, 5) * 0.01, axis=0).astype(np.float32) + 100
gan = TimeGAN(input_dim=5, hidden_dim=24, latent_dim=12, n_layers=1)
print('Training (10 epochs)...')
gan.train(data, epochs=10, batch_size=32, seq_length=24)
print(f'Trained: {gan.trained}')
"
```

### 3. Generate Synthetic Data
```python
python -c "
import numpy as np
from src.gan.timegan import TimeGAN
np.random.seed(42)
data = np.cumsum(np.random.randn(500, 5) * 0.01, axis=0).astype(np.float32) + 100
gan = TimeGAN(input_dim=5, hidden_dim=24, latent_dim=12, n_layers=1)
gan.train(data, epochs=10, batch_size=32, seq_length=24)
synthetic = gan.generate(n_samples=50, seq_length=24)
print(f'Shape: {synthetic.shape}  (50 sequences x 24 days x 5 features)')
print(f'Finite: {np.isfinite(synthetic).all()}')
print(f'Real std: {data[:24].std():.2f}, Synth std: {synthetic.std():.2f}')
"
```

---

## Phase 9: Stress Testing — Manual Testing

### 1. VaR and CVaR
```python
python -c "
import numpy as np
from src.gan.stress import compute_var, compute_cvar
np.random.seed(42)
returns = np.random.normal(0.0005, 0.01, 10000)
print(f'VaR(95%):  {compute_var(returns, 0.95):.4f}')
print(f'VaR(99%):  {compute_var(returns, 0.99):.4f}')
print(f'CVaR(95%): {compute_cvar(returns, 0.95):.4f}')
"
```

### 2. Monte Carlo Simulation
```python
python -c "
import numpy as np
from src.gan.stress import monte_carlo_simulation
np.random.seed(42)
n = 10
weights = np.ones(n) / n
mean_ret = np.random.normal(0.0005, 0.001, n)
cov = np.eye(n) * 0.0001
result = monte_carlo_simulation(weights, mean_ret, cov, n_simulations=5000, n_days=252)
print(f'Mean return: {result.mean_return:.2%}')
print(f'VaR(95%): {result.var_95:.2%}')
print(f'CVaR(95%): {result.cvar_95:.2%}')
print(f'Survival rate: {result.survival_rate:.1%}')
"
```

### 3. All Crash Scenarios
```python
python -c "
import numpy as np
from src.gan.stress import run_all_stress_tests, stress_test_summary
np.random.seed(42)
n = 10
weights = np.ones(n) / n
mean_ret = np.random.normal(0.0005, 0.001, n)
cov = np.eye(n) * 0.0001
results = run_all_stress_tests(weights, mean_ret, cov, n_simulations=2000)
summary = stress_test_summary(results)
for scenario, data in summary.items():
    print(f'--- {scenario.upper()} ---')
    for k, v in data.items():
        print(f'  {k}: {v}')
    print()
"
```
**Expected:** Normal best → 2008 → COVID → Flash Crash progressively worse.

---

## Phase 10: NAS / DARTS — Manual Testing

### 1. Search Space Operations
```python
python -c "
import torch
from src.nas.search_space import create_operation, OPERATION_REGISTRY, MixedOp

x = torch.randn(10, 32)
print('=== 5 Candidate Operations ===')
for name in OPERATION_REGISTRY:
    op = create_operation(name, 32, 64)
    out = op(x)
    params = sum(p.numel() for p in op.parameters())
    print(f'  {name:12s}: input={x.shape} → output={out.shape}, params={params:,}')

print()
mixed = MixedOp(32, 64)
out = mixed(x)
print(f'MixedOp: input={x.shape} → output={out.shape}')
print(f'Alpha weights: {mixed.get_weights()}')
print(f'Selected op: {mixed.get_selected_op()}')
"
```
**Expected:** All 5 ops produce (10, 64), MixedOp blends them with ~equal weights initially.

### 2. TGATSupernet
```python
python -c "
import torch
from torch_geometric.data import Data
from src.nas.darts import TGATSupernet

supernet = TGATSupernet(n_features=21, hidden_dim=32, num_layers=3)
print(f'Supernet params: {supernet.count_parameters():,}')
print(f'Size: {supernet.get_size_mb():.2f} MB')
print(f'Entropy: {supernet.get_alpha_entropy():.3f} (high = uniform alpha)')
print()

# Single graph
x = torch.randn(10, 21)
ei = torch.tensor([[0,1,2,3],[1,0,3,2]], dtype=torch.long)
et = torch.tensor([0,0,1,1], dtype=torch.long)
data = Data(x=x, edge_index=ei, edge_type=et)

supernet.eval()
with torch.no_grad():
    emb = supernet.forward_single(data)
print(f'Single graph: {emb.shape}')

# Sequence
graphs = [Data(x=torch.randn(10,21), edge_index=ei, edge_type=et) for _ in range(5)]
with torch.no_grad():
    emb = supernet(graphs)
print(f'Sequence (5 steps): {emb.shape}')
print(f'Architecture: {supernet.get_architecture()}')
"
```

### 3. Run DARTS Search (Quick Demo)
```python
python -c "
import torch
from torch_geometric.data import Data
from src.nas.darts import DARTSSearcher
from src.utils.seed import set_seed

set_seed(42)
n_stocks, n_features = 10, 21

# Fake graphs
def make_graphs(seq_len=3):
    graphs = []
    for _ in range(seq_len):
        x = torch.randn(n_stocks, n_features)
        ei = torch.tensor([[0,1,2,3],[1,0,3,2]], dtype=torch.long)
        et = torch.tensor([0,0,1,1], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=ei, edge_type=et))
    return graphs

searcher = DARTSSearcher(n_features=n_features, hidden_dim=32,
                          output_dim=32, num_layers=2, device='cpu')
train_g = make_graphs()
val_g = make_graphs()
target = torch.randn(n_stocks, 32)

print('Running DARTS search (20 epochs)...')
result = searcher.search(train_g, val_g, target, target, epochs=20)

print(f'Best val loss: {result.best_val_loss:.4f}')
print(f'Final entropy: {searcher.supernet.get_alpha_entropy():.3f}')
print()

archs = searcher.extract_top_k(k=3)
print('=== Top-3 Architectures ===')
for a in archs:
    print(f'  {a.name}: {a.ops} — {a.description}')
"
```

### 4. RL Policy Grid Search
```python
python -c "
import numpy as np
from src.rl.environment import PortfolioEnv
from src.nas.darts import rl_policy_grid_search, PolicyConfig

np.random.seed(42)
n = 5
features = np.random.randn(n, 200, 21).astype(np.float32)
prices = 100 * np.cumprod(1 + np.random.randn(n, 200) * 0.01, axis=1).astype(np.float32)

def env_fn():
    return PortfolioEnv(features, prices, episode_length=30)

candidates = [
    PolicyConfig(net_arch=[32, 16], name='tiny'),
    PolicyConfig(net_arch=[64, 32], name='small'),
    PolicyConfig(net_arch=[128, 64], name='medium'),
]

print('Grid search over RL policy architectures...')
results = rl_policy_grid_search(env_fn, candidates, train_steps=512, eval_episodes=2)
print()
print('=== Results (ranked by Sharpe) ===')
for r in results:
    print(f'  {r.name:10s} {str(r.net_arch):15s} → Sharpe={r.val_sharpe:.3f}')
"
```

### 5. Generate NAS Report PDF
```python
python -c "
import torch
from torch_geometric.data import Data
from src.nas.darts import DARTSSearcher, generate_nas_report
from src.utils.seed import set_seed

set_seed(42)
searcher = DARTSSearcher(n_features=21, hidden_dim=32, output_dim=32,
                          num_layers=2, device='cpu')

def make_graphs():
    graphs = []
    for _ in range(3):
        x = torch.randn(10, 21)
        ei = torch.tensor([[0,1,2,3],[1,0,3,2]], dtype=torch.long)
        et = torch.tensor([0,0,1,1], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=ei, edge_type=et))
    return graphs

target = torch.randn(10, 32)
result = searcher.search(make_graphs(), make_graphs(), target, target, epochs=20)
searcher.extract_top_k(k=3)

path = generate_nas_report(result, output_path='models/nas_report.pdf')
print(f'Report saved: {path}')
import os
print(f'Size: {os.path.getsize(path):,} bytes')
print('Open models/nas_report.pdf to see convergence plots + architecture heatmap!')
"
```

---

## Phase 12: Quantum ML (QAOA) — Manual Testing

### 1. Build QUBO Matrix
```python
python -c "
import numpy as np
from src.quantum.qaoa import build_qubo, evaluate_cost

mu = np.array([0.001, 0.002, 0.003, 0.004])
sigma = np.eye(4) * 0.0001

Q = build_qubo(mu, sigma, risk_aversion=0.5, k_assets=2)
print('=== QUBO Matrix ===')
print(f'Shape: {Q.shape}')
print(f'Matrix:\n{Q.round(6)}')
print()

# Test some bitstrings
for bs in ['1100', '0011', '1010', '0110']:
    cost = evaluate_cost(bs, Q)
    selected = [i for i, b in enumerate(bs) if b == '1']
    print(f'  {bs} (stocks {selected}): cost = {cost:.6f}')
"
```

### 2. Run QAOA Optimization
```python
python -c "
import numpy as np
from src.quantum.qaoa import run_qaoa
from src.utils.seed import set_seed

set_seed(42)
mu = np.array([0.001, 0.002, 0.003, 0.004])
sigma = np.eye(4) * 0.0001

result = run_qaoa(mu, sigma, k_assets=2, n_layers=2, shots=512, seed=42)

print('=== QAOA Result ===')
print(f'Best bitstring: {result.best_bitstring}')
print(f'Selected assets: {result.selected_assets}')
print(f'Best cost: {result.best_cost:.6f}')
print(f'Converged: {result.optimization_converged}')
print(f'Function evals: {result.n_function_evals}')
print(f'Top 5 counts:')
sorted_counts = sorted(result.all_counts.items(), key=lambda x: -x[1])[:5]
for bs, count in sorted_counts:
    print(f'  {bs}: {count} times')
"
```

### 3. Quantum vs Classical Comparison
```python
python -c "
import numpy as np
from src.quantum.portfolio import quantum_portfolio_optimize
from src.utils.seed import set_seed

set_seed(42)
# Synthetic returns: 8 assets, 500 days
rng = np.random.RandomState(42)
returns = rng.randn(500, 8) * 0.01 + 0.0005

result = quantum_portfolio_optimize(
    returns, n_assets=8, k_select=4,
    qaoa_layers=2, shots=512, seed=42)

print('=== Quantum vs Classical ===')
print(f'Quantum:  assets={result.quantum_assets}, Sharpe={result.quantum_sharpe:.3f}')
print(f'          weights={result.quantum_weights.round(3)}')
print(f'          return={result.quantum_return:.2%}, risk={result.quantum_risk:.2%}')
print()
print(f'Classical: assets={result.classical_assets}, Sharpe={result.classical_sharpe:.3f}')
print(f'           weights={result.classical_weights.round(3)}')
print(f'           return={result.classical_return:.2%}, risk={result.classical_risk:.2%}')
print()
ratio = result.quantum_sharpe / max(result.classical_sharpe, 0.001) * 100
print(f'QAOA achieves {ratio:.0f}% of classical optimal Sharpe')
"
```

### 4. Scaling Benchmark
```python
python -c "
import numpy as np
from src.quantum.portfolio import run_scaling_benchmark
from src.utils.seed import set_seed

set_seed(42)
returns = np.random.RandomState(42).randn(500, 8) * 0.01 + 0.0005

results = run_scaling_benchmark(
    returns, benchmark_sizes=[4, 6, 8],
    qaoa_layers=1, shots=256, seed=42)

print('=== Scaling Benchmark ===')
print(f'{\"N\":>4s} {\"K\":>3s} {\"QAOA Sharpe\":>12s} {\"Classical\":>10s} {\"QAOA Time\":>10s} {\"Class Time\":>10s}')
for r in results:
    print(f'{r.n_assets:4d} {r.k_select:3d} {r.qaoa_sharpe:12.3f} {r.classical_sharpe:10.3f} {r.qaoa_time_sec:10.1f}s {r.classical_time_sec:10.3f}s')
"
```

---

## Phase 11: Federated Learning — Manual Testing

### 1. Create FL Clients (Sector-Based)
```python
python -c "
from src.federated.client import (
    create_fl_clients, get_client_sectors, get_client_tickers, CLIENT_SECTORS
)
import numpy as np
import torch.nn as nn

# Show sector mapping
print('=== 4 FL Clients (Sector-Based) ===')
for cid in range(4):
    sectors = get_client_sectors(cid)
    tickers = get_client_tickers(cid)
    print(f'  Client {cid}: {\" + \".join(sectors)} ({len(tickers)} stocks)')
    print(f'    Tickers: {tickers[:5]}...')
print()

# Create simple model + clients
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
    def forward(self, x): return self.net(x)

model = SimpleModel()
client_data = {}
for cid in range(4):
    np.random.seed(cid)
    X = np.random.randn(80 + cid * 20, 10).astype(np.float32)
    y = np.random.randn(80 + cid * 20, 5).astype(np.float32)
    client_data[cid] = (X, y)

clients = create_fl_clients(model, client_data, lr=0.01)
print(f'Created {len(clients)} clients')
for c in clients:
    print(f'  Client {c.client_id}: {c.data_size} samples')
"
```

### 2. Run Federated Learning (FedAvg)
```python
python -c "
import numpy as np
import torch.nn as nn
from src.federated.client import create_fl_clients
from src.federated.server import FLServer
from src.utils.seed import set_seed

set_seed(42)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
    def forward(self, x): return self.net(x)

model = SimpleModel()
client_data = {i: (np.random.randn(100, 10).astype(np.float32),
                    np.random.randn(100, 5).astype(np.float32)) for i in range(4)}

clients = create_fl_clients(model, client_data, lr=0.01)
server = FLServer(model, strategy='FedAvg')

print('Running FL (20 rounds, FedAvg)...')
result = server.run_fl(clients, n_rounds=20, local_epochs=3)

print(f'=== FL Result ===')
print(f'Strategy: {result.strategy}')
print(f'Clients: {result.num_clients}')
print(f'Rounds: {len(result.rounds)}')
print(f'First loss: {result.rounds[0].avg_train_loss:.4f}')
print(f'Final loss: {result.final_global_loss:.4f}')
print(f'Converged: {result.final_global_loss < result.rounds[0].avg_train_loss}')
"
```

### 3. FedAvg vs FedProx Comparison
```python
python -c "
import numpy as np
import torch.nn as nn
from src.federated.client import create_fl_clients
from src.federated.server import FLServer
from src.utils.seed import set_seed

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
    def forward(self, x): return self.net(x)

client_data = {i: (np.random.randn(80, 10).astype(np.float32),
                    np.random.randn(80, 5).astype(np.float32)) for i in range(4)}

# FedAvg
set_seed(42)
model = SimpleModel()
clients = create_fl_clients(model, client_data, lr=0.01)
server = FLServer(model, strategy='FedAvg')
result_avg = server.run_fl(clients, n_rounds=15, local_epochs=3)

# FedProx
set_seed(42)
model = SimpleModel()
clients = create_fl_clients(model, client_data, lr=0.01)
server = FLServer(model, strategy='FedProx', fedprox_mu=0.01)
result_prox = server.run_fl(clients, n_rounds=15, local_epochs=3)

print(f'=== FedAvg vs FedProx ===')
print(f'FedAvg  final loss: {result_avg.final_global_loss:.4f}')
print(f'FedProx final loss: {result_prox.final_global_loss:.4f}')
winner = 'FedProx' if result_prox.final_global_loss < result_avg.final_global_loss else 'FedAvg'
print(f'Winner: {winner}')
"
```

### 4. Differential Privacy (DP-SGD)
```python
python -c "
import torch
import torch.nn as nn
from src.federated.privacy import DPTrainer, compute_noise_multiplier

# Create DP trainer
dp = DPTrainer(epsilon=8.0, delta=1e-5, max_grad_norm=1.0,
               n_rounds=50, n_samples=1000)

print(f'=== DP-SGD Configuration ===')
print(f'Epsilon (privacy budget): {dp.epsilon}')
print(f'Delta: {dp.delta}')
print(f'Max gradient norm: {dp.max_grad_norm}')
print(f'Noise multiplier: {dp.noise_multiplier:.6f}')
print()

# Train with DP
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
X = torch.randn(50, 10)
y = torch.randn(50, 5)

for step in range(5):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(X), y)
    loss.backward()
    grad_norm = dp.clip_and_noise(model)
    optimizer.step()
    status = dp.get_budget_status()
    print(f'Step {step+1}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}, '
          f'eps_spent={status[\"epsilon_spent\"]:.4f}, budget_left={status[\"budget_remaining_pct\"]:.1f}%')
"
```
**Expected:** Loss decreases, budget gradually consumed, noise doesn't destroy convergence.

### 5. Privacy Budget Exhaustion Demo
```python
python -c "
from src.federated.privacy import DPTrainer

dp = DPTrainer(epsilon=1.0, delta=1e-5, max_grad_norm=1.0,
               n_rounds=10, n_samples=100)

print(f'Small budget: epsilon={dp.epsilon}')
print(f'Exhausted: {dp.is_budget_exhausted()}')
print(f'Status: {dp.get_budget_status()}')
print()
print('After many rounds of noise, budget gets consumed.')
print('In production: stop training when budget exhausted!')
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
python -m pytest tests/test_tgat.py -v     # Phase 5 only
python -m pytest tests/test_env.py -v      # Phase 6 only
python -m pytest tests/test_agent.py -v    # Phase 7 only
python -m pytest tests/test_gan.py -v      # Phase 8-9 only
python -m pytest tests/test_nas.py -v     # Phase 10 only
python -m pytest tests/test_fl.py -v      # Phase 11 only
python -m pytest tests/test_quantum.py -v # Phase 12 only
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
│   ├── models/
│   │   └── tgat.py          # T-GAT: multi-relational GAT + GRU
│   ├── rl/
│   │   ├── environment.py   # Gymnasium portfolio env
│   │   └── agent.py         # PPO + SAC agents (SB3)
│   ├── gan/
│   │   ├── timegan.py       # TimeGAN: 5 components, 3-phase training
│   │   └── stress.py        # VaR, CVaR, Monte Carlo, crash scenarios
│   ├── nas/
│   │   ├── search_space.py  # 5 ops, MixedOp, SearchSpace config
│   │   └── darts.py         # TGATSupernet, DARTSSearcher, RL grid search
│   ├── federated/
│   │   ├── server.py       # FLServer: FedAvg + FedProx aggregation
│   │   ├── client.py       # 4 sector-wise clients, local training
│   │   └── privacy.py      # DP-SGD: clipping + noise + budget
│   ├── quantum/
│   │   ├── qaoa.py         # QAOA: QUBO, Ising, circuit, COBYLA
│   │   └── portfolio.py    # Portfolio encoding, Markowitz, scaling
│   ├── quantum/             # Phase 12: Quantum ML
│   └── api/                 # Phase 13: FastAPI
├── tests/
│   ├── test_phase0.py       # 18 tests
│   ├── test_data.py         # 12 tests
│   ├── test_features.py     # 18 tests
│   ├── test_sentiment.py    # 19 tests
│   ├── test_graph.py        # 20 tests
│   ├── test_tgat.py         # 19 tests
│   ├── test_env.py          # 23 tests
│   ├── test_agent.py        # 16 tests
│   ├── test_gan.py          # 25 tests (TimeGAN + Stress)
│   ├── test_nas.py          # 18 tests (DARTS + RL grid search)
│   ├── test_fl.py           # 17 tests (FL + DP + edge cases)
│   └── test_quantum.py      # 12 tests (QAOA + portfolio + edge cases)
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
