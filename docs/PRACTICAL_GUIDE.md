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
axes[0].fill_between(recent.index, recent['bb_lower'], recent['bb_upper'], alpha=0.1)
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
_(Will be added after Phase 3 is built)_

**Preview of what you'll be able to test:**
```python
# Sentiment score for a financial headline
from src.sentiment.finbert import get_sentiment
score = get_sentiment("Reliance Q3 profit beats estimates, revenue up 15%")
print(score)  # Expected: ~0.7 (positive)

score = get_sentiment("HDFC Bank reports higher NPAs, stock falls 3%")
print(score)  # Expected: ~-0.6 (negative)
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
│   ├── sentiment/           # Phase 3: FinBERT
│   ├── graph/               # Phase 4: Graph construction
│   ├── rl/                  # Phase 6-7: RL environment + agents
│   ├── gan/                 # Phase 8-9: TimeGAN
│   ├── nas/                 # Phase 10: DARTS
│   ├── federated/           # Phase 11: FL
│   ├── quantum/             # Phase 12: Quantum ML
│   └── api/                 # Phase 13: FastAPI
├── tests/
│   ├── test_phase0.py       # 18 tests
│   ├── test_data.py         # 12 tests
│   └── test_features.py     # 18 tests
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
