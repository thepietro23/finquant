# FINQUANT-NEXUS — Phase 0 & Phase 1 Explanation (Hinglish)

> Yeh document har phase ke baad update hoga. Har cheez ka reasoning, kya banaya, kyu banaya, kaise kaam karta hai — sab detail mein.

---

## PHASE 0: Project Scaffolding (Global Setup)

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `configs/base.yaml` | Sab hyperparameters ek jagah | Code mein koi hardcoded number nahi hoga. Experiment change karna ho toh sirf YAML edit karo. Reproducibility ka base. |
| `src/utils/config.py` | YAML config loader | Kisi bhi module mein `get_config('rl')` likhke RL ki settings mil jayengi. Caching bhi hai — ek baar read, baar baar use. |
| `src/utils/seed.py` | Random seed setter | `set_seed(42)` se Python random, NumPy, PyTorch sab same numbers generate karenge. **Kyu?** Agar seed fix nahi kiya toh har run ka result alag aayega — debugging impossible, thesis mein reproducibility claim nahi kar sakte. |
| `src/utils/logger.py` | Logging module | `print()` se debugging bahut mushkil hai production mein. Logger file mein bhi likhta hai, console pe bhi dikhata hai. Timestamp + module name + level (INFO/ERROR). Baad mein bugs trace karna easy. |
| `src/utils/metrics.py` | Financial performance metrics | Yeh 7 metrics poore project mein baar baar use honge — RL reward, backtesting, baselines comparison, thesis results. |
| `.gitignore` | Git ko batata hai kya ignore karna hai | `data/`, `models/`, `venv/`, `.env` — yeh sab git mein nahi jaane chahiye. Data heavy hai, models heavy hain, .env mein secrets hain. |
| `requirements.txt` | Sab Python dependencies ki list | Naya system pe `pip install -r requirements.txt` se sab install ho jayega. Docker mein bhi yahi use hoga. |
| `__init__.py` (12 files) | Python package markers | Bina `__init__.py` Python ko pata nahi chalta ki yeh folder ek package hai. `from src.utils.config import ...` kaam karne ke liye zaruri. |

### Metrics Explained (Detail)

| Metric | Formula Simplified | Kya Batata Hai | Humara Target |
|--------|-------------------|----------------|---------------|
| **Sharpe Ratio** | `(Return - 7%) / Volatility * sqrt(248)` | Risk ke against kitna extra kamaya. 7% = India FD rate (risk-free). Zyada Sharpe = better risk-adjusted return. | > 1.5 |
| **Max Drawdown** | Peak se sabse bada fall (%) | Portfolio ki worst case drop. -25% matlab peak 10L se 7.5L gira. Investors isse dekhte hain. | > -15% |
| **Sortino Ratio** | Sharpe jaisa par sirf downside risk count | Sharpe mein upside volatility bhi penalty deti hai (which is good!). Sortino sirf bad volatility penalize karta hai — fairer measure. | > 2.0 |
| **Calmar Ratio** | Annual Return / Max Drawdown | Long-term sustainability. High return with low drawdown = high Calmar. | > 1.0 |
| **Annualized Return** | Daily returns ko yearly compound | Kitna % per year kamaya. | > 15% |
| **Annualized Volatility** | Daily std * sqrt(248) | Kitna risk liya per year. | < 20% |
| **Portfolio Turnover** | Daily weight changes ka average | Kitna trading kiya. Zyada turnover = zyada transaction cost. Low turnover preferred. | < 0.1 |

### Config (base.yaml) Kyu Important?

```
Sooch tera RL agent ka learning rate 0.0003 hai.
Agar yeh code mein hardcoded hai: lr=0.0003
  - Tujhe 10 files mein dhundhna padega kahan kahan 0.0003 likha hai
  - Ek jagah change kiya, dusri jagah bhool gaya = bug
  - Kaunsa experiment kaunsi setting se tha — yaad nahi rehta

Agar YAML mein hai: rl.lr = 0.0003
  - Ek jagah change, poore project mein apply
  - YAML file git mein hai — history dekhke pata chal jayega kab kya change kiya
  - W&B mein config log kar sakte — har experiment ki setting permanently saved
```

### Tests (18 tests)

| Category | Tests | Kya Check Kiya |
|----------|-------|----------------|
| Config | 3 | YAML load hota hai, sections exist karti hain, values correct hain |
| Seed | 2 | Same seed se same random numbers (PyTorch + NumPy) |
| Logger | 3 | Logger create hota hai, file mein likhta hai, singleton pattern |
| Metrics | 5 | Sharpe, MaxDD, Sortino, Calmar, Turnover — sab correct calculate karte hain |
| Structure | 2 | Sab directories exist karti hain, config file exists |

---

## PHASE 1: Data Pipeline

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/data/stocks.py` | NIFTY 50 stock list + sectors + supply chain | Poore project mein yeh central registry hai — kaunse stocks, kaunsa sector, kaunse supply chain connections. GNN ke edges yahan se banenge. |
| `src/data/download.py` | Yahoo Finance se data download | yfinance API use karke 2015-2025 ka OHLCV data. Per stock CSV + combined Adj Close. Retry mechanism agar internet fail ho. |
| `src/data/quality.py` | Data quality checker + cleaner | Download hone ke baad verify karo — NaN hai? Duplicates hain? Prices negative hain? Stock split handle hua? Sab automated check. |

### stocks.py — Detail Reasoning

**NIFTY50 Dict:**
```python
NIFTY50 = {
    'IT': ['TCS.NS', 'INFY.NS', ...],
    'Banking': ['HDFCBANK.NS', ...],
    ...
}
```
- **Sector mapping kyu?** GNN ke liye chahiye. Same sector ke stocks ko "sector edges" se connect karenge. HDFCBANK aur ICICIBANK dono Banking mein hain = GNN mein connected = information flow.
- **`.NS` suffix kyu?** yfinance NSE stocks ko `.NS` se identify karta hai. BSE ke liye `.BO` hota.

**Supply Chain Edges:**
```python
SUPPLY_CHAIN_EDGES = [
    ('TATASTEEL.NS', 'MARUTI.NS'),  # Steel -> Cars
    ('RELIANCE.NS', 'ONGC.NS'),     # Energy value chain
    ...
]
```
- **Kyu?** Real duniya mein stocks isolated nahi hain. Steel ka price badha toh Maruti ka cost badha = profit gira. Yeh relationship GNN mein capture karni hai.
- **27 edges manually defined** — industry knowledge based. Literature mein bhi yahi approach use hota hai.

**Utility Functions:**
- `get_all_tickers()` — flat list of 47 stocks
- `get_sector(ticker)` — stock ka sector batao
- `get_sector_pairs()` — same sector ke sab pairs (GNN sector edges)
- `get_supply_chain_pairs()` — business relationship edges
- `get_ticker_to_index()` — ticker -> integer index (GNN ke adjacency matrix ke liye consistent ordering)

### download.py — Detail Reasoning

**Retry with Exponential Backoff:**
```python
def download_stock(ticker, retries=3, backoff=2.0):
    for attempt in range(retries):
        try:
            df = yf.download(...)
        except:
            time.sleep(backoff ** attempt)  # 1s, 2s, 4s
```
- **Kyu retry?** yfinance free API hai — kabhi kabhi timeout/rate limit. Ek attempt fail hone pe give up karna galat hai.
- **Kyu exponential?** Pehli retry 1 sec, dusri 2 sec, teesri 4 sec. Server ko time do recover hone ka. Constant retry se server aur overload hoga.

**Adj Close vs Close:**
```
RELIANCE 2020 mein 1:1 stock split hua.
Close price: ...2100, 2100, 1050, 1060... (sudden 50% drop — FAKE!)
Adj Close: ...2100, 2100, 2110, 2120... (smooth — REAL value)

Adj Close automatically splits, dividends, bonus shares adjust karta hai.
Hum HAMESHA Adj Close use karenge returns calculate karne ke liye.
```

**MultiIndex Handling:**
```python
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
```
- yfinance kabhi kabhi multi-level columns return karta hai (ticker as second level). Yeh flatten karta hai.

### quality.py — Detail Reasoning

**7 Quality Checks:**

| # | Check | Kyu |
|---|-------|-----|
| 1 | Min 1000 trading days | 2015-2025 = ~2480 days. 1000 minimum = stock kam se kam 4 years se listed hai. Chhoti history pe model achha nahi seekhega. |
| 2 | Max 5% NaN | Thoda NaN theek hai (holidays, halts). Par 5% se zyada = data problem. |
| 3 | No duplicate dates | Same date pe 2 rows = bug. Quality check + clean function fix karti hai. |
| 4 | No zero/negative prices | Stock ka price 0 ya negative nahi ho sakta. Agar hai toh data corrupt hai. |
| 5 | No extreme returns >50% | Ek din mein 50% return unrealistic hai (India mein circuit limit 20% hai). Agar data mein hai toh data error. Exception: stock split handle na hua ho. |
| 6 | Volume zero <1% | Volume 0 matlab koi trade nahi hua. Thoda OK (halt days), par zyada = illiquid stock. |
| 7 | Chronological order | Dates ascending order mein honi chahiye. Out of order = data corrupt. |

**Clean Function:**
```python
def clean_stock(self, df):
    df = df.sort_index()           # Date order fix
    df = df[~df.index.duplicated()] # Duplicate dates remove
    df = df.ffill()                # Forward fill NaN
    df = df.dropna()               # Remaining NaN drop
```
- **Sort kyu?** Kabhi kabhi data out of order aata hai.
- **ffill kyu?** NSE holiday pe data nahi hoga. Forward fill = last known price carry forward. Yeh standard practice hai finance mein.
- **dropna kyu?** Series ke start mein ffill kaam nahi karega (koi previous value nahi). Woh rows drop.

### Edge Cases Handled

| Case | Real Scenario | Handling |
|------|--------------|----------|
| Stock split | RELIANCE 2020: 1:1 split | Adj Close use karo — smooth prices |
| New listing | ADANIENT 2015 se listed nahi tha | min_days check se flag. Skip with warning, don't crash. |
| NSE holidays | Diwali, Holi, Republic Day | Forward fill — last known price carry |
| API failure | Internet down, yfinance rate limit | 3 retries with exponential backoff (1s, 2s, 4s) |
| Empty response | yfinance returns empty DataFrame | Log error, skip, continue with other stocks |

### Tests (8 tests)

| ID | Test | Kya Check |
|----|------|-----------|
| T1.1 | Stock list count | 45+ tickers in registry |
| T1.2 | CSV columns | OHLCV columns exist in downloaded data |
| T1.3 | NIFTY index | Index downloaded with 1000+ rows |
| T1.4 | Combined prices | all_close_prices.csv has 40+ columns |
| T1.5 | Quality check | RELIANCE passes all 7 checks |
| T1.6 | Clean removes NaN | After clean_stock(), zero NaN |
| T1.7 | No duplicates | After clean_stock(), zero duplicate dates |
| T1.8 | Date range | Data covers 2015 to 2025 |

### File Flow (Pipeline)

```
stocks.py (NIFTY 50 list)
    |
    v
download.py (yfinance se download)
    |
    v
data/*.csv (per stock CSV files)
    |
    v
quality.py (check + clean)
    |
    v
Clean CSVs ready for Phase 2 (Feature Engineering)
```

---

---

## PHASE 2: Feature Engineering

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/data/features.py` | 21 technical indicators + rolling z-score normalization + 3D tensor builder | Raw OHLCV se model nahi seekh sakta. Indicators = derived signals jo market ka "pulse" capture karte hain. Normalization = sab features same scale pe. |
| `tests/test_features.py` | 18 tests (14 unit + 4 edge cases) | Har indicator correct hai? NaN toh nahi? Look-ahead bias toh nahi? Zero division toh nahi? |

### 21 Features — Kya Hain Aur Kyu?

**Sooch aise:** Doctor patient ka checkup karta hai — BP, sugar, heart rate, oxygen level. Koi ek number se poori health nahi pata chalti. Same with stocks — sirf Close price se kuch nahi pata. Multiple "vital signs" chahiye.

#### Category 1: Momentum Indicators (4 features)

| Feature | Kya Karta Hai | Trading Signal |
|---------|--------------|----------------|
| **RSI** (Relative Strength Index) | Last 14 din mein kitna upar gaya vs neeche. Range: 0-100. | >70 = overbought (sell signal), <30 = oversold (buy signal) |
| **MACD** | Short-term trend vs long-term trend ka difference | MACD > Signal line = bullish, MACD < Signal = bearish |
| **MACD Signal** | MACD ki 9-day EMA (smoothed version) | Crossover points are trading signals |
| **MACD Histogram** | MACD minus Signal = momentum strength | Positive = bullish momentum, negative = bearish |

```
RSI ka formula samajh:
  1. Last 14 din ke gains alag karo, losses alag karo
  2. Average gain / Average loss = RS (Relative Strength)
  3. RSI = 100 - (100 / (1 + RS))

Agar 14 mein se 12 din upar gaya → RS bahut bada → RSI ~85-90 → OVERBOUGHT
Agar 14 mein se 12 din neeche gaya → RS bahut chhota → RSI ~15-20 → OVERSOLD
```

#### Category 2: Bollinger Bands (3 features)

| Feature | Formula | Kya Batata Hai |
|---------|---------|----------------|
| **BB Upper** | SMA(20) + 2 * StdDev(20) | Upar ki limit — price yahan se zyada jaaye toh unusual |
| **BB Mid** | SMA(20) | 20-day average — "normal" price |
| **BB Lower** | SMA(20) - 2 * StdDev(20) | Neeche ki limit — price yahan se neeche jaaye toh unusual |

```
Analogy: Highway pe lane markers.
  - BB Mid = center line (average)
  - BB Upper/Lower = boundaries
  - Price boundary todke bahar jaaye = kuch unusual ho raha hai
  - Bands expand = high volatility, Bands contract = low volatility (calm before storm!)
```

#### Category 3: Moving Averages (4 features)

| Feature | Window | Kyu |
|---------|--------|-----|
| **SMA 20** | 20-day simple average | Short-term trend |
| **SMA 50** | 50-day simple average | Medium-term trend. SMA20 > SMA50 = bullish ("Golden Cross") |
| **EMA 12** | 12-day exponential avg | Recent prices ko zyada weight deta hai — faster reaction |
| **EMA 26** | 26-day exponential avg | Slower reaction — trend confirmation |

```
SMA vs EMA:
  SMA = (P1 + P2 + ... + P20) / 20  → Equal weight to all days
  EMA = Recent days ko zyada weight → Faster reaction to new info

  Why both? SMA stable hai par slow. EMA fast hai par noisy.
  RL agent dono dekh ke decide kare — "trend kya bol raha hai?"
```

#### Category 4: Volatility (3 features)

| Feature | Kya | Kyu Important |
|---------|-----|---------------|
| **ATR** (Average True Range) | Average daily price range (High-Low adjusted) | Kitna "swing" ho raha hai. High ATR = volatile, risky. Low ATR = calm. Position sizing ke liye useful. |
| **Volatility 20d** | 20-day annualized volatility | Short-term risk measure. RL agent ko batata hai ki kitna risky hai abhi. |
| **Volatility 60d** | 60-day annualized volatility | Medium-term risk. 20d vs 60d comparison = volatility trend. |

#### Category 5: Stochastic Oscillator (2 features)

| Feature | Kya |
|---------|-----|
| **Stoch %K** | Current price, last 14 days ke range mein kahan hai (0-100) |
| **Stoch %D** | %K ki 3-day average (smoothed) |

```
Sooch: Last 14 din mein price 100 se 120 ke beech tha.
  Aaj price 118 hai.
  %K = (118 - 100) / (120 - 100) * 100 = 90%
  Matlab: Range ke top ke paas hai → overbought signal
```

#### Category 6: Volume (2 features)

| Feature | Kya | Signal |
|---------|-----|--------|
| **Volume SMA** | 20-day average volume | Normal volume level |
| **Volume Ratio** | Today's volume / SMA | >1.5 = unusual activity, <0.5 = low interest |

```
Volume kyu important?
  Price badhne ke 2 cases hain:
  1. Price UP + High Volume = Strong move (sab kharid rahe hain = genuine demand)
  2. Price UP + Low Volume = Weak move (kuch log kharid rahe, baaki wait kar rahe)

  RL agent ko yeh difference samajhna chahiye.
```

#### Category 7: Returns (3 features)

| Feature | Window | Kyu |
|---------|--------|-----|
| **Return 1d** | 1 day | Yesterday se aaj kitna change. Most granular. |
| **Return 5d** | 5 days (1 week) | Weekly momentum. |
| **Return 20d** | 20 days (1 month) | Monthly trend. Positive = uptrend last month. |

### Rolling Z-Score Normalization — Deep Dive

**Problem:** Features different scales pe hain.
```
RSI:         0 to 100
MACD:        -50 to +50 (depends on stock price)
Volume:      1,000,000 to 50,000,000
Return 1d:   -0.10 to +0.10

Neural network ko lagega Volume bahut important hai (bade numbers) aur Return chota.
Par yeh galat hai — numbers bade hone se feature important nahi hota.
```

**Solution: Z-Score**
```
z = (value - mean) / std

Example: RSI aaj 75 hai
  Last 252 days ka mean RSI: 55
  Last 252 days ka std: 10
  z = (75 - 55) / 10 = +2.0

Matlab: RSI "2 standard deviations above normal" hai = quite high = overbought
```

**Kyu ROLLING z-score (not static)?**
```
GALAT approach (data leakage):
  mean = RSI ka mean over ENTIRE 2015-2025 data ← FUTURE DATA USE HO RAHA HAI!
  Agar 2018 mein normalize kar rahe ho toh 2019-2025 ka data use nahi kar sakte.
  Yeh cheating hai — model ko future pata chal jayega.

SAHI approach (rolling):
  Sirf past 252 days (1 year) ka mean/std use karo.
  2018-Jan mein normalize karte waqt sirf 2017-Jan to 2018-Jan ka data.
  No future leakage. Honest evaluation.
```

**Clip [-5, +5] kyu?**
```
Kabhi kabhi z-score extreme hota hai: crash mein -15, rally mein +12.
Neural networks extreme values se explode kar sakte hain (gradient explosion).
Clip karke limit: "Bhai, maximum +5 ya -5. Isse zyada extreme consider nahi karenge."
```

### NaN Handling — Kya Kiya Aur Kyu

```
Rolling windows ko warm-up period chahiye:
  - 252-day rolling z-score → first 252 days NaN
  - 60-day volatility → first 60 days NaN
  - 50-day SMA → first 50 days NaN

Combined effect: ~252 days ka data drop hota hai (worst case).
  2015-Jan start → usable data ~2016-Jan se

Humne DROPNA kiya — NaN rows hata diye.
Kyu? Agar NaN chhoda toh:
  - T-GAT mein NaN propagate hoga → loss NaN → training crash
  - RL environment mein NaN observation → invalid action → crash
  - Better: clean data in, clean results out.

Trade-off: ~1 year ka data lost. But 2016-2025 = 9 years = still enough.
```

### Feature Tensor — 3D Array

```
Shape: (n_stocks, n_timesteps, n_features)
Example: (47, 2200, 21)
  - 47 stocks
  - 2200 common trading days
  - 21 features each

Kyu 3D?
  T-GAT ko chahiye: har stock ka har din ka feature vector.
  RL environment ko chahiye: observation space mein sab stocks ke features.

  tensor[0, 100, :] = Stock 0 ke Day 100 ke saare 21 features
  tensor[:, 100, 0] = Day 100 pe saare stocks ka RSI
```

### Tests: 18/18 PASSING

| ID | Test | Kya Check |
|----|------|-----------|
| T2.1 | All columns present | compute_technical_indicators ke baad 21 columns hain |
| T2.2 | Indicator count | FEATURE_COLUMNS mein 21+ features listed hain |
| T2.3 | Real data works | RELIANCE pe indicators compute without error |
| T2.4 | Z-score clipped | Sab normalized values [-5, +5] ke andar |
| T2.5 | No look-ahead | Truncated data vs full data ka z-score same at cutoff point |
| T2.6 | No NaN output | engineer_stock_features ke baad zero NaN |
| T2.7 | All features in output | Output mein sab 21 columns hain |
| T2.8 | Rows reduced | NaN warm-up rows dropped (output < input) |
| T2.9 | Real pipeline | RELIANCE full pipeline — clean to features, zero NaN, 1000+ rows |
| T2.10 | Tensor shape | 3D shape correct: (n_stocks, n_time, n_features) |
| T2.11 | Tensor no NaN | Final tensor mein zero NaN |
| T2.12 | Tensor dtype | float32 (for FP16 training compatibility) |
| E2.1 | Short history | 300-day stock still produces valid output |
| E2.2 | Zero volume | Volume=0 days don't crash volume_ratio |
| E2.3 | Constant price | Price constant hai toh z-score NaN hota hai — graceful handling |
| E2.4 | Single stock tensor | Tensor works with just 1 stock |
| T2.13 | Config match | FEATURE_COLUMNS matches base.yaml indicators list |
| T2.14 | get_feature_columns | Returns a copy, not the mutable original |

### File Flow (Updated)

```
stocks.py (NIFTY 50 list)
    |
    v
download.py (yfinance se download)
    |
    v
data/*.csv (per stock raw CSV files)
    |
    v
quality.py (check + clean)
    |
    v
features.py (21 indicators + z-score)   ← NEW
    |
    v
data/features/*.csv + all_features.pkl  ← feature output
    |
    v
Feature Tensor (47, ~2200, 21) float32  ← model input ready
    |
    v
Phase 3: FinBERT Sentiment (next)
```

---

> **Next: Phase 3 — FinBERT Sentiment. Google News RSS se headlines → ProsusAI/finBERT se sentiment scores → daily aggregation per stock.**
