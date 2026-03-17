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

---

## PHASE 3: FinBERT Sentiment

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/sentiment/finbert.py` | FinBERT model loading + sentiment scoring + aggregation + decay | Market sirf numbers se nahi chalta. "RBI rate hike" headline se banking stocks girengi — yeh info OHLCV data mein nahi hai. Sentiment feature model ko market mood batata hai. |
| `src/sentiment/news_fetcher.py` | Google News RSS se headlines fetch + SQLite cache | Free news source (no API key). Har stock ke liye "company name + stock NSE" search. Cache se duplicate computation avoid. |
| `tests/test_sentiment.py` | 19 tests covering model, prediction, aggregation, decay | FinBERT sahi classify karta hai? Batch = individual? Decay sahi kaam karta hai? Edge cases handle? |

### FinBERT — Kya Hai Aur Kaise Kaam Karta Hai

```
Normal BERT:
  Input: "The market is bearish"
  BERT samajhta hai: "bearish" = adjective, koi animal related?

FinBERT (ProsusAI/finbert):
  Input: "The market is bearish"
  FinBERT samajhta hai: "bearish" = negative market sentiment = stock prices gir sakte hain

Kyu? FinBERT financial text pe fine-tuned hai — Financial PhraseBank + TRC2 dataset.
3 labels: positive, negative, neutral
```

**Scoring Formula:**
```
score = P(positive) - P(negative)

Example 1: "Reliance Q3 profit beats estimates, revenue up 25%"
  P(positive) = 0.955, P(negative) = 0.023, P(neutral) = 0.021
  score = 0.955 - 0.023 = +0.932 (strongly positive)

Example 2: "Stock crashes 15% after massive losses reported"
  P(positive) = 0.02, P(negative) = 0.95, P(neutral) = 0.03
  score = 0.02 - 0.95 = -0.93 (strongly negative)

Example 3: "Company held annual general meeting"
  P(positive) = 0.10, P(negative) = 0.05, P(neutral) = 0.85
  score = 0.10 - 0.05 = +0.05 (almost neutral)

Range: [-1, +1]. Simple, interpretable. -1 = worst, +1 = best.
```

### News Fetcher — Google News RSS

```
Google News RSS URL:
  https://news.google.com/rss/search?q=Reliance+Industries+stock+NSE&hl=en-IN&gl=IN

Free, no API key needed.
Limitation: Only ~100 recent results. Historical archive nahi milta.
For thesis: Recent sentiment demonstrate karenge, historical ke liye decay mechanism hai.
```

**Ticker to Company Mapping:**
```python
TICKER_TO_COMPANY = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'TCS Tata Consultancy',
    'HDFCBANK.NS': 'HDFC Bank',
    ...
}
# Kyu? "RELIANCE.NS" search karne se news nahi milti.
# "Reliance Industries stock NSE" se relevant results aate hain.
```

### Sentiment Decay — Missing Days ka Solution

```
Problem:
  Monday: 3 headlines → avg sentiment = +0.7
  Tuesday: 0 headlines → sentiment kya ho?
  Wednesday: 0 headlines → ?
  Thursday: 1 headline → sentiment = -0.3

Option 1 (GALAT): Tuesday/Wednesday = 0.0 (neutral)
  Problem: Monday ko positive tha, Tuesday achanak neutral? Misleading.

Option 2 (SAHI): Decay factor = 0.95
  Monday:    +0.700 (actual)
  Tuesday:   +0.700 * 0.95 = +0.665 (decayed)
  Wednesday: +0.665 * 0.95 = +0.632 (more decay)
  Thursday:  -0.300 (new headline resets)

Intuition: Agar koi news nahi hai, toh market sentiment slowly fade hota hai.
95% decay = "yesterday ka mood aaj bhi 95% applicable hai"
After ~60 days: 0.95^60 ≈ 0.046 → almost zero. Purani news irrelevant.
```

### Sentiment Matrix — Model Input

```
Shape: (n_stocks, n_timesteps)
Example: (47, 2200)

  matrix[0, 100] = Stock 0 ka Day 100 ka sentiment score
  matrix[:, 100] = Day 100 pe saare stocks ka sentiment

Yeh feature tensor ke saath combine hoga:
  features:  (47, 2200, 21)  ← technical indicators
  sentiment: (47, 2200)       ← sentiment scores
  Combined:  (47, 2200, 22)  ← 21 indicators + 1 sentiment = 22 features per stock per day

T-GAT aur RL agent ko dono milenge.
```

### SSL/Proxy Fix — College Network Challenge

```
Problem: College/corporate network ka proxy SSL certificates intercept karta hai.
  HuggingFace se model download → proxy ne connection todha → 0-byte file saved → torch.load fail

Solution:
  1. Manually download: requests.get(url, verify=False) se 417MB model saved to data/finbert_local/
  2. Code auto-detects: data/finbert_local/config.json exists? → load local. Nahi? → try HuggingFace Hub.
  3. torch.load patch: torch 2.5.1 default weights_only=True breaks with .bin files → patched to False.

Same SSL issue yfinance mein bhi tha — curl_cffi se fix kiya (Phase 1).
```

### FP16 Memory Optimization

```
FinBERT = BERT-base = 110M parameters
  FP32: ~440 MB VRAM
  FP16: ~220 MB VRAM  ← We use this

model = model.half()  # FP32 → FP16

Kyu? RTX 3050 = 4GB VRAM. FinBERT + T-GAT + RL agent sab load karne hain.
Har jagah FP16 use karke VRAM bachao.

CPU pe testing: FP32 use hota hai (CPU pe FP16 slower hai).
```

### SQLite Cache — Avoid Re-computation

```python
# Schema:
sentiment_scores (ticker, date, headline, score, positive, negative, neutral)
daily_sentiment  (ticker, date, avg_score, num_headlines)

# Kyu?
# 50 stocks × 15 headlines = 750 FinBERT predictions
# ~2 seconds per prediction = 25 minutes
# Cache mein hai? → Instant lookup. Nahi hai? → Predict + cache.
# Next run mein same headlines skip ho jayenge.
```

### Tests: 19/19 PASSING

| ID | Test | Kya Check |
|----|------|-----------|
| T3.1 | Model loads | FinBERT CPU pe load hota hai, error nahi |
| T3.2 | 3 labels | Output 3 classes: positive, negative, neutral |
| T3.3 | Positive text | "Record profits" → positive score (>0) |
| T3.4 | Negative text | "Stock crashes, massive losses" → negative score (<0) |
| T3.5 | Neutral text | "AGM held on Monday" → near-zero score |
| T3.6 | Score range | Sab scores [-1, +1], probs sum to 1.0 |
| T3.7 | Batch count | 3 inputs → 3 outputs |
| T3.8 | Batch = individual | Batch results match one-by-one predictions |
| T3.9 | Aggregation | 2 headlines → correct avg + count |
| T3.10 | Decay fills gaps | Missing days filled with 0.95 decay |
| T3.11 | Headline resets | New headline replaces decayed value |
| T3.12 | Matrix shape | (n_stocks, n_timesteps) float32 |
| E3.1 | Empty text | "" → neutral 0.0 |
| E3.2 | Short text | "Hi" → neutral 0.0 |
| E3.3 | Decay → 0 | After 100 days without news → near-zero |
| E3.4 | Single headline | Single headline aggregation works |
| T3.13 | Company lookup | RELIANCE.NS → "Reliance Industries" |
| T3.14 | Unknown ticker | UNKNOWN.NS → "UNKNOWN" (graceful fallback) |
| T3.15 | DB init | SQLite database creates without error |

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
features.py (21 indicators + z-score)
    |
    v
Feature Tensor (47, ~2200, 21) float32
    |                                    news_fetcher.py (Google News RSS)
    |                                         |
    |                                         v
    |                                    finbert.py (sentiment scoring)
    |                                         |
    |                                         v
    |                                    Sentiment Matrix (47, ~2200) float32
    |                                         |
    +--------------------+--------------------+
                         |
                         v
              Combined Input (47, ~2200, 22)
                         |
                         v
              Phase 4: Graph Construction (next)
```

---

---

## PHASE 4: Graph Construction

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/graph/builder.py` | Multi-relational graph builder — 3 edge types, PyG Data objects, graph sequence | T-GAT ko adjacency matrix chahiye. Kaunse stocks connected hain, kaise connected hain — yeh sab graph define karta hai. |
| `tests/test_graph.py` | 20 tests covering all edge types, full graph, stats, edge cases | Har edge type correct hai? Self-loops toh nahi? Bidirectional hai? Edge cases handle? |

### Graph Kya Hai? — Simple Analogy

```
Sooch: Social media graph.
  - Nodes = users (tum, tumhare dost)
  - Edges = connections (friendship, follows, messages)

Stock market graph:
  - Nodes = 47 stocks (RELIANCE, TCS, HDFC, ...)
  - Edges = relationships between stocks

3 types of edges (relationships):
  1. SECTOR: Same sector = similar business → connected
     HDFCBANK ↔ ICICIBANK (dono Banking)
     TCS ↔ INFY (dono IT)

  2. SUPPLY CHAIN: Business dependency → connected
     TATASTEEL → MARUTI (steel supplier → car maker)
     RELIANCE → ONGC (energy value chain)

  3. CORRELATION: Price co-movement → connected
     Agar do stocks ka |correlation| > 0.6 → connected
     Yeh daily change hota hai (dynamic edge)
```

### 3 Edge Types — Detail

#### Edge Type 0: Sector Edges (Static)

```
Same sector ke stocks ek dusre se connected hain.

Banking sector: HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK, INDUSINDBK
  = C(6,2) = 15 pairs × 2 directions = 30 edges

Kyu? Same sector stocks similar factors se affect hote hain:
  - RBI rate hike → ALL banking stocks girengi
  - IT spending badha → ALL IT stocks badhenge
  GNN yeh "sector effect" capture karta hai edges ke through.

Code:
  for each (stock_a, stock_b) in same_sector:
      add edge a → b
      add edge b → a  (bidirectional)
```

#### Edge Type 1: Supply Chain Edges (Static)

```
27 manually defined business relationships:

TATASTEEL → MARUTI    (steel for cars)
RELIANCE → ONGC       (energy value chain)
HCLTECH → BHARTIARTL  (IT infra for telecom)
BAJFINANCE → MARUTI   (auto loan financing)
...

Kyu? Real world mein steel ka price badha → Maruti ka cost badha → profit gira.
Yeh "dependency effect" sector edges se capture nahi hota.
Supply chain edges explicit business relationships encode karte hain.

Code: Same as sector — both directions added.
```

#### Edge Type 2: Correlation Edges (Dynamic — changes daily!)

```
Har din compute hota hai:
  1. Last 60 trading days ka return data lo
  2. Pairwise correlation matrix compute karo (47×47)
  3. |correlation| > 0.6 → edge add karo

Example:
  Day 100: RELIANCE-TCS correlation = 0.75 → EDGE
  Day 200: RELIANCE-TCS correlation = 0.45 → NO EDGE (dropped below threshold)
  Day 300: RELIANCE-TCS correlation = 0.82 → EDGE again

Kyu dynamic? Market regimes change hoti hain:
  - COVID 2020: Panic selling → sab stocks highly correlated (correlation ~0.9)
  - Normal times: IT aur Pharma uncorrelated (0.2-0.3)
  - Sector rotation: Money moves from IT to Banking → correlations shift

Static correlation galat hogi — "average" relationship batayegi jo kisi bhi time accurate nahi.
Rolling 60-day window = current market regime capture.

Threshold 0.6 kyu?
  - Too low (0.3) = bahut zyada edges = noise, graph dense = slow + noisy
  - Too high (0.9) = bahut kam edges = information loss
  - 0.6 = moderate, financial literature standard
```

### Vectorized Correlation (Fast Version)

```python
# SLOW version (double loop):
for i in range(47):
    for j in range(i+1, 47):
        if abs(corr[i,j]) > 0.6:
            add_edge(i, j)
# 47*46/2 = 1081 iterations per day × 2200 days = 2.3 million iterations = SLOW

# FAST version (vectorized):
mask = np.triu(np.abs(corr) > threshold, k=1)  # Upper triangle, exclude diagonal
sources, targets = np.where(mask)               # All matching pairs at once
# One NumPy call = C-level speed. ~100x faster.
```

### Edge Deduplication — Tricky Part

```
Problem: TATASTEEL aur MARUTI dono Metal sector mein hain AUR supply chain mein bhi.
  → Sector edge: TATASTEEL ↔ MARUTI
  → Supply edge: TATASTEEL ↔ MARUTI
  → Duplicate! Same edge 2 baar.

Solution: _deduplicate_edges()
  Encode: edge_code = source_idx * n + target_idx (unique number per edge)
  Keep first occurrence only.
  First = sector edge → sector type retained.

Kyu important? Duplicate edges = GNN double-counting = biased message passing.
```

### PyG Data Object — Model Input

```python
Data(
    x=node_features,     # tensor (47, 21) — features for this day
    edge_index=edges,    # tensor (2, num_edges) — all connections
    edge_type=types,     # tensor (num_edges,) — 0=sector, 1=supply, 2=corr
)

Kyu PyG Data?
  PyTorch Geometric ka standard format hai.
  T-GAT, GCN, GraphSAGE — sab isse directly accept karte hain.
  x = node features, edge_index = sparse adjacency.
  Reinventing the wheel ki zarurat nahi.
```

### Graph Sequence — One Graph Per Day

```
build_graph_sequence() kya karta hai:
  1. Static edges ek baar compute karo (sector + supply chain) → reuse daily
  2. Har trading day ke liye:
     a. Correlation matrix lo (rolling 60-day window)
     b. Correlation edges compute karo
     c. Static + correlation edges combine karo
     d. Node features attach karo (21 technical indicators for that day)
     e. PyG Data object banao
  3. Result: list of ~2200 Data objects

Memory efficient: Static edges shared, sirf corr edges daily recompute.
```

### Graph Stats — Quick Summary

```python
get_graph_stats(data) → {
    'num_nodes': 47,
    'num_edges': 250,          # Total (all 3 types)
    'density': 0.12,           # edges / possible_edges
    'sector_edges': 160,       # Type 0
    'supply_chain_edges': 54,  # Type 1
    'correlation_edges': 36,   # Type 2
}

Density = edges / n*(n-1) = 250 / (47*46) = 0.12
Matlab: 12% possible connections active. Sparse graph = efficient GNN.
```

### Tests: 20/20 PASSING

| ID | Test | Kya Check |
|----|------|-----------|
| T4.1 | Sector edge count | 2 × valid_pairs = expected edges |
| T4.2 | Sector bidirectional | Every (i,j) has (j,i) |
| T4.3 | Sector no self-loops | No (i,i) edges |
| T4.4 | Supply chain exists | > 0 edges |
| T4.5 | Supply bidirectional | Every (i,j) has (j,i) |
| T4.6 | Supply no self-loops | No (i,i) edges |
| T4.7 | Correlation threshold | Only |corr| > 0.6 pairs included |
| T4.8 | Corr no self-loops | Diagonal excluded |
| T4.9 | Corr bidirectional | Symmetric edges |
| T4.10 | Static has both types | edge_type contains 0 and 1 |
| T4.11 | Type length matches | edge_type.len == edge_index.shape[1] |
| T4.12 | Full graph shape | num_nodes, x.shape correct |
| T4.13 | All 3 types present | With correlation → types 0, 1, 2 all present |
| T4.14 | NumPy auto-convert | np.ndarray → torch.Tensor automatically |
| T4.15 | Stats keys | num_nodes, density, sector_edges in output |
| T4.16 | Density range | 0 ≤ density ≤ 1 |
| E4.1 | Zero correlation | Identity matrix → 0 corr edges |
| E4.2 | Perfect correlation | All 1s → n*(n-1) edges |
| E4.3 | Single stock | 1 stock → 0 corr edges |
| E4.4 | Negative correlation | |corr| > threshold works for negative values too |

### File Flow (Updated)

```
stocks.py (NIFTY 50 list + sectors + supply chain)
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
features.py (21 indicators + z-score)
    |
    v
Feature Tensor (47, ~2200, 21) float32
    |                                    news_fetcher.py (Google News RSS)
    |                                         |
    |                                         v
    |                                    finbert.py (sentiment scoring)
    |                                         |
    |                                         v
    |                                    Sentiment Matrix (47, ~2200) float32
    |                                         |
    +--------------------+--------------------+
                         |
                         v
              Combined Input (47, ~2200, 22)
                         |
                         v
              builder.py (graph construction)          ← NEW
                |              |              |
                v              v              v
          Sector Edges   Supply Chain   Correlation
          (static)       (static)       (dynamic/daily)
                |              |              |
                +------+-------+--------------+
                       |
                       v
              PyG Data Objects (one per day)
              [x=features, edge_index, edge_type]
                       |
                       v
              Phase 5: T-GAT Model (next)
```

---

---

## PHASE 5: T-GAT (Temporal Graph Attention Network)

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/models/tgat.py` | T-GAT model — multi-relational GAT + GRU temporal encoder | Graph + features leke har stock ka "embedding" banata hai. Yeh embedding stock ki apni info + neighborhood ki info encode karta hai. RL agent ko yeh milega. |
| `tests/test_tgat.py` | 19 tests — init, forward pass, gradients, FP16, edge cases | Model sahi shape output karta hai? Gradients flow karte hain? 4GB VRAM mein fit hoga? |

### GNN + Attention — Simple Analogy

```
Sooch: Tu exam mein kisi topic pe stuck hai.
  - Option 1: Sirf apna notes padh (isolation — no graph)
  - Option 2: Sab 47 classmates se puch (fully connected — too noisy)
  - Option 3: SMART approach:
    a. Sirf relevant dost se puch (graph edges = who to ask)
    b. Kisi ka answer zyada trust kar, kisi ka kam (attention = how much to trust)
    c. Alag alag logon se alag cheezein seekh:
       - Same class ke bande (sector peers)
       - Senior jo similar subject padhta tha (supply chain)
       - Jisne same galti ki thi past mein (correlation)

T-GAT = Option 3 for stocks.
  Each stock "asks" its neighbors for information.
  Attention mechanism decides whose info matters more.
  3 edge types = 3 different kinds of relationships.
```

### Architecture — Step by Step

```
1. INPUT PROJECTION
   Raw features (21 indicators) → Hidden representation (64 dimensions)

   Kyu? 21 features ka space mein attention compute karna inefficient.
   Project to 64-dim → richer, learned representation.
   Like translating Hindi notes to a common language sab samajh sakein.

2. RELATIONAL GAT LAYER × 2
   Har stock apne neighbors se info collect karta hai.

   Key insight: 3 edge types = 3 alag GAT convolutions.

   Sector GAT:
     HDFCBANK "asks" ICICIBANK, SBIN, KOTAKBANK
     Attention: ICICIBANK ko 0.4 weight, SBIN ko 0.35, KOTAKBANK ko 0.25
     → "Banking sector ka consensus kya bol raha hai?"

   Supply Chain GAT:
     MARUTI "asks" TATASTEEL (steel supplier)
     → "Mera raw material supplier kaisa perform kar raha hai?"

   Correlation GAT:
     Stock A "asks" Stock B (corr > 0.6)
     → "Jo mere saath move karte hain, woh kya kar rahe hain?"

   Final: Weighted sum of all 3 → combined neighborhood info.
   Weights are LEARNABLE (model decides sector vs supply chain kya zyada important).

3. MULTI-HEAD ATTENTION (4 heads)
   Ek attention head se ek perspective milta hai.
   4 heads = 4 different perspectives simultaneously.

   Head 1: "Momentum similarity" dekhta hai
   Head 2: "Volatility pattern" dekhta hai
   Head 3: "Volume correlation" dekhta hai
   Head 4: "Return pattern" dekhta hai

   (Ye exact kya dekhte hain — model khud seekhta hai training mein)

4. RESIDUAL + LAYER NORM
   residual: output = gat_output + original_input
   norm: stabilize values to prevent explosion

   Kyu residual? Agar GAT layer kuch useful nahi seekha,
   toh at least original info preserve rahegi.
   Like: "Agar neighbors ki advice bekar hai, toh apna notes hi use kar."

5. GRU TEMPORAL ENCODER
   Ab tak: Single day ka graph process hua.
   Par stocks ka behavior time ke saath change hota hai!

   GRU = Gated Recurrent Unit (simpler version of LSTM)
   Input: Sequence of 5-20 daily graph embeddings
   Output: Temporal-aware embedding that captures TREND

   Example:
     Day 1: RELIANCE embedding = [bullish signals from banking neighbors]
     Day 2: RELIANCE embedding = [mixed signals]
     Day 3: RELIANCE embedding = [bearish signals]
     GRU output: "Trend is shifting from bullish to bearish over last 3 days"

   Kyu GRU instead of LSTM?
     GRU: 2 gates (reset + update) — fewer parameters
     LSTM: 3 gates (forget + input + output) — more parameters
     Similar performance, but GRU = less VRAM on our 4GB GPU.

6. OUTPUT PROJECTION
   Hidden (64) → Output embedding (64)

   Final per-stock embedding:
   - Encodes its own features (21 indicators)
   - Encodes neighbor information (via attention)
   - Encodes temporal trend (via GRU)

   Yeh 64-dim vector RL agent ko input jayega as "stock state".
```

### Attention Weights — Interpretability

```python
# T-GAT ka bonus: attention weights extract kar sakte hain!
alpha = model.get_attention_weights(graph, layer_idx=0, relation_idx=0)

# alpha[i] = HDFCBANK ne ICICIBANK ko kitna attention diya
# High alpha = "Yeh neighbor zyada important hai mere liye"

# Thesis mein: attention heatmap bana sakte hain
# "RELIANCE sector attention: IT peers ko 60%, Banking peers ko 40%"
# → Interpretable GNN = examiner ko samjha sakte
```

### Model Size — 4GB VRAM Constraint

```
Parameters: 56,518 (~57K)
FP32 size:  0.22 MB
FP16 size:  0.11 MB (with autocast)

For comparison:
  FinBERT: 110M parameters, ~220 MB (FP16)
  Our T-GAT: 57K parameters, 0.11 MB

T-GAT bahut lightweight hai because:
  - 47 stocks sirf (not millions of nodes like social graphs)
  - 2 GAT layers sufficient (shallow enough for small graph)
  - 64 hidden dim (not 256 or 512)

Budget on 4GB VRAM:
  T-GAT:    ~0.1 MB
  FinBERT:  ~220 MB  (inference only, can unload)
  RL Agent: ~50 MB
  Buffers:  ~500 MB
  Free:     ~3.2 GB → comfortable!
```

### Mixed Precision — Why autocast not .half()

```
Problem: model.half() converts EVERYTHING to FP16.
  But LayerNorm NEEDS FP32 for numerical stability.
  FP16 LayerNorm → RuntimeError on both CPU and CUDA.

Solution: torch.cuda.amp.autocast()
  Smart mixed precision:
  - MatMul, Conv → FP16 (fast, doesn't need precision)
  - LayerNorm, Softmax → FP32 (needs precision)
  - Automatic! PyTorch handles it.

In training:
  scaler = GradScaler()
  with autocast():
      loss = model(...)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
```

### Tests: 19/19 PASSING

| ID | Test | Kya Check |
|----|------|-----------|
| T5.1 | Model creates | TGAT() no error |
| T5.2 | Config from YAML | hidden=64, heads=4, layers=2 from base.yaml |
| T5.3 | Custom config | Manual overrides work |
| T5.4 | Components exist | input_proj, gat_layers, gru, output_proj |
| T5.5 | Sequence shape | (10, 64) output from 5-step sequence of 10 stocks |
| T5.6 | Single shape | (10, 64) from single graph |
| T5.7 | Finite embeddings | No NaN/Inf in sequence output |
| T5.8 | Finite single | No NaN/Inf in single output |
| T5.9 | Gradients flow | Backprop works, parameters get gradients |
| T5.10 | Loss decreases | One optim step reduces MSE loss |
| T5.11 | Missing edge type | Only sector edges → no crash |
| T5.12 | All 3 types | All edge types process correctly |
| T5.13 | Param count | < 1M parameters |
| T5.14 | Model size | < 10 MB |
| T5.15 | FP16 CUDA | autocast works on GPU |
| E5.1 | No edges | 0 edges → isolated node embeddings still work |
| E5.2 | Single node | 1 stock → (1, 64) output |
| E5.3 | Long sequence | 20 timesteps → no memory explosion |
| E5.4 | Empty sequence | [] → raises ValueError |

### File Flow (Updated)

```
stocks.py (NIFTY 50 list + sectors + supply chain)
    |
    v
download.py → quality.py → features.py
    |
    v
Feature Tensor (47, ~2200, 21)     Sentiment Matrix (47, ~2200)
    |                                    |
    +------------------------------------+
    |
    v
builder.py (graph construction)
    |
    v
PyG Data Objects (per day)
[x=features, edge_index, edge_type]
    |
    v
tgat.py (T-GAT model)              ← NEW
    |
    v
Stock Embeddings (47, 64)
[each stock: 64-dim vector encoding
 own features + neighbor info + temporal trend]
    |
    v
Phase 6: RL Environment (next)
[embeddings → observation space → agent decides portfolio weights]
```

---

---

## PHASE 6: RL Environment (Portfolio Gym)

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/rl/environment.py` | Gymnasium-compatible portfolio management environment | RL agent ko ek "game" chahiye jismein woh practice kare. Environment = stock market simulator. Agent har din portfolio rebalance karta hai, returns earn karta hai, costs pay karta hai. |
| `tests/test_env.py` | 23 tests covering init, step, constraints, edge cases | Observation sahi shape hai? Costs lag rahe hain? Stop loss kaam karta hai? Drawdown circuit breaker terminate karta hai? |

### RL Environment — Simple Analogy

```
Sooch: Tu ek ice cream shop ka manager hai.
  - OBSERVATION: Kitni ice cream hai (stock), kitne customers aa rahe hain (features),
                 weather kaisa hai (sentiment), kitna paisa hai (portfolio value)
  - ACTION: Kaunsi flavor kitni banana hai (portfolio weights)
  - REWARD: Profit - waste cost (returns - transaction costs)
  - CONSTRAINTS: Budget limit (max position), spoilage (stop loss)

Stock market version:
  - OBSERVATION: 47 stocks ke 21 features + portfolio state + sentiment
  - ACTION: Kitna invest karna hai har stock mein (0-20% each)
  - REWARD: Risk-adjusted return (Sharpe ratio based)
  - CONSTRAINTS: Max 20%/stock, -5% stop loss, -15% max drawdown
```

### Observation Space — Agent Kya Dekhta Hai

```
Observation = [stock_features | portfolio_state | embeddings | sentiment]

1. Stock Features (n_stocks × 21 = 987 values):
   Har stock ke 21 technical indicators (Phase 2 se)
   RSI, MACD, Bollinger, SMA, volatility, returns...

2. Portfolio State (n_stocks + 2 values):
   - Current weights: [0.15, 0.10, 0, 0.05, ...] (kitna invest hai)
   - Cash ratio: 0.70 (70% cash mein hai)
   - Normalized value: 1.05 (5% profit hua hai start se)

3. T-GAT Embeddings (optional, n_stocks × 64):
   Graph attention se aaye stock embeddings
   "HDFCBANK ka neighborhood kya bol raha hai"

4. Sentiment (optional, n_stocks):
   FinBERT sentiment scores [-1, +1]
   "RELIANCE ki news positive hai ya negative"

Total: ~1050 values (without embeddings) → Agent ka "vision"
```

### Action Space — Agent Kya Karta Hai

```
Action: n_stocks continuous values [-1, +1]
  ↓
Softmax: convert to valid weights [0, 1] summing to 1
  ↓
Clip: max 20% per stock
  ↓
Final weights: [0.15, 0.10, 0.08, 0.20, 0.12, ...]
  Cash = 1 - sum = remaining

Example:
  Agent output: [2.5, 1.0, -0.5, 3.0, 0.1]
  After softmax: [0.30, 0.07, 0.01, 0.50, 0.03]
  After clip(0.20): [0.20, 0.07, 0.01, 0.20, 0.03] → sum=0.51, cash=0.49

Kyu softmax?
  - Ensures all weights positive (can't short sell in our model)
  - Sums to ~1 (all money allocated)
  - Differentiable (PPO needs gradients)
```

### Reward Function — Agent Ko Kaise Grade Karte Hain

```
reward = sharpe_component - drawdown_penalty - turnover_penalty

1. SHARPE COMPONENT (main reward):
   Rolling 20-day Sharpe ratio = mean(returns) / std(returns) × sqrt(248)

   Kyu Sharpe?
   - Raw return reward galat hai: "15% return 50% risk ke saath" BAD
   - Sharpe = risk-adjusted: "15% return 10% risk ke saath" GOOD
   - Agent ko sikhata hai: "Kam risk mein zyada kamao"

2. DRAWDOWN PENALTY:
   penalty = |drawdown| × 0.1

   Kyu?
   - Agar portfolio peak se 10% gira, penalty = 0.01
   - Agent ko sikhata hai: "Peak se mat girao, protect karo"

3. TURNOVER PENALTY:
   penalty = |weight_changes| × 0.01

   Kyu?
   - Har trade mein 0.15% cost lagta hai (STT + brokerage + slippage)
   - Agar agent roz poora portfolio change kare = massive costs
   - Penalty sikhata hai: "Zarurat ke bina mat badlo"
```

### Risk Constraints — India-Specific

```
1. MAX POSITION: 20% per stock
   Kyu? SEBI mutual fund rules: max 10% per stock.
   Hum 20% rakh rahe (slightly aggressive for RL flexibility).
   Agar 50% ek stock mein daale aur woh crash kare = game over.

2. STOP LOSS: -5% per stock per day
   Kyu? NSE circuit limit 20% hai, par hum conservative hain.
   Agar koi stock 5% gire ek din mein → forced exit.
   "Cut your losses short, let profits run."

3. MAX DRAWDOWN: -15% circuit breaker
   Kyu? Agar total portfolio peak se 15% gir gaya → episode terminate.
   Real world mein fund managers ko "max drawdown tolerance" hoti hai.
   Investor ko promise: "Maximum 15% loss hoga."
   Training mein: agent seekhta hai ki -15% se pehle risk reduce karo.

4. TRANSACTION COSTS:
   - 0.1% per trade (STT + brokerage)
   - 0.05% slippage (market impact)
   Total: 0.15% per unit turnover

   Indian context: Zerodha pe intraday 0.03% + STT 0.025% + GST etc.
   0.15% slightly higher = conservative estimate.
```

### Episode Structure

```
1. RESET:
   - Random start date within training data (2016-2021)
   - Initial: 10L cash, 0 positions, 0 history
   - Returns observation + info

2. LOOP (252 steps = 1 year):
   For each trading day:
     a. Agent sees observation → outputs action
     b. Action → target weights (softmax + clip)
     c. Calculate turnover + costs
     d. Move to next day → stock returns
     e. Update portfolio value
     f. Check stop loss + drawdown
     g. Calculate reward
     h. Return (obs, reward, terminated, truncated, info)

3. END:
   - Truncated: 252 steps done (normal end)
   - Terminated: drawdown > -15% (forced end)
   - Summary: total return, Sharpe, max drawdown

4. Random start kyu?
   Agar hamesha 2016-Jan se start kare → agent 2016 ka pattern memorize karega.
   Random start = generalize kare, kisi bhi market condition mein kaam kare.
```

### Tests: 23/23 PASSING

| ID | Test | Kya Check |
|----|------|-----------|
| T6.1 | Env creates | PortfolioEnv() no error |
| T6.2 | Obs space shape | n_stocks×21 + n_stocks + 2 |
| T6.3 | Action space | (n_stocks,) continuous |
| T6.4 | Initial state | 10L cash, 0 positions |
| T6.5 | Reset tuple | (obs, info) returned |
| T6.6 | Reset obs shape | matches observation_space |
| T6.7 | Reset clears | After steps + reset → back to initial |
| T6.8 | Reset deterministic | Same seed → same obs |
| T6.9 | Step tuple | (obs, reward, term, trunc, info) |
| T6.10 | Step obs shape | matches observation_space |
| T6.11 | Value changes | Trading changes portfolio value |
| T6.12 | Costs applied | Flat prices + trade → value decreases |
| T6.13 | Info keys | portfolio_return, turnover, drawdown, etc. |
| T6.14 | Max position | Weight capped at 20% |
| T6.15 | Weights ≤ 1 | Sum never exceeds 1.0 |
| T6.16 | Stop loss | -10% crash → position zeroed |
| T6.17 | Drawdown terminates | -15% drawdown → terminated=True |
| E6.1 | Zero action | No crash on zeros |
| E6.2 | Single stock | 1 stock env works |
| E6.3 | Truncation | Episode ends at episode_length |
| E6.4 | Gym API | Has reset, step, spaces, metadata |
| E6.5 | Summary | End-of-episode stats correct |
| E6.6 | With embeddings | Optional T-GAT + sentiment inputs work |

### File Flow (Updated)

```
stocks.py → download.py → quality.py → features.py
    |
    v
Feature Tensor (47, ~2200, 21)     Sentiment Matrix (47, ~2200)
    |                                    |
    +------------------------------------+
    |
    v
builder.py (graph construction) → PyG Data Objects
    |
    v
tgat.py (T-GAT model) → Stock Embeddings (47, 64)
    |
    +-------- features ----+---- sentiment ---+
    |                      |                  |
    v                      v                  v
environment.py (RL Gym Environment)               ← NEW
    |
    |  Observation: features + weights + cash + embeddings + sentiment
    |  Action: target portfolio weights
    |  Reward: Sharpe - drawdown_penalty - turnover_penalty
    |
    v
Phase 7: Deep RL Agent (PPO/SAC) (next)
[Agent interacts with environment, learns optimal portfolio strategy]
```

---

---

## PHASE 7: Deep RL Agent (PPO + SAC)

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/rl/agent.py` | PPO + SAC agent creation, training, evaluation, comparison, save/load | Environment ready hai (Phase 6). Ab agent chahiye jo environment mein practice karke seekhe ki "optimal portfolio kya hai." PPO primary hai, SAC comparison ke liye. |
| `tests/test_agent.py` | 16 tests covering both algorithms, eval, save/load, edge cases | Agent create hota hai? Train hota hai? Valid actions deta hai? Save/load ke baad same predictions? |

### PPO vs SAC — Simple Analogy

```
PPO (Proximal Policy Optimization):
  Sooch: Tu badminton seekh raha hai.
  PPO approach: "Roz thoda practice kar, zyada change mat kar apne style mein."
  - Conservative updates: policy zyada nahi badalta ek step mein
  - Clip range 0.2: "Maximum 20% change allowed per update"
  - Stable training: rarely diverges
  - Good default choice for continuous actions

SAC (Soft Actor-Critic):
  Sooch: Tu badminton seekh raha hai.
  SAC approach: "Alag alag shots try kar, explore kar, par best wale remember kar."
  - Entropy bonus: encourages exploration (try new things)
  - Replay buffer: past experiences reuse karta hai
  - More sample-efficient (faster learning with less data)
  - But: more complex, more hyperparameters

For thesis: DONO train karo, compare karo, best pick karo.
```

### PPO — How It Works

```
1. COLLECT EXPERIENCE:
   Agent environment mein 2048 steps chalta hai.
   Har step: (observation, action, reward, next_obs)
   Sab store karo.

2. CALCULATE ADVANTAGE:
   "Kya action liya tha usse expected se better hua ya worse?"
   Advantage > 0: "Yeh action achha tha, isko zyada karo"
   Advantage < 0: "Yeh action bura tha, isko kam karo"

3. UPDATE POLICY (10 epochs over collected data):
   Policy = neural network jo obs → action mapping seekhta hai.

   KEY INNOVATION — Clipping:
   ratio = new_policy(action) / old_policy(action)
   clipped_ratio = clip(ratio, 1-0.2, 1+0.2)
   loss = min(ratio × advantage, clipped_ratio × advantage)

   Matlab: "Agar naya policy purane se 20% se zyada different ho,
   toh chhod do. Zyada change dangerous hai."

4. REPEAT:
   New policy se experience collect karo → update → repeat
   500K steps = ~2000 episodes of 252 days

Policy Network: obs (1050 dims) → [128] → [64] → action (47 dims)
  Only 46K parameters. Very lightweight.
```

### SAC — Key Difference

```
SAC has 3 extra things over PPO:

1. REPLAY BUFFER:
   Store 100K past (obs, action, reward, next_obs) tuples.
   Train on random samples from buffer.
   Kyu? "Purane experiences waste nahi karo."
   PPO data use karta hai ek baar aur phir fek deta hai.
   SAC data reuse karta hai = faster learning.

2. ENTROPY BONUS:
   reward_effective = reward + alpha × entropy(policy)
   entropy = "kitni randomness hai policy mein"
   High entropy = exploring different actions
   Kyu? Without exploration, agent local minimum mein fas sakta hai.

3. TWO Q-NETWORKS:
   Min of two Q-values use karta hai.
   "Pessimistic estimate" — overestimation avoid karta hai.
   More stable value estimates.

Trade-off: SAC faster seekhta hai par zyada complex + memory.
```

### Training Pipeline

```
1. Create environment with training data (2016-2021)
2. Create eval environment with validation data (2022-2023)
3. Create PPO agent → train 500K steps
   - Every 5000 steps: evaluate on val set → log Sharpe, return, drawdown
   - Save best model (highest Sharpe on val)
4. Create SAC agent → same training
5. Compare both → pick winner for thesis

PortfolioMetricsCallback:
  Not just "average reward" — we log FINANCIAL metrics:
  - Sharpe ratio (risk-adjusted return)
  - Max drawdown (worst peak-to-trough)
  - Total return (did we make money?)
  These matter for thesis results.
```

### Model Size — VRAM Budget

```
PPO policy:  46K params → ~0.18 MB
SAC policy: 117K params → ~0.47 MB
Replay buffer: 100K × (obs_dim + action_dim + 3) × 4 bytes ≈ 45 MB

Total VRAM usage during training:
  PPO: ~50 MB (model + batch data)
  SAC: ~100 MB (model + buffer)

Budget: 4 GB VRAM → plenty of room!
  Even with T-GAT (0.1 MB) + FinBERT (220 MB) inference.
```

### Tests: 16/16 PASSING

| ID | Test | Kya Check |
|----|------|-----------|
| T7.1 | PPO creates | create_ppo_agent() no error |
| T7.2 | PPO trains | 200 steps training works |
| T7.3 | PPO predicts | Valid action shape, in action_space |
| T7.4 | SAC creates | create_sac_agent() no error |
| T7.5 | SAC trains | 200 steps training works |
| T7.6 | SAC predicts | Valid action shape |
| T7.7 | Eval metrics | mean_return, sharpe, max_dd returned |
| T7.8 | Finite metrics | No NaN/Inf in evaluation |
| T7.9 | Save/load PPO | Same predictions after save → load |
| T7.10 | Save/load SAC | Same predictions after save → load |
| T7.11 | Custom LR | learning_rate=0.001 accepted |
| T7.12 | Custom arch | [64, 32] network works |
| E7.1 | Single stock | 1-stock env, action shape (1,) |
| E7.2 | Short training | 64 steps, no crash |
| E7.3 | Compare agents | Returns winner (PPO or SAC) |
| E7.4 | Eval callback | Training with periodic evaluation |

### File Flow (Updated — Complete P0-P7 Pipeline)

```
Phase 0: config.yaml + seed + logger + metrics
Phase 1: stocks.py → download.py → quality.py → data/*.csv
Phase 2: features.py → Feature Tensor (47, ~2200, 21)
Phase 3: news_fetcher.py → finbert.py → Sentiment Matrix (47, ~2200)
Phase 4: builder.py → PyG Data Objects (per day)
Phase 5: tgat.py → Stock Embeddings (47, 64)
Phase 6: environment.py → Gymnasium PortfolioEnv
Phase 7: agent.py → PPO/SAC trained agents        ← NEW

                    agent.py
                   /        \
              PPO agent    SAC agent
                   \        /
                    compare
                       |
                       v
              Winner → Phase 8: TimeGAN (data augmentation)
```

---

> **Next: Phase 10 — NAS/DARTS. Architecture search for optimal GNN structure.**

---

## PHASE 8: TimeGAN (Synthetic Financial Time Series)

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/gan/timegan.py` | TimeGAN: 5 GRU-based neural networks, 3-phase training, synthetic time series generation | Real market data limited hai — 10 years = ~2500 trading days. RL agent ko zyada diverse scenarios chahiye. TimeGAN realistic synthetic data generate karta hai jo real data jaisa dikhta hai. |
| `src/gan/stress.py` | VaR, CVaR, Monte Carlo simulation, 4 crash scenarios, survival rate | "Model achha hai par 2008 jaisi crash aaye toh kya hoga?" Stress testing portfolio ko extreme conditions mein test karta hai. Banks mein mandatory hai. |
| `tests/test_gan.py` | 25 tests: TimeGAN + Stress Testing + edge cases | GAN train hota hai? Generated data finite hai? VaR correct hai? Crash scenarios expected order mein hain? |

### TimeGAN — Simple Analogy

```
Sooch: Tu ek painter hai jo Monet ki paintings copy karna seekh raha hai.

Regular GAN:
  - Generator: Random noise se painting banao
  - Discriminator: "Yeh asli Monet hai ya fake?"
  - Problem: Static images ke liye kaam karta hai, par
    TIME SERIES mein order matters! Day 5 ka price Day 4 ke baad aana chahiye.

TimeGAN adds:
  - Embedder: Asli data ko "summary space" mein compress karo
  - Recovery: Summary se wapas data banao (autoencoder)
  - Supervisor: Summary mein TEMPORAL PATTERN seekho
    ("Kal price 100 tha, aaj 102, toh kal ~103-104 hona chahiye")
  - Generator: Random noise se summary banao (not direct data!)
  - Discriminator: "Yeh summary real data ka hai ya generated?"

Result: Generated time series mein:
  ✓ Realistic statistics (mean, std similar)
  ✓ Temporal patterns (trends continue, not random jumps)
  ✓ Cross-feature correlations (OHLCV relationships preserved)
```

### 5 Components — Detail

```
1. EMBEDDER (Real → Latent):
   Input:  Real data (batch, seq_len, n_features) e.g. (32, 128, 5)
   GRU:    Learns temporal compression
   FC:     Projects to latent dim
   Output: Latent representation (32, 128, latent_dim) with sigmoid [0,1]

   Kyu sigmoid? Bounded output [0,1] — GAN training mein unbounded values
   explode kar sakte hain. Sigmoid stabilizes.

2. RECOVERY (Latent → Real):
   Input:  Latent representation
   GRU:    Learns to decompress
   FC:     Projects back to original dim
   Output: Reconstructed data (same shape as input)

   Embedder + Recovery = Autoencoder
   Goal: h = Embedder(x), x_hat = Recovery(h), x_hat ≈ x

3. GENERATOR (Noise → Fake Latent):
   Input:  Random noise z ~ N(0,1) (32, 128, latent_dim)
   GRU:    Transforms noise into structured sequence
   FC:     Projects to latent dim + sigmoid
   Output: Fake latent representation (32, 128, latent_dim)

   Generator latent space mein kaam karta hai, data space mein NAHI.
   Kyu? Latent space simpler hai — easier to fool discriminator.

4. DISCRIMINATOR (Real vs Fake):
   Input:  Latent representation (real or fake)
   GRU:    Processes sequence
   FC:     Outputs single logit per timestep
   Output: Real/Fake score (32, 128, 1)

   Standard GAN discriminator — logit output, BCE loss.

5. SUPERVISOR (Temporal Dynamics):
   Input:  Latent representation at time t
   GRU:    Learns "what comes next" in latent space
   FC:     Predicts h(t+1) from h(t)
   Output: Next-step latent prediction

   KEY INNOVATION: Supervisor sikhata hai temporal patterns:
   "Agar latent state [0.3, 0.7, 0.1] hai toh next step [0.35, 0.65, 0.12] hona chahiye"
   Generator ko bhi yeh supervisor guide karta hai.
```

### 3-Phase Training

```
Phase 1: AUTOENCODER (40% epochs)
  Train: Embedder + Recovery
  Loss:  MSE(Recovery(Embedder(x)), x) → minimize reconstruction error
  Goal:  Learn good latent representation of real data

  Analogy: "Pehle real paintings ko ek code mein convert karna seekho,
  phir code se wapas painting banana seekho."

Phase 2: SUPERVISOR (20% epochs)
  Train: Supervisor (Embedder frozen)
  Loss:  MSE(Supervisor(h[t]), h[t+1]) → predict next step in latent space
  Goal:  Learn temporal dynamics in latent space

  Analogy: "Ab code sequences ka pattern samjho —
  agar code [A,B,C] hai toh agla D hona chahiye."

Phase 3: JOINT ADVERSARIAL (40% epochs)
  Train: Generator + Supervisor vs Discriminator
  3 losses combined:

  g_loss = g_adversarial       — fool the discriminator
         + 10 × g_supervisor    — temporal dynamics match
         + 100 × g_moment       — statistics match (mean + std)

  g_adversarial: "Discriminator ko fool karo — fake ko real bolo"
  g_supervisor:  "Generated sequence mein temporal pattern hona chahiye"
  g_moment:      "Real aur fake ka mean/std same hona chahiye"

  Weights: moment (100) >> supervisor (10) >> adversarial (1)
  Kyu? Statistics match karna easier aur zyada important hai.
  Adversarial fine-tuning last mein hota hai.
```

### Generation Pipeline

```
After training:
  1. Sample noise:    z ~ N(0,1) shape (n_samples, seq_length, latent_dim)
  2. Generator:       h_fake = Generator(z) → fake latent sequence
  3. Supervisor:      h_sup = Supervisor(h_fake) → temporal refinement
  4. Recovery:        x_fake = Recovery(h_sup) → synthetic data!

Output: (n_samples, seq_length, n_features)
Example: (100, 128, 5) = 100 synthetic sequences, 128 days each, 5 features
```

---

## PHASE 9: Stress Testing (Portfolio Risk Assessment)

### Kya Hai Stress Testing?

```
Sooch: Tu ek bridge engineer hai.
  Normal testing: "Is bridge pe 10 cars chal sakti hain? Yes!"
  Stress testing: "Agar 1000 cars ek saath aayen? Earthquake aaye? Flood aaye?"

Portfolio version:
  Normal testing: "2020-2024 mein 15% return diya? Great!"
  Stress testing: "2008 crisis dobara aaye toh? COVID jaise crash ho toh?
                   Flash crash ho toh? Kitna loss hoga? Portfolio survive karega?"
```

### Value at Risk (VaR) — "Worst Case at X% Confidence"

```
VaR at 95%:
  "95% chance hai ki 1 year mein loss X se ZYADA nahi hogi."

Technically: 5th percentile of return distribution.

Example:
  10,000 simulated returns sorted: [-25%, -18%, -12%, ..., +30%]
  5th percentile = -12%
  VaR(95%) = -12%
  Matlab: "95% chance hai ki loss 12% se zyada nahi hogi.
           Par 5% chance hai ki 12% se BHI zyada lose karo."

compute_var(returns, 0.95):
  return np.percentile(returns, 5)  # 5th percentile = (1-0.95)*100

VaR(99%) > VaR(95%) always (more conservative):
  99% mein 1st percentile dekhte hain → extreme tail
  "99% confidence se loss 18% se zyada nahi hogi" (example)
```

### Conditional VaR (CVaR / Expected Shortfall) — "Average Worst Case"

```
VaR says:  "95% chance se loss ≤ 12%"
CVaR asks: "OK, par jo 5% WORST cases hain, unka AVERAGE loss kitna hai?"

Example:
  Worst 5% of returns: [-25%, -22%, -18%, -15%, -13%]
  CVaR = average = -18.6%

CVaR ≤ VaR ALWAYS (because CVaR = average of tail, VaR = boundary)

Kyu CVaR better than VaR?
  VaR: "Loss 12% se zyada nahi" → but agar zyada hua toh KITNA zyada? Pata nahi.
  CVaR: "Worst case mein average 18.6% loss" → tail risk capture karta hai.
  Basel III / SEBI: CVaR preferred risk metric.
```

### Monte Carlo Simulation — "Random Future Paths"

```
Process:
  1. Historical statistics calculate karo:
     - Mean daily returns per stock
     - Covariance matrix (stock correlations)

  2. Cholesky decomposition: cov = L @ L.T
     Kyu? To generate CORRELATED random numbers.
     If RELIANCE and HDFCBANK are 0.6 correlated,
     random simulated returns bhi 0.6 correlated hone chahiye.

  3. For each simulation path (10,000 paths):
     a. Generate random: z = random normal (252 days × n_stocks)
     b. Correlated returns: daily_returns = mean + z @ L.T
     c. Portfolio return: weighted sum of stock returns
     d. Compound: total_return = prod(1 + daily_returns) - 1

  4. From 10,000 total returns:
     - VaR(95%), VaR(99%), CVaR(95%)
     - Mean return, max loss
     - Distribution of outcomes

Kyu Monte Carlo?
  VaR from HISTORICAL data → limited to what actually happened.
  Monte Carlo → explores what COULD happen (infinite scenarios).
  "2008 mein market 40% gira. Par 50% bhi gir sakta tha."
```

### Crash Scenarios — Historical Extremes

```
4 Pre-defined scenarios:

1. NORMAL (baseline):
   Daily shock: mean=0%, std=1%
   Duration: 252 days (1 year)
   Correlation boost: 0%
   "Normal market conditions"

2. 2008 GLOBAL FINANCIAL CRISIS:
   Daily shock: mean=-0.3%, std=3.5%
   Duration: 120 days (Sep 2008 - Jan 2009)
   Correlation boost: +30%
   "Lehman crash. Sab stocks gire. Correlations spike."

3. COVID MARCH 2020:
   Daily shock: mean=-0.5%, std=5.0%
   Duration: 30 days (fastest crash)
   Correlation boost: +40%
   "COVID panic. 30% crash in 30 days. Everything correlated."

4. FLASH CRASH:
   Daily shock: mean=-2.0%, std=8.0%
   Duration: 5 days
   Correlation boost: +50%
   "Algorithmic cascade. Extreme for a few days."

Correlation boost kya hai?
  Normal times: IT sector aur Pharma uncorrelated (0.2)
  Crisis: EVERYTHING falls together. Correlation → 0.7+
  "Diversification fails when you need it most."
  Boosting correlations simulates this real phenomenon.

Implementation:
  1. Take base covariance matrix
  2. Extract correlation matrix
  3. Add correlation_boost to off-diagonals
  4. Clip to [-1, 1]
  5. Reconstruct stressed covariance with higher variances
  6. Simulate portfolio under stressed parameters
```

### Survival Rate — "Portfolio Bachega Ya Nahi?"

```
For each simulation:
  Track cumulative returns → peak → drawdown
  If max drawdown > -15% (our threshold): SURVIVED
  Else: FAILED (circuit breaker would have triggered)

Survival rate = survived / total_simulations

Example:
  Normal market: survival 95% → "Safe, mostly fine"
  2008 crash:    survival 40% → "Risky! 60% chance of circuit breaker"
  Flash crash:   survival 20% → "Very dangerous. Hedge immediately."

Portfolio manager ke liye critical metric:
  "If 2008 repeats, 60% chance ki portfolio will hit -15% drawdown."
```

### Tests: 25/25 PASSING

| ID | Test | Kya Check |
|----|------|-----------|
| T8.1 | GAN creates | TimeGAN() with correct architecture |
| T8.2 | Components | embedder, recovery, generator, discriminator, supervisor exist |
| T8.3 | Stats | total_params > 0, trained=False initially |
| T8.4 | Train 2D | 2D data (n_timesteps, n_features) → sliding windows → trains |
| T8.5 | Train 3D | 3D pre-windowed input works |
| T8.6 | Output shape | Generated shape = (n_samples, seq_length, input_dim) |
| T8.7 | Output finite | No NaN/Inf in generated data |
| T8.8 | Statistics | Generated std within 20× of real std |
| T8.9 | Sliding window | _prepare_data creates correct windows |
| T9.1 | VaR 95 | VaR of N(0, 0.01) ≈ -0.0165 |
| T9.2 | CVaR ≤ VaR | CVaR always more conservative |
| T9.3 | VaR 99 > 95 | 99% VaR is worse (more negative) |
| T9.4 | MC result | monte_carlo returns StressResult |
| T9.5 | MC VaR | VaR values present and 99 < 95 |
| T9.6 | All scenarios | 4 scenarios all execute |
| T9.7 | 2008 < normal | Crash mean return worse than normal |
| T9.8 | Survival range | 0 ≤ survival_rate ≤ 1 |
| T9.9 | Summary format | All keys present, % formatted |
| E8.1 | Single feature | TimeGAN with 1 feature works |
| E8.2 | Short training | 1 epoch, no crash |
| E8.3 | No training | generate() before train() → RuntimeError |
| E9.1 | Equal weights | 10-stock equal weight portfolio |
| E9.2 | Concentrated | All money in 1 stock → higher VaR |
| E9.3 | Unknown scenario | 'crash_alien_invasion' → ValueError |
| E9.4 | Zero variance | Flat returns → VaR = 0 |

### File Flow (Updated — Complete P0-P9 Pipeline)

```
Phase 0: config.yaml + seed + logger + metrics
Phase 1: stocks.py → download.py → quality.py → data/*.csv
Phase 2: features.py → Feature Tensor (47, ~2200, 21)
Phase 3: news_fetcher.py → finbert.py → Sentiment Matrix (47, ~2200)
Phase 4: builder.py → PyG Data Objects (per day)
Phase 5: tgat.py → Stock Embeddings (47, 64)
Phase 6: environment.py → Gymnasium PortfolioEnv
Phase 7: agent.py → PPO/SAC trained agents
Phase 8: timegan.py → Synthetic augmented data          ← NEW
Phase 9: stress.py → VaR, CVaR, crash scenarios         ← NEW

                 Real Data (2500 days)
                        |
                    TimeGAN
                        |
              Synthetic Data (+10K days)
                        |
            RL Agent trains on BOTH
                        |
                 Trained Agent
                        |
                 Stress Testing
              /     |       |      \
          Normal  2008   COVID   Flash
              \     |       |      /
               Risk Assessment
          (VaR, CVaR, Survival Rate)
                        |
                  Phase 10: NAS/DARTS
```

---

## PHASE 10: NAS / DARTS (Neural Architecture Search)

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/nas/search_space.py` | 5 candidate operations + MixedOp (softmax-weighted blend) + SearchSpace config | Manually T-GAT design kiya (64 dim, 2 layers, 4 heads) — par kya guarantee hai ki yeh BEST design hai? NAS automatically better architecture dhundh sakta hai. |
| `src/nas/darts.py` | TGATSupernet with MixedOps, DARTSSearcher (bilevel optimization), RL policy grid search, PDF report generation | DARTS = "Differentiable Architecture Search" — gradient-based search. No brute force. Architecture weights (alpha) seekh te hain ki kaunsa operation best hai. |
| `tests/test_nas.py` | 18 tests: supernet, convergence, extraction, report, reproducibility, RL search, edge cases | Supernet fit hota hai? Alpha converge hota hai? Top-3 extract hote hain? Same seed = same result? |

### DARTS — Simple Analogy

```
Sooch: Tu ek restaurant kholna chahta hai.

Manual design (current T-GAT):
  "Mujhe lagta hai Italian best hoga, 2 floors, 4 tables."
  → May not be optimal. Just your guess.

DARTS approach:
  Step 1: Build a "super-restaurant" that serves Italian, Chinese,
          Indian, Mexican, Japanese ALL AT ONCE (supernet).
  Step 2: Track which cuisines customers order most (alpha weights).
  Step 3: Over time, Italian gets 60% orders, Chinese 25%, rest 15%.
  Step 4: Final restaurant: Italian (primary) + Chinese (secondary).

Stock market version:
  Supernet has 5 operations at each layer: linear, conv1d, attention, skip, none.
  During search, alpha weights learn which operations work best.
  After search: extract the winning ops → retrain from scratch.
```

### Two Search Targets

```
1. T-GAT (DARTS — full bilevel optimization):
   Supernet:  input_proj → [MixedOp₁] → [MixedOp₂] → ... → GRU → output_proj
   Each MixedOp blends: linear + conv1d + attention + skip + none
   Alpha weights decide: "Is layer ko attention chahiye ya linear?"

   Bilevel:
     Inner loop: optimize model weights W on TRAINING data
     Outer loop: optimize alpha on VALIDATION data
     → Prevents overfitting! Alpha optimized for generalization.

2. RL Policy (Grid search — simple, keeps SB3):
   5 candidates tested:
     [64, 32]      — lightweight
     [128, 64]     — default (hand-designed)
     [256, 128]    — wide
     [128, 128, 64] — deep 3-layer
     [64, 64]      — square
   Each trained → evaluated → ranked by Sharpe ratio.

Kyu DARTS sirf T-GAT pe?
  SB3 (Stable-Baselines3) ka PPO implementation sealed hai.
  DARTS wrapping is hacky and breaks SB3's internal optimizations.
  T-GAT is OUR model — full control hai. Grid search for RL = pragmatic.
```

### 5 Candidate Operations

```
1. LINEAR:     x → Linear(in, out) → LayerNorm → ELU
   Standard feedforward. Most common. Safe baseline.

2. CONV1D:     x → Conv1d(1×1) → LayerNorm → ELU
   Feature mixing via convolution. Captures local patterns.

3. ATTENTION:  x → MultiheadAttention(self) → proj → LayerNorm → ELU
   Self-attention across ALL nodes. Expensive but powerful.
   "Har stock sabhi stocks ko dekh ke decide kare."

4. SKIP:       x → Linear(in, out) if dims differ, else identity
   Identity connection. "Is layer mein kuch mat karo, bas pass through."
   Important for gradient flow in deep networks.

5. NONE:       → zeros
   Zero output. Effectively REMOVES this layer from the network.
   "Is layer ki zarurat hi nahi hai."
```

### MixedOp — DARTS Ka Core

```
Normal layer: output = Linear(x)  — fixed, one operation.

MixedOp: output = 0.3 × Linear(x)
                + 0.25 × Conv1d(x)
                + 0.2 × Attention(x)
                + 0.15 × Skip(x)
                + 0.1 × None(x)

Weights come from: softmax(alpha)
alpha is LEARNABLE — gradient descent se optimize hota hai.

Training ke dauraan:
  Epoch 1:  alpha = [0.0, 0.0, 0.0, 0.0, 0.0] → softmax → [0.2, 0.2, 0.2, 0.2, 0.2]
            All operations equally weighted (random)
  Epoch 50: alpha = [2.5, 0.1, -0.5, 1.8, -3.0] → softmax → [0.55, 0.05, 0.03, 0.27, 0.01]
            Linear dominates (55%), Skip strong (27%), rest pruned

After search: argmax → "Linear" selected for this layer.
```

### Bilevel Optimization — Kaise Kaam Karta Hai

```
Standard training: minimize L_train(W)
  Problem: model can memorize training data → overfit

DARTS bilevel:
  Inner: W* = argmin L_train(W, α)    — best weights for given architecture
  Outer: α* = argmin L_val(W*, α)     — best architecture on validation

Practical (alternating steps):
  For each epoch:
    Step 1: One gradient step on W using train loss
            W ← W - η × ∇_W L_train(W, α)
    Step 2: One gradient step on α using val loss
            α ← α - λ × ∇_α L_val(W, α)

Kyu better?
  α validation pe optimize hota hai → generalizes
  W training pe optimize hota hai → fits data
  Together: architecture that trains well AND generalizes
```

### Top-3 Architecture Extraction

```
After search:
  Architecture 1 (best): argmax(alpha) at each layer
    e.g., [linear, attention] → "Layer 1: linear, Layer 2: attention"

  Architecture 2 (variant 1): swap layer 0 to its 2nd-best op
    e.g., [conv1d, attention]

  Architecture 3 (variant 2): swap layer 1 to its 2nd-best op
    e.g., [linear, skip]

Kyu top-3?
  Single best may overfit to validation data.
  Top-3 gives robustness + options for downstream tuning.
  Retrain from scratch (new random weights) → fair comparison.
```

### Tests: 18/18 PASSING

| ID | Test | Kya Check |
|----|------|-----------|
| T10.1 | Supernet params | < 50MB, > 0 params |
| T10.2 | Single forward | (10, 64) output from single graph |
| T10.3 | Sequence forward | (10, 64) output from 5-step sequence |
| T10.4 | Param separation | arch_params ∩ weight_params = ∅ |
| T10.5 | Alpha converges | entropy decreases over 30 epochs |
| T10.6 | Extract top-3 | 3 architectures, valid ops, correct layers |
| T10.7 | Comparison | finite val_loss, convergence info available |
| T10.8 | Report PDF | File generated, size > 0 |
| T10.9 | Reproducibility | seed=42 twice → identical architecture |
| T10.10 | RL candidates | 5 PolicyConfig objects with net_arch |
| T10.11 | RL grid search | 2 candidates trained, ranked by Sharpe |
| E10.1 | Tiny search space | 2 ops only → 3 archs still extracted |
| E10.2 | Single layer | num_layers=1 → works, arch len=1 |
| E10.3 | Skip dominance | forced skip alpha → low entropy detected |
| T10.12-15 | Search space | ops create, MixedOp blends, config loads, unknown raises |

### File Flow (Updated — Complete P0-P10 Pipeline)

```
Phase 0:  config.yaml + seed + logger + metrics
Phase 1:  stocks.py → download.py → quality.py → data/*.csv
Phase 2:  features.py → Feature Tensor (47, ~2200, 21)
Phase 3:  news_fetcher.py → finbert.py → Sentiment Matrix (47, ~2200)
Phase 4:  builder.py → PyG Data Objects (per day)
Phase 5:  tgat.py → Stock Embeddings (47, 64)
Phase 6:  environment.py → Gymnasium PortfolioEnv
Phase 7:  agent.py → PPO/SAC trained agents
Phase 8:  timegan.py → Synthetic augmented data
Phase 9:  stress.py → VaR, CVaR, crash scenarios
Phase 10: search_space.py + darts.py → NAS-optimized architecture  ← NEW

           Hand-designed T-GAT          NAS T-GAT
           [64, 2 layers, 4 heads]      [?, ? layers, ? heads]
                    \                      /
                     compare (Sharpe ratio)
                              |
                     Winner → Phase 11: Federated Learning
```

---

> **Next: Phase 13 — API + Docker (FastAPI endpoints).**

---

## PHASE 11: Federated Learning (Privacy-Preserving Collaborative Training)

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/federated/server.py` | FL Server: FedAvg + FedProx aggregation, training coordination | Multiple "institutions" (clients) apna data share nahi karna chahte, par model improve karna chahte hain. Server sirf model weights collect karta hai, data KABHI nahi dekhta. |
| `src/federated/client.py` | FL Client: sector-wise data split, local training, weight sharing | Har client ek "institution" represent karta hai — Banking, IT, Pharma, Energy. Non-IID data split realistic hai kyunki real hedge funds sector-specialize karte hain. |
| `src/federated/privacy.py` | DP-SGD: gradient clipping + Gaussian noise + budget tracking | Sirf weights share karna bhi risky hai — weights se data reverse-engineer ho sakta hai. Differential Privacy guarantee deta hai ki koi individual data point identify nahi ho sakta. |
| `tests/test_fl.py` | 17 tests: client init, convergence, FedAvg vs FedProx, DP, fairness, edge cases | Clients bante hain? FL converge hota hai? DP noise model destroy nahi karta? Byzantine client handle hota hai? |

### Federated Learning — Simple Analogy

```
Sooch: 4 hospitals hain. Sabke paas cancer patients ka data hai.
  - Hospital A: Delhi (1000 patients)
  - Hospital B: Mumbai (800 patients)
  - Hospital C: Chennai (600 patients)
  - Hospital D: Kolkata (500 patients)

Problem: Best cancer prediction model chahiye.
  Option 1: Sab data ek jagah bhejo → ILLEGAL (patient privacy)
  Option 2: Har hospital apna model train kare → limited data = weak model
  Option 3: FEDERATED LEARNING →
    1. Central server ek model design karta hai
    2. Model COPY bhejta hai har hospital ko
    3. Har hospital APNE data pe train karta hai (data stays local!)
    4. Hospitals sirf UPDATED WEIGHTS bhejte hain (not data)
    5. Server weights average karta hai → improved global model
    6. Repeat 50 rounds → sab hospitals ka combined knowledge ek model mein

Result: Best model WITHOUT any hospital sharing patient data.

Stock market version:
  Hospital A → Client 0: Banking + Finance stocks (HDFCBANK, ICICIBANK, SBIN...)
  Hospital B → Client 1: IT + Telecom stocks (TCS, INFY, WIPRO...)
  Hospital C → Client 2: Pharma + FMCG stocks (SUNPHARMA, DRREDDY, ITC...)
  Hospital D → Client 3: Energy + Auto + Metals + Others (RELIANCE, MARUTI, TATASTEEL...)
```

### 4 Clients — Sector-Based Non-IID Split

```
Client 0: Banking + Finance (~10 stocks)
  HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK, INDUSINDBK,
  BAJFINANCE, BAJAJFINSV, HDFCLIFE, SBILIFE
  → Market ke sensitive, RBI rates se affected

Client 1: IT + Telecom (~6 stocks)
  TCS, INFY, WIPRO, HCLTECH, TECHM, BHARTIARTL
  → Dollar rate se affected, global demand

Client 2: Pharma + FMCG (~8 stocks)
  SUNPHARMA, DRREDDY, DIVISLAB, CIPLA, NESTLEIND, HINDUNILVR, ITC, BRITANNIA
  → Defensive stocks, recession mein bhi stable

Client 3: Energy + Auto + Metals + Infra + Others (~23 stocks)
  RELIANCE, ONGC, NTPC, POWERGRID, MARUTI, TATAMOTORS, TATASTEEL,
  JSWSTEEL, HINDALCO, ULTRACEMCO, GRASIM, ADANIENT, ADANIPORTS, etc.
  → Cyclical stocks, commodity prices se affected

Kyu non-IID?
  IID = "Har client ko random 12 stocks do" → easy but unrealistic.
  Non-IID = "Har client ek sector specialist hai" → harder but realistic.
  Real hedge funds specialize: "Goldman = banking focus, Citadel = quant/tech."
  Non-IID challenge: Client 0 sirf banking jaanta hai, Client 1 sirf IT.
  FL ka kaam hai: dono ki knowledge combine kare WITHOUT data share.
```

### FedAvg vs FedProx

```
FedAvg (Federated Averaging):
  Server pe aggregation:
    global_weights = weighted_average(client_weights, by=data_size)

  Example:
    Client 0: 100 samples, weights W₀
    Client 1: 300 samples, weights W₁
    Global = (100 × W₀ + 300 × W₁) / 400

  Intuition: "Bada dataset wale client ki zyada sunno"

  Problem: Non-IID data pe CLIENT DRIFT hota hai.
    Client 0 sirf banking seekhta hai → weights banking ki taraf bias.
    Client 1 sirf IT seekhta hai → weights IT ki taraf bias.
    Average karne pe dono ki expertise partially cancel ho jaati hai!

FedProx (Federated Proximal):
  FedAvg + ek extra term client training mein:

  client_loss = task_loss + (mu/2) × ||W_local - W_global||²

  Yeh extra term kya karta hai?
    "Apna local training kar, PAR global model se zyada door mat jaa."
    mu = 0.01 → halka sa pull towards global (default)
    mu = 1.0 → strong pull (almost no local adaptation)

  Result: Client drift kam hota hai → better convergence on non-IID data.

  Analogy:
    FedAvg: "Hospital ko full freedom do, jo seekhein woh bhejo"
    FedProx: "Hospital ko training do par ek rubber band lagao —
              central knowledge se zyada door mat jao"
```

### Differential Privacy (DP-SGD)

```
Problem:
  Sirf weights share karna bhi unsafe ho sakta hai!
  Adversary model weights analyse karke reconstruct kar sakta hai:
    "Is weight update mein RELIANCE ka data tha ya nahi?"
  This is called "membership inference attack."

Solution: Differential Privacy
  "(epsilon, delta)-DP guarantee:
   With probability 1-delta, an adversary CANNOT distinguish
   whether a specific data point was in the training set."

3 Steps per training round:
  1. GRADIENT CLIPPING (Bound Sensitivity):
     total_grad_norm = ||∇L||₂
     if total_grad_norm > max_norm (1.0):
       ∇L = ∇L × (max_norm / total_grad_norm)

     Kyu? Agar ek data point gradient ko bahut change kare
     (high sensitivity), toh adversary detect kar sakta hai.
     Clipping bounds: "Koi bhi single data point
     gradient ko max 1.0 se zyada affect nahi kar sakta."

  2. NOISE INJECTION (Plausible Deniability):
     ∇L_noisy = ∇L_clipped + N(0, σ²)
     σ = noise_multiplier × max_grad_norm

     Kyu? Clipping ke baad bhi exact gradient se info leak ho sakti hai.
     Gaussian noise add karne se: "Yeh gradient real hai ya noise?
     Adversary ko PATA NAHI."

  3. PRIVACY BUDGET TRACKING:
     Har round mein thoda privacy "spend" hota hai.
     Total budget: epsilon = 8.0
     Per round: epsilon_round = 8 / sqrt(n_rounds)

     Kyu budget? Infinite noise = useless model, zero noise = no privacy.
     epsilon = "kitni privacy sacrifice kar rahe hain."
     epsilon = 0.1 → very private (model learns slowly)
     epsilon = 8.0 → moderate privacy (model still learns well)
     epsilon = ∞ → no privacy (standard training)

     Simple composition: epsilon grows as sqrt(rounds).
     50 rounds at epsilon=8 → still within budget.

Noise Calibration:
  sigma = sensitivity × sqrt(2 × ln(1.25/delta)) / epsilon_per_round
  sensitivity = max_grad_norm / n_samples
  → More samples = less noise needed (averaging effect)
  → More rounds = more noise per round (budget split)
```

### FL Training Loop — Step by Step

```
For each round (50 total):
  1. SERVER → CLIENTS: "Yeh lo global model weights"
     (broadcast: server.get_global_weights() → client.set_weights())

  2. EACH CLIENT TRAINS LOCALLY:
     for epoch in range(5):
       pred = model(local_data)
       loss = MSE(pred, local_target)
       if FedProx: loss += (mu/2) × ||W - W_global||²
       loss.backward()
       if DP: clip_gradients() + add_noise()
       optimizer.step()

  3. CLIENTS → SERVER: "Yeh hain meri updated weights"
     (client.get_weights() → list of state_dicts)

  4. SERVER AGGREGATES:
     for each parameter:
       global[param] = sum(weight_i × client[param]_i) / sum(weight_i)
     where weight_i = client_i.data_size

  5. EVALUATE: Server evaluates new global model
     → Log avg_train_loss, avg_val_loss

Result: After 50 rounds, global model has learned from ALL clients'
  data without any client sharing their raw data.
```

### Tests: 17/17 PASSING

| ID | Test | Kya Check |
|----|------|-----------|
| T11.1 | 4 clients create | create_fl_clients() → 4 FLClient objects, correct data sizes |
| T11.2 | Sector mapping | Client 0=Banking, 1=IT, 2=Pharma, 3=Energy+others |
| T11.3 | Client tickers | Client 0 has HDFCBANK etc. |
| T11.4 | Invalid client ID | client_id=99 → ValueError |
| T11.5 | Loss decreases | 20 FL rounds → last_loss < first_loss |
| T11.6 | FedAvg + FedProx | Both strategies produce finite losses |
| T11.7 | FL vs individual | FL model evaluated, finite loss |
| T11.8 | DP converges | epsilon=8 + noise → training still converges |
| T11.9 | Budget tracking | 5 rounds → rounds_spent=5, epsilon_spent > 0 |
| T11.10 | Noise multiplier | compute_noise_multiplier() → positive, finite |
| T11.11 | Client fairness | All 4 clients benefit (no 10× degradation) |
| T11.12 | Weighted avg | Equal data → average of weights = 2.0 |
| T11.13 | Size-weighted | 10 vs 90 samples → weighted towards larger client |
| E11.1 | Byzantine client | 1 garbage client → FL still converges, finite loss |
| E11.2 | Tiny client | 1 sample client → works, finite loss |
| E11.3 | Small epsilon | epsilon=0.1 → large noise, but no crash |
| E11.4 | Single client | 1 client FL = normal training, converges |

### File Flow (Updated — Complete P0-P11 Pipeline)

```
Phase 0:  config.yaml + seed + logger + metrics
Phase 1:  stocks.py → download.py → quality.py → data/*.csv
Phase 2:  features.py → Feature Tensor (47, ~2200, 21)
Phase 3:  news_fetcher.py → finbert.py → Sentiment Matrix (47, ~2200)
Phase 4:  builder.py → PyG Data Objects (per day)
Phase 5:  tgat.py → Stock Embeddings (47, 64)
Phase 6:  environment.py → Gymnasium PortfolioEnv
Phase 7:  agent.py → PPO/SAC trained agents
Phase 8:  timegan.py → Synthetic augmented data
Phase 9:  stress.py → VaR, CVaR, crash scenarios
Phase 10: search_space.py + darts.py → NAS-optimized architecture
Phase 11: server.py + client.py + privacy.py → Federated Learning  ← NEW

         Client 0          Client 1          Client 2          Client 3
        (Banking)           (IT)            (Pharma)         (Energy+Auto)
            |                 |                |                  |
            +--- local train -+--- local train-+--- local train--+
            |                 |                |                  |
            +---- weights ----+---- weights ---+---- weights ----+
                              |
                        FL Server
                     (FedAvg / FedProx)
                              |
                     Global Model (better than any individual!)
                              |
                     + DP-SGD (privacy guarantee)
                              |
                     Phase 12: Quantum ML (QAOA)
```

---

## PHASE 12: Quantum ML (QAOA Portfolio Optimization)

### Kya Banaya?

| File | Kya Hai | Kyu Banaya |
|------|---------|------------|
| `src/quantum/qaoa.py` | QAOA circuit builder + COBYLA optimizer. QUBO matrix → Ising Hamiltonian → quantum circuit → optimize → decode bitstring | Portfolio mein "kaunse stocks select karein" ek combinatorial optimization problem hai. 47 stocks mein se 8 choose karna = C(47,8) = 314M+ combinations. Quantum computing is angle ko solve kar sakta hai. |
| `src/quantum/portfolio.py` | Portfolio-specific encoding, classical brute-force benchmark, Markowitz weight optimization, scaling study | QAOA sirf stocks SELECT karta hai (binary: haan/na). Weights kitne lagane hain yeh classical Markowitz karta hai. Plus classical brute-force baseline thesis comparison ke liye. |
| `tests/test_quantum.py` | 12 tests: QUBO, circuit, optimization, classical benchmark, comparison, scaling, edge cases | QUBO matrix sahi bani? Circuit chalta hai? QAOA valid result deta hai? Classical se compare ho raha hai? |

### QAOA — Simple Analogy

```
Sooch: Tu ek party plan kar raha hai.
  Tere paas 12 friends hain, 6 ko invite karna hai.
  Rules: Jo aayein unme se zyada se zyada enjoy ho (return), par jhagda kam ho (risk).

Classical approach:
  Sab 924 possible groups check kar. Best group pick kar.
  Works for 12 friends. But 1000 friends? Impossible.

Quantum approach (QAOA):
  Step 1: Har friend = ek qubit (0 = nahi aayega, 1 = aayega)
  Step 2: "Fun function" circuit mein encode kar:
          - Friends jo enjoy karte hain saath = positive
          - Friends jo ladte hain = negative
          - Exactly 6 invite = constraint
  Step 3: Quantum circuit SARE groups explore karta hai SIMULTANEOUSLY
          (superposition = sab states ek saath exist karte hain)
  Step 4: Classical optimizer (COBYLA) circuit tune karta hai
  Step 5: Measure → best group milta hai

Stock market version:
  12 candidate stocks (top by Sharpe from NIFTY 50)
  → Select 6 with highest risk-adjusted return
  → QAOA finds the combination
  → Markowitz weights compute karta hai
```

### QUBO — Optimization Problem Encoding

```
QUBO = Quadratic Unconstrained Binary Optimization
  "Binary" = har variable 0 ya 1 (include stock ya nahi)
  "Quadratic" = objective mein x_i × x_j terms hain (pairs of stocks)

Our QUBO for portfolio:
  minimize: x^T Q x

Where Q matrix has 3 parts:

1. RETURN TERMS (diagonal):
   Q[i,i] += -mu_i
   "Stock i ka expected return zyada hai? Include karo (negative = minimize mein better)"

2. RISK TERMS (off-diagonal):
   Q[i,j] += lambda × sigma[i,j]
   "Stock i aur j correlated hain? Dono include karna risky hai"
   lambda = risk_aversion (0.5 default = balanced)

3. CARDINALITY CONSTRAINT:
   penalty × (sum(x) - K)^2
   "Exactly K stocks select karo. Agar 5 ya 7 select kiye toh BIG penalty"
   Expanded: diagonal + off-diagonal terms mein distribute hota hai

Example (4 stocks, select 2):
  mu = [0.001, 0.002, 0.003, 0.004]  (daily returns)
  sigma = [[0.01, 0.005, ...], ...]    (covariance)

  Q = [ [-0.001 + penalty × (-3), penalty × 2, ...],
        [penalty × 2, -0.002 + penalty × (-3), ...],
        ... ]

  QAOA minimizes x^T Q x → finds best 2 stocks
```

### QUBO → Ising Conversion

```
Quantum computers prefer spins (+1, -1) over binary (0, 1).

Substitution: x_i = (1 - z_i) / 2
  x=0 → z=+1
  x=1 → z=-1

QUBO: minimize x^T Q x
  ↓ substitute
Ising: minimize sum_ij J[i,j] z_i z_j + sum_i h[i] z_i + offset

J = coupling matrix (ZZ interactions between qubits)
h = field vector (Z rotation on each qubit)
offset = constant (doesn't affect optimization)

Kyu convert?
  Qiskit quantum gates naturally work with +1/-1 (Pauli Z eigenvalues).
  RZZ gate = Z_i × Z_j interaction
  RZ gate  = single Z rotation
```

### QAOA Circuit — Step by Step

```
n_qubits = 8 (one per candidate stock)
n_layers = 3 (QAOA depth, p=3)

1. INITIAL STATE: Apply H (Hadamard) to all qubits
   → Creates uniform superposition: all 2^8 = 256 portfolios equally probable

   |0⟩ → H → (|0⟩ + |1⟩)/√2
   Each qubit: 50% chance include, 50% chance exclude

2. For each layer p = 0, 1, 2:

   a. COST UNITARY: exp(-i × gamma_p × C)
      For each pair (i,j): RZZ(2 × gamma × J[i,j]) on qubits i,j
      For each qubit i:    RZ(2 × gamma × h[i]) on qubit i

      "Yeh step HIGH-COST portfolios ko penalize karta hai"
      gamma_p = learnable parameter

   b. MIXER UNITARY: exp(-i × beta_p × B)
      For each qubit i: RX(2 × beta_p) on qubit i

      "Yeh step portfolio space explore karta hai (mixing)"
      beta_p = learnable parameter

3. MEASURE all qubits → bitstring (e.g., "10110010")
   1 = stock selected, 0 = not selected

Total parameters: 2 × n_layers = 6 angles (3 gammas + 3 betas)
```

### Classical Optimizer (COBYLA) — The Outer Loop

```
QAOA is a VARIATIONAL algorithm:
  → Classical optimizer tunes gamma/beta parameters
  → Quantum circuit evaluates the objective
  → Repeat until converged

COBYLA (Constrained Optimization BY Linear Approximation):
  - Gradient-free (doesn't need derivatives)
  - Works with noisy objectives (quantum measurements are probabilistic)
  - ~100-200 iterations to converge

Each iteration:
  1. COBYLA proposes new gamma/beta values
  2. Build circuit with these values
  3. Run circuit 1024 times (shots)
  4. Count bitstrings: "10110010" appeared 42 times, "11001010" appeared 38 times...
  5. Expected cost = sum(cost(bs) × count(bs)) / total_shots
  6. COBYLA uses this to propose better parameters
  7. Repeat

After convergence:
  → Final circuit with optimal gamma/beta
  → Run one more time → pick most common (or lowest cost) bitstring
  → Decode: "10110010" → stocks 0, 2, 3, 5 selected
```

### Classical Brute-Force Benchmark

```
For N ≤ 12, we can enumerate ALL subsets:
  C(8, 4) = 70 combos (8 stocks, select 4)
  C(12, 6) = 924 combos (12 stocks, select 6)

For each combo:
  1. Take subset mu and sigma
  2. Run Markowitz optimization → optimal weights
  3. Compute Sharpe ratio
  4. Track best

This gives the PROVABLY OPTIMAL classical answer.

Thesis comparison:
  "QAOA found Sharpe = 1.8, Classical optimal = 2.0"
  "QAOA was 90% as good as optimal in 5 seconds"

  On real quantum hardware, QAOA would scale polynomially,
  while brute-force scales exponentially.
  On simulator, both are similar — but the algorithm is validated.
```

### Markowitz Portfolio Optimization

```
After QAOA selects K stocks, we compute OPTIMAL WEIGHTS:

maximize: (w^T mu - rf) / sqrt(w^T Sigma w)   (Sharpe ratio)
subject to: sum(w) = 1                         (fully invested)
            0 ≤ w_i ≤ 0.20                     (max 20% per stock)

Uses scipy SLSQP optimizer (sequential least squares).

Example:
  QAOA selects: [HDFCBANK, TCS, RELIANCE, SUNPHARMA]
  Markowitz weights: [0.20, 0.20, 0.15, 0.45]  (Pharma gets most!)
  Sharpe ratio: 1.85

Kyu 20% max? SEBI rules + diversification.
  Agar 60% RELIANCE mein daale aur Reliance crash kare → portfolio dead.
```

### Tests: 12/12 PASSING

| ID | Test | Kya Check |
|----|------|-----------|
| T12.1 | QUBO shape | Q matrix is N×N |
| T12.2 | QUBO finite | All entries finite, no NaN |
| T12.3 | Returns on diagonal | Q[i,i] includes -mu_i terms |
| T12.4 | Circuit builds | Correct qubit count, has parameters |
| T12.5 | Circuit measurements | Classical bits present for readout |
| T12.6 | QAOA valid result | QAOAResult with correct bitstring length |
| T12.7 | Classical benchmark | Brute-force finds portfolio, Sharpe finite |
| T12.8 | Quantum vs Classical | Both produce finite Sharpe ratios |
| T12.9 | Scaling benchmark | Runs at [4, 6] sizes, times positive |
| E12.1 | 2-asset minimum | QAOA works with 2 qubits, selects 1 |
| E12.2 | Identical assets | All same returns → any selection valid |
| E12.3 | Single asset | n=1, k=1 → trivial, weight=1.0 |

### File Flow (Updated — Complete P0-P12 Pipeline)

```
Phase 0:  config.yaml + seed + logger + metrics
Phase 1:  stocks.py → download.py → quality.py → data/*.csv
Phase 2:  features.py → Feature Tensor (47, ~2200, 21)
Phase 3:  news_fetcher.py → finbert.py → Sentiment Matrix (47, ~2200)
Phase 4:  builder.py → PyG Data Objects (per day)
Phase 5:  tgat.py → Stock Embeddings (47, 64)
Phase 6:  environment.py → Gymnasium PortfolioEnv
Phase 7:  agent.py → PPO/SAC trained agents
Phase 8:  timegan.py → Synthetic augmented data
Phase 9:  stress.py → VaR, CVaR, crash scenarios
Phase 10: search_space.py + darts.py → NAS-optimized architecture
Phase 11: server.py + client.py + privacy.py → Federated Learning
Phase 12: qaoa.py + portfolio.py → Quantum portfolio selection  ← NEW

         47 NIFTY stocks
              |
         Top 8 by Sharpe ratio (pre-selection)
              |
       +------+------+
       |              |
    QAOA           Classical
  (quantum)     (brute-force)
       |              |
  Select K=4      Select K=4
       |              |
  Markowitz       Markowitz
   weights         weights
       |              |
       +--- compare --+
              |
         Thesis results:
         "Quantum achieves X% of classical optimal"
              |
         Phase 13: API + Docker
```
