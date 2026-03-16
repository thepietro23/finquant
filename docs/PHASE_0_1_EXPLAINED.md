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

> **Next: Phase 5 — T-GAT (Temporal Graph Attention Network). Graph + features → stock embeddings. Attention mechanism = important neighbors get more weight.**
