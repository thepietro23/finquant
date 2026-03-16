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

> **Next: Phase 2 explanation will cover Feature Engineering — 27 technical indicators, rolling normalization, aur data leakage prevention.**
