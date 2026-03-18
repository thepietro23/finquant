# FINQUANT-NEXUS v4 — Complete Running Guide

> Yeh guide tujhe step-by-step bataegi ki poora project kaise run karna hai — backend, frontend, tests, sab kuch.

---

## Prerequisites (Pehle Yeh Install Hona Chahiye)

| Software | Version | Check Command | Download |
|----------|---------|---------------|----------|
| Python | 3.11.x | `python --version` | python.org |
| Node.js | 18+ | `node --version` | nodejs.org |
| npm | 9+ | `npm --version` | Node ke saath aata hai |
| Git | 2.x | `git --version` | git-scm.com |
| CUDA (optional) | 12.1 | `nvidia-smi` | NVIDIA website |

---

## STEP 1: Python Virtual Environment Setup

```bash
cd d:\Personal\Clg\finquant\fqn1

# Agar venv pehle se hai toh activate karo
.\venv\Scripts\activate

# Agar nahi hai toh banao
python -m venv venv
.\venv\Scripts\activate

# Dependencies install karo
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Verify karo:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import qiskit; print(f'Qiskit: {qiskit.__version__}')"
```

---

## STEP 2: Run All Tests (232 Tests)

**Sabse pehle yeh karo — confirm karo ki sab kuch kaam kar raha hai:**

```bash
cd d:\Personal\Clg\finquant\fqn1
.\venv\Scripts\activate

# Full test suite — sab 14 phases test karo
python -m pytest tests/ -v --tb=short
```

**Expected Output:**
```
232 collected
231 passed, 1 xfail (or minor device assertion)
```

**Individual phase test karna ho toh:**
```bash
python -m pytest tests/test_phase0.py -v     # Phase 0: Config, seed, logger
python -m pytest tests/test_data.py -v       # Phase 1: Data pipeline
python -m pytest tests/test_features.py -v   # Phase 2: Feature engineering
python -m pytest tests/test_sentiment.py -v  # Phase 3: FinBERT sentiment
python -m pytest tests/test_graph.py -v      # Phase 4: Graph construction
python -m pytest tests/test_tgat.py -v       # Phase 5: T-GAT model
python -m pytest tests/test_env.py -v        # Phase 6: RL environment
python -m pytest tests/test_agent.py -v      # Phase 7: PPO/SAC agents
python -m pytest tests/test_gan.py -v        # Phase 8-9: TimeGAN + Stress
python -m pytest tests/test_nas.py -v        # Phase 10: NAS/DARTS
python -m pytest tests/test_fl.py -v         # Phase 11: Federated Learning
python -m pytest tests/test_quantum.py -v    # Phase 12: QAOA Quantum
python -m pytest tests/test_api.py -v        # Phase 13: REST API
```

---

## STEP 3: Start Backend API (FastAPI)

**Terminal 1 kholo:**

```bash
cd d:\Personal\Clg\finquant\fqn1
.\venv\Scripts\activate

uvicorn src.api.main:app --reload --port 8000
```

**Success Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

**Verify karo browser mein:**

| URL | Kya Dikhna Chahiye |
|-----|-------------------|
| http://localhost:8000/api/health | `{"status":"ok","version":"4.0.0","project":"FINQUANT-NEXUS v4"}` |
| http://localhost:8000/docs | Swagger UI — interactive API testing page |
| http://localhost:8000/redoc | ReDoc — beautiful API documentation |

**API Endpoints Test (Swagger mein ya curl se):**

```bash
# Health check
curl http://localhost:8000/api/health

# Stock list
curl http://localhost:8000/api/stocks

# Sentiment analysis
curl -X POST http://localhost:8000/api/sentiment -H "Content-Type: application/json" -d "{\"text\": \"Company profits surged 50 percent\"}"

# Stress test (thoda time lagega)
curl -X POST http://localhost:8000/api/stress-test -H "Content-Type: application/json" -d "{\"n_stocks\": 5, \"n_simulations\": 500}"

# QAOA quantum optimization (10-30 sec lagega)
curl -X POST http://localhost:8000/api/qaoa -H "Content-Type: application/json" -d "{\"n_assets\": 4, \"k_select\": 2, \"qaoa_layers\": 1, \"shots\": 64}"

# Financial metrics
curl -X POST http://localhost:8000/api/metrics -H "Content-Type: application/json" -d "{\"returns\": [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, 0.01, -0.01, 0.03, 0.02]}"
```

---

## STEP 4: Start Frontend Dashboard (React)

**Terminal 2 kholo (naya terminal, Terminal 1 band mat karo):**

```bash
cd d:\Personal\Clg\finquant\fqn1\dashboard

# Pehli baar ho toh dependencies install karo
npm install

# Dev server start karo
npm run dev
```

**Success Output:**
```
VITE v8.0.0  ready in 500ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: http://192.168.x.x:3000/
```

**Browser mein kholo:** http://localhost:3000

---

## STEP 5: Dashboard Pages Check Karo

Sidebar mein har page click karke verify karo:

| # | Page | URL | Kya Check Karo |
|---|------|-----|----------------|
| 1 | **Overview** | `/` | 4 metric cards animate hote hain, performance chart dikhta hai, sector donut, top holdings table |
| 2 | **Portfolio** | `/portfolio` | Stock count 47 dikhta hai, sector weights bar chart, Sharpe/Sortino numbers |
| 3 | **GNN Insights** | `/gnn` | Network graph SVG mein nodes dikhte hain, attention heatmap, edge type legend |
| 4 | **RL Agent** | `/rl` | PPO vs SAC toggle, reward curve chart, portfolio weights bar chart |
| 5 | **Stress Testing** | `/stress` | Monte Carlo paths (faint lines), Click **Generate Stress Test** → table fill hoti hai |
| 6 | **NAS Lab** | `/nas` | Architecture boxes (Linear→Attention→...), alpha convergence chart |
| 7 | **Federated** | `/fl` | 4 client cards (Banking, IT, Pharma, Energy), convergence curves |
| 8 | **Quantum** | `/quantum` | Click **Run QAOA** → circuit diagram SVG, Sharpe bars, weight comparison |
| 9 | **Sentiment** | `/sentiment` | Type text → Click **Analyze** → score bar (-1 to +1), batch headlines |
| 10 | **Graph Viz** | `/graph` | Interactive stock network, click nodes → right panel details, edge filter toggles |

---

## STEP 6: Live API Integration Test

Yeh pages **real API calls** karte hain (backend chalna chahiye):

### Sentiment (Phase 3 — FinBERT)
1. Jao `/sentiment`
2. Type: `Reliance Industries reports record quarterly profit`
3. Click **Analyze**
4. **Expected:** Score > 0, label = "positive", green badge

### Stress Testing (Phase 9 — Monte Carlo)
1. Jao `/stress`
2. Set: Stocks = 5, Simulations = 500
3. Click **Generate Stress Test**
4. **Expected:** 4 scenarios table appear hota hai (normal, crash_2008, crash_covid, flash_crash)

### Quantum (Phase 12 — QAOA)
1. Jao `/quantum`
2. Set: N Assets = 4, K Select = 2, Layers = 1
3. Click **Run QAOA** (10-30 sec lagega)
4. **Expected:** Circuit diagram, quantum vs classical Sharpe bars, weight comparison

### Graph Viz (Extra Page — Stock Network)
1. Jao `/graph`
2. Wait 2-3 sec for force simulation to settle
3. Click any node → right panel shows: ticker, sector, RL weight, connections
4. Toggle edge filters (sector/supply/correlation)
5. **Expected:** Nodes color by sector, size by RL weight, edges filter correctly

---

## STEP 7: Production Build Test

```bash
cd d:\Personal\Clg\finquant\fqn1\dashboard

# Production build
npm run build

# Preview production build
npm run preview
# → http://localhost:4173
```

---

## STEP 8: Docker (Optional — Docker Desktop Chahiye)

```bash
cd d:\Personal\Clg\finquant\fqn1

# Build + start API + PostgreSQL
docker-compose up --build

# Alag terminal mein test karo
curl http://localhost:8000/api/health

# Band karo
docker-compose down
```

---

## Quick Reference Card

### Start Everything (Daily Use)
```bash
# Terminal 1 — Backend
cd d:\Personal\Clg\finquant\fqn1
.\venv\Scripts\activate
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Frontend
cd d:\Personal\Clg\finquant\fqn1\dashboard
npm run dev
```

### Stop Everything
```
Terminal 1: Ctrl + C
Terminal 2: Ctrl + C
```

### Run Tests
```bash
cd d:\Personal\Clg\finquant\fqn1
.\venv\Scripts\activate
python -m pytest tests/ -v
```

---

## Troubleshooting

### Backend Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'src'` | `cd fqn1` karo pehle, uske baad uvicorn chalao |
| `Address already in use :8000` | `npx kill-port 8000` ya Task Manager se python process kill karo |
| `CUDA out of memory` | `configs/base.yaml` mein `device: 'cpu'` karo temporarily |
| `torch not found` | `.\venv\Scripts\activate` bhool gaye — pehle activate karo |
| `SSL/proxy error (yfinance)` | College network pe hai — mobile hotspot try karo |

### Frontend Issues

| Problem | Solution |
|---------|----------|
| `localhost:3000` blank / white page | Backend chal raha hai? `localhost:8000/api/health` check karo |
| Charts empty / "Loading..." stuck | API server down hai — Terminal 1 check karo |
| `npm run dev` fails | `cd dashboard && npm install` pehle run karo |
| `ENOENT package.json` | Galat folder mein ho — `cd d:\Personal\Clg\finquant\fqn1\dashboard` karo |
| Port 3000 busy | `npx kill-port 3000` ya browser tabs band karo |
| `Module not found` error | `npm install` phir se karo |

### Test Issues

| Problem | Solution |
|---------|----------|
| `test_model_loads` assertion fail (cpu vs cuda) | Known issue — GPU available hai toh CUDA pe load hota hai, test expects CPU. Ignore karo. |
| `No module named pytest` | `.\venv\Scripts\activate` pehle karo |
| Timeout on quantum tests | Normal hai — QAOA 10-30 sec le sakta hai |
| FinBERT download fail | Model `data/finbert_local/` mein hona chahiye ya HuggingFace cache mein |

---

## Project Config Files

| File | Kya Hai | Kab Edit Karo |
|------|---------|---------------|
| `configs/base.yaml` | Master config — sab hyperparameters | Model tuning, device change, thresholds |
| `.env` | Environment variables | DB URL, W&B API key, log level |
| `dashboard/vite.config.ts` | Frontend build config | API proxy URL change karna ho |
| `Dockerfile` | Container config | Deployment environment change |
| `docker-compose.yml` | Multi-service orchestration | DB password, port mapping |
| `requirements.txt` | Python dependencies | New package add karna ho |
| `dashboard/package.json` | Node dependencies | New npm package add karna ho |

---

## File Structure Overview

```
fqn1/
├── src/
│   ├── api/           → FastAPI (schemas.py, main.py) — 8 endpoints
│   ├── data/          → NIFTY 50 stocks, download, quality, features
│   ├── sentiment/     → FinBERT model, news fetcher
│   ├── graph/         → 3 edge types, PyG Data objects
│   ├── models/        → T-GAT (multi-relational GAT + GRU)
│   ├── rl/            → PPO/SAC agents, Gym environment
│   ├── gan/           → TimeGAN, stress testing, Monte Carlo
│   ├── nas/           → DARTS architecture search
│   ├── federated/     → FedAvg/FedProx, DP-SGD, 4 clients
│   ├── quantum/       → QAOA, portfolio optimization
│   └── utils/         → Config, seed, logger, metrics
├── tests/             → 14 test files, 232 tests total
├── dashboard/         → React frontend (10 pages)
│   ├── src/pages/     → Overview, Portfolio, GNN, RL, Stress,
│   │                    NAS, FL, Quantum, Sentiment, GraphViz
│   ├── src/components/→ Layout, UI, Charts
│   └── src/lib/       → API client, formatters, animations
├── configs/base.yaml  → All hyperparameters
├── Dockerfile         → Container image
├── docker-compose.yml → API + PostgreSQL
└── docs/              → PROGRESS, EXPLAINED, PRACTICAL_GUIDE
```

---

## Summary Checklist

- [ ] `python -m pytest tests/ -v` → 231+ passed
- [ ] `uvicorn src.api.main:app --reload --port 8000` → running
- [ ] `http://localhost:8000/docs` → Swagger UI dikhta hai
- [ ] `http://localhost:8000/api/health` → `{"status":"ok"}`
- [ ] `cd dashboard && npm run dev` → running on :3000
- [ ] `http://localhost:3000/` → Overview page with charts
- [ ] `/sentiment` → type text, click Analyze → score dikhe
- [ ] `/stress` → click Generate → scenarios table dikhe
- [ ] `/quantum` → click Run QAOA → circuit + Sharpe dikhe
- [ ] `/graph` → stock nodes visible, click → details dikhe
- [ ] `npm run build` → production build successful

**Sab green hai? Congratulations — FINQUANT-NEXUS v4 fully operational!**
