# FINQUANT-NEXUS v4 — Setup Guide

Complete setup instructions to get this project running on a fresh system.

---

## Prerequisites

| Tool | Version | Check Command |
|------|---------|---------------|
| Python | 3.11.x (tested on 3.11.9) | `python --version` |
| Git | Any recent | `git --version` |
| NVIDIA GPU Driver | Latest for your GPU | `nvidia-smi` |
| CUDA Toolkit | 12.1 | `nvcc --version` |
| Docker + Docker Compose | Latest | `docker --version` |
| Node.js | 18+ (for dashboard) | `node --version` |
| npm | 9+ (for dashboard) | `npm --version` |

> **Note:** If you don't have an NVIDIA GPU, you can still run everything on CPU — just skip the CUDA steps and install CPU-only PyTorch instead.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/thepietro23/finquant.git
cd finquant/fqn1
```

---

## Step 2: Create Python Virtual Environment

```bash
python -m venv venv
```

**Activate it:**

```bash
# Windows (CMD)
venv\Scripts\activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

Verify:
```bash
python --version   # Should show 3.11.x
```

---

## Step 3: Install PyTorch (MUST do first, separately)

**With NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Without GPU (CPU only):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify CUDA (GPU only):
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## Step 4: Install All Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs 38+ packages: numpy, pandas, scikit-learn, torch-geometric, transformers, stable-baselines3, qiskit, fastapi, flwr, wandb, etc.

### Troubleshooting: torch-geometric

If `torch-geometric`, `torch-scatter`, or `torch-sparse` fail to install, install them manually:

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric
```

Replace `cu121` with `cpu` if you don't have a GPU. Check your torch version with `python -c "import torch; print(torch.__version__)"` and match accordingly.

---

## Step 5: Setup Environment Variables

Create a `.env` file in the `fqn1/` root:

```bash
# === FINQUANT-NEXUS Environment Variables ===

DATABASE_URL=postgresql://finquant:finquant123@localhost:5432/finquant
MODEL_DIR=./models
LOG_LEVEL=INFO
SEED=42
WANDB_PROJECT=finquant-nexus
# WANDB_API_KEY=your_key_here    # Uncomment and add your key to enable W&B tracking
```

---

## Step 6: Start PostgreSQL Database

**Option A — Docker (recommended):**
```bash
docker-compose up db -d
```

This starts PostgreSQL 16 on port 5432 with:
- User: `finquant`
- Password: `finquant`
- Database: `finquant`

Verify:
```bash
docker ps   # Should show finquant-db running
```

**Option B — Local PostgreSQL:**

If you have PostgreSQL installed locally, create the database:
```sql
CREATE USER finquant WITH PASSWORD 'finquant123';
CREATE DATABASE finquant OWNER finquant;
```

Update `DATABASE_URL` in `.env` accordingly.

---

## Step 7: Download Stock Data

The data pipeline downloads NIFTY 50 stock CSVs (2015–2025) via yfinance:

```bash
python -c "from src.data.download import download_all; download_all()"
```

This creates CSV files in the `data/` folder. Takes ~2-5 minutes depending on internet speed.

> **Note:** If `data/*.csv` files already exist in the repo, you can skip this step.

---

## Step 8: Download FinBERT Model

FinBERT (~417MB) downloads automatically on first use, but you can pre-download it:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('ProsusAI/finbert'); AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"
```

The model gets cached in your Hugging Face cache directory (`~/.cache/huggingface/`). It's also saved locally to `data/finbert_local/` during sentiment analysis.

---

## Step 9: Create Required Directories

```bash
mkdir -p models logs experiments
```

These directories are used for model checkpoints, logs, and experiment tracking.

---

## Step 10: Run Tests

```bash
python -m pytest tests/ -v --tb=short
```

Expected: **232 tests**, all passing (some may be `xfail` for GPU-specific assertions on CPU).

Run individual phase tests:
```bash
python -m pytest tests/test_phase0.py -v     # Config, logging, seed
python -m pytest tests/test_data.py -v       # Data pipeline
python -m pytest tests/test_features.py -v   # Feature engineering
python -m pytest tests/test_sentiment.py -v  # FinBERT sentiment
python -m pytest tests/test_graph.py -v      # Graph construction
python -m pytest tests/test_tgat.py -v       # T-GAT model
python -m pytest tests/test_env.py -v        # RL environment
python -m pytest tests/test_agent.py -v      # PPO/SAC agents
python -m pytest tests/test_gan.py -v        # TimeGAN & stress testing
python -m pytest tests/test_nas.py -v        # NAS/DARTS
python -m pytest tests/test_fl.py -v         # Federated Learning
python -m pytest tests/test_quantum.py -v    # QAOA
python -m pytest tests/test_api.py -v        # REST API
```

---

## Step 11: Start the Backend API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Verify:
- Health check: http://localhost:8000/api/health
- Swagger docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/config` | Current configuration |
| GET | `/api/stocks` | NIFTY 50 stock list |
| POST | `/api/sentiment` | Single text sentiment |
| POST | `/api/batch-sentiment` | Batch sentiment analysis |
| POST | `/api/stress-test` | Monte Carlo stress scenarios |
| POST | `/api/qaoa` | QAOA portfolio optimization |
| POST | `/api/metrics` | Financial metrics calculation |
| GET | `/api/portfolio-summary` | Portfolio overview |
| GET | `/api/rl-summary` | RL agent summary |
| GET | `/api/gnn-summary` | Graph network summary |
| GET | `/api/fl-summary` | Federated learning summary |
| GET | `/api/nas-summary` | NAS results summary |
| GET | `/api/sentiment-portfolio` | Sentiment + holdings |

---

## Step 12: Start the Dashboard (Frontend)

```bash
cd dashboard
npm install        # First time only
npm run dev        # Starts on http://localhost:3000
```

For production build:
```bash
npm run build      # Output in dashboard/dist/
npm run preview    # Preview the built version
```

---

## Docker Deployment (Alternative)

Run everything in Docker (API + PostgreSQL):

```bash
docker-compose up -d
```

This starts:
- **finquant-api** on port 8000 (CPU-only PyTorch)
- **finquant-db** on port 5432 (PostgreSQL 16)

> **Note:** Docker uses CPU-only PyTorch. For GPU training, use the local venv setup.

---

## Project Structure

```
fqn1/
├── src/
│   ├── api/          # FastAPI REST API (Phase 13)
│   ├── data/         # Data pipeline & features (Phases 1-2)
│   ├── sentiment/    # FinBERT NLP (Phase 3)
│   ├── graph/        # Stock correlation graphs (Phase 4)
│   ├── models/       # T-GAT network (Phase 5)
│   ├── rl/           # PPO/SAC agents (Phases 6-7)
│   ├── gan/          # TimeGAN & stress testing (Phases 8-9)
│   ├── nas/          # DARTS architecture search (Phase 10)
│   ├── federated/    # Flower FL framework (Phase 11)
│   ├── quantum/      # Qiskit QAOA optimization (Phase 12)
│   └── utils/        # Config, logging, metrics, seed
├── tests/            # 232 tests across 14 files
├── dashboard/        # React + Vite + Tailwind frontend
├── configs/
│   └── base.yaml     # All hyperparameters
├── data/             # Stock CSVs & sentiment cache
├── models/           # Trained checkpoints (gitignored)
├── docs/             # PROGRESS.md, EXPLAINED.md, PRACTICAL_GUIDE.md
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container image
├── docker-compose.yml
└── .env              # Environment variables (create from template above)
```

---

## Key Configuration (configs/base.yaml)

- **Data split:** Train 2015-2021, Val 2022-2023, Test 2024-2025
- **India-specific:** Risk-free rate 7%, Transaction cost 0.1%, Slippage 0.05%
- **Trading days/year:** 248
- **FP16:** Enabled (mandatory for 4GB VRAM GPUs)
- **Seed:** 42 (reproducibility)

---

## Common Issues & Fixes

### 1. `torch.cuda.is_available()` returns False
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Ensure CUDA 12.1 toolkit is installed: `nvcc --version`
- Reinstall PyTorch with correct CUDA version

### 2. torch-geometric installation fails
- Install scatter/sparse first with explicit wheel URL (see Step 4)

### 3. PostgreSQL connection refused
- Ensure Docker is running: `docker ps`
- Or start the database: `docker-compose up db -d`
- Check port 5432 is not in use: `netstat -an | grep 5432`

### 4. FinBERT download fails
- Check internet connection
- Manually download from https://huggingface.co/ProsusAI/finbert
- Place in `~/.cache/huggingface/hub/models--ProsusAI--finbert/`

### 5. Tests fail with CUDA errors on CPU machine
- Some tests are marked `xfail` for GPU-specific assertions — this is expected
- Ensure `device: cpu` in `configs/base.yaml` if no GPU

### 6. npm install fails for dashboard
- Ensure Node.js 18+ is installed
- Try: `npm cache clean --force && npm install`

---

## W&B Experiment Tracking (Optional)

1. Create account at https://wandb.ai
2. Get API key from https://wandb.ai/settings
3. Add to `.env`: `WANDB_API_KEY=your_key_here`
4. Experiments will auto-log to project `finquant-nexus`

---

## Quick Verification Checklist

```bash
# 1. Python & venv
python --version                              # 3.11.x

# 2. PyTorch & CUDA
python -c "import torch; print(torch.cuda.is_available())"  # True (GPU) or False (CPU)

# 3. All imports work
python -c "import torch_geometric, transformers, qiskit, flwr, fastapi; print('All imports OK')"

# 4. Database
docker ps | grep finquant-db                  # Running

# 5. Tests
python -m pytest tests/ -v --tb=short         # 232 tests

# 6. API
curl http://localhost:8000/api/health          # {"status": "healthy"}

# 7. Dashboard
# Open http://localhost:3000 in browser
```

---

**Happy coding!**
