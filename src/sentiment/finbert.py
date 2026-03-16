"""FinBERT sentiment analysis for financial text.

Uses ProsusAI/finbert — a BERT model fine-tuned on financial text.
Output: sentiment score per headline = P(positive) - P(negative), range [-1, +1].

Pipeline:
  1. Load pre-trained FinBERT model
  2. Score individual headlines
  3. Aggregate daily sentiment per stock
  4. Apply decay for days without news
  5. Build sentiment matrix aligned with feature dates
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger('finbert')

# Global model cache (singleton pattern — load once, use everywhere)
_MODEL_CACHE = {}


def load_finbert(device=None):
    """Load ProsusAI/finbert model and tokenizer.

    Caches globally to avoid reloading. Uses FP16 on GPU.

    Returns:
        tuple: (model, tokenizer, device)
    """
    if 'model' in _MODEL_CACHE:
        return _MODEL_CACHE['model'], _MODEL_CACHE['tokenizer'], _MODEL_CACHE['device']

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = get_config('sentiment')
    model_name = cfg.get('model', 'ProsusAI/finbert')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f'Loading FinBERT ({model_name}) on {device}...')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # FP16 on GPU for memory efficiency (4GB VRAM constraint)
    if device == 'cuda':
        model = model.half()
        logger.info('FinBERT loaded in FP16 mode')

    _MODEL_CACHE['model'] = model
    _MODEL_CACHE['tokenizer'] = tokenizer
    _MODEL_CACHE['device'] = device

    # Log VRAM usage
    if device == 'cuda':
        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
        logger.info(f'FinBERT VRAM usage: {mem_mb:.0f} MB')

    return model, tokenizer, device


def clear_model_cache():
    """Clear cached model (free VRAM)."""
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_sentiment(text, model=None, tokenizer=None, device=None):
    """Get sentiment score for a single text.

    Args:
        text: Financial text/headline string.
        model, tokenizer, device: Optional pre-loaded model components.

    Returns:
        dict: {score: float [-1,+1], positive: float, negative: float, neutral: float}
    """
    if not text or len(text.strip()) < 5:
        return {'score': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

    if model is None or tokenizer is None:
        model, tokenizer, device = load_finbert(device)

    cfg = get_config('sentiment')
    max_length = cfg.get('max_length', 128)

    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=max_length, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if device == 'cuda':
            with autocast(dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    # FinBERT labels: 0=positive, 1=negative, 2=neutral
    probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()[0]

    positive = float(probs[0])
    negative = float(probs[1])
    neutral = float(probs[2])
    score = positive - negative  # Range: [-1, +1]

    return {
        'score': score,
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
    }


def predict_batch(texts, batch_size=16):
    """Score multiple texts efficiently in batches.

    Args:
        texts: List of text strings.
        batch_size: Batch size for inference.

    Returns:
        List of dicts (same format as predict_sentiment).
    """
    if not texts:
        return []

    model, tokenizer, device = load_finbert()
    cfg = get_config('sentiment')
    max_length = cfg.get('max_length', 128)

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Filter empty/short texts
        valid_mask = [bool(t and len(t.strip()) >= 5) for t in batch]
        valid_texts = [t for t, v in zip(batch, valid_mask) if v]

        if not valid_texts:
            results.extend([{'score': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
                           for _ in batch])
            continue

        inputs = tokenizer(valid_texts, return_tensors='pt', truncation=True,
                           max_length=max_length, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if device == 'cuda':
                with autocast(dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

        # Map results back to original batch (including empty texts)
        valid_idx = 0
        for is_valid in valid_mask:
            if is_valid:
                p = probs[valid_idx]
                results.append({
                    'score': float(p[0] - p[1]),
                    'positive': float(p[0]),
                    'negative': float(p[1]),
                    'neutral': float(p[2]),
                })
                valid_idx += 1
            else:
                results.append({'score': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0})

    return results


# ---------------------------------------------------------------------------
# Daily sentiment aggregation
# ---------------------------------------------------------------------------

def aggregate_daily_sentiment(headlines_by_date):
    """Aggregate headline sentiments into daily scores.

    Args:
        headlines_by_date: Dict {date_str: [headline_strings]}

    Returns:
        Dict {date_str: {avg_score, num_headlines}}
    """
    daily = {}
    all_texts = []
    date_headline_map = []

    for date_str, headlines in headlines_by_date.items():
        for h in headlines:
            all_texts.append(h)
            date_headline_map.append(date_str)

    if not all_texts:
        return daily

    # Batch predict all headlines at once
    results = predict_batch(all_texts)

    # Group by date
    date_scores = {}
    for date_str, result in zip(date_headline_map, results):
        if date_str not in date_scores:
            date_scores[date_str] = []
        date_scores[date_str].append(result['score'])

    for date_str, scores in date_scores.items():
        daily[date_str] = {
            'avg_score': float(np.mean(scores)),
            'num_headlines': len(scores),
        }

    return daily


def build_sentiment_series(daily_sentiments, dates, decay_factor=None):
    """Build continuous sentiment time series with decay for missing days.

    Args:
        daily_sentiments: Dict {date_str: {avg_score, num_headlines}} or DataFrame.
        dates: DatetimeIndex — target dates to align with.
        decay_factor: Decay multiplier for days without news (default from config).

    Returns:
        pd.Series aligned with dates, NaN-free.
    """
    cfg = get_config('sentiment')
    if decay_factor is None:
        decay_factor = cfg.get('decay_factor', 0.95)

    # Convert to Series
    if isinstance(daily_sentiments, dict):
        scores = {}
        for date_str, info in daily_sentiments.items():
            scores[pd.Timestamp(date_str)] = info['avg_score']
        sentiment = pd.Series(scores, dtype=float).sort_index()
    elif isinstance(daily_sentiments, pd.DataFrame):
        sentiment = daily_sentiments.set_index('date')['avg_score'].sort_index()
    else:
        sentiment = daily_sentiments

    # Reindex to target dates
    aligned = sentiment.reindex(dates)

    # Forward-fill with decay
    last_val = 0.0
    filled = []
    for val in aligned.values:
        if np.isnan(val):
            last_val = last_val * decay_factor
        else:
            last_val = val
        filled.append(last_val)

    return pd.Series(filled, index=dates, name='sentiment')


def build_sentiment_matrix(sentiment_by_ticker, dates, tickers):
    """Build 2D sentiment matrix (n_stocks x n_timesteps).

    Args:
        sentiment_by_ticker: Dict {ticker: daily_sentiments_dict}
        dates: DatetimeIndex
        tickers: List of ticker strings

    Returns:
        np.ndarray of shape (n_stocks, n_timesteps), dtype float32
    """
    cfg = get_config('sentiment')
    decay = cfg.get('decay_factor', 0.95)

    n_stocks = len(tickers)
    n_time = len(dates)
    matrix = np.zeros((n_stocks, n_time), dtype=np.float32)

    for i, ticker in enumerate(tickers):
        if ticker in sentiment_by_ticker:
            series = build_sentiment_series(
                sentiment_by_ticker[ticker], dates, decay
            )
            matrix[i] = series.values
        # else: stays 0 (no news = neutral)

    return matrix
