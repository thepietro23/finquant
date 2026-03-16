"""Phase 3 Tests: FinBERT Sentiment — model loading, prediction, aggregation.

Tests:
  T3.1: FinBERT model loads successfully (CPU mode for testing)
  T3.2: Positive text gets positive score
  T3.3: Negative text gets negative score
  T3.4: Neutral text gets near-zero score
  T3.5: Batch prediction works correctly
  T3.6: Daily aggregation computes correct averages
  T3.7: Sentiment series with decay fills missing days

Edge Cases:
  E3.1: Empty/short text returns neutral (0.0)
  E3.2: Sentiment decay converges to 0 over many days
  E3.3: Single headline aggregation
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# Skip all tests if transformers not installed or model can't be downloaded
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

pytestmark = pytest.mark.skipif(not HAS_TRANSFORMERS, reason='transformers not installed')


# ===========================
# Fixtures
# ===========================

@pytest.fixture(scope='module')
def finbert_loaded():
    """Load FinBERT once for all tests in this module (saves time)."""
    from src.sentiment.finbert import load_finbert, clear_model_cache
    try:
        model, tokenizer, device = load_finbert(device='cpu')
        yield model, tokenizer, device
    finally:
        clear_model_cache()


# ===========================
# Unit Tests
# ===========================

class TestFinBERTLoading:
    """T3.1: Model loads correctly."""

    def test_model_loads(self, finbert_loaded):
        """FinBERT model loads without error."""
        model, tokenizer, device = finbert_loaded
        assert model is not None
        assert tokenizer is not None
        assert device == 'cpu'

    def test_model_has_3_labels(self, finbert_loaded):
        """FinBERT outputs 3 classes (positive, negative, neutral)."""
        model, _, _ = finbert_loaded
        assert model.config.num_labels == 3


class TestSentimentPrediction:
    """T3.2-T3.4: Sentiment scoring accuracy."""

    def test_positive_text(self, finbert_loaded):
        """T3.2: Positive financial text gets positive score."""
        from src.sentiment.finbert import predict_sentiment
        result = predict_sentiment(
            "Company reports record profits, revenue up 25% year over year",
            *finbert_loaded
        )
        assert result['score'] > 0, f"Expected positive score, got {result['score']}"
        assert result['positive'] > result['negative']

    def test_negative_text(self, finbert_loaded):
        """T3.3: Negative financial text gets negative score."""
        from src.sentiment.finbert import predict_sentiment
        result = predict_sentiment(
            "Stock crashes 15% after company reports massive losses and debt default",
            *finbert_loaded
        )
        assert result['score'] < 0, f"Expected negative score, got {result['score']}"
        assert result['negative'] > result['positive']

    def test_neutral_text(self, finbert_loaded):
        """T3.4: Neutral text gets near-zero score."""
        from src.sentiment.finbert import predict_sentiment
        result = predict_sentiment(
            "The company held its annual general meeting on Monday",
            *finbert_loaded
        )
        # Neutral text: score should be closer to 0 than extremes
        assert abs(result['score']) < 0.8, f"Expected near-neutral, got {result['score']}"

    def test_score_range(self, finbert_loaded):
        """Scores always in [-1, +1] range."""
        from src.sentiment.finbert import predict_sentiment
        texts = [
            "Profits soar to all-time high",
            "Company goes bankrupt, stock delisted",
            "Board meeting scheduled for next week",
        ]
        for text in texts:
            result = predict_sentiment(text, *finbert_loaded)
            assert -1.0 <= result['score'] <= 1.0, f"Score {result['score']} out of range"
            assert 0.0 <= result['positive'] <= 1.0
            assert 0.0 <= result['negative'] <= 1.0
            assert 0.0 <= result['neutral'] <= 1.0
            # Probabilities should sum to ~1
            total = result['positive'] + result['negative'] + result['neutral']
            assert abs(total - 1.0) < 0.01, f"Probs sum to {total}"


class TestBatchPrediction:
    """T3.5: Batch processing works."""

    def test_batch_correct_count(self, finbert_loaded):
        """Batch returns same number of results as inputs."""
        from src.sentiment.finbert import predict_batch
        texts = [
            "Revenue increased by 20%",
            "Stock price declined sharply",
            "Annual meeting concluded",
        ]
        results = predict_batch(texts, batch_size=2)
        assert len(results) == 3

    def test_batch_matches_individual(self, finbert_loaded):
        """Batch results match individual predictions."""
        from src.sentiment.finbert import predict_sentiment, predict_batch
        texts = [
            "Strong quarterly earnings reported",
            "Major fraud detected in accounts",
        ]
        individual = [predict_sentiment(t, *finbert_loaded) for t in texts]
        batch = predict_batch(texts, batch_size=16)

        for i in range(len(texts)):
            assert abs(individual[i]['score'] - batch[i]['score']) < 0.01, \
                f"Mismatch at index {i}: {individual[i]['score']} vs {batch[i]['score']}"


class TestDailyAggregation:
    """T3.6: Daily sentiment aggregation."""

    def test_aggregation(self, finbert_loaded):
        """Multiple headlines per day aggregated correctly."""
        from src.sentiment.finbert import aggregate_daily_sentiment
        headlines = {
            '2024-01-15': [
                "Company reports strong earnings",
                "Stock hits new all-time high",
            ],
            '2024-01-16': [
                "Market crashes on global fears",
            ],
        }
        daily = aggregate_daily_sentiment(headlines)
        assert '2024-01-15' in daily
        assert '2024-01-16' in daily
        assert daily['2024-01-15']['num_headlines'] == 2
        assert daily['2024-01-16']['num_headlines'] == 1
        # Day with positive news should have positive avg
        assert daily['2024-01-15']['avg_score'] > 0


class TestSentimentSeries:
    """T3.7: Sentiment time series with decay."""

    def test_decay_fills_gaps(self):
        """Missing days filled with decayed previous value."""
        from src.sentiment.finbert import build_sentiment_series

        dates = pd.bdate_range('2024-01-01', periods=10)
        daily = {
            '2024-01-01': {'avg_score': 0.8, 'num_headlines': 3},
            # Days 2-9 have no news — should decay
        }
        series = build_sentiment_series(daily, dates, decay_factor=0.9)

        assert len(series) == 10
        assert series.isna().sum() == 0  # No NaN
        assert series.iloc[0] == pytest.approx(0.8, abs=0.01)
        # Day 2 should be 0.8 * 0.9 = 0.72
        assert series.iloc[1] == pytest.approx(0.8 * 0.9, abs=0.01)
        # Each subsequent day decays further
        for i in range(1, len(series)):
            assert abs(series.iloc[i]) <= abs(series.iloc[i - 1]) + 0.01

    def test_new_headline_resets_decay(self):
        """New headline on a day replaces decayed value."""
        from src.sentiment.finbert import build_sentiment_series

        dates = pd.bdate_range('2024-01-01', periods=5)
        daily = {
            '2024-01-01': {'avg_score': 0.5, 'num_headlines': 1},
            # Day 2, 3 decay
            '2024-01-04': {'avg_score': -0.6, 'num_headlines': 2},
            # Day 5 decays from -0.6
        }
        series = build_sentiment_series(daily, dates, decay_factor=0.9)
        # Day 4 (index 3) should be -0.6 (new headline)
        assert series.iloc[3] == pytest.approx(-0.6, abs=0.01)


class TestSentimentMatrix:
    """Sentiment matrix for multiple stocks."""

    def test_matrix_shape(self):
        """Matrix shape is (n_stocks, n_timesteps)."""
        from src.sentiment.finbert import build_sentiment_matrix

        dates = pd.bdate_range('2024-01-01', periods=20)
        tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        sentiment_data = {
            'RELIANCE.NS': {'2024-01-01': {'avg_score': 0.5, 'num_headlines': 2}},
            'TCS.NS': {'2024-01-02': {'avg_score': -0.3, 'num_headlines': 1}},
        }
        matrix = build_sentiment_matrix(sentiment_data, dates, tickers)
        assert matrix.shape == (3, 20)
        assert matrix.dtype == np.float32


# ===========================
# Edge Cases
# ===========================

class TestEdgeCases:
    """Edge case handling."""

    def test_empty_text(self, finbert_loaded):
        """E3.1: Empty text returns neutral."""
        from src.sentiment.finbert import predict_sentiment
        result = predict_sentiment("", *finbert_loaded)
        assert result['score'] == 0.0
        assert result['neutral'] == 1.0

    def test_short_text(self, finbert_loaded):
        """E3.1: Very short text (<5 chars) returns neutral."""
        from src.sentiment.finbert import predict_sentiment
        result = predict_sentiment("Hi", *finbert_loaded)
        assert result['score'] == 0.0

    def test_decay_converges_to_zero(self):
        """E3.2: Sentiment decays to near-zero over many days."""
        from src.sentiment.finbert import build_sentiment_series

        dates = pd.bdate_range('2024-01-01', periods=100)
        daily = {
            '2024-01-01': {'avg_score': 1.0, 'num_headlines': 5},
        }
        series = build_sentiment_series(daily, dates, decay_factor=0.95)
        # After 100 days: 1.0 * 0.95^99 ≈ 0.006
        assert abs(series.iloc[-1]) < 0.01

    def test_single_headline_aggregation(self, finbert_loaded):
        """E3.3: Single headline aggregation works."""
        from src.sentiment.finbert import aggregate_daily_sentiment
        daily = aggregate_daily_sentiment({
            '2024-01-01': ["Revenue up 30% in Q3"],
        })
        assert '2024-01-01' in daily
        assert daily['2024-01-01']['num_headlines'] == 1


class TestNewsFetcher:
    """News fetcher utility tests (no network calls)."""

    def test_company_name_lookup(self):
        """Ticker to company name mapping works."""
        from src.sentiment.news_fetcher import get_company_name
        assert 'Reliance' in get_company_name('RELIANCE.NS')
        assert 'TCS' in get_company_name('TCS.NS')
        assert 'HDFC' in get_company_name('HDFCBANK.NS')

    def test_unknown_ticker_fallback(self):
        """Unknown ticker returns cleaned ticker name."""
        from src.sentiment.news_fetcher import get_company_name
        result = get_company_name('UNKNOWN.NS')
        assert result == 'UNKNOWN'

    def test_sentiment_db_init(self):
        """SQLite DB initializes without error."""
        from src.sentiment.news_fetcher import init_sentiment_db
        import tempfile
        db_path = os.path.join(tempfile.gettempdir(), 'test_sentiment.db')
        try:
            conn = init_sentiment_db(db_path)
            assert conn is not None
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
