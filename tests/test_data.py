"""Phase 1 Tests: Data Pipeline — download, quality, stocks registry."""

import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

from src.data.stocks import get_all_tickers, get_sector, get_sector_pairs, get_supply_chain_pairs


class TestStockRegistry:
    """Test the NIFTY 50 stock registry."""

    def test_stock_list_count(self):
        """T1.1: 40+ stocks in registry."""
        tickers = get_all_tickers()
        assert len(tickers) >= 45, f'Expected 45+ tickers, got {len(tickers)}'

    def test_sector_mapping(self):
        """Sector lookup works for known stocks."""
        assert get_sector('TCS.NS') == 'IT'
        assert get_sector('HDFCBANK.NS') == 'Banking'
        assert get_sector('RELIANCE.NS') == 'Energy'
        assert get_sector('UNKNOWN.NS') == 'Unknown'

    def test_sector_pairs(self):
        """Sector pairs generated correctly."""
        pairs = get_sector_pairs()
        assert len(pairs) > 0
        # Banking has 6 stocks -> C(6,2) = 15 pairs
        banking_pairs = [(a, b) for a, b in pairs
                         if get_sector(a) == 'Banking' and get_sector(b) == 'Banking']
        assert len(banking_pairs) == 15

    def test_supply_chain_exists(self):
        """Supply chain mapping has entries."""
        pairs = get_supply_chain_pairs()
        assert len(pairs) >= 20


class TestDataDownload:
    """Test downloaded CSV files."""

    def test_csv_files_exist(self):
        """T1.1: 40+ stock CSV files in data/ folder."""
        if not os.path.isdir(DATA_DIR):
            pytest.skip('Data not downloaded yet — run download first')
        csv_files = [f for f in os.listdir(DATA_DIR)
                     if f.endswith('.csv') and f not in ('all_close_prices.csv', 'NIFTY50_INDEX.csv')]
        assert len(csv_files) >= 40, f'Expected 40+ CSVs, got {len(csv_files)}'

    def test_csv_columns(self):
        """T1.2: Each CSV has required OHLCV columns."""
        reliance_path = os.path.join(DATA_DIR, 'RELIANCE_NS.csv')
        if not os.path.exists(reliance_path):
            pytest.skip('RELIANCE data not downloaded yet')
        df = pd.read_csv(reliance_path, index_col=0)
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            assert col in df.columns, f'Missing column: {col}'

    def test_nifty_index(self):
        """T1.3: NIFTY50_INDEX.csv exists with 1000+ rows."""
        idx_path = os.path.join(DATA_DIR, 'NIFTY50_INDEX.csv')
        if not os.path.exists(idx_path):
            pytest.skip('NIFTY index not downloaded yet')
        df = pd.read_csv(idx_path, index_col=0)
        assert len(df) >= 1000, f'NIFTY index has only {len(df)} rows'

    def test_combined_close_prices(self):
        """T1.4: all_close_prices.csv has 40+ columns."""
        combined_path = os.path.join(DATA_DIR, 'all_close_prices.csv')
        if not os.path.exists(combined_path):
            pytest.skip('Combined prices not created yet')
        df = pd.read_csv(combined_path, index_col=0)
        assert df.shape[1] >= 40, f'Expected 40+ columns, got {df.shape[1]}'


class TestDataQuality:
    """Test data quality checks."""

    def _load_reliance(self):
        path = os.path.join(DATA_DIR, 'RELIANCE_NS.csv')
        if not os.path.exists(path):
            pytest.skip('RELIANCE data not downloaded yet')
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def test_quality_check_passes(self):
        """T1.5: Quality checker passes on RELIANCE."""
        from src.data.quality import DataQualityChecker
        df = self._load_reliance()
        qc = DataQualityChecker()
        assert qc.check_stock(df, 'RELIANCE.NS') is True

    def test_clean_removes_nan(self):
        """T1.6: Clean function reduces NaN to 0."""
        from src.data.quality import DataQualityChecker
        df = self._load_reliance()
        qc = DataQualityChecker()
        clean = qc.clean_stock(df)
        assert clean.isnull().sum().sum() == 0

    def test_no_duplicates_after_clean(self):
        """T1.7: No duplicate dates after cleaning."""
        from src.data.quality import DataQualityChecker
        df = self._load_reliance()
        qc = DataQualityChecker()
        clean = qc.clean_stock(df)
        assert clean.index.duplicated().sum() == 0

    def test_date_range(self):
        """T1.8: Date range covers 2015 to 2025."""
        df = self._load_reliance()
        start_year = df.index[0].year if hasattr(df.index[0], 'year') else int(str(df.index[0])[:4])
        end_year = df.index[-1].year if hasattr(df.index[-1], 'year') else int(str(df.index[-1])[:4])
        assert start_year <= 2015, f'Data starts too late: {start_year}'
        assert end_year >= 2024, f'Data ends too early: {end_year}'
