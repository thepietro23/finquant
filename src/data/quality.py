"""Data quality checks for NIFTY stock data — NaN, duplicates, splits, outliers."""

import os

import pandas as pd
import numpy as np

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('quality')


class DataQualityChecker:
    """Run quality checks on stock DataFrames and generate reports."""

    def __init__(self, min_days=None, max_nan_pct=None):
        cfg = get_config('data')
        self.min_days = min_days or cfg.get('min_trading_days', 1000)
        self.max_nan_pct = max_nan_pct or cfg.get('max_nan_pct', 0.05)
        self.report = {}

    def check_stock(self, df, ticker):
        """Run all quality checks on a single stock DataFrame.

        Returns True if all critical checks pass, False otherwise.
        """
        issues = []

        # 1. Sufficient data?
        if len(df) < self.min_days:
            issues.append(f'Too few rows: {len(df)} < {self.min_days}')

        # 2. NaN check
        if len(df) > 0:
            nan_pct = df.isnull().sum().max() / len(df)
        else:
            nan_pct = 1.0
        if nan_pct > self.max_nan_pct:
            issues.append(f'Too many NaNs: {nan_pct:.1%}')

        # 3. Duplicate dates?
        if hasattr(df.index, 'duplicated'):
            dupes = df.index.duplicated().sum()
            if dupes > 0:
                issues.append(f'Duplicate dates: {dupes}')

        # 4. Zero/negative prices?
        price_col = 'Close' if 'Close' in df.columns else df.columns[0]
        if (df[price_col] <= 0).any():
            issues.append('Zero or negative prices found')

        # 5. Extreme daily returns (>50% in a day = likely data error)
        returns = df[price_col].pct_change().abs()
        extreme = (returns > 0.50).sum()
        if extreme > 0:
            issues.append(f'Extreme returns (>50%): {extreme} days')

        # 6. Volume zero days (>1% is suspicious)
        if 'Volume' in df.columns:
            zero_vol = (df['Volume'] == 0).sum()
            if zero_vol > len(df) * 0.01:
                issues.append(f'Zero volume days: {zero_vol} ({zero_vol/len(df):.1%})')

        # 7. Chronological order?
        if not df.index.is_monotonic_increasing:
            issues.append('Dates not in chronological order')

        self.report[ticker] = {
            'rows': len(df),
            'date_range': f'{df.index[0]} to {df.index[-1]}' if len(df) > 0 else 'empty',
            'nan_pct': f'{nan_pct:.2%}',
            'issues': issues,
            'status': 'PASS' if len(issues) == 0 else 'FAIL',
        }

        if issues:
            logger.warning(f'{ticker}: FAIL — {issues}')
        else:
            logger.info(f'{ticker}: PASS ({len(df)} rows)')

        return len(issues) == 0

    def clean_stock(self, df):
        """Clean a stock DataFrame: sort, dedup, forward-fill, dropna.

        Returns cleaned DataFrame.
        """
        df = df.copy()
        # Sort by date
        df = df.sort_index()
        # Remove duplicate dates
        df = df[~df.index.duplicated(keep='first')]
        # Forward-fill gaps (NSE holidays etc.)
        df = df.ffill()
        # Drop remaining NaN rows (start of series)
        df = df.dropna()
        return df

    def check_all(self, data_dir='data'):
        """Run quality checks on all stock CSVs in a directory.

        Returns (passed_count, total_count).
        """
        csv_files = [f for f in os.listdir(data_dir)
                     if f.endswith('.csv') and f not in ('all_close_prices.csv', 'NIFTY50_INDEX.csv')]

        for fname in sorted(csv_files):
            path = os.path.join(data_dir, fname)
            ticker = fname.replace('.csv', '').replace('_NS', '.NS').replace('_', '.')
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                self.check_stock(df, ticker)
            except Exception as e:
                logger.error(f'{fname}: failed to read — {e}')
                self.report[ticker] = {'status': 'ERROR', 'issues': [str(e)]}

        passed = sum(1 for v in self.report.values() if v['status'] == 'PASS')
        total = len(self.report)
        return passed, total

    def print_report(self):
        """Print quality report summary."""
        passed = sum(1 for v in self.report.values() if v['status'] == 'PASS')
        total = len(self.report)
        print(f'\n{"=" * 50}')
        print(f'  Data Quality Report: {passed}/{total} PASSED')
        print(f'{"=" * 50}')
        for ticker, info in sorted(self.report.items()):
            status_icon = 'PASS' if info['status'] == 'PASS' else 'FAIL'
            rows = info.get('rows', '?')
            print(f'  [{status_icon}] {ticker}: {rows} rows')
            for issue in info.get('issues', []):
                print(f'         - {issue}')
        print()
