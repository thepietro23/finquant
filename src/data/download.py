"""Download NIFTY 50 stock data from Yahoo Finance via yfinance."""

import os
import time

import pandas as pd

# Fix SSL certificate issues (corporate/college proxy with self-signed certs)
# Must be done BEFORE importing yfinance
try:
    from curl_cffi import requests as _curl_requests
    _orig_init = _curl_requests.Session.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs['verify'] = False
        _orig_init(self, *args, **kwargs)

    _curl_requests.Session.__init__ = _patched_init
except ImportError:
    pass

import yfinance as yf

from src.data.stocks import get_all_tickers, NIFTY_INDEX
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger('download')


def download_stock(ticker, start_date, end_date, retries=5, backoff=3.0):
    """Download single stock with retry + exponential backoff.

    Returns DataFrame or None on failure.
    """
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date,
                             progress=False, auto_adjust=False)
            if df is None or df.empty:
                wait = min(backoff ** attempt, 60)
                logger.warning(f'{ticker}: empty/failed (attempt {attempt + 1}/{retries}). Waiting {wait:.0f}s')
                time.sleep(wait)
                continue

            # Flatten MultiIndex columns if present (yfinance sometimes returns multi-level)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure standard columns
            required = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.warning(f'{ticker}: missing columns {missing}')
                return None

            df.index.name = 'Date'
            logger.info(f'{ticker}: downloaded {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})')
            return df

        except Exception as e:
            wait = min(backoff ** attempt, 60)
            logger.warning(f'{ticker}: attempt {attempt + 1}/{retries} error — {type(e).__name__}: {e}. Waiting {wait:.0f}s')
            time.sleep(wait)

    logger.error(f'{ticker}: all {retries} attempts failed')
    return None


def download_nifty_data(data_dir='data', start_date=None, end_date=None):
    """Download all NIFTY 50 stocks + NIFTY index.

    Returns dict with stats: {success: int, failed: int, failed_tickers: list}
    """
    cfg = get_config('data')
    start_date = start_date or cfg['start_date']
    end_date = end_date or cfg['end_date']

    os.makedirs(data_dir, exist_ok=True)

    tickers = get_all_tickers()
    success = 0
    failed = 0
    failed_tickers = []
    all_close = {}

    # Download individual stocks (with delay to avoid rate limiting)
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(1.0)  # 1 sec between downloads to avoid Yahoo rate limit
        df = download_stock(ticker, start_date, end_date)
        if df is not None and not df.empty:
            # Save per-stock CSV
            safe_name = ticker.replace('^', '').replace('.', '_')
            csv_path = os.path.join(data_dir, f'{safe_name}.csv')
            df.to_csv(csv_path)
            all_close[ticker] = df['Adj Close']
            success += 1
        else:
            failed += 1
            failed_tickers.append(ticker)

    # Download NIFTY 50 Index
    logger.info(f'Downloading NIFTY 50 Index ({NIFTY_INDEX})...')
    idx_df = download_stock(NIFTY_INDEX, start_date, end_date)
    if idx_df is not None and not idx_df.empty:
        idx_df.to_csv(os.path.join(data_dir, 'NIFTY50_INDEX.csv'))
        logger.info(f'NIFTY index saved: {len(idx_df)} rows')
    else:
        logger.error('NIFTY 50 Index download failed!')

    # Save combined Adj Close prices
    if all_close:
        combined = pd.DataFrame(all_close)
        combined.to_csv(os.path.join(data_dir, 'all_close_prices.csv'))
        logger.info(f'Combined close prices saved: {combined.shape}')

    logger.info(f'Download complete: {success} success, {failed} failed')
    if failed_tickers:
        logger.warning(f'Failed tickers: {failed_tickers}')

    return {'success': success, 'failed': failed, 'failed_tickers': failed_tickers}
