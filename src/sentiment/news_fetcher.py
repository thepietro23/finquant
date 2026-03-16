"""Fetch financial news headlines from Google News RSS for NIFTY 50 stocks.

Free, no API key needed. Returns headlines per stock per date range.
Limitations: Google News RSS gives ~100 recent results, not historical archive.
For historical sentiment, we'll use Financial PhraseBank dataset.
"""

import time
import sqlite3
import os
from datetime import datetime, timedelta
from urllib.parse import quote

import feedparser
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger('news_fetcher')

# Mapping from ticker to human-readable company name for news search
TICKER_TO_COMPANY = {
    'TCS.NS': 'TCS Tata Consultancy',
    'INFY.NS': 'Infosys',
    'HCLTECH.NS': 'HCL Technologies',
    'WIPRO.NS': 'Wipro',
    'TECHM.NS': 'Tech Mahindra',
    'HDFCBANK.NS': 'HDFC Bank',
    'ICICIBANK.NS': 'ICICI Bank',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'SBIN.NS': 'State Bank of India SBI',
    'AXISBANK.NS': 'Axis Bank',
    'INDUSINDBK.NS': 'IndusInd Bank',
    'RELIANCE.NS': 'Reliance Industries',
    'ONGC.NS': 'ONGC',
    'NTPC.NS': 'NTPC',
    'POWERGRID.NS': 'Power Grid Corporation',
    'MARUTI.NS': 'Maruti Suzuki',
    'M&M.NS': 'Mahindra and Mahindra',
    'TATAMOTORS.NS': 'Tata Motors',
    'BAJAJ-AUTO.NS': 'Bajaj Auto',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ITC.NS': 'ITC Limited',
    'NESTLEIND.NS': 'Nestle India',
    'BRITANNIA.NS': 'Britannia Industries',
    'SUNPHARMA.NS': 'Sun Pharma',
    'DRREDDY.NS': 'Dr Reddys Laboratories',
    'CIPLA.NS': 'Cipla',
    'DIVISLAB.NS': 'Divi Laboratories',
    'TATASTEEL.NS': 'Tata Steel',
    'HINDALCO.NS': 'Hindalco Industries',
    'JSWSTEEL.NS': 'JSW Steel',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'GRASIM.NS': 'Grasim Industries',
    'LT.NS': 'Larsen Toubro',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'HDFCLIFE.NS': 'HDFC Life Insurance',
    'SBILIFE.NS': 'SBI Life Insurance',
    'TITAN.NS': 'Titan Company',
    'ASIANPAINT.NS': 'Asian Paints',
    'ADANIENT.NS': 'Adani Enterprises',
    'ADANIPORTS.NS': 'Adani Ports',
    'HEROMOTOCO.NS': 'Hero MotoCorp',
    'EICHERMOT.NS': 'Eicher Motors Royal Enfield',
    'APOLLOHOSP.NS': 'Apollo Hospitals',
}


def get_company_name(ticker):
    """Get search-friendly company name for a ticker."""
    return TICKER_TO_COMPANY.get(ticker, ticker.replace('.NS', ''))


def fetch_google_news(query, max_results=20):
    """Fetch news headlines from Google News RSS.

    Args:
        query: Search query string (e.g., "Reliance Industries stock").
        max_results: Maximum headlines to return.

    Returns:
        List of dicts: [{title, published, link}, ...]
    """
    encoded = quote(query)
    url = f'https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en'

    try:
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:max_results]:
            published = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])

            results.append({
                'title': entry.get('title', ''),
                'published': published,
                'link': entry.get('link', ''),
            })
        return results
    except Exception as e:
        logger.warning(f'Google News fetch failed for "{query}": {e}')
        return []


def fetch_stock_news(ticker, max_results=20):
    """Fetch news for a specific stock.

    Searches with company name + "stock NSE" for relevant results.
    """
    company = get_company_name(ticker)
    query = f'{company} stock NSE'
    headlines = fetch_google_news(query, max_results)
    logger.debug(f'{ticker}: fetched {len(headlines)} headlines')
    return headlines


def fetch_all_stock_news(tickers=None, max_per_stock=15, delay=2.0):
    """Fetch news for multiple stocks with rate limiting.

    Args:
        tickers: List of tickers. Defaults to all NIFTY 50.
        max_per_stock: Max headlines per stock.
        delay: Seconds between requests (avoid rate limiting).

    Returns:
        Dict: {ticker: [headline_dicts]}
    """
    if tickers is None:
        from src.data.stocks import get_all_tickers
        tickers = get_all_tickers()

    all_news = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(delay)
        headlines = fetch_stock_news(ticker, max_per_stock)
        all_news[ticker] = headlines
        logger.info(f'[{i+1}/{len(tickers)}] {ticker}: {len(headlines)} headlines')

    return all_news


# ---------------------------------------------------------------------------
# SQLite cache for sentiment scores (avoid re-computing)
# ---------------------------------------------------------------------------

def init_sentiment_db(db_path=None):
    """Initialize SQLite database for caching sentiment scores."""
    if db_path is None:
        db_path = get_config('sentiment').get('cache_db', 'data/sentiment.db')

    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            ticker TEXT,
            date TEXT,
            headline TEXT,
            score REAL,
            positive REAL,
            negative REAL,
            neutral REAL,
            PRIMARY KEY (ticker, date, headline)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            ticker TEXT,
            date TEXT,
            avg_score REAL,
            num_headlines INTEGER,
            PRIMARY KEY (ticker, date)
        )
    ''')
    conn.commit()
    return conn


def save_sentiment_score(conn, ticker, date, headline, score, probs):
    """Save individual headline sentiment to cache."""
    conn.execute('''
        INSERT OR REPLACE INTO sentiment_scores
        (ticker, date, headline, score, positive, negative, neutral)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, date, headline[:500], score,
          probs.get('positive', 0), probs.get('negative', 0), probs.get('neutral', 0)))
    conn.commit()


def save_daily_sentiment(conn, ticker, date, avg_score, num_headlines):
    """Save daily aggregated sentiment."""
    conn.execute('''
        INSERT OR REPLACE INTO daily_sentiment
        (ticker, date, avg_score, num_headlines)
        VALUES (?, ?, ?, ?)
    ''', (ticker, date, avg_score, num_headlines))
    conn.commit()


def load_daily_sentiments(db_path=None):
    """Load all daily sentiments as DataFrame.

    Returns:
        DataFrame with columns [ticker, date, avg_score, num_headlines]
    """
    if db_path is None:
        db_path = get_config('sentiment').get('cache_db', 'data/sentiment.db')

    if not os.path.exists(db_path):
        return pd.DataFrame(columns=['ticker', 'date', 'avg_score', 'num_headlines'])

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM daily_sentiment', conn)
    conn.close()

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df
