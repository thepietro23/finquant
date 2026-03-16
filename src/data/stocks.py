"""NIFTY 50 stock registry — sector mapping, supply chain edges, utility functions."""

# Sector -> list of tickers (yfinance format with .NS suffix)
NIFTY50 = {
    'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS',
                'AXISBANK.NS', 'INDUSINDBK.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS'],
    'Auto': ['MARUTI.NS', 'M&M.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS'],
    'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
    'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS'],
    'Infra': ['ULTRACEMCO.NS', 'GRASIM.NS', 'LT.NS'],
    'Telecom': ['BHARTIARTL.NS'],
    'Finance': ['BAJFINANCE.NS', 'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'SBILIFE.NS'],
    'Others': ['TITAN.NS', 'ASIANPAINT.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
               'HEROMOTOCO.NS', 'EICHERMOT.NS', 'APOLLOHOSP.NS'],
}

NIFTY_INDEX = '^NSEI'

# Supply chain edges — manual mapping of known business relationships
# Format: (supplier/upstream, consumer/downstream)
SUPPLY_CHAIN_EDGES = [
    # Steel -> Auto (steel is raw material for cars)
    ('TATASTEEL.NS', 'TATAMOTORS.NS'),
    ('TATASTEEL.NS', 'MARUTI.NS'),
    ('JSWSTEEL.NS', 'MARUTI.NS'),
    ('JSWSTEEL.NS', 'M&M.NS'),
    ('HINDALCO.NS', 'MARUTI.NS'),       # Aluminium -> Auto
    ('HINDALCO.NS', 'BAJAJ-AUTO.NS'),
    # Energy -> Industrial (fuel/power for manufacturing)
    ('RELIANCE.NS', 'ONGC.NS'),          # Both in energy value chain
    ('NTPC.NS', 'POWERGRID.NS'),         # Power generation -> distribution
    ('RELIANCE.NS', 'TATAMOTORS.NS'),    # Fuel for auto
    # IT services — cross-competition/talent pool
    ('TCS.NS', 'INFY.NS'),
    ('INFY.NS', 'WIPRO.NS'),
    ('WIPRO.NS', 'HCLTECH.NS'),
    ('HCLTECH.NS', 'TECHM.NS'),
    # Banking — interbank/cross-lending
    ('HDFCBANK.NS', 'ICICIBANK.NS'),
    ('SBIN.NS', 'AXISBANK.NS'),
    # Infra -> Cement (construction needs cement)
    ('LT.NS', 'ULTRACEMCO.NS'),
    ('LT.NS', 'GRASIM.NS'),
    # FMCG distribution chain
    ('ITC.NS', 'BRITANNIA.NS'),          # Both FMCG, shared distribution
    ('HINDUNILVR.NS', 'NESTLEIND.NS'),
    # Finance — insurance + banking ecosystem
    ('HDFCBANK.NS', 'HDFCLIFE.NS'),
    ('SBIN.NS', 'SBILIFE.NS'),
    ('BAJFINANCE.NS', 'BAJAJFINSV.NS'),
    # Telecom -> IT (infra for IT services)
    ('BHARTIARTL.NS', 'TCS.NS'),
    ('BHARTIARTL.NS', 'INFY.NS'),
    # Pharma — shared API/raw materials
    ('SUNPHARMA.NS', 'CIPLA.NS'),
    ('DRREDDY.NS', 'DIVISLAB.NS'),       # DIVISLAB supplies APIs to pharma
]


def get_all_tickers():
    """Return flat list of all NIFTY 50 tickers."""
    return [t for stocks in NIFTY50.values() for t in stocks]


def get_sector(ticker):
    """Get sector name for a ticker. Returns 'Unknown' if not found."""
    for sector, stocks in NIFTY50.items():
        if ticker in stocks:
            return sector
    return 'Unknown'


def get_sector_pairs():
    """Return all (stock_a, stock_b) pairs within same sector — for GNN sector edges."""
    pairs = []
    for stocks in NIFTY50.values():
        for i in range(len(stocks)):
            for j in range(i + 1, len(stocks)):
                pairs.append((stocks[i], stocks[j]))
    return pairs


def get_supply_chain_pairs():
    """Return supply chain edges as list of (stock_a, stock_b) tuples."""
    return SUPPLY_CHAIN_EDGES.copy()


def get_ticker_to_index():
    """Return dict: ticker -> integer index. Consistent ordering for GNN."""
    tickers = sorted(get_all_tickers())
    return {t: i for i, t in enumerate(tickers)}
