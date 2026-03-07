import numpy as np


def sharpe_ratio(returns, rf=0.07, periods=248):
    """Annualized Sharpe Ratio. rf=7% (India govt bond)."""
    excess = returns - rf / periods
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / std)


def max_drawdown(portfolio_values):
    """Maximum drawdown from peak. Returns negative value (e.g., -0.15 = 15% drawdown)."""
    values = np.asarray(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / np.where(peak == 0, 1, peak)
    return float(drawdown.min())


def sortino_ratio(returns, rf=0.07, periods=248):
    """Annualized Sortino Ratio. Penalizes only downside volatility."""
    excess = returns - rf / periods
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / downside.std())


def calmar_ratio(returns, portfolio_values, rf=0.07):
    """Calmar Ratio = Annualized return / Max drawdown."""
    if len(returns) == 0:
        return 0.0
    annual_ret = (1 + returns).prod() ** (248 / len(returns)) - 1
    mdd = abs(max_drawdown(portfolio_values))
    if mdd == 0:
        return 0.0
    return float((annual_ret - rf) / mdd)


def annualized_return(returns, periods=248):
    """Annualized return from daily returns."""
    if len(returns) == 0:
        return 0.0
    total = (1 + returns).prod()
    n_years = len(returns) / periods
    if n_years == 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def annualized_volatility(returns, periods=248):
    """Annualized volatility from daily returns."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(periods))


def portfolio_turnover(weights_history):
    """Average daily turnover (sum of absolute weight changes)."""
    if len(weights_history) < 2:
        return 0.0
    w = np.asarray(weights_history)
    daily_turnover = np.abs(np.diff(w, axis=0)).sum(axis=1)
    return float(daily_turnover.mean())
