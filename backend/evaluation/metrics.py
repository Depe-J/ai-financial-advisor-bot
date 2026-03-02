# metrics.py - all the performance calculations are kept here
# imported by backtester.py so there's only one place to change if i need to fix something

from __future__ import annotations
import numpy as np

TRADING_DAYS = 252    # standard number of trading days in a year
N_BOOTSTRAP = 1000    # resamples for bootstrap CI
CI_LEVEL = 0.95
RANDOM_SEED = 42


def total_return(portfolio_values: list[float]) -> float:
    if len(portfolio_values) < 2:
        return 0.0
    return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]


def total_return_from_daily(daily_returns: np.ndarray) -> float:
    # reconstruct total return by compounding daily returns
    return float(np.prod(1 + daily_returns) - 1)


def sharpe_ratio(daily_returns: np.ndarray) -> float:
    # annualised sharpe ratio, assuming risk-free rate of 0
    if len(daily_returns) < 2:
        return 0.0
    mu = daily_returns.mean()
    sig = daily_returns.std(ddof=1)
    if sig < 1e-10:
        return 0.0
    return float(mu / sig * np.sqrt(TRADING_DAYS))


def max_drawdown(portfolio_values: np.ndarray) -> float:
    # biggest peak to trough drop - negative number, closer to 0 is better
    if len(portfolio_values) < 2:
        return 0.0
    pv = np.asarray(portfolio_values, dtype=float)
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / (peak + 1e-8)
    return float(dd.min())


def win_rate(trades: list[dict]) -> float:
    # only count SELL trades since BUY trades dont have a pnl yet
    sells = [t for t in trades if t.get("action") == "SELL"]
    if not sells:
        return 0.0
    wins = sum(1 for t in sells if t.get("pnl", 0) > 0)
    return wins / len(sells)


def bootstrap_ci(daily_returns, stat_fn, n=N_BOOTSTRAP, ci=CI_LEVEL, seed=RANDOM_SEED):
    # resample n times to build a confidence interval for whatever stat we pass in
    # using bootstrap because return distributions arent normal
    rng = np.random.default_rng(seed)
    stats = np.array([
        stat_fn(rng.choice(daily_returns, size=len(daily_returns), replace=True))
        for _ in range(n)
    ])
    alpha = (1 - ci) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))
