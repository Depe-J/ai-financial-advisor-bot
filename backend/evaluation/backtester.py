import logging
import csv
import os
import pandas as pd
import numpy as np
import math
from datetime import datetime

# import all the metric functions from metrics.py
# keeping calculations in one place makes it easier to update
from evaluation.metrics import (
    sharpe_ratio,
    max_drawdown,
    win_rate,
    total_return_from_daily,
    bootstrap_ci,
    N_BOOTSTRAP,
    CI_LEVEL,
)

# transaction cost of 0.1% applied on both buy and sell
# this simulates real brokerage fees so results are more realistic
log = logging.getLogger(__name__)

TRANSACTION_COST = 0.001

# ─── EXPERIMENT LOG ───────────────────────────────────────────────────────────
# Path for CSV experiment log.
# Previously, RL experiment results were only observed in the terminal and not
# saved, making it impossible to compare reward configurations retrospectively.
# This log ensures every backtest run is recorded so results can be compared.
EXPERIMENT_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "evaluation", "experiment_log.csv"
)
EXPERIMENT_LOG_FIELDS = [
    "timestamp", "ticker", "strategy",
    "return_pct", "sharpe", "max_drawdown_pct",
    "win_rate_pct", "total_trades",
    "sharpe_ci_low", "sharpe_ci_high",
    "return_ci_low", "return_ci_high",
    "n_bootstrap", "seed",
]

def _log_experiment(ticker, strategy, return_pct, sharpe, mdd, wr, trades,
                    sharpe_ci, return_ci):
    """Append one row to the experiment CSV log."""
    log_path = os.path.normpath(EXPERIMENT_LOG_PATH)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":       datetime.utcnow().isoformat(),
            "ticker":          ticker,
            "strategy":        strategy,
            "return_pct":      round(return_pct, 4),
            "sharpe":          round(sharpe, 4),
            "max_drawdown_pct":round(mdd * 100, 4),
            "win_rate_pct":    round(wr * 100, 4),
            "total_trades":    trades,
            "sharpe_ci_low":   round(sharpe_ci[0], 4),
            "sharpe_ci_high":  round(sharpe_ci[1], 4),
            "return_ci_low":   round(return_ci[0] * 100, 4),
            "return_ci_high":  round(return_ci[1] * 100, 4),
            "n_bootstrap":     N_BOOTSTRAP,
            "seed":            42,
        })
# ──────────────────────────────────────────────────────────────────────────────


def _run_single(df: pd.DataFrame, initial_cash: float = 10000.0):
    # runs the backtest for a single strategy (based on whatever is in df['signal'])
    # signals are: 1 = buy, -1 = sell, 0 = hold
    # starts with 10k and simulates buying/selling shares day by day

    df = df.copy().reset_index(drop=True)
    cash = initial_cash
    position = 0
    buy_price = None  # price paid when we bought (with transaction cost baked in)
    trade_log = []
    equity_curve = []
    portfolio_values = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row['close'])
        signal = row['signal']
        date = str(row['date'])[:10]

        # BUY logic - only buy if we dont already have a position
        if signal == 1 and position == 0:
            effective_buy = price * (1 + TRANSACTION_COST)
            position = int(cash // effective_buy)
            if position > 0:
                buy_price = effective_buy
                cash -= position * effective_buy
                trade_log.append({
                    "date": date,
                    "action": "BUY",
                    "price": round(price, 2),
                    "shares": position,
                    "cash": round(cash, 2),
                })

        elif signal == -1 and position > 0:
            effective_sell = price * (1 - TRANSACTION_COST)
            proceeds = position * effective_sell
            pnl = 0.0
            if buy_price is not None:
                raw_pnl = (effective_sell - buy_price) * position
                pnl = 0.0 if pd.isna(raw_pnl) or math.isnan(raw_pnl) else round(raw_pnl, 2)
            cash += proceeds
            trade_log.append({
                "date": date,
                "action": "SELL",
                "price": round(price, 2),
                "shares": position,
                "cash": round(cash, 2),
                "pnl": pnl,
            })
            position = 0
            buy_price = None

        # track portfolio value every day (cash + value of any shares held)
        equity = cash + (position * price)
        portfolio_values.append(equity)
        equity_curve.append({
            "date": date,
            "equity": round(equity, 2),
            "close": round(price, 2),
        })

    # if we still have shares at the end, sell them at the last price
    if position > 0:
        final_price = float(df.iloc[-1]['close'])
        cash += position * final_price * (1 - TRANSACTION_COST)

    returns_pct = ((cash - initial_cash) / initial_cash) * 100

    return trade_log, equity_curve, portfolio_values, round(returns_pct, 2), cash


def backtest_strategy(df: pd.DataFrame, initial_cash: float = 10000.0,
                      ticker: str = "UNKNOWN"):
    # runs the backtest for all 3 strategies and returns results for the hybrid
    # the three strategies are:
    #   1. MA-Only  - just uses the moving average crossover signal
    #   2. RL-Only  - just uses the Q-learning signal
    #   3. Hybrid   - uses RL signal wherever RL is non-zero, MA signal otherwise
    # this lets us compare all three in the evaluation panel

    df = df.copy().reset_index(drop=True)

    # evaluate only on the final 30% of data (held-out test window)
    # the RL agent trains on the full df before this function is called,
    # so slicing here ensures backtest results don't include the training period
    split_idx = int(len(df) * 0.70)
    df = df.iloc[split_idx:].reset_index(drop=True)

    # work out which column has the MA signal
    # basic_strategy.py saves it as 'ma_signal' before RL overwrites 'signal'
    has_ma_signal = 'ma_signal' in df.columns
    ma_col = df['ma_signal'] if has_ma_signal else df['signal']
    rl_col = df['signal']

    # ── HYBRID SIGNAL CONSTRUCTION (FIXED) ───────────────────────────────────
    # PREVIOUS (BUGGY) VERSION:
    #   hybrid_col = ma_col.copy()
    #   hybrid_col[(rl_col != 0) & (rl_col == ma_col)] = rl_col[...]
    #
    # The bug: only replaced values where RL and MA *already agreed*, so any
    # disagreement silently fell back to MA.  The result was that the Hybrid
    # strategy produced identical signals — and therefore identical results —
    # to MA-Only across every tested equity (TSLA, BABA, GOOG).
    #
    # FIXED VERSION:
    # Start from the MA signal as the baseline (it always gives a direction),
    # then override with the RL signal wherever RL has a non-zero opinion.
    # This means RL can now independently influence the final trade decision
    # even when it disagrees with MA.
    hybrid_col = ma_col.copy()
    hybrid_col[rl_col != 0] = rl_col[rl_col != 0]
    # ─────────────────────────────────────────────────────────────────────────

    # create three separate dataframes, one per strategy
    df_ma = df.copy(); df_ma['signal'] = ma_col
    df_rl = df.copy(); df_rl['signal'] = rl_col
    df_hybrid = df.copy(); df_hybrid['signal'] = hybrid_col

    # run the simulation for each strategy
    ma_trades, ma_equity, ma_pv, ma_ret, _ = _run_single(df_ma, initial_cash)
    rl_trades, rl_equity, rl_pv, rl_ret, _ = _run_single(df_rl, initial_cash)
    hyb_trades, hyb_equity, hyb_pv, hyb_ret, hyb_cash = _run_single(df_hybrid, initial_cash)

    # calculate risk metrics for the hybrid (main strategy)
    hyb_pv_arr = np.array(hyb_pv, dtype=float)
    daily_ret = np.diff(hyb_pv_arr) / (hyb_pv_arr[:-1] + 1e-8)

    hyb_sharpe = sharpe_ratio(daily_ret)
    hyb_mdd = max_drawdown(hyb_pv_arr)
    hyb_wr = win_rate(hyb_trades)

    # bootstrap confidence intervals - resample 1000 times to get 95% CI
    # this gives a range for the Sharpe ratio and return rather than just a point estimate
    sharpe_ci = bootstrap_ci(daily_ret, sharpe_ratio)
    return_ci = bootstrap_ci(daily_ret, total_return_from_daily)

    # helper to get metrics for the comparison strategies
    def _quick_metrics(pv, trades):
        arr  = np.array(pv, dtype=float)
        dret = np.diff(arr) / (arr[:-1] + 1e-8)
        return sharpe_ratio(dret), max_drawdown(arr), win_rate(trades)

    ma_sharpe, ma_mdd, ma_wr = _quick_metrics(ma_pv, ma_trades)
    rl_sharpe, rl_mdd, rl_wr = _quick_metrics(rl_pv, rl_trades)

    profitable = sum(1 for t in hyb_trades if t.get("pnl", 0) > 0)
    losing = sum(1 for t in hyb_trades if t.get("pnl", 0) < 0)

    # ── LOG ALL THREE STRATEGIES TO EXPERIMENT CSV ────────────────────────────
    # Previously results were only visible in the terminal.
    # Now every run is recorded so configurations can be compared retrospectively.
    try:
        ma_sc  = bootstrap_ci(np.diff(np.array(ma_pv,  dtype=float)) /
                               (np.array(ma_pv,  dtype=float)[:-1] + 1e-8), sharpe_ratio)
        ma_rc  = bootstrap_ci(np.diff(np.array(ma_pv,  dtype=float)) /
                               (np.array(ma_pv,  dtype=float)[:-1] + 1e-8), total_return_from_daily)
        rl_sc  = bootstrap_ci(np.diff(np.array(rl_pv,  dtype=float)) /
                               (np.array(rl_pv,  dtype=float)[:-1] + 1e-8), sharpe_ratio)
        rl_rc  = bootstrap_ci(np.diff(np.array(rl_pv,  dtype=float)) /
                               (np.array(rl_pv,  dtype=float)[:-1] + 1e-8), total_return_from_daily)
        _log_experiment(ticker, "MA-Only",       ma_ret,  ma_sharpe,  ma_mdd,  ma_wr,  len(ma_trades),  ma_sc,  ma_rc)
        _log_experiment(ticker, "RL-Only",       rl_ret,  rl_sharpe,  rl_mdd,  rl_wr,  len(rl_trades),  rl_sc,  rl_rc)
        _log_experiment(ticker, "Hybrid",        hyb_ret, hyb_sharpe, hyb_mdd, hyb_wr, len(hyb_trades), sharpe_ci, return_ci)
    except Exception as e:
        log.warning(f"Experiment log write failed: {e}")
    # ─────────────────────────────────────────────────────────────────────────

    summary = {
        "Initial Cash": round(initial_cash, 2),
        "Final Value": round(hyb_cash, 2),
        "Return (%)": hyb_ret,
        "Total Trades": len(hyb_trades),
        "Profitable Trades": profitable,
        "Losing Trades": losing,
        "Sharpe Ratio": round(hyb_sharpe, 2),
        "Max Drawdown (%)": round(hyb_mdd * 100, 2),
        "Win Rate (%)": round(hyb_wr * 100, 2),
        "Sharpe 95% CI": f"[{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]",
        "Return 95% CI": f"[{return_ci[0]*100:.1f}%, {return_ci[1]*100:.1f}%]",
        "Bootstrap N": N_BOOTSTRAP,
        "Strategy Comparison": [
            {"Strategy": "MA-Only",       "Return (%)": round(ma_ret, 2),  "Sharpe": round(ma_sharpe, 2),  "Max DD (%)": round(ma_mdd * 100, 2),  "Win Rate (%)": round(ma_wr * 100, 2),  "Trades": len(ma_trades)},
            {"Strategy": "RL-Only",       "Return (%)": round(rl_ret, 2),  "Sharpe": round(rl_sharpe, 2),  "Max DD (%)": round(rl_mdd * 100, 2),  "Win Rate (%)": round(rl_wr * 100, 2),  "Trades": len(rl_trades)},
            {"Strategy": "Hybrid (Final)","Return (%)": hyb_ret,           "Sharpe": round(hyb_sharpe, 2), "Max DD (%)": round(hyb_mdd * 100, 2), "Win Rate (%)": round(hyb_wr * 100, 2), "Trades": len(hyb_trades)},
        ],
    }

    trade_df  = pd.DataFrame(hyb_trades).fillna(0)
    equity_df = pd.DataFrame(hyb_equity).fillna(0)

    log.debug(f"Hybrid — BUYs: {sum(1 for t in hyb_trades if t['action'] == 'BUY')}, SELLs: {sum(1 for t in hyb_trades if t['action'] == 'SELL')}")
    log.debug(f"Sharpe: {hyb_sharpe:.2f}, Max DD: {hyb_mdd*100:.1f}%, Bootstrap N: {N_BOOTSTRAP}")

    return summary, trade_df, equity_df
