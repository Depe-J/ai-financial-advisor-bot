"""
test_backtester.py
──────────────────
Automated tests for the backtesting engine.

Previously the project had no unit tests, which meant the hybrid strategy
logic flaw went undetected during development.  The flaw was:

    hybrid_col = ma_col.copy()
    hybrid_col[(rl_col != 0) & (rl_col == ma_col)] = rl_col[...]

This only replaced values where RL and MA *already agreed*, so any
disagreement silently defaulted to MA, producing results identical to
MA-Only across every tested equity.

The test below (test_hybrid_differs_from_ma) is a property-based check
that would have caught this immediately: if hybrid is always identical to
MA, something is wrong with the construction logic.

Run with:  python -m pytest backend/tests/ -v
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

# Allow imports from backend/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.backtester import backtest_strategy
from strategies.basic_strategy import compute_signals
from strategies.q_learning_strategy import QLearningTrader


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _make_synthetic_df(n=200, seed=0):
    """
    Generate a synthetic OHLCV DataFrame with a simple price trend so that
    SMA10 and SMA50 are meaningful and the RL agent has something to learn.
    """
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.clip(close, 50, 300)
    df = pd.DataFrame({
        "date":   pd.date_range("2023-01-01", periods=n, freq="B"),
        "open":   close * rng.uniform(0.99, 1.01, n),
        "high":   close * rng.uniform(1.00, 1.02, n),
        "low":    close * rng.uniform(0.98, 1.00, n),
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, n),
    })
    return df


def _prepare_df(df):
    """Run the same pipeline as app.py: MA signals then RL training."""
    df = compute_signals(df)             # adds sma_short, sma_long, ma_signal, signal
    trader = QLearningTrader(seed=42)
    df = trader.train(df, episodes=10)   # overwrites 'signal' with RL decisions
    return df


# ── TESTS ─────────────────────────────────────────────────────────────────────

class TestHybridSignalConstruction:
    """
    Core property test: the Hybrid strategy must be meaningfully different
    from MA-Only.  If they are identical, the hybrid construction is broken.
    """

    def test_hybrid_differs_from_ma(self):
        """
        PROPERTY TEST — hybrid signals must not be identical to MA signals.

        If this test fails it means the hybrid column is being initialised
        from MA and only updated where RL already agrees with MA, which
        defeats the purpose of combining the two strategies.
        """
        df = _make_synthetic_df(n=200, seed=42)
        df = _prepare_df(df)

        # Run backtest (uses the fixed hybrid construction)
        summary, _, _ = backtest_strategy(df, ticker="SYNTHETIC")

        # Extract strategy comparison rows
        comparison = summary["Strategy Comparison"]
        ma_row  = next(r for r in comparison if r["Strategy"] == "MA-Only")
        hyb_row = next(r for r in comparison if "Hybrid" in r["Strategy"])

        # Hybrid must produce at least one different trade count or return
        # compared to MA-Only.  They can be close, but they must not be
        # bit-for-bit identical — that would indicate the signal flaw is back.
        signals_differ = (
            ma_row["Trades"]      != hyb_row["Trades"] or
            ma_row["Return (%)"]  != hyb_row["Return (%)"] or
            ma_row["Sharpe"]      != hyb_row["Sharpe"]
        )
        assert signals_differ, (
            "Hybrid strategy is producing results identical to MA-Only.\n"
            f"  MA-Only:  trades={ma_row['Trades']}, return={ma_row['Return (%)']}\n"
            f"  Hybrid:   trades={hyb_row['Trades']}, return={hyb_row['Return (%)']}\n"
            "Check the hybrid signal construction in backtester.py — "
            "RL signals should override MA wherever rl_col != 0."
        )

    def test_hybrid_trade_count_within_reasonable_range(self):
        """
        Sanity check: Hybrid should produce at least 1 trade.
        If it produces zero trades, both MA and RL are flat — something
        upstream is broken (e.g. signal column all zeros).
        """
        df = _make_synthetic_df(n=200, seed=7)
        df = _prepare_df(df)
        summary, _, _ = backtest_strategy(df, ticker="SYNTHETIC_SANITY")
        comparison = summary["Strategy Comparison"]
        hyb_row = next(r for r in comparison if "Hybrid" in r["Strategy"])
        assert hyb_row["Trades"] >= 1, (
            "Hybrid strategy produced zero trades — "
            "check that the signal column is being populated correctly."
        )


class TestBacktesterOutputShape:
    """
    Structural tests: verify that backtest_strategy returns the expected
    keys and data types regardless of input size.
    """

    def test_summary_keys_present(self):
        df = _make_synthetic_df(n=150, seed=1)
        df = _prepare_df(df)
        summary, trade_df, equity_df = backtest_strategy(df, ticker="SHAPE_TEST")

        required_keys = [
            "Initial Cash", "Final Value", "Return (%)", "Total Trades",
            "Profitable Trades", "Losing Trades", "Sharpe Ratio",
            "Max Drawdown (%)", "Win Rate (%)",
            "Sharpe 95% CI", "Return 95% CI", "Bootstrap N",
            "Strategy Comparison",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key in summary: '{key}'"

    def test_strategy_comparison_has_three_rows(self):
        df = _make_synthetic_df(n=150, seed=2)
        df = _prepare_df(df)
        summary, _, _ = backtest_strategy(df, ticker="COMPARE_TEST")
        assert len(summary["Strategy Comparison"]) == 3, (
            "Strategy Comparison should contain exactly 3 rows "
            "(MA-Only, RL-Only, Hybrid)."
        )

    def test_equity_curve_is_non_empty(self):
        df = _make_synthetic_df(n=150, seed=3)
        df = _prepare_df(df)
        _, _, equity_df = backtest_strategy(df, ticker="EQUITY_TEST")
        assert len(equity_df) > 0, "Equity curve DataFrame is empty."

    def test_initial_cash_preserved(self):
        df = _make_synthetic_df(n=150, seed=4)
        df = _prepare_df(df)
        summary, _, _ = backtest_strategy(df, initial_cash=5000.0, ticker="CASH_TEST")
        assert summary["Initial Cash"] == 5000.0


class TestBootstrapCI:
    """
    Verify that the confidence interval values are plausible.
    """

    def test_sharpe_ci_ordering(self):
        """Lower bound of Sharpe CI must be <= upper bound."""
        df = _make_synthetic_df(n=200, seed=5)
        df = _prepare_df(df)
        summary, _, _ = backtest_strategy(df, ticker="CI_TEST")
        ci_str = summary["Sharpe 95% CI"]   # e.g. "[-0.45, 1.23]"
        lo, hi = [float(x) for x in ci_str.strip("[]").split(",")]
        assert lo <= hi, (
            f"Sharpe CI lower bound ({lo}) is greater than upper bound ({hi})."
        )
