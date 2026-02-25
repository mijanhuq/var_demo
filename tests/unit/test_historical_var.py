"""
tests/unit/test_historical_var.py
-----------------------------------
Unit tests for var/historical.py — full-revaluation Historical VaR and ES.

Test IDs (from docs/test_plan.md §4.4)
---------------------------------------
VAR-01  VaR is positive (loss expressed as positive number)
VAR-02  VaR(99%) > VaR(95%)
VAR-03  VaR of a zero-position portfolio is zero
VAR-04  Long call VaR ≤ premium paid (bounded downside from options)
VAR-05  VaR is computed from exactly 252 scenarios
VAR-06  Sorted P&L at 99th percentile equals VaR(99%) — 3rd worst loss
VAR-07  Doubling all position sizes doubles VaR
VAR-08  Fully hedged (long + short same option) portfolio has VaR ≈ 0
VAR-09  Crisis-window VaR > non-crisis VaR (for a long call position)
VAR-10  Historical scenarios applied to spot prices preserve log-return magnitudes

Additional tests:
  - InsufficientDataError raised when market data has fewer rows than lookback
  - ES(99%) ≥ VaR(99%) and ES(95%) ≥ VaR(95%)
  - pnl_sorted property returns ascending-sorted vector
  - VaR(99%) and VaR(95%) are both non-negative
  - Performance: 120-position × 252-scenario run completes in < 5 s
"""

import math
import time

import numpy as np
import pytest

from portfolio.portfolio import MarketSnapshot, Portfolio, build_from_config
from portfolio.position import OptionPosition, OptionType
from var.historical import (
    InsufficientDataError,
    VaRResult,
    _build_shocked_snapshot,
    _var_es_at_confidence,
    compute_historical_var,
    compute_pnl_scenarios,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _long_call(ticker: str, S: float, quantity: int = 1) -> OptionPosition:
    """Single long OTM call: strike = 1.05 × S, 3-month maturity."""
    return OptionPosition(
        ticker=ticker,
        option_type=OptionType.CALL,
        strike=round(S * 1.05, 4),
        tenor_years=0.25,
        quantity=quantity,
        notional_per_contract=100,
    )


def _short_call(ticker: str, S: float, quantity: int = 1) -> OptionPosition:
    """Short call with identical parameters to _long_call."""
    return OptionPosition(
        ticker=ticker,
        option_type=OptionType.CALL,
        strike=round(S * 1.05, 4),
        tenor_years=0.25,
        quantity=-quantity,
        notional_per_contract=100,
    )


# ===========================================================================
# VAR-01 / VAR-02 / VAR-03: Basic properties
# ===========================================================================

class TestVaRBasicProperties:
    """VAR-01, VAR-02, VAR-03."""

    def test_var01_var_is_positive(self, var_result_small):
        """VAR-01: VaR is a positive number (loss expressed as a gain)."""
        assert var_result_small.var_99 > 0.0
        assert var_result_small.var_95 > 0.0

    def test_var02_var99_exceeds_var95(self, var_result_small):
        """VAR-02: VaR(99%) is strictly larger than VaR(95%)."""
        assert var_result_small.var_99 > var_result_small.var_95

    def test_var03_empty_portfolio_var_is_zero(self, snapshot, market_data_small):
        """VAR-03: Empty portfolio → VaR = ES = 0."""
        result = compute_historical_var(
            Portfolio([]), snapshot, market_data_small, lookback_window=252
        )
        assert result.var_99 == 0.0
        assert result.var_95 == 0.0
        assert result.es_99 == 0.0
        assert result.es_95 == 0.0

    def test_var_non_negative(self, var_result_small):
        """VaR and ES are always ≥ 0."""
        assert var_result_small.var_99 >= 0.0
        assert var_result_small.var_95 >= 0.0
        assert var_result_small.es_99 >= 0.0
        assert var_result_small.es_95 >= 0.0

    def test_es_exceeds_var(self, var_result_small):
        """ES(α) ≥ VaR(α) for both confidence levels."""
        assert var_result_small.es_99 >= var_result_small.var_99
        assert var_result_small.es_95 >= var_result_small.var_95


# ===========================================================================
# VAR-04: Options limit downside
# ===========================================================================

class TestVaROptionBound:
    """VAR-04: Long call VaR is bounded by the premium paid."""

    def test_var04_long_call_var_bounded_by_premium(self, snapshot, market_data_small):
        """
        VAR-04: The maximum loss for a long call position is the premium paid
        (the call can only go to zero — it cannot go below zero).
        Therefore VaR(99%) ≤ base_value (initial portfolio value).
        """
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        port = Portfolio([_long_call(ticker, S, quantity=1)])

        result = compute_historical_var(port, snapshot, market_data_small, lookback_window=252)

        # Long call: VaR cannot exceed the premium paid
        assert result.var_99 <= result.base_value * (1 + 1e-9)

    def test_long_call_base_value_positive(self, snapshot, market_data_small):
        """Sanity: long call portfolio always has positive base value."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        port = Portfolio([_long_call(ticker, S, quantity=1)])
        result = compute_historical_var(port, snapshot, market_data_small, lookback_window=252)
        assert result.base_value > 0.0


# ===========================================================================
# VAR-05 / VAR-06 / VAR-10: Scenario mechanics
# ===========================================================================

class TestVaRScenarios:
    """VAR-05, VAR-06, VAR-10."""

    def test_var05_exactly_252_scenarios(self, var_result_small):
        """VAR-05: VaR is computed from exactly 252 historical scenarios."""
        assert var_result_small.n_scenarios == 252

    def test_var05_pnl_vector_length(self, var_result_small):
        """pnl_vector has the same length as n_scenarios."""
        assert len(var_result_small.pnl_vector) == var_result_small.n_scenarios

    def test_var06_third_worst_pnl_equals_var99(self, var_result_small):
        """
        VAR-06: For 252 scenarios at 99% confidence:
            n_exceed = ceil(0.01 × 252) = 3
            VaR(99%) = −pnl_sorted[2]  (3rd worst P&L)
        """
        pnl_sorted = var_result_small.pnl_sorted
        n_exceed = math.ceil(0.01 * 252)  # = 3
        expected_var99 = max(0.0, -float(pnl_sorted[n_exceed - 1]))
        assert var_result_small.var_99 == pytest.approx(expected_var99, abs=1e-10)

    def test_var06_thirteenth_worst_pnl_equals_var95(self, var_result_small):
        """
        VAR-06 (95% variant): n_exceed = ceil(0.05 × 252) = 13
            VaR(95%) = −pnl_sorted[12]  (13th worst P&L)
        """
        pnl_sorted = var_result_small.pnl_sorted
        n_exceed = math.ceil(0.05 * 252)  # = 13
        expected_var95 = max(0.0, -float(pnl_sorted[n_exceed - 1]))
        assert var_result_small.var_95 == pytest.approx(expected_var95, abs=1e-10)

    def test_pnl_sorted_property_is_ascending(self, var_result_small):
        """pnl_sorted returns values in non-decreasing order."""
        pnl_sorted = var_result_small.pnl_sorted
        assert np.all(np.diff(pnl_sorted) >= 0.0)

    def test_var10_shocked_spot_equals_current_times_exp_return(
        self, snapshot, market_data_small
    ):
        """
        VAR-10: Shocked price = S_current × exp(historical_log_return).

        Test _build_shocked_snapshot for one scenario row.
        """
        row = market_data_small.log_returns.iloc[-1]  # last day's returns
        shocked = _build_shocked_snapshot(snapshot, row)

        for ticker in market_data_small.tickers:
            S_current = float(snapshot.spot_prices[ticker])
            log_ret = float(row[ticker])
            S_expected = S_current * np.exp(log_ret)
            assert float(shocked.spot_prices[ticker]) == pytest.approx(S_expected, rel=1e-10)

    def test_var10_shocked_vol_surface_unchanged(self, snapshot, market_data_small):
        """VAR-10 (vol): vol surface is frozen at current snapshot in each scenario."""
        row = market_data_small.log_returns.iloc[-1]
        shocked = _build_shocked_snapshot(snapshot, row)
        assert shocked.vol_surface is snapshot.vol_surface


# ===========================================================================
# VAR-07: Linear scaling
# ===========================================================================

class TestVaRLinearScaling:
    """VAR-07: Doubling all position sizes doubles VaR."""

    def test_var07_doubling_quantity_doubles_var99(self, snapshot, market_data_small):
        """VAR-07: P&L scales linearly with notional — doubling quantity doubles VaR."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])

        port_1 = Portfolio([_long_call(ticker, S, quantity=1)])
        port_2 = Portfolio([_long_call(ticker, S, quantity=2)])

        result_1 = compute_historical_var(port_1, snapshot, market_data_small, lookback_window=252)
        result_2 = compute_historical_var(port_2, snapshot, market_data_small, lookback_window=252)

        if result_1.var_99 > 0.0:
            assert result_2.var_99 == pytest.approx(2.0 * result_1.var_99, rel=1e-6)

    def test_var07_doubling_quantity_doubles_var95(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])

        port_1 = Portfolio([_long_call(ticker, S, quantity=1)])
        port_2 = Portfolio([_long_call(ticker, S, quantity=2)])

        result_1 = compute_historical_var(port_1, snapshot, market_data_small, lookback_window=252)
        result_2 = compute_historical_var(port_2, snapshot, market_data_small, lookback_window=252)

        if result_1.var_95 > 0.0:
            assert result_2.var_95 == pytest.approx(2.0 * result_1.var_95, rel=1e-6)

    def test_var07_pnl_vector_scales_linearly(self, snapshot, market_data_small):
        """The full P&L vector doubles when quantity doubles."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])

        scenarios = market_data_small.log_returns.iloc[-252:]
        port_1 = Portfolio([_long_call(ticker, S, quantity=1)])
        port_2 = Portfolio([_long_call(ticker, S, quantity=2)])

        pnl_1 = compute_pnl_scenarios(port_1, snapshot, scenarios)
        pnl_2 = compute_pnl_scenarios(port_2, snapshot, scenarios)

        np.testing.assert_allclose(pnl_2, 2.0 * pnl_1, rtol=1e-9)


# ===========================================================================
# VAR-08: Fully hedged portfolio
# ===========================================================================

class TestVaRHedging:
    """VAR-08: Long + short of the identical option gives VaR ≈ 0."""

    def test_var08_fully_hedged_portfolio_var99_is_zero(self, snapshot, market_data_small):
        """VAR-08: Perfect hedge (long call + short call, same K/T) → P&L ≡ 0."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])

        port = Portfolio([
            _long_call(ticker, S, quantity=1),
            _short_call(ticker, S, quantity=1),
        ])

        result = compute_historical_var(port, snapshot, market_data_small, lookback_window=252)
        assert result.var_99 == pytest.approx(0.0, abs=1e-8)
        assert result.var_95 == pytest.approx(0.0, abs=1e-8)

    def test_var08_fully_hedged_pnl_all_zero(self, snapshot, market_data_small):
        """P&L is exactly zero for every scenario in a perfectly hedged portfolio."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])

        port = Portfolio([
            _long_call(ticker, S, quantity=1),
            _short_call(ticker, S, quantity=1),
        ])
        scenarios = market_data_small.log_returns.iloc[-252:]
        pnl = compute_pnl_scenarios(port, snapshot, scenarios)
        np.testing.assert_allclose(pnl, 0.0, atol=1e-8)


# ===========================================================================
# VAR-09: Stress (crisis) window
# ===========================================================================

class TestVaRStressWindow:
    """
    VAR-09: Crisis-window VaR > non-crisis VaR for a long (bullish) position.

    Strategy:
    - Use market_data_standard (1510 days, crisis injected at rows 630–882).
    - Build a SINGLE LONG CALL on TECH_01 (bullish: suffers in crisis downturns).
    - Non-crisis VaR: last 252 rows of log_returns (post-crisis, calmer period).
    - Crisis VaR:     log_returns rows 630–881 (252 rows, crisis period).
    - The crisis returns include large negative daily moves → bigger losses
      for a long call → crisis_var > non_crisis_var.
    """

    def test_var09_crisis_var_exceeds_non_crisis_var(self, market_data_standard):
        """VAR-09: Crisis-period VaR(99%) > non-crisis VaR(99%) for a long call."""
        ticker = market_data_standard.tickers[0]  # TECH_01 — Technology sector

        # Snapshot at last date (post-crisis, calm market)
        snap_std = MarketSnapshot.from_market_data(market_data_standard, date_idx=-1)
        S = float(snap_std.spot_prices[ticker])

        # Single long OTM call — bullish, loses value in crisis downturns
        port = Portfolio([_long_call(ticker, S, quantity=10)])

        # Non-crisis VaR: last 252 scenarios (recent, post-crisis)
        non_crisis_result = compute_historical_var(
            port, snap_std, market_data_standard, lookback_window=252
        )

        # Crisis VaR: rows 630–881 (252 crisis scenarios applied to current spot)
        c_start = market_data_standard.crisis_start_idx  # 630
        c_end   = market_data_standard.crisis_end_idx    # 882
        crisis_returns = market_data_standard.log_returns.iloc[c_start:c_end]
        assert len(crisis_returns) == 252, "Crisis window should be exactly 252 rows"

        crisis_pnl = compute_pnl_scenarios(port, snap_std, crisis_returns)
        crisis_pnl_sorted = np.sort(crisis_pnl)
        n_exceed = math.ceil(0.01 * len(crisis_pnl_sorted))
        crisis_var99 = max(0.0, -float(crisis_pnl_sorted[n_exceed - 1]))

        assert crisis_var99 > non_crisis_result.var_99, (
            f"Crisis VaR {crisis_var99:.2f} should exceed non-crisis VaR "
            f"{non_crisis_result.var_99:.2f}"
        )


# ===========================================================================
# Edge cases
# ===========================================================================

class TestVaREdgeCases:
    """EDG-06: InsufficientDataError; other edge cases."""

    def test_insufficient_data_raises_error(self, full_portfolio, snapshot, minimal_config):
        """EDG-06: lookback_window > available rows raises InsufficientDataError."""
        from data.generator import MarketDataGenerator

        tiny_config = {**minimal_config, "simulation": {**minimal_config["simulation"], "n_trading_days": 100}}
        md_tiny = MarketDataGenerator(config=tiny_config, seed=42).generate()

        with pytest.raises(InsufficientDataError):
            compute_historical_var(full_portfolio, snapshot, md_tiny, lookback_window=252)

    def test_custom_lookback_window(self, snapshot, market_data_small):
        """A shorter lookback window (e.g. 100 scenarios) produces valid results."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        port = Portfolio([_long_call(ticker, S)])
        result = compute_historical_var(port, snapshot, market_data_small, lookback_window=100)
        assert result.n_scenarios == 100
        assert result.var_99 >= 0.0

    def test_var_es_at_confidence_helper(self):
        """_var_es_at_confidence produces correct VaR/ES for a known P&L vector."""
        # P&L vector: -100, -80, -60, ..., 0, ..., +100 (21 values)
        pnl_sorted = np.linspace(-100, 100, 21)  # sorted ascending
        # 99% confidence: n_exceed = ceil(0.01 * 21) = ceil(0.21) = 1
        var, es = _var_es_at_confidence(pnl_sorted, 0.99)
        assert var == pytest.approx(100.0)   # worst P&L = −100 → VaR = 100
        assert es == pytest.approx(100.0)    # only 1 tail scenario

        # 90% confidence: n_exceed = ceil(0.10 * 21) = ceil(2.1) = 3
        var, es = _var_es_at_confidence(pnl_sorted, 0.90)
        assert var == pytest.approx(80.0)    # 3rd worst P&L = −80
        assert es == pytest.approx(90.0)    # mean of (−100, −90, −80) = −90 → es=90

    def test_pnl_scenarios_empty_portfolio(self, snapshot, market_data_small):
        """compute_pnl_scenarios returns all zeros for an empty portfolio."""
        scenarios = market_data_small.log_returns.iloc[-10:]
        pnl = compute_pnl_scenarios(Portfolio([]), snapshot, scenarios)
        assert pnl.shape == (10,)
        np.testing.assert_array_equal(pnl, 0.0)


# ===========================================================================
# Performance
# ===========================================================================

class TestVaRPerformance:
    """PRF-01: 120-position portfolio, 252 scenarios completes in < 5 s."""

    def test_prf01_full_portfolio_var_within_time_limit(
        self, full_portfolio, snapshot, market_data_small
    ):
        start = time.perf_counter()
        _ = compute_historical_var(full_portfolio, snapshot, market_data_small, lookback_window=252)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"compute_historical_var took {elapsed:.2f}s (limit 5s)"
