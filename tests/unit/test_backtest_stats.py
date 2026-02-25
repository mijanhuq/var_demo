"""
tests/unit/test_backtest_stats.py
----------------------------------
Unit tests for backtest/statistics.py (BT-01–09) and backtest/engine.py.

Statistics tests use synthetic numpy arrays only — no market data fixtures
needed, so these are fast and fully independent.

Engine tests use the session-scoped fixtures from conftest.py
(market_data_small, full_portfolio).
"""

import numpy as np
import pytest

from backtest.engine import BacktestResult, run_backtest
from backtest.statistics import (
    ChristoffersenResult,
    KupiecResult,
    christoffersen_independence_test,
    count_exceptions,
    exception_rate,
    kupiec_pof_test,
    traffic_light,
)
from var.historical import InsufficientDataError


# ===========================================================================
# Helpers
# ===========================================================================

def _make_pnl_and_var(n: int, n_breach: int, var_level: float = 100.0):
    """
    Build (actual_pnl, var_estimates) arrays with exactly n_breach exceptions.

    Strategy: first n_breach days have P&L = -(var_level + 1) (breach),
    remaining days have P&L = 0 (no breach).
    All VaR estimates are var_level.
    """
    pnl = np.zeros(n)
    pnl[:n_breach] = -(var_level + 1.0)
    var_est = np.full(n, var_level)
    return pnl, var_est


# ===========================================================================
# BT-01 / BT-02 — Exception counting
# ===========================================================================

class TestExceptions:
    def test_bt01_count_exceptions_correct(self):
        """BT-01: count_exceptions returns exact breach count."""
        pnl, var_est = _make_pnl_and_var(100, 7)
        assert count_exceptions(pnl, var_est) == 7

    def test_bt02_exception_rate_correct(self):
        """BT-02: exception_rate = count / n."""
        pnl, var_est = _make_pnl_and_var(200, 10)
        assert exception_rate(pnl, var_est) == pytest.approx(0.05)

    def test_zero_exceptions(self):
        """count_exceptions returns 0 when P&L always > -VaR."""
        pnl = np.full(50, 10.0)
        var_est = np.full(50, 5.0)
        assert count_exceptions(pnl, var_est) == 0

    def test_all_exceptions(self):
        """count_exceptions returns n when every day is a breach."""
        n = 40
        pnl = np.full(n, -200.0)
        var_est = np.full(n, 100.0)
        assert count_exceptions(pnl, var_est) == n

    def test_exception_rate_zero(self):
        """exception_rate returns 0.0 when no exceptions."""
        pnl = np.zeros(100)
        var_est = np.full(100, 1.0)
        assert exception_rate(pnl, var_est) == 0.0

    def test_boundary_exactly_at_var(self):
        """P&L == -VaR is NOT a breach (strict inequality)."""
        pnl = np.array([-100.0, -99.0, 0.0])
        var_est = np.array([100.0, 100.0, 100.0])
        # -100 == -100 is NOT < -100, so 0 breaches
        assert count_exceptions(pnl, var_est) == 0


# ===========================================================================
# BT-03 / BT-04 / BT-05 — Traffic light
# ===========================================================================

class TestTrafficLight:
    def test_bt03_green(self):
        """BT-03: 4 exceptions → green."""
        assert traffic_light(4) == "green"

    def test_bt04_yellow(self):
        """BT-04: 7 exceptions → yellow."""
        assert traffic_light(7) == "yellow"

    def test_bt05_red(self):
        """BT-05: 10 exceptions → red."""
        assert traffic_light(10) == "red"

    def test_zero_exceptions_green(self):
        """0 exceptions → green."""
        assert traffic_light(0) == "green"

    def test_boundary_5_is_yellow(self):
        """Exactly 5 exceptions is the boundary: yellow."""
        assert traffic_light(5) == "yellow"

    def test_boundary_9_is_yellow(self):
        """9 exceptions is still yellow (10 needed for red)."""
        assert traffic_light(9) == "yellow"

    def test_boundary_10_is_red(self):
        """Exactly 10 exceptions → red."""
        assert traffic_light(10) == "red"

    def test_large_count_red(self):
        """50 exceptions → red."""
        assert traffic_light(50) == "red"


# ===========================================================================
# BT-06 / BT-07 — Kupiec POF test
# ===========================================================================

class TestKupiec:
    def test_bt06_rejects_5pct_exception_rate_at_99pct_var(self):
        """
        BT-06: 5% exception rate vs 99% VaR should be rejected.
        250 days, 12 exceptions (4.8%) — well above p0 = 1%.
        """
        result = kupiec_pof_test(n_exceptions=12, n_days=250, confidence=0.99)
        assert isinstance(result, KupiecResult)
        assert result.reject is True
        assert result.p_value < 0.05

    def test_bt07_does_not_reject_1pct_exception_rate_at_99pct_var(self):
        """
        BT-07: 1% exception rate vs 99% VaR should NOT be rejected.
        250 days, 2 exceptions (0.8%) — consistent with p0 = 1%.
        """
        result = kupiec_pof_test(n_exceptions=2, n_days=250, confidence=0.99)
        assert isinstance(result, KupiecResult)
        assert result.reject is False
        assert result.p_value > 0.05

    def test_kupiec_zero_exceptions_does_not_reject_at_99pct(self):
        """0 exceptions at 99% confidence with a small sample does not reject.

        With n=50 days and p0=1%: LR = -2*50*ln(0.99) ≈ 1.005 → p ≈ 0.316.
        Note: 0 exceptions in 250 days DOES reject (model over-conservative),
        but with only 50 observations the test lacks power.
        """
        result = kupiec_pof_test(n_exceptions=0, n_days=50, confidence=0.99)
        assert result.reject is False

    def test_kupiec_result_fields(self):
        """KupiecResult contains expected fields."""
        result = kupiec_pof_test(n_exceptions=5, n_days=250, confidence=0.99)
        assert result.n_exceptions == 5
        assert result.n_days == 250
        assert result.statistic >= 0.0
        assert 0.0 <= result.p_value <= 1.0

    def test_kupiec_statistic_nonnegative(self):
        """LR statistic is always non-negative."""
        for x in [0, 1, 3, 10, 20]:
            result = kupiec_pof_test(n_exceptions=x, n_days=250, confidence=0.99)
            assert result.statistic >= 0.0

    def test_kupiec_reject_flag_consistent_with_pvalue(self):
        """reject == (p_value < 0.05)."""
        result = kupiec_pof_test(n_exceptions=12, n_days=250, confidence=0.99)
        assert result.reject == (result.p_value < 0.05)

    def test_kupiec_custom_significance(self):
        """Custom significance level is respected."""
        # With significance=0.01, a borderline case may not reject
        result = kupiec_pof_test(
            n_exceptions=5, n_days=250, confidence=0.99, significance=0.01
        )
        assert result.reject == (result.p_value < 0.01)

    def test_kupiec_at_95pct_confidence(self):
        """Kupiec test works at 95% confidence level."""
        # 25 exceptions in 250 days = 10%, expected 5% → should reject
        result = kupiec_pof_test(n_exceptions=25, n_days=250, confidence=0.95)
        assert result.reject is True


# ===========================================================================
# BT-08 / BT-09 — Christoffersen independence test
# ===========================================================================

class TestChristoffersen:
    def test_bt08_rejects_clustered_exceptions(self):
        """
        BT-08: Clustered exceptions (block of 20 in the middle) should be flagged.

        250 days: 100 non-exception, 20 consecutive exceptions, 130 non-exception.
        This creates clear clustering: pi_11 ≈ 0.95 vs pi_01 ≈ 0.01.
        The block must NOT start at day 0 (that would give n01=0 → degenerate).
        """
        exceptions = np.zeros(250, dtype=int)
        exceptions[115:135] = 1  # 20 consecutive exceptions in the middle
        result = christoffersen_independence_test(exceptions)
        assert isinstance(result, ChristoffersenResult)
        assert result.reject is True

    def test_bt09_does_not_reject_uniform_exceptions(self):
        """
        BT-09: Uniformly spaced exceptions should NOT be flagged.
        250 days, every 25th day is an exception (10 total, no clustering).
        """
        exceptions = np.zeros(250, dtype=int)
        exceptions[::25] = 1  # days 0, 25, 50, ..., 225
        result = christoffersen_independence_test(exceptions)
        assert result.reject is False

    def test_christoffersen_transition_counts(self):
        """Transition counts n00/n01/n10/n11 sum to n-1."""
        exceptions = np.array([0, 0, 1, 1, 0, 1, 0])
        result = christoffersen_independence_test(exceptions)
        total = result.n00 + result.n01 + result.n10 + result.n11
        assert total == len(exceptions) - 1

    def test_christoffersen_result_fields(self):
        """ChristoffersenResult contains expected fields."""
        exceptions = np.zeros(100, dtype=int)
        exceptions[:5] = 1
        result = christoffersen_independence_test(exceptions)
        assert result.statistic >= 0.0
        assert 0.0 <= result.p_value <= 1.0
        assert isinstance(result.reject, (bool, np.bool_))

    def test_christoffersen_degenerate_one_exception(self):
        """Single exception → degenerate case, LR = 0, no rejection."""
        exceptions = np.zeros(100, dtype=int)
        exceptions[50] = 1
        result = christoffersen_independence_test(exceptions)
        assert result.reject is False
        assert result.statistic == pytest.approx(0.0)

    def test_christoffersen_degenerate_zero_exceptions(self):
        """No exceptions → degenerate case, LR = 0, no rejection."""
        exceptions = np.zeros(100, dtype=int)
        result = christoffersen_independence_test(exceptions)
        assert result.reject is False
        assert result.statistic == pytest.approx(0.0)

    def test_christoffersen_degenerate_short_array(self):
        """Array of length < 2 returns degenerate result."""
        result = christoffersen_independence_test(np.array([1]))
        assert result.reject is False

    def test_christoffersen_boolean_input(self):
        """Accepts boolean numpy array."""
        exceptions = np.array([False, True, True, False, True, False])
        result = christoffersen_independence_test(exceptions)
        assert result.n00 + result.n01 + result.n10 + result.n11 == 5


# ===========================================================================
# Engine tests — TestBacktestEngine
# ===========================================================================

class TestBacktestEngine:
    """Tests for backtest/engine.py run_backtest."""

    def test_engine_returns_backtest_result(self, full_portfolio, market_data_small):
        """run_backtest returns a BacktestResult instance."""
        result = run_backtest(
            full_portfolio, market_data_small,
            lookback_window=252, backtest_window=47,
        )
        assert isinstance(result, BacktestResult)

    def test_engine_n_backtest_days(self, full_portfolio, market_data_small):
        """n_backtest_days == backtest_window - 1 (last day has no t+1)."""
        result = run_backtest(
            full_portfolio, market_data_small,
            lookback_window=252, backtest_window=47,
        )
        assert result.n_backtest_days == 46

    def test_engine_array_lengths_match(self, full_portfolio, market_data_small):
        """All output arrays have the same length."""
        result = run_backtest(
            full_portfolio, market_data_small,
            lookback_window=252, backtest_window=47,
        )
        n = result.n_backtest_days
        assert len(result.var_estimates_99) == n
        assert len(result.var_estimates_95) == n
        assert len(result.actual_pnl) == n
        assert len(result.exceptions_99) == n
        assert len(result.exceptions_95) == n
        assert len(result.dates) == n

    def test_engine_var99_nonnegative(self, full_portfolio, market_data_small):
        """VaR(99%) is always non-negative."""
        result = run_backtest(
            full_portfolio, market_data_small,
            lookback_window=252, backtest_window=47,
        )
        assert np.all(result.var_estimates_99 >= 0.0)

    def test_engine_var99_ge_var95(self, full_portfolio, market_data_small):
        """VaR(99%) >= VaR(95%) for every backtest day."""
        result = run_backtest(
            full_portfolio, market_data_small,
            lookback_window=252, backtest_window=47,
        )
        assert np.all(result.var_estimates_99 >= result.var_estimates_95)

    def test_engine_exceptions_consistent_with_pnl(self, full_portfolio, market_data_small):
        """exceptions_99 == (actual_pnl < -var_estimates_99)."""
        result = run_backtest(
            full_portfolio, market_data_small,
            lookback_window=252, backtest_window=47,
        )
        expected = result.actual_pnl < -result.var_estimates_99
        np.testing.assert_array_equal(result.exceptions_99, expected)

    def test_engine_n_exceptions_properties(self, full_portfolio, market_data_small):
        """n_exceptions_99 and n_exceptions_95 match sum of exception arrays."""
        result = run_backtest(
            full_portfolio, market_data_small,
            lookback_window=252, backtest_window=47,
        )
        assert result.n_exceptions_99 == int(np.sum(result.exceptions_99))
        assert result.n_exceptions_95 == int(np.sum(result.exceptions_95))

    def test_engine_insufficient_data_raises(self, full_portfolio, market_data_small):
        """InsufficientDataError raised when data too short."""
        # market_data_small has 300 log-return rows
        # lookback=252, backtest=200 → requires 452 > 300
        with pytest.raises(InsufficientDataError):
            run_backtest(
                full_portfolio, market_data_small,
                lookback_window=252, backtest_window=200,
            )

    def test_engine_var95_nonnegative(self, full_portfolio, market_data_small):
        """VaR(95%) is always non-negative."""
        result = run_backtest(
            full_portfolio, market_data_small,
            lookback_window=252, backtest_window=47,
        )
        assert np.all(result.var_estimates_95 >= 0.0)
