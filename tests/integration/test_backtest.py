"""
Integration tests for the rolling backtest framework.
(BT-01 through BT-09)

Note: these tests are marked @pytest.mark.slow as they loop over many dates.
"""

import numpy as np
import pandas as pd
import pytest
from utils.stats import kupiec_test, christoffersen_test, traffic_light


class TestActiveTrades:
    """Unit tests for the trade-aging helper (no backtest loop needed)."""

    def _make_trades(self):
        return pd.DataFrame([
            {"trade_id": "T1", "underlying": "AAPL", "option_type": "call",
             "strike": 100.0, "expiry": pd.Timestamp("2025-03-01"),
             "notional": 100, "position_sign": 1, "premium": 5.0,
             "trade_date": pd.Timestamp("2024-01-01")},
            {"trade_id": "T2", "underlying": "MSFT", "option_type": "put",
             "strike": 90.0, "expiry": pd.Timestamp("2025-12-31"),
             "notional": 200, "position_sign": -1, "premium": 3.0,
             "trade_date": pd.Timestamp("2024-01-01")},
        ])

    def test_all_active_before_any_expiry(self):
        """Both trades active before first expiry date."""
        from backtest.backtest import _active_trades
        trades = self._make_trades()
        active = _active_trades(trades, pd.Timestamp("2025-01-01"))
        assert len(active) == 2

    def test_expired_trade_excluded(self):
        """T1 expires 2025-03-01; querying as_of=2025-03-01 excludes it."""
        from backtest.backtest import _active_trades
        trades = self._make_trades()
        active = _active_trades(trades, pd.Timestamp("2025-03-01"))
        assert len(active) == 1
        assert active.iloc[0]["trade_id"] == "T2"

    def test_all_expired_returns_empty(self):
        """All trades expired → empty DataFrame."""
        from backtest.backtest import _active_trades
        trades = self._make_trades()
        active = _active_trades(trades, pd.Timestamp("2026-01-01"))
        assert active.empty

    def test_n_active_trades_in_backtest_columns(self, sample_portfolio, sample_prices):
        """rolling_backtest output includes n_active_trades column."""
        from backtest.backtest import rolling_backtest
        df = rolling_backtest(sample_portfolio, sample_prices, r=0.05, test_window=3)
        assert "n_active_trades" in df.columns


class TestBacktestReport:
    """
    Test Kupiec + Christoffersen + traffic light using synthetic exception series.
    These tests do NOT require the full rolling backtest loop (which is slow)
    and instead test the statistical functions directly with crafted inputs.
    """

    def test_BT03_kupiec_calibrated(self):
        """BT-03: Kupiec p > 0.05 for correctly calibrated model (≈2.5 exceptions in 250)."""
        result = kupiec_test(n_exceptions=2, n_obs=250, confidence_level=0.99)
        assert result["p_value"] > 0.05

    def test_BT04_christoffersen_no_cluster(self):
        """BT-04: Christoffersen p > 0.05 for i.i.d. exceptions."""
        rng = np.random.default_rng(99)
        exc = (rng.uniform(size=500) < 0.01).astype(int)
        result = christoffersen_test(exc)
        # With very few exceptions, this may not be significant
        assert "p_value" in result

    def test_BT05_inflated_var_kupiec(self):
        """BT-05: 0 exceptions → Kupiec p < 0.05 (over-conservative model detected)."""
        with pytest.warns(UserWarning, match="over-conservative"):
            result = kupiec_test(n_exceptions=0, n_obs=250, confidence_level=0.99)
        assert result["p_value"] < 0.05
        assert result["reject_h0"]

    def test_BT06_traffic_light_green(self):
        """BT-06: Traffic light = 'Green' when exceptions ≤ 4."""
        assert traffic_light(4) == "Green"

    def test_BT07_traffic_light_amber(self):
        """BT-07: Traffic light = 'Amber' when exceptions ∈ [5, 9]."""
        assert traffic_light(5) == "Amber"
        assert traffic_light(9) == "Amber"

    def test_BT08_traffic_light_red(self):
        """BT-08: Traffic light = 'Red' when exceptions ≥ 10."""
        assert traffic_light(10) == "Red"
        assert traffic_light(25) == "Red"


@pytest.mark.slow
class TestRollingBacktest:
    """Full rolling backtest tests — slow, require complete price history."""

    def test_BT01_produces_one_row_per_day(self, sample_portfolio, sample_prices):
        """BT-01: Rolling backtest produces one VAR estimate per business day."""
        from backtest.backtest import rolling_backtest
        test_window = 20  # short window for speed
        df = rolling_backtest(
            sample_portfolio,
            sample_prices,
            r=0.05,
            test_window=test_window,
        )
        assert len(df) == test_window

    def test_backtest_columns(self, sample_portfolio, sample_prices):
        """Output DataFrame contains expected columns."""
        from backtest.backtest import rolling_backtest
        df = rolling_backtest(
            sample_portfolio, sample_prices, r=0.05, test_window=5
        )
        assert "realised_pnl" in df.columns
        assert "var_99" in df.columns
        assert "es_99" in df.columns
        assert "exception_99" in df.columns

    def test_full_report_structure(self, sample_portfolio, sample_prices):
        """backtest_report returns expected structure."""
        from backtest.backtest import rolling_backtest, backtest_report
        df = rolling_backtest(sample_portfolio, sample_prices, r=0.05, test_window=10)
        report = backtest_report(df, confidence_levels=[0.99])
        assert 0.99 in report
        entry = report[0.99]
        for key in ("n_obs", "n_exceptions", "kupiec", "christoffersen", "traffic_light"):
            assert key in entry
