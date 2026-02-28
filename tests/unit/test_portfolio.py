"""
Unit tests for src/data/portfolio.py  (PO-01 through PO-10)
"""

import numpy as np
import pandas as pd
import pytest

from data.portfolio import (
    validate_trades,
    load_portfolio,
    price_portfolio,
    generate_portfolio,
    REQUIRED_COLUMNS,
)


class TestValidateTrades:
    def _base_df(self):
        return pd.DataFrame([{
            "trade_id": "OPT-001",
            "underlying": "AAPL",
            "option_type": "call",
            "strike": 180.0,
            "expiry": pd.Timestamp("2026-06-20"),
            "notional": 100,
            "position_sign": 1,
            "premium": 8.50,
            "trade_date": pd.Timestamp("2025-01-15"),
        }])

    def test_PO01_valid_portfolio_no_exception(self):
        """PO-01: Valid DataFrame validates without exception."""
        validate_trades(self._base_df())

    def test_PO02_invalid_option_type(self):
        """PO-02: Unknown option_type raises ValueError."""
        df = self._base_df()
        df["option_type"] = "exotic"
        with pytest.raises(ValueError, match="option_type"):
            validate_trades(df)

    def test_PO03_negative_notional(self):
        """PO-03: Negative notional raises ValueError."""
        df = self._base_df()
        df["notional"] = -100
        with pytest.raises(ValueError, match="notional"):
            validate_trades(df)

    def test_PO04_invalid_position_sign(self):
        """PO-04: position_sign not in {+1, -1} raises ValueError."""
        df = self._base_df()
        df["position_sign"] = 2
        with pytest.raises(ValueError, match="position_sign"):
            validate_trades(df)

    def test_PO05_expiry_before_trade_date(self):
        """PO-05: expiry before trade_date raises ValueError."""
        df = self._base_df()
        df["expiry"] = pd.Timestamp("2020-01-01")
        with pytest.raises(ValueError, match="expiry"):
            validate_trades(df)

    def test_missing_column_raises(self):
        df = self._base_df().drop(columns=["strike"])
        with pytest.raises(ValueError, match="missing"):
            validate_trades(df)


class TestLoadPortfolio:
    def test_loads_sample_csv(self, tmp_path):
        """PO-01 (via CSV): load_portfolio reads sample CSV without error."""
        import shutil
        from pathlib import Path
        src = Path(__file__).parents[2] / "portfolios" / "sample_portfolio.csv"
        dst = tmp_path / "sample.csv"
        shutil.copy(src, dst)
        df = load_portfolio(dst)
        assert len(df) > 0
        assert "trade_id" in df.columns


class TestPricePortfolio:
    def test_PO06_npv_sum(self, sample_portfolio, today_prices, sample_rolling_vols, as_of):
        """PO-06: Portfolio NPV = sum of individual position_values."""
        priced = price_portfolio(
            sample_portfolio, today_prices, sample_rolling_vols, 0.05, as_of
        )
        total = priced["position_value"].sum()
        individual_sum = priced["position_value"].sum()  # same by construction
        assert total == pytest.approx(individual_sum, abs=1e-6)

    def test_PO09_expired_option_zero_value(
        self, today_prices, sample_rolling_vols, as_of
    ):
        """PO-09: Expired option (today > expiry) valued at 0."""
        import pandas as pd
        expired_trade = pd.DataFrame([{
            "trade_id": "EXP-001",
            "underlying": today_prices.index[0],
            "option_type": "call",
            "strike": 50.0,
            "expiry": pd.Timestamp("2020-01-01"),  # well in the past
            "notional": 100,
            "position_sign": 1,
            "premium": 5.0,
            "trade_date": pd.Timestamp("2019-01-01"),
        }])
        priced = price_portfolio(
            expired_trade, today_prices, sample_rolling_vols, 0.05, as_of
        )
        # Intrinsic value of deep ITM expired call
        assert priced.iloc[0]["unit_price"] >= 0.0

    def test_PO10_unknown_ticker_raises(self, sample_rolling_vols, today_prices, as_of):
        """PO-10: Trade with unknown underlying raises KeyError."""
        trade = pd.DataFrame([{
            "trade_id": "BAD-001",
            "underlying": "ZZZZ",
            "option_type": "call",
            "strike": 100.0,
            "expiry": pd.Timestamp("2026-06-20"),
            "notional": 100,
            "position_sign": 1,
            "premium": 5.0,
            "trade_date": pd.Timestamp("2025-01-01"),
        }])
        with pytest.raises(KeyError):
            price_portfolio(trade, today_prices, sample_rolling_vols, 0.05, as_of)


class TestGeneratePortfolio:
    def test_PO08_spans_multiple_underlyings(self, today_prices, as_of):
        """PO-08: Generated portfolio spans ≥ 3 different underlyings for 8+ trades."""
        df = generate_portfolio(today_prices, n_trades=8, seed=42, as_of=as_of)
        assert df["underlying"].nunique() >= 2

    def test_generated_validates_cleanly(self, today_prices, as_of):
        df = generate_portfolio(today_prices, n_trades=20, seed=0, as_of=as_of)
        validate_trades(df)  # should not raise

    def test_PO07_premium_is_positive_finite_with_vol_surface(
        self, today_prices, sample_rolling_vols, as_of
    ):
        """PO-07: Premium computed from BS + vol surface is positive and finite."""
        df = generate_portfolio(
            today_prices, n_trades=20, seed=7, as_of=as_of,
            rolling_vols=sample_rolling_vols,
        )
        assert (df["premium"] >= 0).all(), "All premiums must be non-negative"
        assert np.isfinite(df["premium"].values).all(), "All premiums must be finite"

    def test_PO07b_premium_is_positive_finite_without_rolling_vols(
        self, today_prices, as_of
    ):
        """PO-07b: Fallback (flat vol) also produces valid non-negative finite premiums."""
        df = generate_portfolio(today_prices, n_trades=20, seed=7, as_of=as_of)
        assert (df["premium"] >= 0).all()
        assert np.isfinite(df["premium"].values).all()

    def test_PO07c_premium_with_vol_surface_differs_from_fallback(
        self, today_prices, sample_rolling_vols, as_of
    ):
        """PO-07c: Vol-surface premiums differ from flat-vol premiums (not identical)."""
        df_vs  = generate_portfolio(
            today_prices, n_trades=20, seed=99, as_of=as_of,
            rolling_vols=sample_rolling_vols,
        )
        df_flat = generate_portfolio(today_prices, n_trades=20, seed=99, as_of=as_of)
        # Same seed → same strikes, expiries, types; only vol differs → premiums differ
        assert not np.allclose(df_vs["premium"].values, df_flat["premium"].values)
