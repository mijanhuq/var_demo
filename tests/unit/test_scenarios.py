"""
Unit tests for src/var/scenarios.py  (SC-01 through SC-09)
"""

import numpy as np
import pytest
import pandas as pd

from var.scenarios import (
    compute_log_returns,
    build_scenario_set,
    build_age_weights,
    apply_scenario,
    build_stressed_price_matrix,
)


class TestLogReturns:
    def test_SC01_formula(self, sample_prices):
        """SC-01: log-returns = log(S_t / S_{t-1}) to machine precision."""
        returns = compute_log_returns(sample_prices)
        manual = np.log(sample_prices / sample_prices.shift(1)).dropna()
        np.testing.assert_allclose(returns.values, manual.values, atol=1e-12)

    def test_no_nan_after_dropna(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        assert not returns.isna().any().any()

    def test_no_inf(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        assert np.isfinite(returns.values).all()


class TestBuildScenarioSet:
    def test_SC02_252_rows(self, sample_log_returns):
        """SC-02: scenario set has exactly 252 rows."""
        s = build_scenario_set(sample_log_returns, lookback=252)
        assert len(s) == 252

    def test_SC03_columns_match(self, sample_log_returns):
        """SC-03: columns match input return columns."""
        s = build_scenario_set(sample_log_returns, lookback=252)
        assert list(s.columns) == list(sample_log_returns.columns)

    def test_SC04_no_nan_or_inf(self, sample_log_returns):
        """SC-04: no NaN or inf in scenario frame."""
        s = build_scenario_set(sample_log_returns, lookback=252)
        assert not s.isna().any().any()
        assert np.isfinite(s.values).all()

    def test_SC09_raises_when_insufficient_history(self):
        """SC-09: ValueError when lookback > available history."""
        small_df = pd.DataFrame({"A": [0.01, -0.01, 0.02]})
        with pytest.raises(ValueError):
            build_scenario_set(small_df, lookback=252)


class TestBuildAgeWeights:
    def test_SC06_sum_to_one(self):
        """SC-05: Age weights sum to 1.0."""
        w = build_age_weights(252, lambda_=0.97)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_SC07_monotone_increasing_toward_recent(self):
        """SC-06: Weights monotonically increase toward most recent (last index)."""
        w = build_age_weights(252, lambda_=0.97)
        assert np.all(np.diff(w) >= 0)

    def test_SC08_uniform_when_lambda_one(self):
        """SC-08: λ close to 1 → nearly uniform weights (within 5% spread)."""
        w = build_age_weights(252, lambda_=0.9999)
        # With λ=0.9999 and n=252 the max/min ratio is 0.9999^251 ≈ 0.975
        # so weights span ~2.5% — check max/min ratio rather than allclose
        assert w.max() / w.min() < 1.05

    def test_raises_on_invalid_lambda(self):
        with pytest.raises(ValueError):
            build_age_weights(252, lambda_=1.5)


class TestApplyScenario:
    def test_SC05_stressed_spots_positive(self, today_prices, sample_scenario_set):
        """SC-05: All stressed spots > 0."""
        for _, scenario_row in sample_scenario_set.iterrows():
            stressed = apply_scenario(today_prices, scenario_row)
            assert (stressed > 0).all()

    def test_no_shock_gives_today(self, today_prices):
        """Zero return scenario → stressed == today."""
        zero_returns = pd.Series(0.0, index=today_prices.index)
        stressed = apply_scenario(today_prices, zero_returns)
        pd.testing.assert_series_equal(stressed, today_prices, check_names=False)


class TestBuildStressedPriceMatrix:
    def test_shape(self, today_prices, sample_scenario_set):
        matrix = build_stressed_price_matrix(today_prices, sample_scenario_set)
        assert matrix.shape == sample_scenario_set.shape

    def test_all_positive(self, today_prices, sample_scenario_set):
        matrix = build_stressed_price_matrix(today_prices, sample_scenario_set)
        assert (matrix.values > 0).all()
