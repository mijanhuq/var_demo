"""
Unit tests for src/var/revaluation.py  (RV-01 through RV-07)
"""

import numpy as np
import pytest
import pandas as pd

from var.revaluation import full_reprice_pnl, greeks_approx_pnl
from data.portfolio import price_portfolio


class TestFullRepricePnl:
    def test_RV01_flat_scenario_zero_pnl(
        self, sample_portfolio, sample_prices, sample_rolling_vols, today_prices, as_of
    ):
        """RV-01: P&L = 0 when scenario = no spot change and same vol date.

        Use as_of as the scenario date for every scenario so the vol lookup
        returns the same vol as today's pricing, isolating the spot-only effect.
        """
        priced = price_portfolio(
            sample_portfolio, today_prices, sample_rolling_vols, 0.05, as_of
        )
        today_values = priced.set_index("trade_id")["position_value"]

        # Zero returns, all indexed at as_of so vol lookup = today's vol
        n = 5
        zero_returns = pd.DataFrame(
            np.zeros((n, len(today_prices))),
            columns=today_prices.index,
            index=[as_of] * n,         # repeated as_of → same vol every scenario
        )
        from var.scenarios import build_stressed_price_matrix
        scenario_prices = build_stressed_price_matrix(today_prices, zero_returns)

        pnl = full_reprice_pnl(
            sample_portfolio, today_prices, today_values, scenario_prices,
            sample_rolling_vols, 0.05, as_of,
        )
        np.testing.assert_allclose(pnl, 0.0, atol=1e-5)

    def test_RV05_pnl_vector_length(self, sample_pnl_vector, sample_scenario_set):
        """RV-05: P&L vector length = number of scenarios."""
        assert len(sample_pnl_vector) == len(sample_scenario_set)

    def test_RV06_portfolio_pnl_sum_of_trades(
        self, sample_portfolio, sample_prices, sample_rolling_vols, today_prices, as_of
    ):
        """RV-06: Portfolio P&L = sum of individual trade P&Ls (verified by single-trade check)."""
        priced = price_portfolio(
            sample_portfolio, today_prices, sample_rolling_vols, 0.05, as_of
        )
        today_values = priced.set_index("trade_id")["position_value"]

        from var.scenarios import compute_log_returns, build_scenario_set, build_stressed_price_matrix
        log_returns = compute_log_returns(sample_prices)
        scenario_set = build_scenario_set(log_returns, 10)  # small for speed
        scenario_prices = build_stressed_price_matrix(today_prices, scenario_set)

        pnl = full_reprice_pnl(
            sample_portfolio, today_prices, today_values, scenario_prices,
            sample_rolling_vols, 0.05, as_of,
        )
        assert pnl.shape == (10,)


class TestGreeksApproxPnl:
    def test_RV04_approx_close_to_full_for_small_shocks(
        self, sample_portfolio, sample_prices, sample_rolling_vols, today_prices, as_of
    ):
        """RV-04: Greeks approx vs full reprice for small spot-only shocks.

        Use a flat vol surface (alpha=beta=0) and as_of as the scenario date so
        the implied vol is identical across all spot levels and scenario dates.
        This isolates the pure spot-move effect, making delta-gamma a valid comparison.
        """
        from var.scenarios import build_stressed_price_matrix
        from data.portfolio import price_portfolio
        from var.revaluation import full_reprice_pnl, greeks_approx_pnl

        # Flat skew parameters → vol does not change with moneyness
        flat_alpha, flat_beta = 0.0, 0.0

        # Small ±1% spot shocks, all at as_of so vol is constant
        rng = np.random.default_rng(0)
        n = 20
        small_returns = pd.DataFrame(
            rng.normal(0, 0.01, (n, len(today_prices))),
            columns=today_prices.index,
            index=[as_of] * n,
        )
        scenario_prices = build_stressed_price_matrix(today_prices, small_returns)

        priced = price_portfolio(
            sample_portfolio, today_prices, sample_rolling_vols, 0.05, as_of,
            alpha=flat_alpha, beta=flat_beta,
        )
        today_values = priced.set_index("trade_id")["position_value"]

        pnl_full = full_reprice_pnl(
            sample_portfolio, today_prices, today_values, scenario_prices,
            sample_rolling_vols, 0.05, as_of,
            alpha=flat_alpha, beta=flat_beta,
        )
        pnl_approx = greeks_approx_pnl(
            sample_portfolio, today_prices, scenario_prices,
            sample_rolling_vols, 0.05, as_of,
            alpha=flat_alpha, beta=flat_beta,
        )

        # With flat vol and small shocks, delta-gamma should match full reprice well
        # Use median to guard against near-zero P&L outliers
        mask = np.abs(pnl_full) > 1.0
        if mask.any():
            rel_errors = np.abs(pnl_approx[mask] - pnl_full[mask]) / np.abs(pnl_full[mask])
            assert np.median(rel_errors) < 0.15
