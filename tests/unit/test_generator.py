"""
tests/unit/test_generator.py
-----------------------------
Unit tests for data/generator.py.

Test IDs follow the GEN-## scheme defined in docs/test_plan.md.
All tests are deterministic: fixtures use fixed seeds.

Run with:
    pytest tests/unit/test_generator.py -v
"""

import numpy as np
import pandas as pd
import pytest

from data import (
    MONEYNESS_LABELS,
    MONEYNESS_LEVELS,
    N_ASSETS,
    N_PER_SECTOR,
    SECTORS,
    TENOR_LABELS,
    TENOR_YEARS,
)
from data.generator import (
    CRISIS_VOL_MULTIPLIER,
    RHO_INTRA,
    MarketData,
    MarketDataGenerator,
)
from data.schemas import DataValidationError, validate_market_data


# ===========================================================================
# GEN-01 to GEN-10: Core correctness
# ===========================================================================

class TestGeneratorCorrectness:

    def test_gen01_price_series_correct_length(self, market_data_small, minimal_config):
        """GEN-01: Generated prices DataFrame has exactly n_trading_days rows."""
        expected = minimal_config["simulation"]["n_trading_days"]
        assert len(market_data_small.prices) == expected

    def test_gen02_prices_strictly_positive(self, market_data_small):
        """GEN-02: All generated equity prices are strictly positive."""
        assert (market_data_small.prices.values > 0).all(), (
            "Found non-positive prices in generated data."
        )

    def test_gen03_log_returns_realistic_moments(self, market_data_standard):
        """
        GEN-03: In the normal (non-crisis) regime, annualised mean return
        is in [-20%, +30%] and annualised vol is in [5%, 65%] for every asset.
        """
        c_start = market_data_standard.crisis_start_idx
        c_end = market_data_standard.crisis_end_idx
        lr = market_data_standard.log_returns

        # Exclude crisis period
        normal_returns = pd.concat([lr.iloc[:c_start], lr.iloc[c_end:]])

        ann_mean = normal_returns.mean() * 252
        ann_vol = normal_returns.std() * np.sqrt(252)

        assert (ann_mean > -0.20).all(), (
            f"Some assets have ann. mean < -20%:\n{ann_mean[ann_mean <= -0.20]}"
        )
        assert (ann_mean < 0.30).all(), (
            f"Some assets have ann. mean > +30%:\n{ann_mean[ann_mean >= 0.30]}"
        )
        assert (ann_vol > 0.05).all(), (
            f"Some assets have ann. vol < 5%:\n{ann_vol[ann_vol <= 0.05]}"
        )
        assert (ann_vol < 0.65).all(), (
            f"Some assets have ann. vol > 65% outside crisis:\n{ann_vol[ann_vol >= 0.65]}"
        )

    def test_gen04_within_sector_correlation_preserved(self, market_data_standard):
        """
        GEN-04: Mean within-sector correlation of log-returns is within ±0.20
        of the target RHO_INTRA = 0.65.
        """
        returns = market_data_standard.log_returns
        corr = returns.corr().values

        within_corrs = []
        for s in range(len(SECTORS)):
            start = s * N_PER_SECTOR
            end = start + N_PER_SECTOR
            block = corr[start:end, start:end]
            mask = ~np.eye(N_PER_SECTOR, dtype=bool)
            within_corrs.extend(block[mask].tolist())

        mean_rho = np.mean(within_corrs)
        assert abs(mean_rho - RHO_INTRA) < 0.20, (
            f"Mean within-sector correlation {mean_rho:.3f} is too far from "
            f"target RHO_INTRA={RHO_INTRA}. Difference: {abs(mean_rho - RHO_INTRA):.3f}"
        )

    def test_gen05_atm_vol_close_to_sector_base(self, market_data_standard):
        """
        GEN-05: Mean ATM vol in the pre-crisis period is within ±10 vol points
        of each sector's base vol.
        """
        from data.generator import SECTOR_PARAMS

        c_start = market_data_standard.crisis_start_idx
        atm = market_data_standard.atm_vols.iloc[:c_start]

        for i, sector in enumerate(SECTORS):
            start = i * N_PER_SECTOR
            tickers = market_data_standard.tickers[start : start + N_PER_SECTOR]
            base_vol = SECTOR_PARAMS[sector]["base_vol"]
            mean_atm = atm[tickers].values.mean()
            assert abs(mean_atm - base_vol) < 0.10, (
                f"Sector {sector}: mean ATM vol {mean_atm:.3f} deviates > 10 vol pts "
                f"from base {base_vol:.3f}."
            )

    def test_gen06_vol_surface_put_skew(self, market_data_small):
        """
        GEN-06: OTM put implied vol (moneyness=0.85) > ATM implied vol (moneyness=1.00)
        on average, for every ticker and tenor.
        """
        vs = market_data_small.vol_surface
        tickers = market_data_small.tickers

        for ticker in tickers:
            for tenor in TENOR_LABELS:
                otm_put_mean = vs[(ticker, "0.85", tenor)].mean()
                atm_mean = vs[(ticker, "1.00", tenor)].mean()
                assert otm_put_mean > atm_mean, (
                    f"Put skew violation: {ticker} {tenor}: "
                    f"0.85-put vol ({otm_put_mean:.4f}) <= ATM vol ({atm_mean:.4f})"
                )

    def test_gen07_vol_surface_term_structure(self, market_data_small):
        """
        GEN-07: Mean 12m ATM implied vol >= mean 1m ATM implied vol
        for every ticker (upward-sloping term structure).
        """
        vs = market_data_small.vol_surface
        for ticker in market_data_small.tickers:
            vol_1m = vs[(ticker, "1.00", "1m")].mean()
            vol_12m = vs[(ticker, "1.00", "12m")].mean()
            assert vol_12m >= vol_1m, (
                f"Term structure violation for {ticker}: "
                f"12m vol ({vol_12m:.4f}) < 1m vol ({vol_1m:.4f})"
            )

    def test_gen08_crisis_produces_significant_drawdown(self, market_data_standard):
        """
        GEN-08: The mean equity price at the end of the crisis period is at
        least 25% below the price at the start of the crisis period.
        """
        prices = market_data_standard.prices
        c_start = market_data_standard.crisis_start_idx
        c_end = market_data_standard.crisis_end_idx

        # Price just before crisis vs at the end of the crisis
        pre_crisis = prices.iloc[max(0, c_start - 1)]
        at_crisis_end = prices.iloc[min(c_end - 1, len(prices) - 1)]

        drawdown = (at_crisis_end - pre_crisis) / pre_crisis
        mean_drawdown = drawdown.mean()

        assert mean_drawdown < -0.25, (
            f"Crisis injection did not produce sufficient drawdown. "
            f"Mean drawdown = {mean_drawdown:.2%} (expected < -25%)"
        )

    def test_gen09_dataset_covers_requested_days(self, market_data_standard, standard_config):
        """GEN-09: Prices and log_returns cover at least n_trading_days rows."""
        requested = standard_config["simulation"]["n_trading_days"]
        assert len(market_data_standard.prices) >= requested
        assert len(market_data_standard.log_returns) >= requested
        assert len(market_data_standard.vol_surface) >= requested

    def test_gen10_reproducibility_same_seed(self, minimal_config):
        """
        GEN-10: Two MarketDataGenerator instances with the same seed
        produce byte-identical prices, log returns, and vol surface.
        """
        gen1 = MarketDataGenerator(config=minimal_config, seed=99)
        gen2 = MarketDataGenerator(config=minimal_config, seed=99)
        md1 = gen1.generate()
        md2 = gen2.generate()

        pd.testing.assert_frame_equal(md1.prices, md2.prices)
        pd.testing.assert_frame_equal(md1.log_returns, md2.log_returns)
        pd.testing.assert_frame_equal(md1.vol_surface, md2.vol_surface)


# ===========================================================================
# Structural / edge-case tests
# ===========================================================================

class TestGeneratorStructure:

    def test_prices_have_correct_number_of_assets(self, market_data_small):
        """prices DataFrame has exactly N_ASSETS (25) columns."""
        assert market_data_small.prices.shape[1] == N_ASSETS

    def test_log_returns_shape_matches_prices(self, market_data_small):
        """log_returns has the same shape as prices."""
        assert market_data_small.log_returns.shape == market_data_small.prices.shape

    def test_vol_surface_multiindex_structure(self, market_data_small):
        """
        Vol surface has a 3-level MultiIndex (ticker, moneyness, tenor)
        with the expected number of columns.
        """
        vs = market_data_small.vol_surface
        assert isinstance(vs.columns, pd.MultiIndex)
        assert vs.columns.nlevels == 3
        expected_cols = N_ASSETS * len(MONEYNESS_LABELS) * len(TENOR_LABELS)
        assert vs.shape[1] == expected_cols, (
            f"Expected {expected_cols} vol surface columns, got {vs.shape[1]}"
        )

    def test_vol_surface_moneyness_labels(self, market_data_small):
        """Vol surface columns contain exactly the expected moneyness labels."""
        vs = market_data_small.vol_surface
        labels_in_cols = vs.columns.get_level_values("moneyness").unique().tolist()
        assert sorted(labels_in_cols) == sorted(MONEYNESS_LABELS)

    def test_vol_surface_tenor_labels(self, market_data_small):
        """Vol surface columns contain exactly the expected tenor labels."""
        vs = market_data_small.vol_surface
        labels_in_cols = vs.columns.get_level_values("tenor").unique().tolist()
        assert sorted(labels_in_cols) == sorted(TENOR_LABELS)

    def test_trading_dates_are_business_days(self, market_data_small):
        """All entries in trading_dates are weekdays (Mon=0 … Fri=4)."""
        dow = market_data_small.trading_dates.day_of_week
        assert (dow < 5).all(), "Non-business days found in trading_dates."

    def test_trading_dates_length_matches_prices(self, market_data_small):
        """trading_dates has the same length as the prices DataFrame."""
        assert len(market_data_small.trading_dates) == len(market_data_small.prices)

    def test_prices_index_equals_trading_dates(self, market_data_small):
        """prices.index is identical to market_data.trading_dates."""
        pd.testing.assert_index_equal(
            market_data_small.prices.index,
            pd.DatetimeIndex(market_data_small.trading_dates),
        )

    def test_log_returns_consistent_with_prices(self, market_data_small):
        """
        For t >= 1: log_returns.iloc[t] == log(prices.iloc[t] / prices.iloc[t-1])
        up to floating-point tolerance.
        """
        ticker = market_data_small.tickers[0]
        prices = market_data_small.prices[ticker]
        returns = market_data_small.log_returns[ticker]

        for t in range(1, 6):
            expected = np.log(prices.iloc[t] / prices.iloc[t - 1])
            actual = returns.iloc[t]
            assert abs(expected - actual) < 1e-10, (
                f"Log return inconsistency at day {t}: "
                f"expected {expected:.8f}, got {actual:.8f}"
            )

    def test_tickers_list_length(self, market_data_small):
        """tickers list contains exactly N_ASSETS entries."""
        assert len(market_data_small.tickers) == N_ASSETS

    def test_sector_mapping_covers_all_tickers(self, market_data_small):
        """sectors dict covers all tickers with no overlaps."""
        sectors = market_data_small.sectors
        all_from_sectors = []
        for sector_tickers in sectors.values():
            all_from_sectors.extend(sector_tickers)
        assert sorted(all_from_sectors) == sorted(market_data_small.tickers)

    def test_dividend_yields_indexed_by_tickers(self, market_data_small):
        """dividend_yields Series index matches the tickers list."""
        assert list(market_data_small.dividend_yields.index) == market_data_small.tickers

    def test_dividend_yields_positive(self, market_data_small):
        """All dividend yields are non-negative."""
        assert (market_data_small.dividend_yields >= 0).all()

    def test_no_crisis_injection(self, market_data_no_crisis):
        """
        Generator produces valid data when crisis injection is disabled
        and prices remain positive throughout.
        """
        md = market_data_no_crisis
        assert (md.prices.values > 0).all()
        # Without crisis, crisis_start_idx and crisis_end_idx should be equal
        # (or both == n_days), indicating no active crisis window.
        assert md.crisis_start_idx >= md.crisis_end_idx or md.crisis_start_idx >= md.n_days

    def test_atm_vols_spike_during_crisis(self, market_data_standard):
        """
        Mean ATM vol during the crisis period is significantly higher
        than in the pre-crisis period (at least 1.5× the multiplier direction).
        """
        atm = market_data_standard.atm_vols
        c_start = market_data_standard.crisis_start_idx
        c_end = market_data_standard.crisis_end_idx

        pre_crisis_vol = atm.iloc[:c_start].mean().mean()
        crisis_vol = atm.iloc[c_start:c_end].mean().mean()

        # Crisis vol should be substantially higher (at least 1.5× pre-crisis)
        assert crisis_vol > 1.5 * pre_crisis_vol, (
            f"Crisis vol ({crisis_vol:.3f}) is not sufficiently above "
            f"pre-crisis vol ({pre_crisis_vol:.3f})"
        )

    def test_different_seeds_produce_different_data(self, minimal_config):
        """Two different seeds produce different price paths."""
        md1 = MarketDataGenerator(config=minimal_config, seed=1).generate()
        md2 = MarketDataGenerator(config=minimal_config, seed=2).generate()
        assert not md1.prices.equals(md2.prices)


# ===========================================================================
# Schema validation tests
# ===========================================================================

class TestSchemaValidation:

    def test_valid_market_data_passes_validation(self, market_data_small):
        """validate_market_data raises no error on well-formed MarketData."""
        validate_market_data(market_data_small)  # must not raise

    def test_n_days_property(self, market_data_small, minimal_config):
        """MarketData.n_days returns the correct count."""
        assert market_data_small.n_days == minimal_config["simulation"]["n_trading_days"]

    def test_n_assets_property(self, market_data_small):
        """MarketData.n_assets returns N_ASSETS (25)."""
        assert market_data_small.n_assets == N_ASSETS
