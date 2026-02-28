"""
Unit tests for src/data/vol_surface.py  (VS-01 through VS-11)
"""

import numpy as np
import pytest
import pandas as pd
from data.vol_surface import (
    compute_rolling_vol,
    get_atm_vol,
    skew_vol,
    surface_vol,
    VOL_WINDOWS,
    VOL_CAP,
    VOL_FLOOR,
)


class TestRollingVol:
    def test_VS01_shape(self, sample_prices, sample_rolling_vols):
        """VS-01: Rolling vol output shape = (n_days, n_tickers) per window."""
        for label, df in sample_rolling_vols.items():
            assert df.shape == sample_prices.shape

    def test_VS02_positive_values(self, sample_rolling_vols):
        """VS-02: All rolling vol values > 0 after warm-up."""
        window = max(VOL_WINDOWS.values())  # longest warm-up
        for label, df in sample_rolling_vols.items():
            valid = df.iloc[window:].dropna()
            assert (valid > 0).all().all()

    def test_VS03_realistic_range(self, sample_rolling_vols):
        """VS-03: Annualised vol ∈ (0, 3.0] for realistic equity data."""
        for label, df in sample_rolling_vols.items():
            valid = df.dropna()
            assert (valid <= 3.0).all().all()

    def test_VS06_no_negative_vols(self, sample_rolling_vols):
        """VS-07: No negative vols anywhere."""
        for label, df in sample_rolling_vols.items():
            valid = df.dropna()
            assert (valid > 0).all().all()


class TestGetAtmVol:
    def test_short_tenor_uses_21d_window(self, sample_rolling_vols, sample_prices):
        """VS-04: ATM vol for short tenor (< 63/252 yr) uses 21-day window."""
        ticker = sample_prices.columns[0]
        date = sample_prices.index[-1]
        T_short = 0.1  # < 63/252 ≈ 0.25

        atm = get_atm_vol(sample_rolling_vols, ticker, date, T_short)
        expected = float(sample_rolling_vols["short"].loc[date, ticker])
        assert atm == pytest.approx(expected, abs=1e-6)

    def test_long_tenor_uses_126d_window(self, sample_rolling_vols, sample_prices):
        """VS-05: ATM vol for long tenor (>= 126/252 yr) uses 126-day window."""
        ticker = sample_prices.columns[0]
        date = sample_prices.index[-1]
        T_long = 0.6  # >= 126/252 ≈ 0.5

        atm = get_atm_vol(sample_rolling_vols, ticker, date, T_long)
        expected = float(sample_rolling_vols["long"].loc[date, ticker])
        assert atm == pytest.approx(expected, abs=1e-6)

    def test_positive_atm_vol(self, sample_rolling_vols, sample_prices):
        ticker = sample_prices.columns[0]
        date = sample_prices.index[-1]
        atm = get_atm_vol(sample_rolling_vols, ticker, date, 0.5)
        assert atm > 0


class TestSkewVol:
    def test_VS08_atm_equals_sigma_atm(self):
        """VS-08: Skew at k=0 (ATM) equals σ_ATM."""
        atm = 0.20
        S = 100.0
        # K = F at ATM (k=0); approximate with K ≈ S (small r, small T)
        T = 0.01   # very short T so F ≈ S
        sigma = skew_vol(atm, K=100.0, S=S, r=0.0, T=T, alpha=-0.15, beta=0.05)
        assert abs(sigma - atm) < 0.01  # should be close for near-zero moneyness

    def test_VS09_negative_skew_otm_put(self):
        """VS-09: OTM put (K < F) has higher vol than ATM when alpha < 0."""
        atm = 0.20
        S = 100.0
        r = 0.05
        T = 1.0
        F = S * np.exp(r * T)
        K_otm_put = F * 0.85  # below forward → k < 0 → σ increases with alpha < 0
        sigma_otm = skew_vol(atm, K=K_otm_put, S=S, r=r, T=T, alpha=-0.15, beta=0.05)
        assert sigma_otm > atm  # negative alpha → more vol for k < 0

    def test_vol_floored_at_positive(self):
        """Skew vol never goes negative."""
        sigma = skew_vol(0.05, K=200.0, S=100.0, r=0.05, T=1.0, alpha=-2.0, beta=0.0)
        assert sigma >= 1e-4

    def test_VS11_analytical_formula(self):
        """VS-11: σ(k, T) matches analytical formula."""
        atm = 0.20
        S, K, r, T = 100.0, 95.0, 0.05, 0.5
        alpha, beta = -0.15, 0.05
        F = S * np.exp(r * T)
        k = np.log(K / F)
        raw = atm + alpha * k + beta * k**2
        expected = float(np.clip(raw, VOL_FLOOR, VOL_CAP))
        result = skew_vol(atm, K, S, r, T, alpha=alpha, beta=beta)
        assert result == pytest.approx(expected, abs=1e-8)

    def test_vol_capped_at_vol_cap(self):
        """VS-12: Skew vol never exceeds VOL_CAP even for extreme high ATM vol."""
        sigma = skew_vol(10.0, K=50.0, S=100.0, r=0.05, T=1.0, alpha=5.0, beta=0.0)
        assert sigma <= VOL_CAP

    def test_vol_bounded_above_vol_floor(self):
        """VS-13: Skew vol never falls below VOL_FLOOR."""
        sigma = skew_vol(0.001, K=500.0, S=100.0, r=0.05, T=1.0, alpha=-5.0, beta=0.0)
        assert sigma >= VOL_FLOOR
