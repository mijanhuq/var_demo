"""
Unit tests for src/utils/black_scholes.py  (BS-01 through BS-14)
"""

import numpy as np
import pytest
from utils.black_scholes import bs_price, bs_delta, bs_gamma, bs_vega, bs_theta, bs_greeks


# Shared baseline parameters
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestBSPrice:
    def test_BS01_atm_call_value(self):
        """BS-01: ATM call price ≈ 10.45 (well-known analytical value)."""
        price = bs_price(S, K, T, r, sigma, "call")
        assert abs(price - 10.4506) < 0.01

    def test_BS02_put_call_parity(self):
        """BS-02: C - P = S - K·e^{-rT}  (no dividends)."""
        call = bs_price(S, K, T, r, sigma, "call")
        put  = bs_price(S, K, T, r, sigma, "put")
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-8

    def test_BS03_deep_itm_call_intrinsic(self):
        """BS-03: Deep ITM call → intrinsic value as σ→0 (r=0 to avoid discounting)."""
        price = bs_price(200.0, K, T, r=0.0, sigma=1e-6, option_type="call")
        intrinsic = max(200.0 - K, 0.0)
        assert abs(price - intrinsic) < 0.01

    def test_BS04_deep_otm_call_near_zero(self):
        """BS-04: Deep OTM call → ~0 as σ→0."""
        price = bs_price(50.0, K, T, r, 1e-6, "call")
        assert price < 1e-4

    def test_BS10_put_non_negative(self):
        """BS-10: Put price ≥ 0 for a range of inputs."""
        for spot in [50, 100, 150, 200]:
            for strike in [80, 100, 120]:
                p = bs_price(float(spot), float(strike), T, r, sigma, "put")
                assert p >= 0.0

    def test_BS11_vectorised_shape(self):
        """BS-09/BS-11: Vectorised call over strike array returns correct shape."""
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = bs_price(S, strikes, T, r, sigma, "call")
        assert prices.shape == (5,)

    def test_BS12_zero_rate_no_exception(self):
        """BS-12: r=0 produces finite, non-negative prices."""
        p = bs_price(S, K, T, 0.0, sigma, "call")
        assert np.isfinite(p) and p >= 0.0

    def test_BS13_very_short_dated(self):
        """BS-13: T=1/252 — no division by zero."""
        p = bs_price(S, K, 1 / 252, r, sigma, "call")
        assert np.isfinite(p) and p >= 0.0

    def test_BS14_very_long_dated(self):
        """BS-14: T=10 — finite price."""
        p = bs_price(S, K, 10.0, r, sigma, "call")
        assert np.isfinite(p) and p >= 0.0

    def test_expired_call_intrinsic(self):
        """T=0: expired call returns max(S-K, 0)."""
        assert bs_price(110.0, 100.0, 0.0, r, sigma, "call") == pytest.approx(10.0)
        assert bs_price(90.0, 100.0, 0.0, r, sigma, "call") == pytest.approx(0.0)

    def test_expired_put_intrinsic(self):
        """T=0: expired put returns max(K-S, 0)."""
        assert bs_price(90.0, 100.0, 0.0, r, sigma, "put") == pytest.approx(10.0)
        assert bs_price(110.0, 100.0, 0.0, r, sigma, "put") == pytest.approx(0.0)


class TestBSGreeks:
    def test_BS05_call_delta_range(self):
        """BS-05: Call delta ∈ (0, 1)."""
        d = bs_delta(S, K, T, r, sigma, "call")
        assert 0 < d < 1

    def test_BS06_put_delta_range(self):
        """BS-06: Put delta ∈ (-1, 0)."""
        d = bs_delta(S, K, T, r, sigma, "put")
        assert -1 < d < 0

    def test_BS07_gamma_positive(self):
        """BS-07: Gamma > 0."""
        g = bs_gamma(S, K, T, r, sigma)
        assert g > 0

    def test_BS08_vega_positive(self):
        """BS-08: Vega > 0."""
        v = bs_vega(S, K, T, r, sigma)
        assert v > 0

    def test_BS09_theta_negative_long_call(self):
        """BS-09: Theta < 0 for a long call."""
        th = bs_theta(S, K, T, r, sigma, "call")
        assert th < 0

    def test_greeks_dict_keys(self):
        """bs_greeks returns all expected keys."""
        g = bs_greeks(S, K, T, r, sigma, "call")
        assert set(g.keys()) == {"delta", "gamma", "vega", "theta"}

    def test_call_put_same_gamma_vega(self):
        """Gamma and Vega are identical for call and put (BS symmetry)."""
        g_call = bs_gamma(S, K, T, r, sigma)
        g_put  = bs_gamma(S, K, T, r, sigma)
        v_call = bs_vega(S, K, T, r, sigma)
        v_put  = bs_vega(S, K, T, r, sigma)
        assert g_call == pytest.approx(g_put, rel=1e-9)
        assert v_call == pytest.approx(v_put, rel=1e-9)

    def test_delta_call_plus_put_equals_one(self):
        """Δ_call - Δ_put = 1 (put-call parity for delta)."""
        dc = bs_delta(S, K, T, r, sigma, "call")
        dp = bs_delta(S, K, T, r, sigma, "put")
        assert abs(dc - dp - 1.0) < 1e-9
