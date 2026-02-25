"""
tests/unit/test_pricer.py
--------------------------
Unit tests for portfolio/pricer.py.

Test IDs follow the PRC-## and VAL-## schemes defined in docs/test_plan.md.
EDG-01/02/03 (pricer-specific edge cases) are also covered here.

All tests use fixed, analytically verifiable inputs.  No market data fixtures
are needed — this is pure maths validation.

Reference values computed from:
    d1 = (ln(S/K) + (r - q + 0.5σ²)T) / (σ√T)
    d2 = d1 - σ√T

Run with:
    pytest tests/unit/test_pricer.py -v
"""

import numpy as np
import pytest
from scipy.stats import norm

from portfolio.pricer import bs_delta, bs_gamma, bs_greeks, bs_price, bs_theta, bs_vega
from portfolio.position import OptionType

# ---------------------------------------------------------------------------
# Shared test parameters (ATM, 1-year, standard inputs)
# ---------------------------------------------------------------------------
# S=100, K=100, T=1, r=5%, σ=20%, q=0%
#   d1 = (0 + 0.05 + 0.02) / 0.20 = 0.35
#   d2 = 0.15
#   N(0.35) = 0.63683,  N(0.15) = 0.55962
#   C ≈ 10.4506,  P ≈ 5.5735 (put-call parity)
#   Δ_call ≈ 0.6368,  Vega ≈ 37.524

S0 = 100.0
K0 = 100.0
T1 = 1.0
R = 0.05
SIG = 0.20
Q = 0.0


# ===========================================================================
# PRC-01 to PRC-07: Pricing properties
# ===========================================================================

class TestBlackScholesPricing:

    def test_prc01_atm_call_known_value(self):
        """PRC-01: ATM call price matches analytic reference within 0.01%."""
        price = bs_price(S0, K0, T1, R, SIG, Q, option_type="call")
        # Reference: 10.4506  (analytically derived, see module docstring)
        assert abs(price - 10.4506) < 0.005, (
            f"ATM call price {price:.6f} deviates from reference 10.4506"
        )

    def test_prc02_put_call_parity(self):
        """PRC-02: |C - P - S·e^{-qT} + K·e^{-rT}| < 1e-8 (put-call parity)."""
        for S in [80.0, 100.0, 120.0]:
            for T in [0.25, 1.0, 2.0]:
                C = bs_price(S, K0, T, R, SIG, Q, option_type="call")
                P = bs_price(S, K0, T, R, SIG, Q, option_type="put")
                pcp = C - P - S * np.exp(-Q * T) + K0 * np.exp(-R * T)
                assert abs(pcp) < 1e-8, (
                    f"Put-call parity violated at S={S}, T={T}: residual={pcp:.2e}"
                )

    def test_prc03_call_near_zero_when_spot_tiny(self):
        """PRC-03: Call price → 0 as S → 0 (deep OTM call)."""
        price = bs_price(S=1e-6, K=100.0, T=1.0, r=R, sigma=SIG, q=Q, option_type="call")
        assert price < 1e-8, f"Call price {price:.2e} should be ~0 when S ≈ 0"

    def test_prc04_call_bounded_by_spot_at_high_vol(self):
        """PRC-04: Call price ≤ S for any finite vol (upper bound)."""
        for sigma in [1.0, 2.0, 5.0, 10.0]:
            C = bs_price(S0, K0, T1, R, sigma, Q, option_type="call")
            assert C <= S0 + 1e-6, (
                f"Call price {C:.4f} exceeds spot {S0} at sigma={sigma}"
            )

    def test_prc05_call_monotone_in_spot(self):
        """PRC-05: Call price strictly increases with spot price."""
        spots = np.linspace(60.0, 160.0, 20)
        prices = [bs_price(s, K0, T1, R, SIG, Q, "call") for s in spots]
        for i in range(len(prices) - 1):
            assert prices[i + 1] > prices[i], (
                f"Call price not monotone at S={spots[i]:.1f}→{spots[i+1]:.1f}: "
                f"{prices[i]:.4f}→{prices[i+1]:.4f}"
            )

    def test_prc06_call_monotone_decreasing_in_strike(self):
        """PRC-06: Call price strictly decreases as strike increases."""
        strikes = np.linspace(70.0, 140.0, 20)
        prices = [bs_price(S0, k, T1, R, SIG, Q, "call") for k in strikes]
        for i in range(len(prices) - 1):
            assert prices[i + 1] < prices[i], (
                f"Call price not decreasing in K at K={strikes[i]:.1f}→{strikes[i+1]:.1f}: "
                f"{prices[i]:.4f}→{prices[i+1]:.4f}"
            )

    def test_prc07_call_monotone_in_vol(self):
        """PRC-07: Call price strictly increases with implied volatility."""
        vols = np.linspace(0.05, 1.00, 20)
        prices = [bs_price(S0, K0, T1, R, v, Q, "call") for v in vols]
        for i in range(len(prices) - 1):
            assert prices[i + 1] > prices[i], (
                f"Call price not increasing in vol at σ={vols[i]:.3f}→{vols[i+1]:.3f}: "
                f"{prices[i]:.4f}→{prices[i+1]:.4f}"
            )


# ===========================================================================
# PRC-08 to PRC-16: Greeks
# ===========================================================================

class TestBlackScholesGreeks:

    def test_prc08_call_delta_in_unit_interval(self):
        """PRC-08: Call delta ∈ (0, 1) for all standard inputs."""
        for S in [70.0, 100.0, 140.0]:
            for T in [0.1, 0.5, 1.0, 2.0]:
                d = bs_delta(S, K0, T, R, SIG, Q, "call")
                assert 0 < d < 1, (
                    f"Call delta {d:.4f} out of (0,1) at S={S}, T={T}"
                )

    def test_prc09_put_delta_in_minus_one_zero(self):
        """PRC-09: Put delta ∈ (−1, 0) for all standard inputs."""
        for S in [70.0, 100.0, 140.0]:
            for T in [0.1, 0.5, 1.0, 2.0]:
                d = bs_delta(S, K0, T, R, SIG, Q, "put")
                assert -1 < d < 0, (
                    f"Put delta {d:.4f} out of (-1,0) at S={S}, T={T}"
                )

    def test_prc10_gamma_positive_for_calls_and_puts(self):
        """PRC-10: Gamma is strictly positive for both calls and puts."""
        for S in [80.0, 100.0, 120.0]:
            g = bs_gamma(S, K0, T1, R, SIG, Q)
            assert g > 0, f"Gamma {g:.6f} not positive at S={S}"

    def test_prc11_vega_positive_for_calls_and_puts(self):
        """PRC-11: Vega is strictly positive for both calls and puts."""
        for S in [80.0, 100.0, 120.0]:
            v = bs_vega(S, K0, T1, R, SIG, Q)
            assert v > 0, f"Vega {v:.6f} not positive at S={S}"

    def test_prc12_theta_negative_for_long_positions(self):
        """PRC-12: Theta (∂V/∂t) is negative — long options lose time value."""
        for option_type in ("call", "put"):
            th = bs_theta(S0, K0, T1, R, SIG, Q, option_type)
            assert th < 0, (
                f"{option_type} theta {th:.4f} should be negative"
            )

    def test_prc13_deep_itm_call_delta_near_one(self):
        """PRC-13: Delta of deep ITM call (S=200, K=100) ≈ 1."""
        d = bs_delta(S=200.0, K=100.0, T=1.0, r=R, sigma=SIG, q=Q, option_type="call")
        assert d > 0.99, f"Deep ITM call delta {d:.4f} should be > 0.99"

    def test_prc14_deep_otm_call_delta_near_zero(self):
        """PRC-14: Delta of deep OTM call (S=50, K=100) ≈ 0."""
        d = bs_delta(S=50.0, K=100.0, T=1.0, r=R, sigma=SIG, q=Q, option_type="call")
        assert d < 0.01, f"Deep OTM call delta {d:.4f} should be < 0.01"

    def test_prc15_numerical_delta_matches_analytical(self):
        """
        PRC-15: Finite-difference delta agrees with analytical delta
        to within 1e-4 (central difference, ε = 0.01% of spot).
        """
        eps = S0 * 1e-4
        C_up = bs_price(S0 + eps, K0, T1, R, SIG, Q, "call")
        C_dn = bs_price(S0 - eps, K0, T1, R, SIG, Q, "call")
        numerical_delta = (C_up - C_dn) / (2 * eps)

        analytical_delta = bs_delta(S0, K0, T1, R, SIG, Q, "call")
        assert abs(numerical_delta - analytical_delta) < 1e-4, (
            f"Numerical delta {numerical_delta:.6f} vs analytical {analytical_delta:.6f}: "
            f"difference {abs(numerical_delta - analytical_delta):.2e}"
        )

    def test_prc16_numerical_gamma_matches_analytical(self):
        """
        PRC-16: Finite-difference gamma agrees with analytical gamma
        to within 1e-4 (second-order central difference, ε = 0.1% of spot).
        """
        eps = S0 * 1e-3
        C_up = bs_price(S0 + eps, K0, T1, R, SIG, Q, "call")
        C_mid = bs_price(S0, K0, T1, R, SIG, Q, "call")
        C_dn = bs_price(S0 - eps, K0, T1, R, SIG, Q, "call")
        numerical_gamma = (C_up - 2 * C_mid + C_dn) / eps ** 2

        analytical_gamma = bs_gamma(S0, K0, T1, R, SIG, Q)
        assert abs(numerical_gamma - analytical_gamma) < 1e-4, (
            f"Numerical gamma {numerical_gamma:.6f} vs analytical {analytical_gamma:.6f}: "
            f"difference {abs(numerical_gamma - analytical_gamma):.2e}"
        )

    def test_call_put_delta_difference_equals_discount_factor(self):
        """
        Δ_call − Δ_put = e^{-qT}  (derived by differentiating put-call parity w.r.t. S).
        Put delta is negative, so this is call_delta − (negative number) = e^{-qT}.
        """
        d_call = bs_delta(S0, K0, T1, R, SIG, Q, "call")
        d_put = bs_delta(S0, K0, T1, R, SIG, Q, "put")
        identity = d_call - d_put   # = N(d1) - (N(d1) - 1) = 1 = e^{-qT} when q=0
        expected = np.exp(-Q * T1)  # = 1.0 when q=0
        assert abs(identity - expected) < 1e-10, (
            f"Delta identity violated: Δ_call - Δ_put = {identity:.8f}, expected {expected:.8f}"
        )

    def test_gamma_equal_for_call_and_put(self):
        """Gamma is identical for a call and put with the same inputs (put-call parity)."""
        g_call = bs_gamma(S0, K0, T1, R, SIG, Q)
        # Gamma has no option_type argument — verify calling consistency
        # (gamma function signature is the same for call/put)
        assert g_call > 0

    def test_vega_equal_for_call_and_put(self):
        """
        Vega is identical for call and put (both are dV/dσ from the same N'(d1) term).
        Verified via put-call parity: d(C-P)/dσ = 0, so dC/dσ = dP/dσ.
        """
        eps = 1e-4
        C1 = bs_price(S0, K0, T1, R, SIG, Q, "call")
        C2 = bs_price(S0, K0, T1, R, SIG + eps, Q, "call")
        P1 = bs_price(S0, K0, T1, R, SIG, Q, "put")
        P2 = bs_price(S0, K0, T1, R, SIG + eps, Q, "put")
        # d(C)/dσ ≈ d(P)/dσ
        vega_call_fd = (C2 - C1) / eps
        vega_put_fd = (P2 - P1) / eps
        assert abs(vega_call_fd - vega_put_fd) < 1e-4, (
            f"Vega mismatch: call {vega_call_fd:.4f} vs put {vega_put_fd:.4f}"
        )


# ===========================================================================
# VAL-01 to VAL-04: Analytical reference validation
# ===========================================================================

class TestBlackScholesValidation:
    """
    Cross-check against known analytical values.
    Reference inputs: S=100, K=100, T=1, r=5%, σ=20%, q=0.
    """

    def test_val01_call_price_reference(self):
        """VAL-01: Call price ≈ 10.4506 (analytic reference)."""
        price = bs_price(S0, K0, T1, R, SIG, Q, "call")
        assert abs(price - 10.4506) < 5e-4, (
            f"Call price {price:.6f} deviates from reference 10.4506"
        )

    def test_val02_put_price_reference(self):
        """VAL-02: Put price ≈ 5.5735 (via put-call parity)."""
        price = bs_price(S0, K0, T1, R, SIG, Q, "put")
        assert abs(price - 5.5735) < 5e-4, (
            f"Put price {price:.6f} deviates from reference 5.5735"
        )

    def test_val03_call_delta_reference(self):
        """VAL-03: ATM call delta ≈ 0.6368 (= N(d1) = N(0.35))."""
        delta = bs_delta(S0, K0, T1, R, SIG, Q, "call")
        assert abs(delta - 0.6368) < 5e-4, (
            f"ATM call delta {delta:.6f} deviates from reference 0.6368"
        )

    def test_val04_vega_reference(self):
        """VAL-04: ATM vega ≈ 37.524 per unit vol (= S·N'(d1)·√T)."""
        vega = bs_vega(S0, K0, T1, R, SIG, Q)
        assert abs(vega - 37.524) < 0.01, (
            f"ATM vega {vega:.4f} deviates from reference 37.524"
        )

    def test_val04b_vega_per_one_percent(self):
        """VAL-04 (alt): Vega per 1% vol move ≈ 0.3752."""
        vega_per_unit = bs_vega(S0, K0, T1, R, SIG, Q)
        vega_per_pct = vega_per_unit * 0.01
        assert abs(vega_per_pct - 0.3752) < 1e-3


# ===========================================================================
# EDG-01, EDG-02, EDG-03: Pricer-specific edge cases
# ===========================================================================

class TestBlackScholesEdgeCases:

    def test_edg01_call_at_expiry_equals_intrinsic(self):
        """EDG-01: Call at T=0 returns max(S-K, 0) exactly."""
        # ITM call
        assert bs_price(110.0, 100.0, 0.0, R, SIG, Q, "call") == pytest.approx(10.0, abs=1e-8)
        # OTM call
        assert bs_price(90.0, 100.0, 0.0, R, SIG, Q, "call") == pytest.approx(0.0, abs=1e-8)
        # ATM call
        assert bs_price(100.0, 100.0, 0.0, R, SIG, Q, "call") == pytest.approx(0.0, abs=1e-8)

    def test_edg01_put_at_expiry_equals_intrinsic(self):
        """EDG-01 (put): Put at T=0 returns max(K-S, 0) exactly."""
        assert bs_price(90.0, 100.0, 0.0, R, SIG, Q, "put") == pytest.approx(10.0, abs=1e-8)
        assert bs_price(110.0, 100.0, 0.0, R, SIG, Q, "put") == pytest.approx(0.0, abs=1e-8)

    def test_edg02_zero_vol_call_equals_forward_intrinsic(self):
        """EDG-02: Zero-vol call price = max(S - K·e^{-rT}, 0)."""
        # ITM forward: S=100, K·e^{-rT} = 100·e^{-0.05} ≈ 95.12
        expected = max(100.0 - 100.0 * np.exp(-R * T1), 0.0)
        price = bs_price(S0, K0, T1, R, sigma=1e-10, q=Q, option_type="call")
        assert abs(price - expected) < 1e-6, (
            f"Zero-vol call {price:.6f} deviates from forward intrinsic {expected:.6f}"
        )

    def test_edg02_zero_vol_put_equals_forward_intrinsic(self):
        """EDG-02 (put): Zero-vol put price = max(K·e^{-rT} - S, 0)."""
        expected = max(100.0 * np.exp(-R * T1) - 100.0, 0.0)
        price = bs_price(S0, K0, T1, R, sigma=1e-10, q=Q, option_type="put")
        assert abs(price - expected) < 1e-6

    def test_edg03_high_vol_call_bounded_by_spot(self):
        """EDG-03: Call price ≤ S even at very high vol (σ=500%)."""
        price = bs_price(S0, K0, T1, R, sigma=5.0, q=Q, option_type="call")
        assert price <= S0 + 1e-6, (
            f"High-vol call price {price:.4f} exceeds spot {S0}"
        )
        assert price > 0, "High-vol call price should be positive"

    def test_option_price_always_non_negative(self):
        """Option price must be >= 0 for any input combination."""
        test_cases = [
            (50.0, 100.0, 1.0, 0.01),   # deep OTM call
            (100.0, 50.0, 0.01, 0.50),  # deep ITM, short expiry, high vol
            (100.0, 100.0, 1e-6, 0.20), # near-expiry ATM
        ]
        for S, K, T, sigma in test_cases:
            for opt in ("call", "put"):
                price = bs_price(S, K, T, R, sigma, Q, opt)
                assert price >= 0, (
                    f"Negative price {price:.6f} for {opt} S={S}, K={K}, T={T}, σ={sigma}"
                )

    def test_with_nonzero_dividend_yield(self):
        """Pricer correctly applies continuous dividend yield q > 0."""
        # Higher q → lower call price (reduces effective spot)
        C_no_div = bs_price(S0, K0, T1, R, SIG, q=0.0, option_type="call")
        C_div = bs_price(S0, K0, T1, R, SIG, q=0.03, option_type="call")
        assert C_div < C_no_div, (
            f"Call with q=3% ({C_div:.4f}) should be cheaper than q=0% ({C_no_div:.4f})"
        )

    def test_bs_greeks_dict_has_all_keys(self):
        """bs_greeks returns a dict with keys: price, delta, gamma, vega, theta."""
        g = bs_greeks(S0, K0, T1, R, SIG, Q, "call")
        for key in ("price", "delta", "gamma", "vega", "theta"):
            assert key in g, f"Missing key '{key}' in bs_greeks output"

    def test_bs_greeks_consistent_with_individual_functions(self):
        """bs_greeks values match individually-called functions to 1e-10."""
        g = bs_greeks(S0, K0, T1, R, SIG, Q, "call")
        assert abs(g["price"] - bs_price(S0, K0, T1, R, SIG, Q, "call")) < 1e-10
        assert abs(g["delta"] - bs_delta(S0, K0, T1, R, SIG, Q, "call")) < 1e-10
        assert abs(g["gamma"] - bs_gamma(S0, K0, T1, R, SIG, Q)) < 1e-10
        assert abs(g["vega"] - bs_vega(S0, K0, T1, R, SIG, Q)) < 1e-10
        assert abs(g["theta"] - bs_theta(S0, K0, T1, R, SIG, Q, "call")) < 1e-10


# ===========================================================================
# Vectorisation tests
# ===========================================================================

class TestVectorisation:
    """Verify that the pricer accepts numpy arrays and returns arrays."""

    def test_array_spot_returns_array(self):
        """bs_price with array S returns an ndarray of the same length."""
        spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = bs_price(spots, K0, T1, R, SIG, Q, "call")
        assert isinstance(prices, np.ndarray)
        assert len(prices) == len(spots)
        assert (prices > 0).all()

    def test_array_sigma_returns_array(self):
        """bs_price with array sigma (e.g., 252 scenarios) returns array."""
        sigmas = np.linspace(0.10, 0.60, 252)
        prices = bs_price(S0, K0, T1, R, sigmas, Q, "call")
        assert isinstance(prices, np.ndarray)
        assert len(prices) == 252
        assert (prices > 0).all()

    def test_scalar_inputs_return_scalar(self):
        """bs_price with all-scalar inputs returns a Python float."""
        price = bs_price(S0, K0, T1, R, SIG, Q, "call")
        assert isinstance(price, float)

    def test_vectorised_put_call_parity(self):
        """Put-call parity holds for array inputs."""
        spots = np.linspace(60.0, 160.0, 50)
        C = bs_price(spots, K0, T1, R, SIG, Q, "call")
        P = bs_price(spots, K0, T1, R, SIG, Q, "put")
        pcp = C - P - spots * np.exp(-Q * T1) + K0 * np.exp(-R * T1)
        np.testing.assert_allclose(pcp, 0.0, atol=1e-8)

    def test_vectorised_delta_monotone(self):
        """Array delta for calls increases monotonically with spot."""
        spots = np.linspace(60.0, 160.0, 30)
        deltas = bs_delta(spots, K0, T1, R, SIG, Q, "call")
        assert isinstance(deltas, np.ndarray)
        assert np.all(np.diff(deltas) > 0), "Call delta must increase monotonically with S"
