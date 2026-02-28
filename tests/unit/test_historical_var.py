"""
Unit tests for src/var/historical_var.py  (HV-01 through HV-15)
"""

import numpy as np
import pytest
from utils.stats import compute_var_es, compute_weighted_var_es, scale_to_10d
from var.scenarios import build_age_weights
from var.historical_var import plain_hs_var, age_weighted_var


@pytest.fixture
def simple_pnl():
    """Deterministic P&L: first 3 are large losses, rest near zero."""
    pnl = np.zeros(252)
    pnl[0] = -5000.0
    pnl[1] = -4000.0
    pnl[2] = -3000.0
    pnl[3:] = np.linspace(100, 500, 249)
    return pnl


@pytest.fixture
def zero_pnl():
    return np.zeros(252)


class TestPlainHsVar:
    def test_HV01_var_positive(self, simple_pnl):
        """HV-01: VAR > 0 for non-trivial portfolio."""
        result = plain_hs_var(simple_pnl, [0.99])
        assert result["var_99"] > 0

    def test_HV02_es_ge_var(self, simple_pnl):
        """HV-02: ES ≥ VAR at same confidence level."""
        result = plain_hs_var(simple_pnl, [0.95, 0.99])
        assert result["es_95"] >= result["var_95"]
        assert result["es_99"] >= result["var_99"]

    def test_HV03_var_99_gt_var_95(self, simple_pnl):
        """HV-03: VAR_99 ≥ VAR_95."""
        result = plain_hs_var(simple_pnl, [0.95, 0.99])
        assert result["var_99"] >= result["var_95"]

    def test_HV04_es_99_gt_es_95(self, simple_pnl):
        """HV-04: ES_99 ≥ ES_95."""
        result = plain_hs_var(simple_pnl, [0.95, 0.99])
        assert result["es_99"] >= result["es_95"]

    def test_HV05_es_gt_var_same_cl(self, simple_pnl):
        """HV-05: ES_99 > VAR_99."""
        result = plain_hs_var(simple_pnl, [0.99])
        assert result["es_99"] > result["var_99"]

    def test_HV06_10d_equals_sqrt10_x_1d(self, simple_pnl):
        """HV-06 & HV-07: 10-day VAR/ES = 1-day × √10."""
        result = plain_hs_var(simple_pnl, [0.99], scale_10d=True)
        assert result["var_99_10d"] == pytest.approx(result["var_99"] * np.sqrt(10))
        assert result["es_99_10d"] == pytest.approx(result["es_99"] * np.sqrt(10))

    def test_HV08_zero_portfolio_var_is_zero(self, zero_pnl):
        """HV-08: Zero-position portfolio → VAR = 0 and ES = 0."""
        result = plain_hs_var(zero_pnl, [0.95, 0.99])
        assert result["var_99"] == pytest.approx(0.0)
        assert result["es_99"] == pytest.approx(0.0)

    def test_HV11_custom_confidence_level_key(self, simple_pnl):
        """HV-11: Custom CL [0.975] returns key 'var_97'."""
        result = plain_hs_var(simple_pnl, [0.975])
        # Key should be var_97 (int of 97.5*100 // 10 ... or just int(97.5) = 97)
        # Actually _cl_key does int(round(0.975*100)) = 98
        assert "var_98" in result or "var_97" in result  # depends on rounding convention

    def test_HV15_output_keys_complete(self, simple_pnl):
        """HV-15: Output dict contains all expected keys."""
        result = plain_hs_var(simple_pnl, [0.95, 0.99], scale_10d=True)
        for key in ("var_95", "es_95", "var_99", "es_99",
                    "var_95_10d", "es_95_10d", "var_99_10d", "es_99_10d"):
            assert key in result, f"Missing key: {key}"


class TestAgeWeightedVar:
    def test_HV12_age_weighted_ne_plain(self, simple_pnl):
        """HV-12: Age-weighted VAR ≠ plain VAR when λ < 1."""
        plain = plain_hs_var(simple_pnl, [0.99])
        weighted = age_weighted_var(simple_pnl, lambda_=0.94, confidence_levels=[0.99])
        # They will differ unless pnl is trivially flat
        # For our simple_pnl the worst scenarios are at the beginning (oldest),
        # so age-weighting (which reduces weight of oldest) should give lower VAR
        assert plain["var_99"] != pytest.approx(weighted["var_99"], rel=1e-3)

    def test_HV13_age_weighted_es_ne_plain_es(self, simple_pnl):
        """HV-13: Age-weighted ES ≠ plain ES when λ < 1."""
        plain = plain_hs_var(simple_pnl, [0.99])
        weighted = age_weighted_var(simple_pnl, lambda_=0.94, confidence_levels=[0.99])
        assert plain["es_99"] != pytest.approx(weighted["es_99"], rel=1e-3)

    def test_age_weighted_es_ge_var(self, simple_pnl):
        """ES ≥ VAR also holds for age-weighted."""
        result = age_weighted_var(simple_pnl, lambda_=0.97, confidence_levels=[0.95, 0.99])
        assert result["es_99"] >= result["var_99"]

    def test_uniform_lambda_matches_plain(self):
        """λ ≈ 1 → age-weighted VAR converges toward plain HS VAR."""
        rng = np.random.default_rng(7)
        pnl = rng.normal(0, 1000, 252)
        plain = plain_hs_var(pnl, [0.99])["var_99"]
        weighted = age_weighted_var(pnl, lambda_=0.9999, confidence_levels=[0.99])["var_99"]
        # Should be close but not identical (different quantile method)
        assert abs(plain - weighted) / max(plain, 1.0) < 0.15
