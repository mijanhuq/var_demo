"""
Unit tests for src/var/filtered_var.py  (FV-01 through FV-07)

Note: GARCH fitting requires the `arch` library.
Tests that invoke GARCH are marked @pytest.mark.slow.
"""

import numpy as np
import pytest
from utils.stats import compute_var_es


@pytest.mark.slow
class TestFitGarch:
    def test_FV01_garch_stationarity(self):
        """FV-01: GARCH(1,1) parameters α + β < 1 (stationarity condition)."""
        from var.filtered_var import fit_garch
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        result = fit_garch(returns)
        assert result["alpha"] + result["beta"] < 1.0

    def test_FV02_conditional_vol_positive(self):
        """FV-02: Conditional vol series is positive throughout."""
        from var.filtered_var import fit_garch
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        result = fit_garch(returns)
        assert (result["conditional_vol"] > 0).all()

    def test_FV03_standardised_residuals_unit_variance(self):
        """FV-03: Standardised residuals have approximately unit variance."""
        from var.filtered_var import fit_garch
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        result = fit_garch(returns)
        std = result["standardised_resid"].std()
        assert abs(std - 1.0) < 0.25  # relaxed tolerance for finite sample


@pytest.mark.slow
class TestFilteredVarHighLevel:
    def test_FV06_filtered_es_ge_var(self, sample_portfolio, sample_prices, as_of):
        """FV-06: Filtered ES ≥ filtered VAR."""
        from var.filtered_var import compute_filtered_var
        result = compute_filtered_var(
            sample_portfolio, sample_prices, r=0.05, as_of=as_of,
            confidence_levels=[0.95, 0.99],
        )
        assert result["es_99"] >= result["var_99"]
        assert result["es_95"] >= result["var_95"]

    def test_FV07_output_keys(self, sample_portfolio, sample_prices, as_of):
        """FV-07: Output contains VAR and ES keys at all confidence levels."""
        from var.filtered_var import compute_filtered_var
        result = compute_filtered_var(
            sample_portfolio, sample_prices, r=0.05, as_of=as_of,
            confidence_levels=[0.95, 0.99],
        )
        for key in ("var_95", "es_95", "var_99", "es_99",
                    "var_95_10d", "es_95_10d", "var_99_10d", "es_99_10d"):
            assert key in result, f"Missing: {key}"
