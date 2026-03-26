"""
Integration tests: end-to-end plain historical simulation pipeline.
(E2E-01 through E2E-05)
"""

import numpy as np
import pytest
from var.historical_var import compute_historical_var


CONFIDENCE_LEVELS = [0.95, 0.99]
EXPECTED_KEYS = [
    "var_95", "es_95", "var_99", "es_99",
    "var_95_10d", "es_95_10d", "var_99_10d", "es_99_10d",
]


class TestPlainHSPipeline:
    def test_E2E01_runs_without_exception(
        self, sample_portfolio, sample_prices, as_of
    ):
        """E2E-01: Full pipeline runs without exception."""
        result = compute_historical_var(
            sample_portfolio, sample_prices, r=0.05, as_of=as_of,
            confidence_levels=CONFIDENCE_LEVELS,
        )
        assert result is not None

    def test_E2E02_output_keys_present(
        self, sample_portfolio, sample_prices, as_of
    ):
        """E2E-02: Output contains VAR and ES at 95%, 99%, 1-day and 10-day."""
        result = compute_historical_var(
            sample_portfolio, sample_prices, r=0.05, as_of=as_of,
            confidence_levels=CONFIDENCE_LEVELS,
        )
        for key in EXPECTED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_E2E03_all_values_finite_positive(
        self, sample_portfolio, sample_prices, as_of
    ):
        """E2E-03: All VAR/ES values are finite positive floats."""
        result = compute_historical_var(
            sample_portfolio, sample_prices, r=0.05, as_of=as_of,
            confidence_levels=CONFIDENCE_LEVELS,
        )
        for key in EXPECTED_KEYS:
            val = result[key]
            assert np.isfinite(val), f"{key} is not finite"
            assert val >= 0.0, f"{key} is negative"

    def test_E2E04_pnl_vector_length(
        self, sample_portfolio, sample_prices, as_of
    ):
        """P&L vector has exactly 252 scenarios."""
        result = compute_historical_var(
            sample_portfolio, sample_prices, r=0.05, as_of=as_of,
            confidence_levels=CONFIDENCE_LEVELS,
        )
        assert len(result["pnl_vector"]) == 252

    def test_method_label(self, sample_portfolio, sample_prices, as_of):
        """Method label is set correctly."""
        result = compute_historical_var(
            sample_portfolio, sample_prices, method="plain",
            r=0.05, as_of=as_of,
        )
        assert result["method"] == "plain_hs"

    def test_age_weighted_method_label(self, sample_portfolio, sample_prices, as_of):
        result = compute_historical_var(
            sample_portfolio, sample_prices, method="age_weighted",
            r=0.05, as_of=as_of,
        )
        assert result["method"] == "age_weighted_hs"

    def test_unknown_method_raises(self, sample_portfolio, sample_prices, as_of):
        with pytest.raises(ValueError, match="Unknown method"):
            compute_historical_var(
                sample_portfolio, sample_prices, method="magic",
                r=0.05, as_of=as_of,
            )
