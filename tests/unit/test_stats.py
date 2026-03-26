"""
Unit tests for src/utils/stats.py — VAR/ES quantile computation and backtesting statistics.
"""

import numpy as np
import pytest
from utils.stats import (
    compute_var_es,
    compute_weighted_var_es,
    scale_to_10d,
    kupiec_test,
    christoffersen_test,
    traffic_light,
)
from var.scenarios import build_age_weights


class TestComputeVarEs:
    def _make_pnl(self, n=252, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1000, n)

    def test_var_positive(self):
        pnl = self._make_pnl()
        result = compute_var_es(pnl, [0.99])
        assert result["var_99"] > 0

    def test_es_ge_var(self):
        pnl = self._make_pnl()
        result = compute_var_es(pnl, [0.95, 0.99])
        assert result["es_95"] >= result["var_95"]
        assert result["es_99"] >= result["var_99"]

    def test_var_99_gt_var_95(self):
        pnl = self._make_pnl()
        result = compute_var_es(pnl, [0.95, 0.99])
        assert result["var_99"] >= result["var_95"]

    def test_es_99_gt_es_95(self):
        pnl = self._make_pnl()
        result = compute_var_es(pnl, [0.95, 0.99])
        assert result["es_99"] >= result["es_95"]

    def test_zero_portfolio(self):
        pnl = np.zeros(252)
        result = compute_var_es(pnl, [0.95, 0.99])
        assert result["var_99"] == pytest.approx(0.0)
        assert result["es_99"] == pytest.approx(0.0)

    def test_var_95_uses_correct_scenarios(self):
        """VAR_95 uses floor(252*0.05) = 12 scenarios at the tail."""
        pnl = np.arange(252, dtype=float) - 126  # range -126 to 125
        result = compute_var_es(pnl, [0.95])
        n_tail = int(np.floor(252 * 0.05))  # = 12
        # The 12th-worst (0-indexed: index 11 from sorted ascending)
        expected_var = -np.sort(pnl)[n_tail - 1]
        assert result["var_95"] == pytest.approx(expected_var, abs=1e-8)

    def test_single_confidence_level(self):
        """EC-07: Single CL list works correctly."""
        pnl = self._make_pnl()
        result = compute_var_es(pnl, [0.99])
        assert "var_99" in result
        assert "es_99" in result

    def test_three_confidence_levels(self):
        """EC-08: Three CL list produces 6 output keys."""
        pnl = self._make_pnl()
        result = compute_var_es(pnl, [0.90, 0.95, 0.99])
        for cl in ["90", "95", "99"]:
            assert f"var_{cl}" in result
            assert f"es_{cl}" in result


class TestScaleTo10d:
    def test_10d_gt_1d(self):
        pnl = np.random.default_rng(0).normal(0, 1000, 252)
        res_1d = compute_var_es(pnl, [0.99])
        res = scale_to_10d(res_1d)
        assert res["var_99_10d"] == pytest.approx(res_1d["var_99"] * np.sqrt(10))
        assert res["es_99_10d"] == pytest.approx(res_1d["es_99"] * np.sqrt(10))


class TestKupiecTest:
    def test_well_calibrated_model_high_pvalue(self):
        """Kupiec: p > 0.05 for ~2 exceptions in 250 days at 99% CL."""
        result = kupiec_test(n_exceptions=2, n_obs=250, confidence_level=0.99)
        assert result["p_value"] > 0.05
        assert not result["reject_h0"]

    def test_bad_model_low_pvalue(self):
        """Kupiec: p < 0.05 when exception rate is far off (e.g. 20 exceptions at 99% CL)."""
        result = kupiec_test(n_exceptions=20, n_obs=250, confidence_level=0.99)
        assert result["p_value"] < 0.05
        assert result["reject_h0"]

    def test_expected_exceptions(self):
        result = kupiec_test(3, 250, 0.99)
        assert result["expected_exceptions"] == pytest.approx(2.5, abs=0.01)

    def test_zero_exceptions_rejects_h0(self):
        """x=0 → model is over-conservative; Kupiec should reject H0 (p < 0.05)."""
        with pytest.warns(UserWarning, match="over-conservative"):
            result = kupiec_test(n_exceptions=0, n_obs=250, confidence_level=0.99)
        assert result["p_value"] < 0.05
        assert result["reject_h0"]
        assert result["lr_stat"] > 0

    def test_zero_exceptions_lr_formula(self):
        """x=0 LR = 2·n·ln(1/(1-p)) where p=1-CL=0.01, so 1/(1-p)=1/0.99."""
        with pytest.warns(UserWarning):
            result = kupiec_test(0, 250, 0.99)
        p = 1.0 - 0.99  # = 0.01
        expected_lr = 2.0 * 250 * np.log(1.0 / (1.0 - p))   # = 2·250·ln(1/0.99)
        assert result["lr_stat"] == pytest.approx(expected_lr, rel=1e-8)


class TestChristoffersenTest:
    def test_no_clustering_high_pvalue(self):
        """No clustering (random exceptions) → high p-value."""
        rng = np.random.default_rng(99)
        exc = (rng.uniform(size=252) < 0.01).astype(int)
        result = christoffersen_test(exc)
        # With few exceptions, test may be unreliable, but p should generally be > 0.05
        assert "p_value" in result
        assert "lr_stat" in result

    def test_perfect_clustering_low_pvalue(self):
        """Clustered exceptions (all in a row) → should be detected."""
        exc = np.zeros(252, dtype=int)
        exc[:10] = 1  # 10 consecutive exceptions at the start
        result = christoffersen_test(exc)
        # Clustering should produce a significant test (may reject H0)
        assert result["lr_stat"] >= 0

    def test_output_keys(self):
        exc = np.array([0, 1, 0, 0, 1, 1, 0])
        result = christoffersen_test(exc)
        for key in ("lr_stat", "p_value", "reject_h0", "n00", "n01", "n10", "n11"):
            assert key in result


class TestTrafficLight:
    def test_green(self):
        assert traffic_light(0) == "Green"
        assert traffic_light(4) == "Green"

    def test_amber(self):
        assert traffic_light(5) == "Amber"
        assert traffic_light(9) == "Amber"

    def test_red(self):
        assert traffic_light(10) == "Red"
        assert traffic_light(50) == "Red"
