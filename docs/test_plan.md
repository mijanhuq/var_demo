# Test Plan — Historical VaR Model
## Equities Options Portfolio

**Version:** 0.1
**Date:** 2026-02-23
**Status:** Awaiting Review
**Linked Requirement:** [requirements.md](requirements.md)

---

## 1. Testing Philosophy

This project follows **test-driven development (TDD)**. Tests are written before or alongside implementation. The goals of the test suite are:

1. **Correctness** — all financial calculations produce mathematically verifiable results
2. **Robustness** — the model behaves predictably at boundary conditions and with edge-case portfolios
3. **Regression safety** — no previously passing scenario breaks as the codebase evolves
4. **Interpretability** — test names read as specifications; failures point directly to the broken behaviour

All tests live in `tests/`, mirroring the module structure of the source code.

---

## 2. Test Scope by Component

| Module | Unit | Integration | Validation |
|---|---|---|---|
| Data generator | Yes | Yes | Yes |
| Black-Scholes pricer | Yes | — | Yes |
| Greeks calculation | Yes | — | Yes |
| Portfolio construction | Yes | Yes | — |
| Historical VaR | Yes | Yes | Yes |
| Delta-Gamma VaR | Yes | Yes | Yes |
| Stress VaR | Yes | Yes | — |
| Factor VaR | Yes | Yes | — |
| Backtesting engine | Yes | Yes | Yes |
| Backtesting statistics | Yes | — | Yes |

---

## 3. Test Directory Structure

```
tests/
├── conftest.py                  # Shared fixtures (sample portfolio, market data)
├── unit/
│   ├── test_generator.py        # Synthetic data generator
│   ├── test_pricer.py           # Black-Scholes pricing and Greeks
│   ├── test_portfolio.py        # Portfolio construction and aggregation
│   ├── test_historical_var.py   # Full revaluation HVaR
│   ├── test_delta_gamma_var.py  # Delta-Gamma-Vega approximation VaR
│   ├── test_stress_var.py       # Stress VaR period identification
│   ├── test_factor_var.py       # Factor regression and factor VaR
│   └── test_backtest_stats.py   # Kupiec, Christoffersen, traffic light
└── integration/
    ├── test_pipeline_phase1.py  # End-to-end: generate → price → VaR → backtest
    └── test_pipeline_phase2.py  # End-to-end with stress VaR + factor VaR
```

---

## 4. Unit Tests

### 4.1 `test_generator.py` — Synthetic Data Generator

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| GEN-01 | GBM price series has correct length | Returns exactly N trading days |
| GEN-02 | GBM prices are strictly positive | No zero or negative values |
| GEN-03 | GBM log-returns have realistic mean and std | μ within ±3σ of input; σ within 20% of input |
| GEN-04 | Multi-asset correlation is preserved | Pearson correlation of generated log-returns matches input correlation matrix within tolerance |
| GEN-05 | Vol surface ATM vol is close to realized vol | ATM implied vol ≈ realized vol ± 5 vol points |
| GEN-06 | Vol surface skew is negative for puts | OTM put vol > ATM vol for all tenors |
| GEN-07 | Term structure has upward slope | 12m ATM vol ≥ 1m ATM vol |
| GEN-08 | Crisis injection produces a drawdown period | Maximum rolling 252-day drawdown ≥ 40% |
| GEN-09 | Dataset covers requested number of trading days | `len(data) >= requested_days` |
| GEN-10 | Reproducibility: same seed → same output | Two runs with same seed produce identical data |

### 4.2 `test_pricer.py` — Black-Scholes Pricer and Greeks

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| PRC-01 | Call price ATM, short expiry, known vol | Within 0.01% of analytic B-S value |
| PRC-02 | Put-call parity holds | \|C − P − S + K·e^{−rT}\| < 1e-8 |
| PRC-03 | Call price is zero when S → 0 | price < 1e-10 |
| PRC-04 | Call price → S − K·e^{−rT} when vol → ∞ | price converges to intrinsic + time value |
| PRC-05 | Call price monotonically increases with S | ∂C/∂S > 0 for all test cases |
| PRC-06 | Call price monotonically decreases with K | ∂C/∂K < 0 for all test cases |
| PRC-07 | Call price monotonically increases with vol | ∂C/∂σ > 0 (positive vega) |
| PRC-08 | Delta of call ∈ (0, 1) | 0 < Δ < 1 for all call cases |
| PRC-09 | Delta of put ∈ (−1, 0) | −1 < Δ < 0 for all put cases |
| PRC-10 | Gamma is positive for both calls and puts | Γ > 0 |
| PRC-11 | Vega is positive for both calls and puts | ν > 0 |
| PRC-12 | Theta is negative (time decay) | Θ < 0 for long positions |
| PRC-13 | Delta of deep ITM call ≈ 1 | Δ > 0.99 for S/K = 2.0 |
| PRC-14 | Delta of deep OTM call ≈ 0 | Δ < 0.01 for S/K = 0.5 |
| PRC-15 | Numerical delta ≈ analytical delta | Finite-difference delta within 1e-4 of analytic delta |
| PRC-16 | Numerical gamma ≈ analytical gamma | Finite-difference gamma within 1e-4 of analytic gamma |

### 4.3 `test_portfolio.py` — Portfolio Construction

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| PFL-01 | Portfolio value is sum of position values | Σ position_value = portfolio_value |
| PFL-02 | Portfolio delta is sum of position deltas × notional | Aggregated correctly |
| PFL-03 | Portfolio gamma is sum of position gammas × notional | Aggregated correctly |
| PFL-04 | Short position has negated Greeks | short call delta = −long call delta |
| PFL-05 | Empty portfolio has zero value and zero Greeks | All metrics = 0 |
| PFL-06 | Delta-neutral portfolio has portfolio delta ≈ 0 | Straddle example: \|Δ_portfolio\| < 1e-4 |
| PFL-07 | Loading portfolio from config file produces correct positions | Positions match YAML spec |

### 4.4 `test_historical_var.py` — Full Revaluation HVaR

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| VAR-01 | VaR is positive (loss expressed as positive number) | VaR > 0 |
| VAR-02 | VaR(99%) > VaR(95%) | Strictly true |
| VAR-03 | VaR of a zero-position portfolio is zero | VaR = 0 |
| VAR-04 | Long call VaR < underlying equity VaR at same notional | Options limit downside |
| VAR-05 | VaR is computed from exactly 252 scenarios | Scenario count = 252 |
| VAR-06 | Sorted P&L at 99th percentile equals VaR | 3rd worst P&L = VaR(99%) |
| VAR-07 | Doubling all position sizes doubles VaR | Linear scaling test |
| VAR-08 | Fully hedged (long + short same option) portfolio has VaR ≈ 0 | \|VaR\| < 1e-6 |
| VAR-09 | VaR with a stressed crisis window is higher than base VaR | Stress VaR > base VaR (for long portfolio) |
| VAR-10 | Historical scenarios applied to equity prices preserve log-return magnitudes | Shocked price = S_t × exp(historical_return) |

### 4.5 `test_delta_gamma_var.py` — Delta-Gamma Approximation

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| DGV-01 | DG VaR is positive | VaR > 0 |
| DGV-02 | DG VaR(99%) > DG VaR(95%) | Strictly true |
| DGV-03 | DG VaR ≈ Full revaluation VaR for small moves | Within 10% for normal vol regimes |
| DGV-04 | DG VaR diverges from full reval for large moves (expected) | DG VaR < Full reval for crisis scenarios (gamma effect) |
| DGV-05 | DG approximation P&L matches second-order Taylor expansion | Residual < 1e-6 per scenario |

### 4.6 `test_stress_var.py` — Stress VaR

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| SVR-01 | Identified stress window covers exactly 252 trading days | `len(stress_window) == 252` |
| SVR-02 | Stress window produces the worst portfolio loss in the dataset | No other 252-day window produces higher VaR |
| SVR-03 | Stress VaR ≥ base VaR | For a long portfolio under any scenario |
| SVR-04 | Injected crisis period is selected as the stress window | Stress window overlaps with injected crisis dates |
| SVR-05 | Stress window search is deterministic | Two runs on same dataset return same window |

### 4.7 `test_factor_var.py` — Factor VaR

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| FAC-01 | Beta regression converges for all underlyings | No NaN betas |
| FAC-02 | R² ≥ 0 and ≤ 1 for all regressions | Sanity check |
| FAC-03 | Systematic + idiosyncratic VaR components sum to approximately total VaR | Diversification benefit: total ≤ systematic + idiosyncratic |
| FAC-04 | A market-neutral portfolio has near-zero systematic VaR | \|systematic VaR\| < 5% of total VaR |
| FAC-05 | Factor returns explain index variance correctly | Reproduced index variance within 1% |

### 4.8 `test_backtest_stats.py` — Backtesting Statistics

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| BT-01 | Exception count is correct | Manual count matches function output |
| BT-02 | Exception rate for a perfect model is ≤ 1% | Injected model with known exceptions passes |
| BT-03 | Traffic light is Green when exceptions ≤ 4 / 250 | Correct zone label returned |
| BT-04 | Traffic light is Yellow when exceptions = 7 / 250 | Correct zone label returned |
| BT-05 | Traffic light is Red when exceptions ≥ 10 / 250 | Correct zone label returned |
| BT-06 | Kupiec test rejects at p=0.05 when exception rate is 5% against 99% VaR model | `p_value < 0.05` |
| BT-07 | Kupiec test does not reject when exception rate is exactly 1% | `p_value > 0.05` |
| BT-08 | Christoffersen test flags clustered exceptions | Detects 5 consecutive exceptions as non-independent |
| BT-09 | Christoffersen test passes for uniformly distributed exceptions | Uniform spacing accepted |

---

## 5. Integration Tests

### 5.1 Phase 1 Pipeline (`test_pipeline_phase1.py`)

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| INT-01 | Full pipeline: generate data → build portfolio → compute VaR → backtest | Completes without error; VaR > 0; exception rate < 5% |
| INT-02 | Pipeline runs with a single-underlying, single-option portfolio | No crashes; sensible output |
| INT-03 | Pipeline produces consistent results with a fixed random seed | Output identical across two runs |
| INT-04 | VaR output includes both 99% and 95% figures | Both values present in output |
| INT-05 | Backtesting exceptions plotted for 3-year window | Chart generated without error |

### 5.2 Phase 2 Pipeline (`test_pipeline_phase2.py`)

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| INT-06 | Stress VaR is computed and reported alongside base VaR | Both values in output; stress > base |
| INT-07 | Factor VaR decomposition is reported for all 4 factors | Per-factor VaR table has 4 rows |
| INT-08 | Factor VaR + stress VaR pipeline runs end-to-end | No errors; output complete |

---

## 6. Validation Tests

These tests verify financial correctness against known results.

| Test ID | Test Description | Expected Result |
|---|---|---|
| VAL-01 | B-S call price: S=100, K=100, T=1, r=5%, σ=20% | ≈ $10.451 |
| VAL-02 | B-S put price: same inputs | ≈ $5.574 (via put-call parity) |
| VAL-03 | Delta of ATM call (T=1, σ=20%) | ≈ 0.6368 |
| VAL-04 | Vega of ATM option (T=1, σ=20%) | ≈ 0.3752 per 1% vol move |
| VAL-05 | HVaR of long equity position ≈ σ × 2.326 × S (normal approximation) | Within 15% of parametric estimate for normal market data |
| VAL-06 | Exception rate on 3-year backtest window ≤ 2% at 99% confidence | ≤ 5 exceptions per 250 days on average |
| VAL-07 | VaR decreases when portfolio is delta-hedged | Hedged VaR < unhedged VaR |

---

## 7. Edge Case & Boundary Tests

| Test ID | Test Description | Pass Criterion |
|---|---|---|
| EDG-01 | Option at expiry (T=0): call value = max(S−K, 0) | Intrinsic value only |
| EDG-02 | Zero volatility: call value = max(S−Ke^{−rT}, 0) | Matches forward intrinsic |
| EDG-03 | Very large volatility (σ=500%): price bounded by S | Call price ≤ S |
| EDG-04 | Portfolio with 100 positions runs without timeout | Completes in < 10 seconds |
| EDG-05 | Data with NaN values raises a clear error | `ValueError` or equivalent raised |
| EDG-06 | Lookback window with fewer than 252 data points raises clear error | `InsufficientDataError` raised |
| EDG-07 | Deep ITM put: price ≈ K·e^{−rT} − S | Within 0.01% of formula |
| EDG-08 | Negative portfolio delta (net short): VaR is loss on up-move | VaR is tied to rising price scenario |

---

## 8. Performance Tests

| Test ID | Test Description | Acceptance Threshold |
|---|---|---|
| PRF-01 | Full revaluation HVaR for 10-underlying, 30-option portfolio over 252 scenarios | < 5 seconds on standard hardware |
| PRF-02 | 3-year rolling backtest (756 VaR calculations) | < 60 seconds |
| PRF-03 | Synthetic data generation for 1,510 trading days, 10 underlyings | < 2 seconds |
| PRF-04 | Stress VaR window search over 1,260 days | < 10 seconds |

---

## 9. Test Data & Fixtures

Shared test fixtures will be defined in `tests/conftest.py`:

```python
# Key fixtures (pseudocode)

@pytest.fixture
def sample_equity_prices():
    """1260 days × 5 underlyings, fixed seed, calibrated to SPX-like params."""

@pytest.fixture
def sample_vol_surface():
    """3D surface: 5 strikes × 4 tenors × 1260 days."""

@pytest.fixture
def sample_portfolio():
    """5-underlying portfolio: 1 long call + 1 short put per name + 1 index hedge."""

@pytest.fixture
def atm_european_call():
    """S=100, K=100, T=1, r=0.05, sigma=0.20"""

@pytest.fixture
def crisis_market_data():
    """252-day window with embedded 50% drawdown, vol spike to 70%."""
```

---

## 10. Test Execution & Reporting

| Command | Purpose |
|---|---|
| `pytest tests/unit/` | Run all unit tests |
| `pytest tests/integration/` | Run all integration tests |
| `pytest tests/ -v` | Verbose run of all tests |
| `pytest tests/ --cov=. --cov-report=html` | Coverage report |
| `pytest tests/ -m validation` | Run validation tests only |
| `pytest tests/ -k "VAR"` | Run all VaR-related tests |

**Coverage target:** ≥ 85% line coverage across `data/`, `portfolio/`, `var/`, `backtest/`.

---

## 11. Test Outcomes Log

Test results will be recorded in the [Journal.md](../Journal.md) after each test run. Format:

```
## Test Run — <date>
Phase: <Phase 1 / Phase 2>
Tests run: N
Passed: N
Failed: N
Skipped: N
Coverage: N%
Failures: <list of failing test IDs and brief description>
```

---

## 12. Known Gaps & Deferred Tests

| Item | Status | Notes |
|---|---|---|
| American option pricing tests | Deferred to phase 2 | Binomial model not in phase 1 scope |
| Implied vol inversion (from market price) | Deferred | Newton-Raphson IV solver test needed in phase 2 |
| Real data ingestion tests (yfinance) | Deferred | Requires live/cached network access; mock in phase 1 |
| FRTB Expected Shortfall tests | Out of scope v1 | Different metric; separate workstream |
