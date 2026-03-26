# Test Plan — Historical VAR Model
## Equities Options Portfolio

**Author:** Claude (Quant Developer Assistant)
**Date:** 2026-02-27
**Status:** Approved — Implementation Ready

---

## 1. Overview

This test plan covers the full VAR system described in `requirements.md`. Tests are grouped by layer: data loading, vol surface, portfolio, pricing, scenarios, VAR/ES computation, backtesting, and integration. All tests live in `tests/` and are executed with `pytest`.

**Primary testing objectives:**
- Correctness of mathematical implementations (BS pricing, VAR/ES quantiles, GARCH)
- Integrity of the yfinance data loading and vol surface construction pipeline
- Correctness of configurable confidence levels for both VAR and ES
- Statistical validity of the VAR model under backtesting
- Regression safety — changes must not silently break existing behaviour

---

## 2. Test Structure

```
tests/
├── conftest.py                       # Shared fixtures (prices, portfolio, market snapshot)
│
├── unit/
│   ├── test_black_scholes.py         # BS pricer: price, Greeks, edge cases
│   ├── test_loader.py                # yfinance download, fill, cache
│   ├── test_vol_surface.py           # Rolling vol, skew model, surface properties
│   ├── test_portfolio.py             # Trade format, portfolio construction, validation
│   ├── test_scenarios.py             # Log-returns, scenario set, age weights
│   ├── test_revaluation.py           # Full reprice + Greeks approx
│   ├── test_historical_var.py        # Methods A & B — VAR and ES
│   ├── test_filtered_var.py          # Method C — GARCH-filtered VAR and ES
│   ├── test_factor_var.py            # Method D — factor decomposition, VAR and ES
│   └── test_stress_var.py            # Stress window selection and stressed VAR/ES
│
├── integration/
│   ├── test_pipeline_plain_hs.py     # End-to-end: data → plain HS VAR + ES
│   ├── test_pipeline_factor.py       # End-to-end: data → factor VAR + ES
│   └── test_backtest.py              # Rolling backtest + Kupiec/Christoffersen
│
└── fixtures/
    └── sample_data.py                # Deterministic datasets (fixed seed=42)
```

---

## 3. Unit Tests

### 3.1 Black-Scholes Pricer (`test_black_scholes.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| BS-01 | ATM call price (S=K=100, T=1, r=5%, σ=20%) | Within 0.01 of known analytical value (10.45) |
| BS-02 | Put-call parity: C − P = S·e^(−q·T) − K·e^(−r·T) | Difference < 1e-8 |
| BS-03 | Deep ITM call → intrinsic value as σ→0 | abs(C − max(S−K,0)) < 1e-4 |
| BS-04 | Deep OTM call → 0 as σ→0 | C < 1e-6 |
| BS-05 | Call delta ∈ (0, 1) for all valid inputs | Assertion |
| BS-06 | Put delta ∈ (−1, 0) for all valid inputs | Assertion |
| BS-07 | Gamma > 0 for both calls and puts | Assertion |
| BS-08 | Vega > 0 for both calls and puts | Assertion |
| BS-09 | Theta < 0 for long option (time decay) | Assertion |
| BS-10 | Put price ≥ 0 for all inputs | Assertion |
| BS-11 | Vectorised pricing over strike array returns correct shape | Shape match |
| BS-12 | Zero interest rate (r=0) produces finite, non-negative prices | No exception, price ≥ 0 |
| BS-13 | Very short-dated option (T=1/252) — no division by zero | Finite price |
| BS-14 | Very long-dated option (T=10) — finite price | Finite price |

### 3.2 Data Loader (`test_loader.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| DL-01 | Download returns a DataFrame with correct tickers as columns | Column set match |
| DL-02 | No NaN in adjusted close after forward-fill + back-fill | isna().sum() == 0 |
| DL-03 | All prices > 0 after loading | min > 0 |
| DL-04 | Date index is a DatetimeIndex sorted ascending | Sorted check |
| DL-05 | Requesting 5 years returns ≥ 1,200 trading days | len ≥ 1,200 |
| DL-06 | Ticker with > 5% missing data emits a warning | warnings.warn called |
| DL-07 | Cache file written on first download; second call reads cache, not network | File exists; no HTTP call |
| DL-08 | Index tickers (^GSPC, ^NDX, ^RUT, ^VIX) download without error | No exception |

### 3.3 Vol Surface (`test_vol_surface.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| VS-01 | Rolling 21-day vol output shape = (n_days, n_tickers) | Shape assertion |
| VS-02 | All rolling vol values > 0 after initial warm-up period | min > 0 |
| VS-03 | 21-day vol ≥ 0 and ≤ 3.0 (300% annualised) for any realistic equity | Range check |
| VS-04 | ATM vol for long-dated tenor uses 126-day window | Equality check |
| VS-05 | ATM vol for short-dated tenor uses 21-day window | Equality check |
| VS-06 | Vol surface shape = (n_days, n_strikes, n_tenors) | Shape assertion |
| VS-07 | No negative vols anywhere in surface | min(σ) > 0 |
| VS-08 | Skew at k=0 (ATM) equals σ_ATM | Exact match |
| VS-09 | Skew is negative for k < 0 (OTM put) when α < 0 | σ(k<0) > σ_ATM |
| VS-10 | Calendar-spread arbitrage-free: total variance non-decreasing in T | All diffs ≥ 0 |
| VS-11 | σ(k, T) for custom (α, β) matches analytical formula | Numerical equality |

### 3.4 Portfolio (`test_portfolio.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| PO-01 | Portfolio loads from valid CSV without exception | No exception |
| PO-02 | Unknown option_type ("exotic") raises ValueError | ValueError raised |
| PO-03 | Negative notional raises ValueError | ValueError raised |
| PO-04 | position_sign not in {+1, −1} raises ValueError | ValueError raised |
| PO-05 | Expiry in the past (relative to trade_date) raises ValueError | ValueError raised |
| PO-06 | Portfolio NPV = sum of (position_sign × price × notional) per trade | abs(diff) < 1e-6 |
| PO-07 | Long call has positive delta; long put has negative delta | Sign check |
| PO-08 | Generated synthetic portfolio spans ≥ 10 different underlyings | len(underlyings) ≥ 10 |
| PO-09 | Expired option (today > expiry) is valued at 0 | V == 0 |
| PO-10 | Trade with underlying not in equity universe raises KeyError | KeyError raised |

### 3.5 Scenario Construction (`test_scenarios.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| SC-01 | Log-returns computed correctly vs manual: log(S_t / S_{t-1}) | Max abs error < 1e-12 |
| SC-02 | 252-day scenario set has exactly 252 rows | len == 252 |
| SC-03 | Scenario frame has one column per equity + index | Column count match |
| SC-04 | No NaN or inf in scenario frame | isna + isinf == 0 |
| SC-05 | Applying scenario return to today's spot produces stressed spot > 0 | All stressed spots > 0 |
| SC-06 | Age weights sum to 1.0 | abs(sum − 1.0) < 1e-10 |
| SC-07 | Age weights monotonically increase toward most recent date | Assertion |
| SC-08 | λ=1.0 produces uniform weights (plain HS) | All weights == 1/252 |
| SC-09 | Requesting lookback > available history raises ValueError | ValueError raised |

### 3.6 Revaluation (`test_revaluation.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| RV-01 | Full reprice P&L = 0 when scenario = flat (no moves) | abs(PnL) < 1e-8 |
| RV-02 | Large spot up shock (+20%): call P&L > 0, put P&L < 0 | Sign check |
| RV-03 | Large spot down shock (−20%): put P&L > 0, call P&L < 0 | Sign check |
| RV-04 | Greeks approx vs full reprice: relative error < 5% for ±2% spot moves | Relative error |
| RV-05 | P&L vector length = number of scenarios | Shape assertion |
| RV-06 | Portfolio P&L = sum of individual trade P&Ls | abs(diff) < 1e-6 |
| RV-07 | Long call + short call same strike → P&L ≈ 0 for any shock | abs(PnL) < 1e-4 |

### 3.7 Historical VAR & ES — Plain HS and Age-Weighted (`test_historical_var.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| HV-01 | VAR > 0 for non-trivial portfolio (loss convention) | VAR > 0 |
| HV-02 | ES ≥ VAR at same confidence level | Inequality |
| HV-03 | VAR_99 > VAR_95 | Inequality |
| HV-04 | ES_99 > ES_95 | Inequality |
| HV-05 | ES_99 > VAR_99 | Inequality |
| HV-06 | 10-day VAR = 1-day VAR × √10 | abs(diff) < 1e-8 |
| HV-07 | 10-day ES = 1-day ES × √10 | abs(diff) < 1e-8 |
| HV-08 | Zero-position portfolio → VAR = 0 and ES = 0 | Exact zero |
| HV-09 | VAR_95 uses floor(252 × 0.05) = 12 scenarios at the tail | Index check |
| HV-10 | ES_99 is mean of worst 2 scenarios (floor(252 × 0.01) = 2) | Numerical check |
| HV-11 | Custom confidence level [0.975] returns output keyed "var_97.5_1d" | Key exists |
| HV-12 | Age-weighted VAR ≠ plain VAR when λ < 1 | Inequality |
| HV-13 | Age-weighted ES ≠ plain ES when λ < 1 | Inequality |
| HV-14 | In high-vol recent period: age-weighted VAR > plain VAR | Directional |
| HV-15 | Output dict contains all expected keys for both measures and both horizons | Key set assertion |

### 3.8 Filtered VAR — GARCH (`test_filtered_var.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| FV-01 | GARCH(1,1) parameters: α + β < 1 (stationarity) | Constraint |
| FV-02 | Conditional vol series is positive throughout | min(σ_t) > 0 |
| FV-03 | Standardised residuals have approximately unit variance | abs(std − 1) < 0.15 |
| FV-04 | Filtered VAR > plain VAR in high-vol regime | Directional |
| FV-05 | Filtered VAR < plain VAR in low-vol regime | Directional |
| FV-06 | Filtered ES ≥ filtered VAR at same confidence | Inequality |
| FV-07 | Output contains VAR and ES keys at all confidence levels | Key set assertion |

### 3.9 Factor VAR (`test_factor_var.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| FA-01 | OLS regression R² > 0.3 for a market-correlated equity | R² check |
| FA-02 | Beta to ^GSPC is positive for a long large-cap equity | β > 0 |
| FA-03 | Factor VAR for β=1, single stock, single index ≈ plain HS VAR | Within 10% |
| FA-04 | Factor ES ≥ factor VAR at same confidence | Inequality |
| FA-05 | Portfolio with β≈0 (market-neutral) → near-zero factor VAR | VAR < threshold |
| FA-06 | Residual idiosyncratic P&L has near-zero correlation with factor P&L | abs(corr) < 0.05 |
| FA-07 | Output contains VAR and ES keys at all confidence levels | Key set assertion |

### 3.10 Stress VAR (`test_stress_var.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| SV-01 | Stress window start/end dates are within the full 5-year history | Date bounds |
| SV-02 | Stress window has exactly 252 trading days | len == 252 |
| SV-03 | Stressed VAR ≥ plain HS VAR (stress is worst-case) | Inequality |
| SV-04 | Stressed ES ≥ stressed VAR | Inequality |
| SV-05 | Injecting an artificial extreme-loss period → stress window selects it | Exact match |
| SV-06 | Stress window is the same regardless of confidence level input | Stability check |

---

## 4. Integration Tests

### 4.1 End-to-End — Plain HS (`test_pipeline_plain_hs.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| E2E-01 | Full pipeline runs without exception on 25-name universe | No exception |
| E2E-02 | Output contains VAR and ES at 95% and 99%, 1-day and 10-day | All 8 keys present |
| E2E-03 | All output values are finite positive floats | isfinite and > 0 |
| E2E-04 | Doubling all notionals doubles VAR and ES | Ratio ≈ 2.0 |
| E2E-05 | Perfectly hedged portfolio (long call + short call, same strike, same notional) → VAR ≈ 0 | VAR < 1e-4 |

### 4.2 End-to-End — Factor VAR (`test_pipeline_factor.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| FA-E2E-01 | Factor pipeline runs without exception on 25-name universe | No exception |
| FA-E2E-02 | Factor VAR and plain HS VAR are in the same order of magnitude | Ratio ∈ (0.3, 3.0) |
| FA-E2E-03 | Factor ES ≥ factor VAR in all outputs | Inequality throughout |

### 4.3 Backtesting (`test_backtest.py`)

| Test ID | Description | Pass Criterion |
|---|---|---|
| BT-01 | Rolling backtest produces one VAR estimate per business day over test window | len == n_test_days |
| BT-02 | For a correctly calibrated synthetic dataset, exception count ∈ [1, 7] over 250 days | Basel Green/Amber |
| BT-03 | Kupiec test: p-value > 0.05 for correctly calibrated model (seed=42) | p > 0.05 |
| BT-04 | Christoffersen test: p-value > 0.05 (no clustering) for correctly calibrated model | p > 0.05 |
| BT-05 | Artificially inflated VAR (1000× actual) → 0 exceptions → Kupiec p < 0.05 | p < 0.05 |
| BT-06 | Traffic light = "Green" when exceptions ≤ 4 | String match |
| BT-07 | Traffic light = "Amber" when exceptions ∈ [5, 9] | String match |
| BT-08 | Traffic light = "Red" when exceptions ≥ 10 | String match |
| BT-09 | ES backtest: average loss in exception days > ES estimate (conservative check) | Directional |

---

## 5. Edge Case & Robustness Tests

| Test ID | Description | Pass Criterion |
|---|---|---|
| EC-01 | Single-trade portfolio (one call) — pipeline runs without error | No exception |
| EC-02 | All options expire on the same date — correct aggregation | No exception |
| EC-03 | Portfolio with mix of calls and puts on same underlying | No exception |
| EC-04 | Very deep OTM option (K = 2×S) priced correctly | Price ≥ 0, finite |
| EC-05 | Very deep ITM option (K = 0.5×S) priced correctly | Price ≥ intrinsic |
| EC-06 | 25-name, 100-option portfolio end-to-end completes in < 60s | Timing assertion |
| EC-07 | Confidence levels list with a single value [0.99] works correctly | No exception |
| EC-08 | Confidence levels list with three values [0.90, 0.95, 0.99] works correctly | All 12 output keys present |
| EC-09 | Missing ticker in yfinance (delisted) triggers a clear error or skip with warning | Exception or warning raised |

---

## 6. Test Fixtures

All unit tests use **fixed random seed (seed=42)** for reproducibility.

**`conftest.py` provides session-scoped fixtures:**

```python
sample_prices         # 5-year daily price DataFrame, 5 equities, seed=42
sample_index_prices   # 5-year daily prices for ^GSPC, ^NDX, ^RUT, ^VIX
sample_vol_surface    # Vol surface matching sample_prices tickers
sample_portfolio      # 8-trade options book (3 calls, 3 puts, 2 spreads)
today_market          # Market snapshot (spot, vol surface) for revaluation tests
```

**Integration test fixtures** (session-scoped, generated once):
```python
full_prices           # 5-year, 25 equities + indices
full_portfolio        # 50-trade synthetic book across 10 underlyings
```

**`fixtures/sample_data.py`** contains hard-coded, deterministic small datasets for tests that must not touch the network.

---

## 7. Test Execution

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only (slower, touches more data)
pytest tests/integration/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run a specific file
pytest tests/unit/test_black_scholes.py -v

# Run a specific test by ID pattern
pytest tests/ -k "BS-02 or HV-05" -v

# Skip network-dependent tests
pytest tests/ -m "not network" -v
```

Mark network-dependent tests (yfinance calls) with `@pytest.mark.network` so they can be skipped in offline CI.

---

## 8. Acceptance Criteria (Definition of Done)

Before the base model is considered implementation-complete:

- [ ] All unit tests pass (0 failures, 0 errors)
- [ ] All integration tests pass
- [ ] Code coverage ≥ 80% across `src/`
- [ ] Both VAR and ES outputs present at 95% and 99% for all four methods
- [ ] Configurable confidence level list passes through correctly (EC-08)
- [ ] Backtesting: Kupiec p-value > 0.05 on synthetic dataset (seed=42, 99% CL)
- [ ] Rolling 252-day backtest exception count in Basel Green or Amber zone (≤ 9)
- [ ] End-to-end pipeline (25 names, 50 options) completes in < 60 seconds

---

*End of test_plan.md*
