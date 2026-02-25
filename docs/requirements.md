# Historical VaR Model — Equities Options Portfolio
## Model Proposal & Requirements

**Version:** 0.2 (Approved)
**Date:** 2026-02-23
**Status:** Approved — Phase 1 Implementation Authorised

---

## 1. Executive Summary

This document proposes a Historical Value-at-Risk (HVaR) framework for an equity options portfolio. The system will:

- Generate (or ingest) multi-year historical equity and option price data
- Compute portfolio P&L under each historical scenario via full revaluation
- Report VaR at 99% and 95% confidence, and Expected Shortfall (ES/CVaR) at both percentiles, over a 1-day holding period using a 1-year (252 trading day) lookback window
- Support a 3–5 year data history for backtesting and Stress VaR
- Offer an optional factor-based VaR decomposition using equity indices

The design follows Basel III internal model approach conventions and is suitable for regulatory reporting extensions.

---

## 2. Scope

| Item | In Scope | Out of Scope (v1) |
|---|---|---|
| European vanilla calls & puts (25 underlyings) | Yes | Exotic options, barriers, American options |
| Equity underlyings (single names) | Yes | FX, rates, credit |
| Equity index factors (4 indices) | Yes (phase 2) | Fama-French factors, macro factors |
| Historical Simulation VaR (99%, 95%) | Yes | |
| Expected Shortfall ES (99%, 95%) | Yes | |
| Delta-Gamma VaR (approximation) | Yes (phase 2) | Monte Carlo VaR |
| Stress VaR (worst 12-month window) | Yes (phase 2) | |
| Factor VaR (index decomposition) | Yes (phase 2) | PCA factor model |
| Backtesting (Kupiec, Christoffersen) | Yes | |
| Richer reporting (charts + tables) | Yes | PDF/HTML export (phase 3) |
| Greeks: Delta, Gamma, Vega, Theta | Yes | Rho, Vanna, Volga |
| Real data ingestion (yfinance) | Phase 2 | |

---

## 3. Data Requirements

### 3.1 Equities

| Field | Description |
|---|---|
| Daily closing price | Adjusted for splits and dividends |
| Daily return | Log return: ln(S_t / S_{t-1}) |
| Historical volatility | 21-day rolling realized vol (annualized) |
| Dividend yield | Required for option pricing |

### 3.2 Options

| Field | Description |
|---|---|
| Strike (K) | Sampled across moneyness: 0.80–1.20 × S |
| Expiry (T) | 1m, 3m, 6m, 12m tenors |
| Type | Call, Put |
| Implied volatility | Surface generated from historical vol ± smile |
| Price | Black-Scholes (European), no early exercise |
| Greeks | Delta, Gamma, Vega, Theta (computed analytically) |

### 3.3 Indices (Factor Model)

| Index | Role |
|---|---|
| Broad market (e.g. SPX) | Systematic/market factor |
| Sector indices (e.g. XLK, XLF) | Sector factor |

### 3.4 Historical Window

| Purpose | Data Length | Trading Days |
|---|---|---|
| VaR lookback | 1 year | 252 days |
| Backtesting | 3 years rolling | ~756 days |
| Stress VaR search window | 5 years | ~1,260 days |
| Total data generated | 5 years + 1 year buffer | ~1,510 days |

---

## 4. Data Generation Strategy

Two modes will be supported, selectable via config:

### Mode A — Synthetic Data Generator (default)
- Equity prices simulated via **Geometric Brownian Motion (GBM)**:
  - `dS = μS dt + σS dW`
  - Calibrate μ and σ to realistic market parameters (e.g. SPX long-run values)
  - Add stochastic volatility regime switches to model vol clustering and crisis periods
  - Correlations across underlyings via Cholesky decomposition of a covariance matrix
- Implied volatility surface generated as:
  - Flat ATM vol = realized vol × (1 + noise)
  - Skew: OTM puts carry a volatility premium (parametric skew)
  - Term structure: slight upward slope for longer tenors
- Crisis scenario injection: embed a 6–12 month stress period (e.g. 50% equity drawdown, vol spike to 60–80%) to ensure Stress VaR exercises are meaningful

### Mode B — Real Data Ingestion (optional)
- Download via `yfinance`: daily OHLCV for selected tickers and indices
- Implied vol approximated from historical realized vol (no live options chain required)
- Option prices computed from historical equity prices + approximated vol surface using Black-Scholes

---

## 5. Portfolio Construction

The portfolio consists of **25 single-name equity underlyings** spread across multiple sectors, with European calls and puts at varying maturities and strikes. A synthetic index overlay will also be included.

```
Portfolio specification:
  Underlyings: 25 single-name equities
    - 5 Technology (e.g. AAPL-like)
    - 5 Financials (e.g. JPM-like)
    - 5 Healthcare (e.g. JNJ-like)
    - 5 Energy (e.g. XOM-like)
    - 5 Consumer (e.g. AMZN-like)

  Options per underlying:
    Maturities:  1m, 3m, 6m, 12m
    Strikes:     0.85×S, 0.95×S, 1.00×S (ATM), 1.05×S, 1.15×S
    Types:       Call and Put at each (strike, maturity) node
    Positions:   Mix of long and short; defined in portfolio.yaml

  Index overlay:
    - 1 synthetic broad-market index (equal-weighted of 25 names)
    - Long calls at 1.05×S (3m and 6m) — tail hedge

  Total options: ~25 × 4 × 5 × 2 = 1,000 option legs (before netting)
  Active positions (post config): ~150–200 legs (sparse book)
```

The portfolio module will:
1. Read positions from `config/portfolio.yaml`
2. Value each option using current market data via Black-Scholes
3. Report current Greeks (portfolio delta, gamma, vega, theta) per underlying and in aggregate

---

## 6. VaR Methodologies

### 6.1 Primary: Full Revaluation Historical Simulation (HVaR)

**Method:**
1. Collect 252 daily log-returns for each risk factor (equity prices, implied vols)
2. For each of the 252 historical scenarios:
   - Shift today's equity prices and vol surface by the historical scenario return
   - Reprice every option using Black-Scholes under the shifted inputs
   - Compute scenario P&L = new portfolio value − current portfolio value
3. Sort the 252 P&L scenarios
4. VaR(99%) = loss at the 2.52nd worst scenario (floor to 3rd worst)
5. VaR(95%) = loss at the 12.6th worst scenario (floor to 13th worst)

**Advantages:** Captures full non-linearity; no distributional assumptions; handles skew, gamma, vega correctly.

**Limitations:** Computationally heavier; bounded by the historical window.

### 6.2 Secondary: Delta-Gamma-Vega Approximation

**Method:**
- Approximate option P&L per scenario using second-order Taylor expansion:
  - `ΔP ≈ Δ·ΔS + ½·Γ·(ΔS)² + ν·Δσ`
- Aggregate across positions using portfolio Greeks
- Apply 252 historical factor shocks to compute P&L distribution

**Advantages:** Fast (vectorized); useful as a cross-check and for large portfolios.

**Limitations:** Approximation breaks down for large moves; ignores higher-order Greeks.

### 6.3 Expected Shortfall (ES / CVaR)

Expected Shortfall is computed alongside VaR for both confidence levels. ES captures the average severity of losses beyond the VaR threshold (the "tail mean").

**Formula:**
- ES(α) = −E[P&L | P&L < −VaR(α)]
- ES(99%) over 252 scenarios = mean of the worst ~3 scenario P&Ls
- ES(95%) over 252 scenarios = mean of the worst ~13 scenario P&Ls

ES is more sensitive than VaR to the shape of the tail and is preferred under FRTB for regulatory capital.

### 6.4 Confidence Levels & Holding Period

| Metric | Value |
|---|---|
| VaR confidence levels | 99% and 95% (1-day) |
| ES confidence levels | 99% and 95% (1-day) |
| Holding period | 1 day |
| Lookback window | 252 trading days |
| Scaling to 10-day VaR | √10 scaling (regulatory, phase 2) |

---

## 7. Stress VaR (Phase 2)

Following Basel II.5 / Basel III conventions:

1. Using the full 5-year dataset, identify the **worst continuous 252-trading-day window** for the current portfolio
2. Re-run the full revaluation HVaR calculation over that stressed window
3. Report Stressed VaR(99%) alongside base VaR(99%)
4. The stressed period selection is updated periodically (e.g. quarterly)

**Predefined crisis periods** (injectable for testing):
- 2008-09-01 to 2009-09-01 (Global Financial Crisis)
- 2020-02-01 to 2020-12-31 (COVID crash)
- 2022-01-01 to 2022-12-31 (Rate shock)

Since we use synthetic data, we will embed a synthetic stress period into the generated history that mimics the severity of a real crisis.

---

## 8. Factor-Based VaR (Phase 2)

### 8.1 Motivation
Decompose portfolio VaR into systematic (index-driven) and idiosyncratic components. Useful for:
- Risk attribution by factor
- Hedging analysis (how much does SPX hedge reduce VaR?)
- Understanding correlation contribution

### 8.2 Methodology

**Step 1 — Factor Model for Equities:**
- Regress each equity's historical returns against index returns:
  - `r_i = α_i + β_i · r_index + ε_i`
- For a multi-factor model, include sector indices

**Step 2 — Factor P&L:**
- Decompose each historical scenario return into factor component and residual
- Apply factor shocks separately to compute systematic vs idiosyncratic P&L

**Step 3 — Factor VaR:**
- Compute VaR of systematic P&L (factor VaR)
- Compute VaR of idiosyncratic P&L
- Report combined, systematic, and idiosyncratic components

**Proposed Factor Set (configurable):**

| Factor | Proxy |
|---|---|
| Market | SPX or synthetic broad market index |
| Technology | XLK proxy |
| Financials | XLF proxy |
| Energy | XLE proxy |

---

## 9. Backtesting Framework

Regulatory-grade backtesting will be implemented:

| Test | Description |
|---|---|
| **Exception counting** | Count days where actual P&L < −VaR |
| **Exception rate** | exceptions / total days (target: ≤1% for 99% VaR) |
| **Kupiec POF test** | Likelihood ratio test on exception rate |
| **Christoffersen test** | Tests for independence of exceptions (clustering) |
| **Traffic light** | Green (≤4 exceptions/250), Yellow (5–9), Red (≥10) |

Backtesting will run over the full 3-year backtest window, with results plotted as:
- P&L vs VaR time series
- Exception scatter chart
- Rolling exception rate

---

## 10. Proposed System Architecture

```
var_demo/
├── data/
│   ├── generator.py          # Synthetic data generation (GBM + vol surface)
│   ├── loader.py             # Real data ingestion (yfinance)
│   └── schemas.py            # Data validation / type definitions
├── portfolio/
│   ├── position.py           # Option/equity position definitions
│   ├── pricer.py             # Black-Scholes pricer + Greeks
│   └── portfolio.py          # Aggregation, current valuation
├── var/
│   ├── historical.py         # Full revaluation HVaR
│   ├── delta_gamma.py        # Delta-Gamma-Vega approximation VaR
│   ├── stress.py             # Stress VaR (phase 2)
│   └── factor.py             # Factor VaR (phase 2)
├── backtest/
│   ├── engine.py             # Rolling backtest runner
│   └── statistics.py         # Kupiec, Christoffersen, traffic light
├── reporting/
│   └── report.py             # Output tables + charts
├── config/
│   ├── portfolio.yaml        # Portfolio positions
│   └── model.yaml            # Model parameters (window, confidence, etc.)
├── docs/
│   ├── requirements.md       # This document
│   └── test_plan.md
└── tests/
    └── ...
```

---

## 11. Technology Stack

| Component | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Option pricing | Custom Black-Scholes (scipy.stats) |
| Real data (optional) | `yfinance` |
| Statistical tests | `scipy.stats`, `statsmodels` |
| Visualization | `matplotlib`, `seaborn` |
| Config management | `pyyaml` |
| Testing | `pytest`, `pytest-cov` |
| Dependency management | `pip` + `requirements.txt` |

---

## 12. Proposed Implementation Phases

| Phase | Deliverables |
|---|---|
| **Phase 1 — Foundation** | Synthetic data generator (25 underlyings, 5yr history); Black-Scholes pricer + Greeks; portfolio construction (config-driven, 25 names, varying maturities/strikes); full revaluation HVaR + ES at 99%/95%; backtesting (Kupiec, Christoffersen, traffic light); richer reporting (charts + formatted tables) |
| **Phase 2 — Extensions** | Stress VaR (worst-window search + crisis injection); factor-based VaR (4 index factors, systematic/idiosyncratic split); real data ingestion (yfinance Mode B); delta-gamma-vega approximation VaR |
| **Phase 3 — Hardening** | Performance optimisation for large portfolios; PDF/HTML report export; √10 scaling to 10-day VaR; extended backtesting statistics |

---

## 13. Key Design Decisions & Trade-offs

| Decision | Chosen Approach | Rationale |
|---|---|---|
| Option pricing model | Black-Scholes (European only) | Tractable, analytically differentiable for Greeks; American options explicitly out of scope |
| Vol surface | Parametric skew + term structure | Avoids requiring a live options chain; realistic enough for HVaR |
| Historical scenario application | Log-return shock to S, additive shock to σ | Standard industry practice |
| VaR aggregation | Full portfolio revaluation per scenario | More accurate than portfolio-level Greeks for nonlinear books |
| Correlation handling | Implicit in historical scenarios | No need for correlation matrix; historical returns capture co-movements naturally |
| Stress period | Embedded in synthetic data + predefined windows | Allows deterministic testing even without real data |

---

## 14. Confirmed Design Decisions (v0.1 → v0.2)

All open questions from v0.1 have been resolved:

| # | Question | Decision |
|---|---|---|
| 1 | Portfolio composition | **25 single-name underlyings**; European calls and puts at 1m/3m/6m/12m maturities; strikes at 0.85/0.95/1.00/1.05/1.15 × S |
| 2 | Real vs synthetic data | **Synthetic (Mode A) for Phase 1**; yfinance (Mode B) deferred to Phase 2 |
| 3 | Output format | **Richer reporting**: matplotlib charts (P&L distribution, VaR time series, backtesting exceptions) + formatted summary tables |
| 4 | Confidence levels | **VaR and ES at both 95% and 99%**; 10-day scaling deferred to Phase 3 |
| 5 | American options | **Out of scope for all phases** |
| 6 | Factor set | **Four equity index factors confirmed** (market, tech, financials, energy) |
