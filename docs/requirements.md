# Historical VAR Model — Equities Options Portfolio
## Model Requirements (Final)

**Author:** Claude (Quant Developer Assistant)
**Date:** 2026-02-27
**Status:** Approved — Implementation Ready

---

## 1. Executive Summary

This document specifies a **Historical Simulation Value-at-Risk (HS-VAR)** framework for a European equities options portfolio. The system is built in three layers:

1. **Data Layer** — real historical equity and index prices from `yfinance`, combined with a synthetic implied volatility surface for options repricing
2. **VAR & ES Engine** — multiple methodologies producing both VAR quantiles and Expected Shortfall at configurable confidence levels
3. **Backtesting & Stress Extension** — statistical validation and stressed scenario capability

### Finalised Parameters

| Parameter | Decision |
|---|---|
| Options type | European only |
| Equity universe | 25 single-name equities + equity indices |
| Data source | `yfinance` (real historical prices); no synthetic fallback for equity data |
| Historical lookback (VAR) | 252 trading days |
| Full data history | 5 years (~1,260 trading days) — for backtesting and Stress VAR |
| Confidence levels | Configurable; defaults 95% and 99% |
| Risk measures | VAR (quantile) **and** ES (Expected Shortfall / CVaR) at every confidence level |
| 10-day horizon | √10 scaling of 1-day VAR |
| Pricer | Black-Scholes (European closed-form) |
| Options historical pricing | Hybrid approach — see Section 2.3 |

---

## 2. Data Layer

### 2.1 Equity Universe

**25 single-name equities** downloaded from `yfinance`. Suggested diversified universe (configurable):

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, NVDA, GOOGL, META |
| Financials | JPM, GS, BAC, MS, BLK |
| Healthcare | JNJ, UNH, PFE, ABBV, MRK |
| Consumer | AMZN, TSLA, HD, NKE, MCD |
| Energy / Industrials | XOM, CVX, CAT, BA, GE |

**Data spec:**
- Field: adjusted close price (accounts for splits and dividends)
- Frequency: daily
- History: 5 years from download date (~1,260 rows per ticker)
- Missing days: forward-filled then back-filled; any ticker with > 5% missing data raises a warning

### 2.2 Equity Index Universe

Indices serve dual purposes: portfolio underlyings (index options) and risk factors (factor VAR).

| yfinance Ticker | Description | Use |
|---|---|---|
| `^GSPC` | S&P 500 | Market factor, underlying |
| `^NDX` | NASDAQ-100 | Tech factor, underlying |
| `^RUT` | Russell 2000 | Size factor |
| `^VIX` | CBOE VIX | Volatility factor |
| `XLK` | Tech Select Sector ETF | Sector factor |
| `XLF` | Financial Select Sector ETF | Sector factor |
| `XLE` | Energy Select Sector ETF | Sector factor |

### 2.3 Options Historical Pricing — Hybrid Approach

**The problem:** `yfinance` provides historical equity spot prices but **no historical options chains**. For historical VAR, we need a daily option value for every trade across the full 5-year history.

**Solution: Historical Repricing with Synthetic Vol Surface**

On each historical date `t`, we reprice every option in the book using Black-Scholes, with the following inputs constructed from real price data:

#### Step 1 — Realised Vol as ATM IV Proxy

Compute rolling realised volatility from log-returns of the equity/index spot price. Use tenor-matched windows:

| Option tenor remaining | Realised vol window used as IV |
|---|---|
| < 3 months (63 days) | 21-day rolling realised vol |
| 3–6 months | 63-day rolling realised vol |
| > 6 months | 126-day rolling realised vol |

For tenors between bands, linearly interpolate between adjacent vol estimates.

**Rationale:** Realised vol is a well-established proxy for ATM implied vol, especially for single-name equities where the implied-realised vol spread is relatively stable. It anchors the surface to genuine market-observed price dynamics.

#### Step 2 — Synthetic Skew Model

Apply a quadratic skew in log-moneyness to generate the full strike surface:

```
k = log(K / F)          where F = S · exp(r · T)  [log-moneyness]

σ(k, T) = σ_ATM(T) + α · k + β · k²
```

Default parameters (configurable):
- `α = -0.15` — negative skew (downside puts more expensive; standard for equities)
- `β = 0.05` — smile curvature

These produce a realistic equity vol skew. In Phase 2, `α` and `β` can be calibrated to current VIX skew or current options chain if available.

#### Step 3 — Reprice Each Option

For each historical date `t` and each trade in the book, compute:

```
V_t = BS_price(S=spot_t, K=strike, T=T_remaining, r=r_f, σ=σ(k, T_remaining))
```

`T_remaining` decreases by 1/252 per day (business day count).

#### Step 4 — Compute Historical P&L

```
PnL_t = V_t - V_today
```

This produces a 252-element P&L vector (one per scenario date) for each option in the portfolio. Portfolio P&L = sum of individual P&Ls weighted by notional and position sign.

**What this captures:** actual historical equity spot moves (from yfinance) feeding into real option Greeks; vol changes driven by actual realised vol dynamics. The skew shape is synthetic but anchored to real level.

**What it does not capture:** implied vol surface moves beyond the realised vol proxy (e.g., sudden IV spikes like VIX jumps on event days). This is an accepted limitation of the base model; the GARCH-filtered extension (Method C) partially addresses it.

### 2.4 Options Portfolio — Trade Format

The portfolio is represented as a CSV file (or in-memory DataFrame) with the following schema:

```
trade_id       str       Unique identifier, e.g. "OPT-001"
underlying     str       yfinance ticker, e.g. "AAPL" or "^GSPC"
option_type    str       "call" or "put"
strike         float     Strike price (same currency as spot)
expiry         date      YYYY-MM-DD
notional       float     Number of contracts × contract multiplier (e.g. 100 shares/contract)
position_sign  int       +1 = long, -1 = short
premium        float     Original traded premium per unit (for P&L attribution)
trade_date     date      YYYY-MM-DD
```

Example:
```
trade_id,underlying,option_type,strike,expiry,notional,position_sign,premium,trade_date
OPT-001,AAPL,call,180.00,2025-06-20,100,1,8.50,2025-01-15
OPT-002,AAPL,put,160.00,2025-06-20,100,-1,5.20,2025-01-15
OPT-003,^GSPC,call,5000.00,2025-12-19,100,1,45.00,2025-01-15
OPT-004,MSFT,put,350.00,2025-09-19,100,1,12.75,2025-01-15
```

The portfolio generator will create a synthetic book spanning 25 underlyings with a mix of calls, puts, and spreads. The format also allows real trades to be loaded from a CSV.

---

## 3. Risk Measures

Every VAR engine outputs **both** VAR and ES at every specified confidence level.

### 3.1 VAR (Quantile)

```
VAR(α) = -Quantile(PnL distribution, 1 - α)
```

Convention: VAR is reported as a **positive number** representing a loss.

For plain HS with 252 scenarios:

| Confidence | Scenarios in tail | Scenario index (0-based, sorted ascending) |
|---|---|---|
| 95% | 12–13 | 12 (floor(252 × 0.05)) |
| 99% | 2–3 | 2 (floor(252 × 0.01)) |

### 3.2 Expected Shortfall (ES / CVaR)

ES is the mean loss of all scenarios **beyond** the VAR quantile:

```
ES(α) = -E[PnL | PnL < -VAR(α)]
       = mean of the worst floor(252 × (1 - α)) scenario P&Ls (sign-flipped)
```

ES is always ≥ VAR at the same confidence level. For regulatory purposes (FRTB / Basel III), ES at 97.5% is the primary risk measure; we compute it alongside 95% and 99%.

### 3.3 Configuration Object

All confidence levels are passed as a list:

```python
var_config = {
    "confidence_levels": [0.95, 0.99],   # configurable
    "horizon_days": 1,                    # 1-day base
    "scale_to_10d": True,                 # apply √10 scaling
    "measures": ["var", "es"],            # both measures always computed
    "lookback_days": 252,                 # rolling window
}
```

### 3.4 10-Day Horizon

10-day VAR and ES are computed by scaling 1-day figures:

```
VAR_10d = VAR_1d × √10
ES_10d  = ES_1d  × √10
```

---

## 4. VAR Methodologies

### 4.1 Method A — Plain Historical Simulation (Base Model)

**Steps:**
1. Collect 252 daily log-return vectors `r_t` for all equity spots and vol levels
2. For each scenario `t`, apply the return shock to today's market data → stressed market state
3. Reprice all European options under the stressed state using BS + synthetic vol surface (full revaluation)
4. Compute portfolio P&L: `PnL_t = V_stressed - V_today`
5. Sort 252 P&Ls ascending; read VAR and ES at each confidence level

**Output per run:**
```python
{
  "method": "plain_hs",
  "var_95_1d":  float,  "es_95_1d":  float,
  "var_99_1d":  float,  "es_99_1d":  float,
  "var_95_10d": float,  "es_95_10d": float,
  "var_99_10d": float,  "es_99_10d": float,
  "pnl_vector": np.ndarray,   # 252 values, for audit
}
```

---

### 4.2 Method B — Age-Weighted Historical Simulation

Exponential decay weights recent observations more heavily (λ typically 0.94–0.99):

```
w_t = (1 - λ) · λ^(T - t)    normalised so Σ w_t = 1
```

Weighted CDF replaces the empirical CDF for quantile and ES calculation.

---

### 4.3 Method C — Filtered Historical Simulation (GARCH-Filtered)

Standardise historical returns by conditional GARCH(1,1) vol, then rescale to today's regime:

```
ε_t = r_t / σ_t          # standardised innovation (zero mean, unit vol)
scenario_return_t = ε_t × σ_today
```

Filtered scenarios are then used identically to Method A.

---

### 4.4 Method D — Factor-Based Historical VAR

Decompose equity returns into systematic index factors and residuals:

```
r_i,t = α_i + Σ_k β_i,k · F_k,t + ε_i,t
```

Portfolio P&L under each factor scenario is computed via delta-adjusted Greeks:

```
ΔV_option ≈ Δ · ΔS + ½Γ · (ΔS)²    where ΔS driven by factor shocks
```

Factor set (configurable subset of index universe):
- Market (`^GSPC`), Tech (`^NDX`), Size (`^RUT`), Volatility (`^VIX`), Sector ETFs

---

### 4.5 Method Comparison

| | Plain HS | Age-Weighted | GARCH-Filtered | Factor VAR |
|---|---|---|---|---|
| Distributional assumption | None | None | None (GARCH) | Linear factor |
| Vol regime adaptive | No | Partial | Yes | Partial |
| Options treatment | Full reprice | Full reprice | Full reprice | Greeks approx. |
| ES supported | Yes | Yes | Yes | Yes |
| Recommended phase | Phase 1 base | Phase 1 alt | Phase 1 alt | Phase 1 ext |

---

## 5. Backtesting Framework

| Test | What it checks |
|---|---|
| **Kupiec POF** | Is observed exception rate consistent with (1-CL)? |
| **Christoffersen Independence** | Are exceptions clustered (serial dependence)? |
| **Basel Traffic Light** | Green (≤4) / Amber (5–9) / Red (≥10) exceptions over 250 days |

Rolling backtest: for each date in years 2–5, compute 1-day 99% VAR using the prior 252-day window; compare to next-day realised P&L; count exceptions. Backtest also computed for ES.

---

## 6. Stress VAR Extension (Phase 2)

- Scan the full 5-year history using a rolling 252-day window
- Select the window that maximises 99% VAR
- Stress VAR = that window's 99th-percentile loss
- Mirrors Basel 2.5 / FRTB IMA Stressed VAR requirement

---

## 7. Workflow

```
┌────────────────────────────────────────┐
│  1. DATA DOWNLOAD                      │
│  • yfinance: 25 equities + indices     │
│  • 5-year adjusted close prices        │
│  • Forward-fill gaps                   │
└──────────────────┬─────────────────────┘
                   │
┌──────────────────▼─────────────────────┐
│  2. VOL SURFACE CONSTRUCTION           │
│  • Rolling realised vol (21/63/126d)   │
│  • Quadratic skew model                │
│  • Per-ticker, per historical date     │
└──────────────────┬─────────────────────┘
                   │
┌──────────────────▼─────────────────────┐
│  3. OPTIONS PORTFOLIO SETUP            │
│  • Load / generate trade book (CSV)    │
│  • Reprice book at today's market      │
└──────────────────┬─────────────────────┘
                   │
┌──────────────────▼─────────────────────┐
│  4. SCENARIO CONSTRUCTION              │
│  • Compute daily log-returns           │
│  • Select 252-day lookback window      │
│  • (Optional) GARCH filter             │
│  • (Optional) Factor decomposition     │
└──────────────────┬─────────────────────┘
                   │
┌──────────────────▼─────────────────────┐
│  5. PORTFOLIO REVALUATION              │
│  • Full BS reprice per scenario        │
│  • Aggregate portfolio P&L vector      │
└──────────────────┬─────────────────────┘
                   │
┌──────────────────▼─────────────────────┐
│  6. VAR & ES COMPUTATION               │
│  • Quantile → VAR at 95%, 99%          │
│  • Tail mean → ES at 95%, 99%          │
│  • Scale to 10-day via √10             │
└──────────────────┬─────────────────────┘
                   │
┌──────────────────▼─────────────────────┐
│  7. BACKTESTING                        │
│  • Rolling 1-day 99% VAR over years 2–5│
│  • Kupiec + Christoffersen tests       │
│  • Basel traffic light report          │
└──────────────────┬─────────────────────┘
                   │
┌──────────────────▼─────────────────────┐
│  8. (PHASE 2) STRESS VAR               │
│  • Identify worst 252-day window       │
│  • Output stressed VAR and ES          │
└────────────────────────────────────────┘
```

---

## 8. Project Structure

```
var_demo/
├── CLAUDE.md
├── Journal.md
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── loader.py                # yfinance download, cache, fill
│   │   ├── vol_surface.py           # Rolling vol + quadratic skew surface
│   │   └── portfolio.py             # Trade format, portfolio builder
│   │
│   ├── var/
│   │   ├── scenarios.py             # Log-returns, scenario set construction
│   │   ├── revaluation.py           # Full BS reprice + Greeks approx
│   │   ├── historical_var.py        # Methods A (plain) and B (age-weighted)
│   │   ├── filtered_var.py          # Method C (GARCH-filtered)
│   │   ├── factor_var.py            # Method D (factor decomposition)
│   │   └── stress_var.py            # Phase 2: stressed VAR
│   │
│   ├── backtest/
│   │   └── backtest.py              # Rolling backtest, Kupiec, Christoffersen
│   │
│   └── utils/
│       ├── black_scholes.py         # BS pricer: price, delta, gamma, vega, theta
│       └── stats.py                 # VAR/ES quantile functions, stat helpers
│
├── data/
│   ├── raw/                         # yfinance cache (gitignored)
│   └── processed/                   # Cleaned price frames
│
├── portfolios/                      # Sample trade CSVs
│
├── notebooks/                       # Analysis notebooks
│
├── tests/
│   └── (see test_plan.md)
│
└── docs/
    ├── requirements.md              # This file
    └── test_plan.md
```

---

## 9. Technology Stack

| Component | Library |
|---|---|
| Numerical computation | `numpy`, `scipy` |
| Data manipulation | `pandas` |
| Real market data | `yfinance` |
| GARCH estimation | `arch` |
| Visualisation | `matplotlib`, `seaborn` |
| Testing | `pytest`, `pytest-cov` |

Minimum Python version: **3.10**

---

## 10. Decisions Log

| # | Question | Decision |
|---|---|---|
| 1 | Confidence levels | 95% and 99% (configurable list); both VAR and ES computed |
| 2 | Equity universe size | 25 names + equity indices |
| 3 | Options book source | Synthetic generator using defined trade format; CSV loadable |
| 4 | Options pricer | Black-Scholes (European closed-form only) |
| 5 | Historical options data | Hybrid: real spots from yfinance + rolling realised vol + synthetic skew |
| 6 | 10-day horizon | √10 scaling |
| 7 | Options type | European only |
| 8 | Indices | Included as both underlyings and risk factors |

---

*End of requirements.md*
