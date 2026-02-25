"""
var/historical.py
-----------------
Full-revaluation Historical Value-at-Risk and Expected Shortfall.

Public API
----------
    VaRResult               — container for VaR/ES output and the P&L vector
    InsufficientDataError   — raised when the lookback window is too long
    compute_pnl_scenarios   — apply any set of log-return scenarios to a portfolio
    compute_historical_var  — compute 1-day HVaR + ES from MarketData

Methodology
-----------
For each of the N most recent 1-day historical log-return scenarios:

    1. Shocked spot:  S_shocked_i = S_current × exp(log_return_i)

    2. Shocked vol:   implied vol re-interpolated from the CURRENT vol surface
                      at the new moneyness K / S_shocked_i (sticky-moneyness
                      assumption — the shape of the surface is unchanged).

    3. Revalue:       portfolio value at (S_shocked, vol_shocked) via Black-Scholes.

    4. P&L_i:         V_shocked_i − V_current

The P&L vector is sorted ascending (worst loss first).  VaR and ES are then:

    n_exceed(α) = ceil( (1 − α) × N )               — tail scenario count
    VaR(α)      = −pnl_sorted[ n_exceed − 1 ]       — N-th worst loss, ≥ 0
    ES(α)        = −mean( pnl_sorted[:n_exceed] )    — mean of worst N losses, ≥ 0

For N = 252 and α = 99%:  n_exceed = ceil(2.52) = 3  → VaR = 3rd worst loss.
For N = 252 and α = 95%:  n_exceed = ceil(12.6) = 13 → VaR = 13th worst loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data import MONEYNESS_LABELS, MONEYNESS_LEVELS, TENOR_LABELS, TENOR_YEARS
from portfolio.portfolio import MarketSnapshot, Portfolio
from portfolio.pricer import bs_price


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class InsufficientDataError(ValueError):
    """
    Raised when the lookback window requires more scenarios than are available
    in the market data's log-return history.
    """


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VaRResult:
    """
    Container for 1-day Historical VaR and Expected Shortfall results.

    Parameters
    ----------
    var_95 : float
        VaR at 95% confidence (positive number = loss in dollars).
    var_99 : float
        VaR at 99% confidence (positive number = loss in dollars).
    es_95 : float
        Expected Shortfall at 95% confidence (mean of worst 5% P&Ls, dollars).
    es_99 : float
        Expected Shortfall at 99% confidence (mean of worst 1% P&Ls, dollars).
    pnl_vector : np.ndarray
        Unsorted 1-D P&L array, one entry per historical scenario.
        Negative values = portfolio losses; positive values = gains.
    n_scenarios : int
        Number of historical scenarios used (typically 252).
    base_value : float
        Portfolio MTM value at the current pricing snapshot.
    """

    var_95: float
    var_99: float
    es_95: float
    es_99: float
    pnl_vector: np.ndarray
    n_scenarios: int
    base_value: float

    @property
    def pnl_sorted(self) -> np.ndarray:
        """P&L vector sorted ascending (worst loss first, best gain last)."""
        return np.sort(self.pnl_vector)


# ---------------------------------------------------------------------------
# Internal: vectorised vol lookup over a batch of shocked spot prices
# ---------------------------------------------------------------------------

def _get_vol_batch(
    vol_surface: pd.Series,
    ticker: str,
    K: float,
    S_vec: np.ndarray,
    tenor_years: float,
) -> np.ndarray:
    """
    Vectorised implied-vol lookup for a vector of spot prices.

    Selects the nearest tenor label (no tenor interpolation in Phase 1) then
    uses ``np.interp`` for linear moneyness interpolation with automatic
    boundary clamping — identical semantics to the scalar ``_get_vol`` in
    portfolio.portfolio but operating on a full scenario batch at once.

    Parameters
    ----------
    vol_surface : pd.Series
        Single-date vol surface row with MultiIndex (ticker, moneyness, tenor).
    ticker : str
        Underlying identifier.
    K : float
        Fixed strike price (same across all scenarios).
    S_vec : np.ndarray
        Shocked spot prices, one per scenario — shape (n_scenarios,).
    tenor_years : float
        Time to expiry in years.

    Returns
    -------
    np.ndarray of shape (n_scenarios,) — annualised implied vols (decimal).
    """
    tenor_label = min(TENOR_LABELS, key=lambda lbl: abs(TENOR_YEARS[lbl] - tenor_years))
    grid_vols = np.array(
        [float(vol_surface[ticker, ml, tenor_label]) for ml in MONEYNESS_LABELS]
    )
    moneyness = K / S_vec  # one per scenario; shape (n_scenarios,)
    # np.interp clamps to grid_vols[0] / grid_vols[-1] outside [xp[0], xp[-1]]
    return np.interp(moneyness, MONEYNESS_LEVELS, grid_vols)


# ---------------------------------------------------------------------------
# Internal: single-scenario shocked snapshot (used by tests, not the engine)
# ---------------------------------------------------------------------------

def _build_shocked_snapshot(
    snapshot: MarketSnapshot,
    log_returns_row: pd.Series,
) -> MarketSnapshot:
    """
    Apply one historical log-return row to produce a shocked MarketSnapshot.

    Shocked spot: S_shocked = S_current × exp(log_return).
    Vol surface is frozen at the current snapshot (sticky-moneyness assumption).

    This helper is exposed for unit testing (VAR-10).  The vectorised engine
    ``compute_pnl_scenarios`` does not call it — it handles batching directly.
    """
    shocked_spots = snapshot.spot_prices * np.exp(log_returns_row)
    return MarketSnapshot(
        spot_prices=shocked_spots,
        vol_surface=snapshot.vol_surface,
        risk_free_rate=snapshot.risk_free_rate,
        dividend_yields=snapshot.dividend_yields,
        date=None,
    )


# ---------------------------------------------------------------------------
# Internal: VaR and ES from a sorted P&L vector
# ---------------------------------------------------------------------------

def _var_es_at_confidence(
    pnl_sorted: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    """
    Compute VaR and ES at a given confidence level from a sorted P&L array.

    Parameters
    ----------
    pnl_sorted : np.ndarray
        P&L vector sorted ascending (worst loss first).
    confidence : float
        Confidence level (e.g. 0.95, 0.99).

    Returns
    -------
    (var, es) — both floored at 0.0 (a portfolio that gains in all scenarios
    has VaR = ES = 0).
    """
    n = len(pnl_sorted)
    n_exceed = max(1, math.ceil((1.0 - confidence) * n))
    var = max(0.0, -float(pnl_sorted[n_exceed - 1]))
    es = max(0.0, -float(pnl_sorted[:n_exceed].mean()))
    return var, es


# ---------------------------------------------------------------------------
# Public: P&L computation across a set of historical scenarios
# ---------------------------------------------------------------------------

def compute_pnl_scenarios(
    portfolio: Portfolio,
    snapshot: MarketSnapshot,
    scenario_returns: pd.DataFrame,
) -> np.ndarray:
    """
    Full-revaluation P&L for a set of historical log-return scenarios.

    For each row in ``scenario_returns``:
        S_shocked  = snapshot.spot_prices × exp(row)
        sigma_vec  = re-interpolated from current vol surface at new moneyness
        P&L        = Σ_legs [bs_price(S_shocked, …) × notional_shares] − base_value

    The computation is vectorised over scenarios: each leg is priced across
    all N scenarios simultaneously using numpy broadcasting.

    Parameters
    ----------
    portfolio : Portfolio
        Option legs to reprice.
    snapshot : MarketSnapshot
        Current market state (the pricing "today").
    scenario_returns : pd.DataFrame
        Historical log-return scenarios, one per row.
        Columns must include every ticker that appears in the portfolio.
        Shape: (n_scenarios, n_tickers_or_more).

    Returns
    -------
    np.ndarray of shape (n_scenarios,).
    Negative = loss, positive = gain.
    """
    if not portfolio.positions:
        return np.zeros(len(scenario_returns))

    n = len(scenario_returns)
    tickers = list(snapshot.spot_prices.index)

    # --- Shocked spot matrix (n_scenarios, n_tickers) ---
    spot_current = snapshot.spot_prices.reindex(tickers).values       # (n_tickers,)
    return_mat   = scenario_returns.reindex(columns=tickers).values   # (n, n_tickers)
    shocked_mat  = spot_current * np.exp(return_mat)                  # (n, n_tickers)

    ticker_col = {t: i for i, t in enumerate(tickers)}

    # --- Base portfolio value (once) ---
    base_value = portfolio.value(snapshot)

    # --- Accumulate scenario values across all legs ---
    scenario_values = np.zeros(n)
    for pos in portfolio.positions:
        col    = ticker_col[pos.ticker]
        S_vec  = shocked_mat[:, col]                                  # (n,)
        q      = float(snapshot.dividend_yields[pos.ticker])
        n_shr  = pos.notional_shares                                  # signed int

        # Vectorised vol lookup: each scenario may have different moneyness
        sigma_vec = _get_vol_batch(
            snapshot.vol_surface, pos.ticker, pos.strike, S_vec, pos.tenor_years
        )

        # Vectorised Black-Scholes price across all n scenarios
        prices = bs_price(
            S_vec, pos.strike, pos.tenor_years,
            snapshot.risk_free_rate, sigma_vec, q,
            pos.option_type.value,
        )

        scenario_values += prices * n_shr

    return scenario_values - base_value


# ---------------------------------------------------------------------------
# Public: full Historical VaR computation from MarketData
# ---------------------------------------------------------------------------

def compute_historical_var(
    portfolio: Portfolio,
    snapshot: MarketSnapshot,
    market_data,
    lookback_window: int = 252,
) -> VaRResult:
    """
    Compute 1-day Historical VaR and Expected Shortfall via full revaluation.

    Uses the most recent ``lookback_window`` rows of ``market_data.log_returns``
    as historical scenarios (i.e. the last trading year).

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio to evaluate.
    snapshot : MarketSnapshot
        Current market state.  The spot prices here are used as the base; the
        historical scenarios are applied as relative shocks on top of them.
    market_data : MarketData
        Full generated (or loaded) market dataset.
    lookback_window : int
        Number of 1-day historical scenarios.  Default 252 (one trading year).

    Returns
    -------
    VaRResult

    Raises
    ------
    InsufficientDataError
        If fewer than ``lookback_window`` rows are available.
    """
    available = len(market_data.log_returns)
    if available < lookback_window:
        raise InsufficientDataError(
            f"Lookback window requires {lookback_window} scenarios but "
            f"market_data has only {available} log-return rows."
        )

    scenario_returns = market_data.log_returns.iloc[-lookback_window:]
    pnl = compute_pnl_scenarios(portfolio, snapshot, scenario_returns)

    pnl_sorted = np.sort(pnl)
    var_95, es_95 = _var_es_at_confidence(pnl_sorted, 0.95)
    var_99, es_99 = _var_es_at_confidence(pnl_sorted, 0.99)

    return VaRResult(
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99,
        pnl_vector=pnl,
        n_scenarios=len(pnl),
        base_value=portfolio.value(snapshot),
    )
