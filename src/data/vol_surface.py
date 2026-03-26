"""
Implied volatility surface construction.

Approach (requirements §2.3):
  1. Compute rolling realised volatility from yfinance equity log-returns.
  2. Use tenor-matched windows as ATM IV proxy.
  3. Apply a quadratic skew in log-moneyness to get the full strike surface.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rolling realised vol parameters
# ---------------------------------------------------------------------------

VOL_WINDOWS = {
    "short":  21,   # ~1 month  — used for T < 3 months
    "medium": 63,   # ~3 months — used for 3 ≤ T < 6 months
    "long":   126,  # ~6 months — used for T ≥ 6 months
}

# Tenor cutoffs in years
_SHORT_CUTOFF  = 63 / 252   # 3 months
_MEDIUM_CUTOFF = 126 / 252  # 6 months

# Default quadratic skew parameters (configurable per ticker in future)
DEFAULT_ALPHA = -0.15   # negative skew (downside puts more expensive)
DEFAULT_BETA  =  0.05   # smile curvature

# No-arbitrage vol bounds
# Lower: vol cannot be zero or negative (enforced by BS formula)
# Upper: cap at 5.0 (500%) — prevents numerical explosion at extreme strikes.
#   Full Dupire butterfly constraint (d²(σ²T)/dk² ≥ 0) is not enforced here;
#   the quadratic form is well-behaved for |k| < 1 (typical liquid strikes).
VOL_FLOOR = 1e-4
VOL_CAP   = 5.0


# ---------------------------------------------------------------------------
# Rolling vol computation
# ---------------------------------------------------------------------------


def compute_rolling_vol(
    prices: pd.DataFrame,
    windows: dict[str, int] | None = None,
    annualisation_factor: int = 252,
) -> dict[str, pd.DataFrame]:
    """
    Compute rolling annualised realised volatility for each window.

    Parameters
    ----------
    prices              : DataFrame of adjusted close prices (n_days × n_tickers)
    windows             : dict mapping window label to rolling window size in days
    annualisation_factor: trading days per year (252)

    Returns
    -------
    dict[str, DataFrame] — one DataFrame per window label, same shape as prices.
    NaN in the first (window - 1) rows of each column (warm-up period).
    """
    if windows is None:
        windows = VOL_WINDOWS

    log_returns = np.log(prices / prices.shift(1))

    rolling_vols: dict[str, pd.DataFrame] = {}
    for label, window in windows.items():
        rv = log_returns.rolling(window).std() * np.sqrt(annualisation_factor)
        rolling_vols[label] = rv

    return rolling_vols


def get_atm_vol(
    rolling_vols: dict[str, pd.DataFrame],
    ticker: str,
    date: pd.Timestamp | str,
    T_remaining: float,
) -> float:
    """
    Return the tenor-matched ATM IV proxy for a single ticker on a given date.

    Parameters
    ----------
    rolling_vols : output of compute_rolling_vol
    ticker       : equity ticker string
    date         : historical date to look up
    T_remaining  : time to expiry in years

    Returns
    -------
    Annualised ATM implied vol (float)
    """
    date = pd.Timestamp(date)

    if T_remaining < _SHORT_CUTOFF:
        label = "short"
    elif T_remaining < _MEDIUM_CUTOFF:
        label = "medium"
    else:
        label = "long"

    vol = rolling_vols[label].loc[date, ticker]

    if np.isnan(vol):
        # Warm-up period: fall back to the next available window
        for fallback in ("short", "medium", "long"):
            v = rolling_vols[fallback].loc[date, ticker]
            if not np.isnan(v):
                vol = v
                break

    if np.isnan(vol):
        # TODO: decide on a better fallback (e.g. cross-sectional average or VIX-derived)
        raise ValueError(
            f"No realised vol available for {ticker} on {date.date()} — "
            "increase warm-up period or check data."
        )

    # No-arbitrage bounds: floor at VOL_FLOOR, cap at VOL_CAP
    return float(np.clip(vol, VOL_FLOOR, VOL_CAP))


# ---------------------------------------------------------------------------
# Skew model
# ---------------------------------------------------------------------------


def skew_vol(
    atm_vol: float,
    K: float,
    S: float,
    r: float,
    T: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> float:
    """
    Quadratic skew model in log-moneyness.

    σ(K, T) = σ_ATM + α · k + β · k²
    where k = log(K / F), F = S · exp(r · T)

    Parameters
    ----------
    atm_vol : ATM implied vol (output of get_atm_vol)
    K       : option strike
    S       : current spot price
    r       : risk-free rate
    T       : time to expiry in years
    alpha   : skew coefficient (typically negative for equities)
    beta    : smile curvature coefficient

    Returns
    -------
    Smile-adjusted implied vol (floored at 1e-4)
    """
    if T <= 0:
        return atm_vol

    F = S * np.exp(r * T)
    k = np.log(K / F)
    vol = atm_vol + alpha * k + beta * k**2

    # No-arbitrage bounds: floor prevents negative vol; cap prevents explosion
    # at extreme deep-OTM strikes. Full Dupire constraint omitted (out of scope).
    return float(np.clip(vol, VOL_FLOOR, VOL_CAP))


# ---------------------------------------------------------------------------
# Convenience: build full surface on one date for one ticker
# ---------------------------------------------------------------------------


def surface_vol(
    rolling_vols: dict[str, pd.DataFrame],
    ticker: str,
    date: pd.Timestamp | str,
    K: float,
    S: float,
    r: float,
    T: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> float:
    """
    Full vol surface lookup: tenor-matched ATM vol + quadratic skew.

    Parameters
    ----------
    rolling_vols : output of compute_rolling_vol
    ticker       : equity ticker
    date         : valuation date
    K            : strike
    S            : spot price
    r            : risk-free rate
    T            : time to expiry in years

    Returns
    -------
    Implied vol for (K, T) on the given date.
    """
    atm = get_atm_vol(rolling_vols, ticker, date, T)
    return skew_vol(atm, K, S, r, T, alpha=alpha, beta=beta)
