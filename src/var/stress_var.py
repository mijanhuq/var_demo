"""
Phase 2 — Stressed VAR.

Identifies the worst 252-day rolling window in the full historical price history
and computes VAR and ES over that stress period.

Mirrors the Basel 2.5 / FRTB Stressed VAR concept.
"""

import numpy as np
import pandas as pd

from data.portfolio import price_portfolio
from data.vol_surface import compute_rolling_vol, VOL_WINDOWS
from var.scenarios import (
    compute_log_returns,
    build_stressed_price_matrix,
)
from var.revaluation import full_reprice_pnl
from utils.stats import compute_var_es, scale_to_10d


def find_stress_window(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    rolling_vols: dict,
    today_prices: pd.Series,
    today_values: pd.Series,
    r: float,
    as_of: pd.Timestamp,
    window: int = 252,
    alpha: float = -0.15,
    beta: float = 0.05,
    primary_cl: float = 0.99,
) -> dict:
    """
    Scan all rolling `window`-day periods in the historical data and return
    the one that produces the highest VAR at `primary_cl`.

    Parameters
    ----------
    prices        : full historical price DataFrame (n_days × n_tickers)
    trades        : options portfolio DataFrame
    rolling_vols  : pre-computed rolling vol dict
    today_prices  : current spot prices (Series)
    today_values  : current portfolio position values (Series, index = trade_id)
    r             : risk-free rate
    as_of         : valuation date (last price date or user-specified)
    window        : rolling window size in days (default 252)
    alpha, beta   : vol skew parameters
    primary_cl    : confidence level used to score each window (default 0.99)

    Returns
    -------
    dict with:
        start_date, end_date — stress window boundaries
        best_var             — VAR at primary_cl for the stress window
        window_pnl           — P&L vector for the selected window
    """
    log_returns = compute_log_returns(prices)

    # Trim the rolling-vol warm-up period.
    # The longest vol window (126 days) first becomes valid at prices.index[126],
    # which corresponds to log_returns.index[125] (log_returns drops the first NaN
    # row from prices.shift(1)).  Scenarios before this date cause get_atm_vol to
    # raise ValueError because the rolling vol is still NaN.
    vol_warm_up = max(VOL_WINDOWS.values())  # 126
    log_returns = log_returns.iloc[vol_warm_up - 1:]

    n = len(log_returns)

    if n < window:
        raise ValueError(
            f"Only {n} return observations after vol warm-up; "
            f"need at least {window} for stress scan."
        )

    # -----------------------------------------------------------------------
    # Vectorised approach: one full_reprice_pnl call over all n scenarios,
    # then find the worst rolling window by scanning the resulting P&L array.
    #
    # Speedup: O(n) BS pricings instead of O(n_windows × window) calls.
    # The rolling window scan is pure numpy and fast.
    # -----------------------------------------------------------------------

    # 1. Price all n historical scenarios at once
    all_scenario_prices = build_stressed_price_matrix(today_prices, log_returns)
    all_pnl = full_reprice_pnl(
        trades, today_prices, today_values, all_scenario_prices,
        rolling_vols, r, as_of, alpha, beta,
    )  # shape (n,)

    # 2. Build a (n_windows, window) matrix of P&L values via stride trick
    n_windows = n - window + 1
    row_idx = np.arange(window)[np.newaxis, :] + np.arange(n_windows)[:, np.newaxis]
    windowed_pnl = all_pnl[row_idx]          # shape (n_windows, window)

    # 3. Compute VAR for each window: sort ascending, take n_tail-th value
    n_tail = max(1, int(np.floor(window * (1.0 - primary_cl))))
    sorted_windows = np.sort(windowed_pnl, axis=1)   # each row sorted ascending
    window_vars = -sorted_windows[:, n_tail - 1]      # VAR per window (positive)

    # 4. Select the worst window
    best_start_idx = int(np.argmax(window_vars))
    best_end_idx = best_start_idx + window
    best_var = float(window_vars[best_start_idx])
    best_pnl = windowed_pnl[best_start_idx]          # shape (window,)

    start_date = log_returns.index[best_start_idx]
    end_date = log_returns.index[best_end_idx - 1]

    return {
        "start_date": start_date,
        "end_date":   end_date,
        "best_var":   best_var,
        "window_pnl": best_pnl,
    }


def compute_stress_var(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    r: float = 0.05,
    as_of: pd.Timestamp | None = None,
    window: int = 252,
    confidence_levels: list[float] | None = None,
    alpha: float = -0.15,
    beta: float = 0.05,
) -> dict:
    """
    End-to-end Stressed VAR pipeline.

    Parameters
    ----------
    trades            : options portfolio DataFrame
    prices            : full historical price DataFrame
    r                 : risk-free rate
    as_of             : valuation date; defaults to last price date
    window            : stress window size (default 252)
    confidence_levels : defaults to [0.95, 0.99]
    alpha, beta       : vol skew parameters

    Returns
    -------
    dict with method, stress_window, pnl_vector, var/es figures.
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    if as_of is None:
        as_of = prices.index[-1]
    as_of = pd.Timestamp(as_of)

    rolling_vols = compute_rolling_vol(prices)
    today_prices = prices.loc[as_of]

    priced = price_portfolio(trades, today_prices, rolling_vols, r, as_of, alpha, beta)
    today_values = priced.set_index("trade_id")["position_value"]

    stress_info = find_stress_window(
        prices, trades, rolling_vols, today_prices, today_values, r, as_of,
        window=window, alpha=alpha, beta=beta, primary_cl=max(confidence_levels),
    )

    pnl = stress_info["window_pnl"]
    results = compute_var_es(pnl, confidence_levels)
    results = scale_to_10d(results)
    results["method"] = "stress_var"
    results["pnl_vector"] = pnl
    results["stress_start"] = stress_info["start_date"]
    results["stress_end"] = stress_info["end_date"]
    results["as_of"] = as_of
    results["n_scenarios"] = len(pnl)

    return results
