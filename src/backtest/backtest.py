"""
Rolling backtest, exception counting, and statistical validation.

Runs a rolling 1-day VAR over the historical dataset, compares to realised P&L,
and applies Kupiec, Christoffersen, and Basel traffic-light tests.

Trade lifecycle:
  - Options are aged each day (T decreases as as_of advances).
  - Options with expiry <= as_of are excluded from that day's P&L and VAR.
  - Cash settlement of expired options is not tracked (out of scope).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.vol_surface import compute_rolling_vol, VOL_WINDOWS
from data.portfolio import price_portfolio
from var.scenarios import compute_log_returns, build_scenario_set, build_stressed_price_matrix
from var.revaluation import full_reprice_pnl
from utils.stats import (
    compute_var_es,
    kupiec_test,
    christoffersen_test,
    traffic_light,
)


# ---------------------------------------------------------------------------
# Trade aging helper
# ---------------------------------------------------------------------------


def _active_trades(trades: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Return only trades that have not yet expired as of `as_of`.

    A trade is active if its expiry is strictly after `as_of`.
    Expired options are excluded from VAR and realised P&L on and after their
    expiry date (cash settlement is not tracked).
    """
    expiry = pd.to_datetime(trades["expiry"])
    return trades[expiry > as_of].copy()


# ---------------------------------------------------------------------------
# Core rolling backtest
# ---------------------------------------------------------------------------


def rolling_backtest(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    r: float = 0.05,
    lookback: int = 252,
    test_window: int | None = None,
    confidence_levels: list[float] | None = None,
    alpha: float = -0.15,
    beta: float = 0.05,
) -> pd.DataFrame:
    """
    Compute rolling 1-day VAR and next-day realised P&L for each date
    in the test window.

    Trade lifecycle: options are aged on each backtest date; expired options
    (expiry <= as_of) are excluded from that day's VAR and P&L computation.

    Parameters
    ----------
    trades            : options portfolio DataFrame
    prices            : full historical price DataFrame (n_days × n_tickers)
    r                 : risk-free rate
    lookback          : scenarios per VAR calculation (252)
    test_window       : number of days to backtest; defaults to all days after warm-up
    confidence_levels : defaults to [0.95, 0.99]
    alpha, beta       : vol skew parameters

    Returns
    -------
    DataFrame with columns:
        date, var_{cl}, es_{cl} (for each CL), realised_pnl, exception_{cl},
        n_active_trades
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    rolling_vols = compute_rolling_vol(prices)
    log_returns = compute_log_returns(prices)

    # Usable test dates: start after the rolling vol warm-up period AND the lookback window.
    # The longest vol window is 126 days; scenario slices must not include pre-warm-up dates.
    vol_warm_up = max(VOL_WINDOWS.values())  # 126 days
    first_test = lookback + vol_warm_up + 1  # 0-indexed into prices
    all_dates = prices.index.tolist()

    if test_window is not None:
        test_dates = all_dates[first_test: first_test + test_window]
    else:
        test_dates = all_dates[first_test:-1]  # exclude very last (no next day)

    records = []

    for as_of in test_dates:
        as_of_ts = pd.Timestamp(as_of)
        as_of_idx = prices.index.get_loc(as_of_ts)

        # Next business day — realised P&L
        if as_of_idx + 1 >= len(prices):
            break
        next_date = prices.index[as_of_idx + 1]
        next_ts = pd.Timestamp(next_date)

        # --- Filter to live trades on this date ---
        live_trades = _active_trades(trades, as_of_ts)
        if live_trades.empty:
            row: dict = {
                "date": as_of_ts, "realised_pnl": 0.0, "n_active_trades": 0,
            }
            for cl in confidence_levels:
                cl_key = str(int(round(cl * 100)))
                row[f"var_{cl_key}"] = 0.0
                row[f"es_{cl_key}"] = 0.0
                row[f"exception_{cl_key}"] = 0
            records.append(row)
            continue

        # --- VAR calculation ---
        today_prices = prices.loc[as_of_ts]
        priced_today = price_portfolio(
            live_trades, today_prices, rolling_vols, r, as_of_ts, alpha, beta
        )
        today_values = priced_today.set_index("trade_id")["position_value"]

        scenario_set = build_scenario_set(
            log_returns.iloc[:as_of_idx], lookback=lookback
        )
        scenario_prices = build_stressed_price_matrix(today_prices, scenario_set)

        pnl_vec = full_reprice_pnl(
            live_trades, today_prices, today_values, scenario_prices,
            rolling_vols, r, as_of_ts, alpha, beta,
        )
        var_es = compute_var_es(pnl_vec, confidence_levels)

        # --- Realised P&L ---
        # Use the same live_trades; trades expiring between as_of and next_date
        # are valued at intrinsic by price_portfolio (T→0 branch).
        next_prices = prices.loc[next_date]
        priced_next = price_portfolio(
            live_trades, next_prices, rolling_vols, r, next_ts, alpha, beta
        )
        next_values = priced_next.set_index("trade_id")["position_value"]
        realised_pnl = float(next_values.sum() - today_values.sum())

        row = {
            "date": as_of_ts,
            "realised_pnl": realised_pnl,
            "n_active_trades": len(live_trades),
        }
        row.update(var_es)

        for cl in confidence_levels:
            cl_key = str(int(round(cl * 100)))
            row[f"exception_{cl_key}"] = int(realised_pnl < -var_es[f"var_{cl_key}"])

        records.append(row)

    return pd.DataFrame(records).set_index("date")


# ---------------------------------------------------------------------------
# Statistical tests and summary report
# ---------------------------------------------------------------------------


def count_exceptions(backtest_df: pd.DataFrame, cl_key: str) -> int:
    """Count exceptions for a given confidence level key (e.g. '99')."""
    col = f"exception_{cl_key}"
    if col not in backtest_df.columns:
        raise KeyError(f"Column {col!r} not found in backtest DataFrame.")
    return int(backtest_df[col].sum())


def backtest_report(
    backtest_df: pd.DataFrame,
    confidence_levels: list[float] | None = None,
) -> dict:
    """
    Produce a full backtest report for each confidence level.

    Parameters
    ----------
    backtest_df       : output of rolling_backtest
    confidence_levels : defaults to [0.95, 0.99]

    Returns
    -------
    dict keyed by confidence level, each containing:
        n_obs, n_exceptions, kupiec, christoffersen, traffic_light
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    n_obs = len(backtest_df)
    report = {}

    for cl in confidence_levels:
        cl_key = str(int(round(cl * 100)))
        exc_col = f"exception_{cl_key}"

        if exc_col not in backtest_df.columns:
            continue

        exceptions_series = backtest_df[exc_col].values.astype(int)
        n_exc = int(exceptions_series.sum())

        kupiec = kupiec_test(n_exc, n_obs, cl)
        christoffersen = christoffersen_test(exceptions_series)
        tl = traffic_light(n_exc)

        report[cl] = {
            "confidence_level": cl,
            "n_obs": n_obs,
            "n_exceptions": n_exc,
            "expected_exceptions": round(n_obs * (1.0 - cl), 2),
            "exception_rate": round(n_exc / n_obs, 4) if n_obs > 0 else 0.0,
            "kupiec": kupiec,
            "christoffersen": christoffersen,
            "traffic_light": tl,
        }

    return report
