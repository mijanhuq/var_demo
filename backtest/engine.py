"""
backtest/engine.py
------------------
Rolling 1-day VaR backtest engine.

For each day t in the backtest window the engine:
  1. Computes 1-day Historical VaR (99% and 95%) using the preceding
     ``lookback_window`` log-return scenarios.
  2. Measures the realised portfolio P&L between t and t+1 by full
     revaluation at the next day's market snapshot.
  3. Flags exception days where realised loss exceeds the VaR forecast.

Public API
----------
    BacktestResult  — container for rolling VaR, P&L, and exception arrays
    run_backtest    — execute the rolling backtest
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portfolio.portfolio import MarketSnapshot, Portfolio
from var.historical import (
    InsufficientDataError,
    _var_es_at_confidence,
    compute_pnl_scenarios,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Container for rolling VaR backtest output.

    Parameters
    ----------
    var_estimates_99 : np.ndarray
        1-day VaR at 99% confidence for each backtest day, shape (n,).
        Positive numbers (loss magnitudes).
    var_estimates_95 : np.ndarray
        1-day VaR at 95% confidence, shape (n,).
    actual_pnl : np.ndarray
        Realised 1-day portfolio P&L for each backtest day, shape (n,).
        Negative = loss.
    exceptions_99 : np.ndarray
        Boolean array, True where actual_pnl < -var_estimates_99.
    exceptions_95 : np.ndarray
        Boolean array, True where actual_pnl < -var_estimates_95.
    dates : list
        Trading dates corresponding to each backtest day (the "as of" date
        on which VaR was estimated, before the P&L day).
    n_backtest_days : int
        Number of days in the backtest window.
    """

    var_estimates_99: np.ndarray
    var_estimates_95: np.ndarray
    actual_pnl: np.ndarray
    exceptions_99: np.ndarray
    exceptions_95: np.ndarray
    dates: list
    n_backtest_days: int

    @property
    def n_exceptions_99(self) -> int:
        """Number of 99% VaR exceptions."""
        return int(np.sum(self.exceptions_99))

    @property
    def n_exceptions_95(self) -> int:
        """Number of 95% VaR exceptions."""
        return int(np.sum(self.exceptions_95))


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------

def run_backtest(
    portfolio: Portfolio,
    market_data,
    lookback_window: int = 252,
    backtest_window: int = 756,
) -> BacktestResult:
    """
    Execute a rolling 1-day VaR backtest via full revaluation.

    For each day t in ``range(t_start, n_total - 1)`` where
    ``t_start = n_total - backtest_window``:

        VaR(t)      = HVaR computed from log_returns[t-lookback:t]
        actual_pnl  = portfolio.value(snapshot_t1) - portfolio.value(snapshot_t)

    This yields ``backtest_window - 1`` comparable (VaR, P&L) pairs.

    Parameters
    ----------
    portfolio : Portfolio
        The option portfolio to backtest.  Positions (strikes, tenors) are
        held fixed throughout; only market prices and vols change.
    market_data : MarketData
        Full generated or loaded market dataset.
    lookback_window : int
        Historical scenario window for each VaR estimate.  Default 252.
    backtest_window : int
        Number of trading days in the backtest period.  Default 756 (~3 yrs).
        The actual number of (VaR, P&L) pairs is ``backtest_window - 1``.

    Returns
    -------
    BacktestResult

    Raises
    ------
    InsufficientDataError
        If ``len(market_data.log_returns) < lookback_window + backtest_window``.
    """
    n_total = len(market_data.log_returns)
    required = lookback_window + backtest_window
    if n_total < required:
        raise InsufficientDataError(
            f"Backtest requires {required} log-return rows "
            f"(lookback={lookback_window} + backtest={backtest_window}) "
            f"but market_data has only {n_total}."
        )

    t_start = n_total - backtest_window
    # We iterate t from t_start to n_total-2 inclusive so we can always
    # look at t+1 for the realised P&L.
    n_days = n_total - 1 - t_start  # = backtest_window - 1

    var_99_list: list[float] = []
    var_95_list: list[float] = []
    pnl_list: list[float] = []
    dates: list = []

    for t in range(t_start, n_total - 1):
        snapshot_t = MarketSnapshot.from_market_data(market_data, date_idx=t)
        scenario_returns = market_data.log_returns.iloc[t - lookback_window: t]

        pnl_scenarios = compute_pnl_scenarios(portfolio, snapshot_t, scenario_returns)
        pnl_sorted = np.sort(pnl_scenarios)

        var_99, _ = _var_es_at_confidence(pnl_sorted, 0.99)
        var_95, _ = _var_es_at_confidence(pnl_sorted, 0.95)

        snapshot_t1 = MarketSnapshot.from_market_data(market_data, date_idx=t + 1)
        actual_pnl = portfolio.value(snapshot_t1) - portfolio.value(snapshot_t)

        var_99_list.append(var_99)
        var_95_list.append(var_95)
        pnl_list.append(actual_pnl)
        dates.append(market_data.trading_dates[t])

    var_99_arr = np.array(var_99_list)
    var_95_arr = np.array(var_95_list)
    pnl_arr = np.array(pnl_list)

    return BacktestResult(
        var_estimates_99=var_99_arr,
        var_estimates_95=var_95_arr,
        actual_pnl=pnl_arr,
        exceptions_99=pnl_arr < -var_99_arr,
        exceptions_95=pnl_arr < -var_95_arr,
        dates=dates,
        n_backtest_days=n_days,
    )
