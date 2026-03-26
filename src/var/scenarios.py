"""
Scenario construction: log-return computation, lookback window slicing,
age-weighting, and scenario application to today's market data.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Log returns
# ---------------------------------------------------------------------------


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns: r_t = log(S_t / S_{t-1}).

    Parameters
    ----------
    prices : DataFrame of adjusted close prices (DatetimeIndex, columns = tickers)

    Returns
    -------
    DataFrame of log-returns (first row is NaN, dropped).
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


# ---------------------------------------------------------------------------
# Scenario set construction
# ---------------------------------------------------------------------------


def build_scenario_set(
    returns: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Slice the most recent `lookback` days from the returns DataFrame.

    Parameters
    ----------
    returns  : log-returns DataFrame (output of compute_log_returns)
    lookback : number of historical scenarios to include (default 252 = 1 year)

    Returns
    -------
    DataFrame of shape (lookback, n_tickers).

    Raises
    ------
    ValueError if fewer rows than lookback are available.
    """
    if len(returns) < lookback:
        raise ValueError(
            f"Only {len(returns)} return observations available; "
            f"need at least {lookback}."
        )
    return returns.iloc[-lookback:].copy()


def build_age_weights(n: int, lambda_: float = 0.97) -> np.ndarray:
    """
    Compute exponential decay age weights for Boudoukh-Richardson-Whitelaw
    age-weighted historical simulation.

    Weight for the most recent observation (index n-1) is proportional to 1,
    and weights decay geometrically back in time.

    Parameters
    ----------
    n       : number of scenarios (equal to lookback, typically 252)
    lambda_ : decay factor ∈ (0, 1). Higher = slower decay (more uniform).
              Typical values: 0.94–0.99.

    Returns
    -------
    1-D array of length n, summing to 1.
    Index 0 = oldest scenario (lowest weight); index n-1 = most recent.
    """
    if not (0 < lambda_ < 1):
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")
    # w_t ∝ λ^(n-1-t) for t = 0, ..., n-1
    exponents = np.arange(n - 1, -1, -1, dtype=float)  # [n-1, n-2, ..., 0]
    raw = lambda_**exponents
    return raw / raw.sum()


# ---------------------------------------------------------------------------
# Apply scenarios to today's market
# ---------------------------------------------------------------------------


def apply_scenario(
    today_prices: pd.Series,
    scenario_returns: pd.Series,
) -> pd.Series:
    """
    Apply a single scenario's log-returns to today's spot prices.

    S_stressed = S_today × exp(r_scenario)

    Parameters
    ----------
    today_prices     : current spot prices (Series, index = tickers)
    scenario_returns : log-returns for one historical day (Series, index = tickers)

    Returns
    -------
    Series of stressed spot prices (same index as today_prices).
    Only tickers present in both Series are stressed; others retain today's value.
    """
    common = today_prices.index.intersection(scenario_returns.index)
    stressed = today_prices.copy()
    stressed[common] = today_prices[common] * np.exp(scenario_returns[common])
    return stressed


def build_stressed_price_matrix(
    today_prices: pd.Series,
    scenario_set: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply every scenario to today's prices, producing a matrix of stressed prices.

    Parameters
    ----------
    today_prices : current spot prices (Series, index = tickers)
    scenario_set : log-return DataFrame (n_scenarios × n_tickers)

    Returns
    -------
    DataFrame of shape (n_scenarios × n_tickers) with stressed spot prices.
    """
    common = today_prices.index.intersection(scenario_set.columns)
    shocked = today_prices[common].values * np.exp(scenario_set[common].values)
    return pd.DataFrame(shocked, index=scenario_set.index, columns=common)
