"""
Portfolio revaluation under historical scenarios.

Two methods:
  1. Full revaluation — reprice every option under each stressed market state (BS).
  2. Greeks approximation — delta-gamma-vega approximation (faster, less accurate).
"""

import numpy as np
import pandas as pd

from utils.black_scholes import bs_price, bs_greeks
from data.vol_surface import surface_vol, get_atm_vol
from data.portfolio import _time_to_expiry


# ---------------------------------------------------------------------------
# Full revaluation
# ---------------------------------------------------------------------------


def _price_one(
    trade: pd.Series,
    spot: float,
    rolling_vols: dict,
    ticker: str,
    r: float,
    as_of: pd.Timestamp,
    scenario_date: pd.Timestamp,
    alpha: float,
    beta: float,
) -> float:
    """
    Price one trade under a stressed spot using the vol surface on `scenario_date`.
    The ATM vol is taken from the historical scenario date (rolling vol on that day),
    and the skew is applied relative to the stressed forward.
    """
    K = float(trade["strike"])
    T = _time_to_expiry(as_of, pd.Timestamp(trade["expiry"]))
    opt_type = trade["option_type"]

    if T <= 0:
        return max(spot - K, 0.0) if opt_type == "call" else max(K - spot, 0.0)

    # Use historical ATM vol level from the scenario date
    # This captures historical vol dynamics driving option P&L
    sigma = surface_vol(
        rolling_vols, ticker, scenario_date, K, spot, r, T, alpha=alpha, beta=beta
    )
    return bs_price(spot, K, T, r, sigma, option_type=opt_type)


def full_reprice_pnl(
    trades: pd.DataFrame,
    today_prices: pd.Series,
    today_values: pd.Series,
    scenario_prices: pd.DataFrame,
    rolling_vols: dict,
    r: float,
    as_of: pd.Timestamp,
    alpha: float = -0.15,
    beta: float = 0.05,
) -> np.ndarray:
    """
    Compute portfolio P&L for each scenario using full BS revaluation.

    Parameters
    ----------
    trades          : trade DataFrame (from portfolio.py)
    today_prices    : current spot prices (index = tickers)
    today_values    : current portfolio value per trade (Series, index = trade_id)
    scenario_prices : stressed spot prices (n_scenarios × n_tickers, from scenarios.py)
    rolling_vols    : output of compute_rolling_vol (used for scenario-date vol)
    r               : risk-free rate
    as_of           : valuation date (today)
    alpha, beta     : vol skew parameters

    Returns
    -------
    1-D ndarray of shape (n_scenarios,) — portfolio P&L per scenario.
    Losses are negative.
    """
    n_scenarios = len(scenario_prices)
    portfolio_pnl = np.zeros(n_scenarios)

    for _, trade in trades.iterrows():
        ticker = trade["underlying"]
        if ticker not in scenario_prices.columns:
            # TODO: handle index options whose ticker may not be in scenario_prices
            continue

        # Today's position value
        today_val = float(today_values[trade["trade_id"]])
        notional = float(trade["notional"])
        pos_sign = int(trade["position_sign"])

        for i, (scenario_date, row) in enumerate(scenario_prices.iterrows()):
            stressed_spot = float(row[ticker])
            unit_price = _price_one(
                trade, stressed_spot, rolling_vols, ticker, r, as_of,
                scenario_date, alpha, beta,
            )
            stressed_val = unit_price * notional * pos_sign
            portfolio_pnl[i] += stressed_val - today_val

    return portfolio_pnl


# ---------------------------------------------------------------------------
# Greeks approximation (delta-gamma)
# ---------------------------------------------------------------------------


def greeks_approx_pnl(
    trades: pd.DataFrame,
    today_prices: pd.Series,
    scenario_prices: pd.DataFrame,
    rolling_vols: dict,
    r: float,
    as_of: pd.Timestamp,
    alpha: float = -0.15,
    beta: float = 0.05,
) -> np.ndarray:
    """
    Approximate portfolio P&L using delta-gamma expansion.

    ΔV ≈ Δ · ΔS + ½Γ · (ΔS)²

    Vol changes (vega contribution) are not modelled here.
    This is faster but less accurate for large moves or vega-heavy books.

    TODO: add vega component once vol scenario evolution is incorporated.

    Parameters
    ----------
    trades          : trade DataFrame
    today_prices    : current spot prices
    scenario_prices : stressed spot prices (n_scenarios × n_tickers)
    rolling_vols    : for Greek computation at today's market
    r               : risk-free rate
    as_of           : valuation date

    Returns
    -------
    1-D ndarray of shape (n_scenarios,) — approx portfolio P&L.
    """
    from data.vol_surface import surface_vol

    n_scenarios = len(scenario_prices)
    portfolio_pnl = np.zeros(n_scenarios)

    for _, trade in trades.iterrows():
        ticker = trade["underlying"]
        if ticker not in scenario_prices.columns:
            continue

        S0 = float(today_prices[ticker])
        K = float(trade["strike"])
        T = _time_to_expiry(as_of, pd.Timestamp(trade["expiry"]))
        opt_type = trade["option_type"]
        notional = float(trade["notional"])
        pos_sign = int(trade["position_sign"])

        if T <= 0:
            continue

        sigma = surface_vol(rolling_vols, ticker, as_of, K, S0, r, T, alpha=alpha, beta=beta)
        g = bs_greeks(S0, K, T, r, sigma, option_type=opt_type)
        delta = g["delta"]
        gamma = g["gamma"]

        S_stressed = scenario_prices[ticker].values  # shape (n_scenarios,)
        dS = S_stressed - S0
        approx_pnl = (delta * dS + 0.5 * gamma * dS**2) * notional * pos_sign
        portfolio_pnl += approx_pnl

    return portfolio_pnl
