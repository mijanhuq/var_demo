"""
Historical Simulation VAR and ES.

Method A — Plain Historical Simulation (equal-weight scenarios).
Method B — Age-Weighted Historical Simulation (exponential decay).

Both methods produce VAR and ES at all configured confidence levels,
for 1-day and 10-day horizons (√10 scaling).
"""

import numpy as np
import pandas as pd

from data.portfolio import price_portfolio
from data.vol_surface import compute_rolling_vol
from var.scenarios import (
    compute_log_returns,
    build_scenario_set,
    build_age_weights,
    build_stressed_price_matrix,
)
from var.revaluation import full_reprice_pnl
from utils.stats import compute_var_es, compute_weighted_var_es, scale_to_10d


# ---------------------------------------------------------------------------
# Low-level: compute VAR/ES from a pre-built P&L vector
# ---------------------------------------------------------------------------


def plain_hs_var(
    pnl: np.ndarray,
    confidence_levels: list[float],
    scale_10d: bool = True,
) -> dict:
    """
    Compute plain historical simulation VAR and ES.

    Parameters
    ----------
    pnl               : 1-D P&L array (losses negative)
    confidence_levels : list of floats, e.g. [0.95, 0.99]
    scale_10d         : if True, also return √10-scaled 10-day figures

    Returns
    -------
    dict with keys var_{cl}, es_{cl} [, var_{cl}_10d, es_{cl}_10d]
    """
    results = compute_var_es(pnl, confidence_levels)
    if scale_10d:
        results = scale_to_10d(results)
    return results


def age_weighted_var(
    pnl: np.ndarray,
    lambda_: float = 0.97,
    confidence_levels: list[float] = (0.95, 0.99),
    scale_10d: bool = True,
) -> dict:
    """
    Compute age-weighted historical simulation VAR and ES.

    Parameters
    ----------
    pnl               : 1-D P&L array (losses negative)
    lambda_           : decay factor ∈ (0, 1)
    confidence_levels : list of floats
    scale_10d         : if True, also return 10-day figures

    Returns
    -------
    dict with keys var_{cl}, es_{cl} [, var_{cl}_10d, es_{cl}_10d]
    """
    weights = build_age_weights(len(pnl), lambda_=lambda_)
    results = compute_weighted_var_es(pnl, weights, list(confidence_levels))
    if scale_10d:
        results = scale_to_10d(results)
    return results


# ---------------------------------------------------------------------------
# High-level: full pipeline from prices + portfolio → VAR report
# ---------------------------------------------------------------------------


def compute_historical_var(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    r: float = 0.05,
    as_of: pd.Timestamp | None = None,
    lookback: int = 252,
    confidence_levels: list[float] | None = None,
    method: str = "plain",
    lambda_: float = 0.97,
    alpha: float = -0.15,
    beta: float = 0.05,
) -> dict:
    """
    End-to-end historical VAR computation pipeline.

    Parameters
    ----------
    trades            : options portfolio DataFrame (from portfolio.py)
    prices            : full historical price DataFrame (n_days × n_tickers)
    r                 : risk-free rate
    as_of             : valuation date; defaults to last date in prices
    lookback          : number of historical scenarios (default 252)
    confidence_levels : list of confidence levels; defaults to [0.95, 0.99]
    method            : 'plain' (Method A) or 'age_weighted' (Method B)
    lambda_           : age-weight decay factor (only used for method='age_weighted')
    alpha, beta       : vol skew parameters

    Returns
    -------
    dict with keys:
        method, pnl_vector, var_95, es_95, var_99, es_99,
        var_95_10d, es_95_10d, var_99_10d, es_99_10d
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    if as_of is None:
        as_of = prices.index[-1]
    as_of = pd.Timestamp(as_of)

    # 1. Build vol surface from full price history
    rolling_vols = compute_rolling_vol(prices)

    # 2. Price portfolio at today's market
    today_prices = prices.loc[as_of]
    priced = price_portfolio(trades, today_prices, rolling_vols, r, as_of, alpha, beta)
    today_values = priced.set_index("trade_id")["position_value"]

    # 3. Build scenario set (last 252 days of returns)
    log_returns = compute_log_returns(prices)
    scenario_set = build_scenario_set(log_returns, lookback=lookback)

    # 4. Build stressed price matrix
    scenario_prices = build_stressed_price_matrix(today_prices, scenario_set)

    # 5. Full revaluation → P&L vector
    pnl = full_reprice_pnl(
        trades, today_prices, today_values, scenario_prices,
        rolling_vols, r, as_of, alpha, beta,
    )

    # 6. Compute VAR and ES
    if method == "plain":
        results = plain_hs_var(pnl, confidence_levels, scale_10d=True)
        results["method"] = "plain_hs"
    elif method == "age_weighted":
        results = age_weighted_var(pnl, lambda_=lambda_,
                                   confidence_levels=confidence_levels, scale_10d=True)
        results["method"] = "age_weighted_hs"
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'plain' or 'age_weighted'.")

    results["pnl_vector"] = pnl
    results["as_of"] = as_of
    results["n_scenarios"] = len(pnl)

    return results
