"""
Method C — Filtered Historical Simulation (GARCH-filtered).

Standardises historical returns by their conditional GARCH(1,1) volatility,
then rescales to today's conditional vol regime before applying as scenarios.
"""

import warnings

import numpy as np
import pandas as pd

from data.portfolio import price_portfolio
from data.vol_surface import compute_rolling_vol
from var.scenarios import (
    compute_log_returns,
    build_scenario_set,
    build_stressed_price_matrix,
)
from var.revaluation import full_reprice_pnl
from utils.stats import compute_var_es, scale_to_10d


def fit_garch(returns: np.ndarray) -> dict:
    """
    Fit a GARCH(1,1) model to a return series.

    Parameters
    ----------
    returns : 1-D array of log-returns

    Returns
    -------
    dict with keys:
        omega, alpha, beta   — GARCH(1,1) parameters
        conditional_vol      — 1-D array of conditional volatilities (same length as returns)
        standardised_resid   — 1-D array of standardised residuals ε_t = r_t / σ_t

    Raises
    ------
    ImportError if the `arch` library is not installed.
    """
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError(
            "The 'arch' package is required for GARCH-filtered VAR. "
            "Install it with: pip install arch"
        ) from exc

    # Rescale returns to percentage for numerical stability in arch
    returns_pct = returns * 100.0
    am = arch_model(returns_pct, vol="GARCH", p=1, q=1, dist="normal", rescale=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = am.fit(disp="off", show_warning=False)

    params = result.params
    omega = float(params.get("omega", params.iloc[1]))
    alpha = float(params.get("alpha[1]", params.iloc[2]))
    beta  = float(params.get("beta[1]", params.iloc[3]))

    # Extract conditional vols (in percentage terms) → convert back
    cond_vol_pct = result.conditional_volatility
    cond_vol = cond_vol_pct.values / 100.0
    std_resid = returns / cond_vol

    return {
        "omega":             omega,
        "alpha":             alpha,
        "beta":              beta,
        "conditional_vol":   cond_vol,
        "standardised_resid": std_resid,
        "arch_result":       result,
        # TODO: add AIC/BIC, log-likelihood for model diagnostics
    }


def filter_returns(
    historical_returns: np.ndarray,
    garch_fit: dict,
    sigma_today: float,
) -> np.ndarray:
    """
    Produce filtered (regime-adjusted) scenario returns.

    ε_t = r_t / σ_t           # standardised innovations from GARCH fit
    scenario_return_t = ε_t × σ_today   # rescaled to today's vol

    Parameters
    ----------
    historical_returns : 1-D log-return array (length = lookback)
    garch_fit          : output of fit_garch
    sigma_today        : today's conditional vol estimate (from GARCH forecast)

    Returns
    -------
    1-D array of filtered scenario returns.
    """
    std_resid = garch_fit["standardised_resid"][-len(historical_returns):]
    return std_resid * sigma_today


def compute_filtered_var(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    r: float = 0.05,
    as_of: pd.Timestamp | None = None,
    lookback: int = 252,
    confidence_levels: list[float] | None = None,
    alpha: float = -0.15,
    beta: float = 0.05,
) -> dict:
    """
    End-to-end GARCH-filtered historical VAR pipeline.

    For each equity/index in the scenario set, returns are standardised by their
    GARCH(1,1) conditional vol and rescaled to today's vol before being applied
    as stressed scenarios.

    Parameters
    ----------
    trades            : options portfolio DataFrame
    prices            : full historical price DataFrame (n_days × n_tickers)
    r                 : risk-free rate
    as_of             : valuation date; defaults to last price date
    lookback          : number of scenarios
    confidence_levels : defaults to [0.95, 0.99]
    alpha, beta       : vol skew parameters

    Returns
    -------
    dict with method, pnl_vector, var/es figures at 1d and 10d horizons.
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

    log_returns = compute_log_returns(prices)
    # Use more history for GARCH estimation (full available history)
    full_returns_df = log_returns
    scenario_returns_df = build_scenario_set(log_returns, lookback=lookback)

    # Filter returns per ticker
    filtered_scenarios = pd.DataFrame(
        index=scenario_returns_df.index, columns=scenario_returns_df.columns, dtype=float
    )

    for ticker in scenario_returns_df.columns:
        col_returns = full_returns_df[ticker].dropna().values
        if len(col_returns) < 50:
            # Not enough data to fit GARCH — fall back to unfiltered
            # TODO: log a warning per ticker
            filtered_scenarios[ticker] = scenario_returns_df[ticker]
            continue

        try:
            garch_fit = fit_garch(col_returns)
        except Exception:
            # TODO: log exception details; fall back gracefully
            filtered_scenarios[ticker] = scenario_returns_df[ticker]
            continue

        # Today's conditional vol: last value from GARCH conditional vol series
        sigma_today = float(garch_fit["conditional_vol"][-1])
        if sigma_today <= 0:
            filtered_scenarios[ticker] = scenario_returns_df[ticker]
            continue

        # Historical returns for the lookback window
        hist_returns = scenario_returns_df[ticker].values
        filtered_scenarios[ticker] = filter_returns(hist_returns, garch_fit, sigma_today)

    # Stressed prices using filtered returns
    scenario_prices = build_stressed_price_matrix(today_prices, filtered_scenarios)

    pnl = full_reprice_pnl(
        trades, today_prices, today_values, scenario_prices,
        rolling_vols, r, as_of, alpha, beta,
    )

    results = compute_var_es(pnl, confidence_levels)
    results = scale_to_10d(results)
    results["method"] = "filtered_hs"
    results["pnl_vector"] = pnl
    results["as_of"] = as_of
    results["n_scenarios"] = len(pnl)

    return results
