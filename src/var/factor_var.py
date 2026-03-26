"""
Method D — Factor-Based Historical VAR.

Decomposes equity returns into systematic index factors via OLS.
Portfolio P&L is split into:
  - Systematic component: delta-gamma from factor shocks
  - Idiosyncratic component: delta-gamma from OLS residuals
Both components are combined scenario-by-scenario to preserve joint distribution.
"""

import numpy as np
import pandas as pd

from data.portfolio import price_portfolio, portfolio_greeks
from data.vol_surface import compute_rolling_vol
from var.scenarios import compute_log_returns, build_scenario_set
from utils.stats import compute_var_es, scale_to_10d


# ---------------------------------------------------------------------------
# Factor model fitting
# ---------------------------------------------------------------------------


def fit_factor_model(
    equity_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit OLS factor models for each equity against the chosen index factors.

    r_i,t = α_i + Σ_k β_i,k · F_k,t + ε_i,t

    Parameters
    ----------
    equity_returns : DataFrame (n_days × n_equities) of equity log-returns
    factor_returns : DataFrame (n_days × n_factors) of index log-returns
                     (must share the same DatetimeIndex)

    Returns
    -------
    DataFrame with index = equity tickers, columns = ['alpha'] + factor tickers + ['r_squared'].
    """
    # Align on common dates
    common_idx = equity_returns.index.intersection(factor_returns.index)
    eq = equity_returns.loc[common_idx]
    fac = factor_returns.loc[common_idx]

    X = np.column_stack([np.ones(len(fac)), fac.values])  # (n, 1 + k)
    results = []

    for ticker in eq.columns:
        y = eq[ticker].values
        # OLS: β = (X'X)^{-1} X'y
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            # TODO: add fallback (ridge regression) if X is near-singular
            coeffs = np.zeros(X.shape[1])

        alpha = coeffs[0]
        betas = coeffs[1:]
        y_hat = X @ coeffs
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        row = {"ticker": ticker, "alpha": alpha, "r_squared": r2}
        for i, factor_name in enumerate(fac.columns):
            row[factor_name] = betas[i]
        results.append(row)

    return pd.DataFrame(results).set_index("ticker")


# ---------------------------------------------------------------------------
# Factor scenario P&L
# ---------------------------------------------------------------------------


def compute_factor_pnl(
    trades: pd.DataFrame,
    today_prices: pd.Series,
    factor_betas: pd.DataFrame,
    factor_scenarios: pd.DataFrame,
    today_greeks: pd.DataFrame,
) -> np.ndarray:
    """
    Estimate portfolio P&L under factor scenarios using delta-gamma approximation.

    For each scenario:
      ΔS_i = S_i · (Σ_k β_i,k · ΔF_k)          # spot change driven by factors
      ΔV_trade ≈ (Δ · ΔS + ½Γ · ΔS²) · notional · pos_sign

    Parameters
    ----------
    trades           : trade DataFrame
    today_prices     : current spot prices (Series, index = tickers)
    factor_betas     : output of fit_factor_model (index = equity tickers)
    factor_scenarios : (n_scenarios × n_factors) factor log-return scenarios
    today_greeks     : output of portfolio_greeks (index = trade_id)

    Returns
    -------
    1-D ndarray of shape (n_scenarios,).
    """
    n_scenarios = len(factor_scenarios)
    portfolio_pnl = np.zeros(n_scenarios)

    greeks_by_id = today_greeks.set_index("trade_id")

    for _, trade in trades.iterrows():
        ticker = trade["underlying"]
        trade_id = trade["trade_id"]

        if ticker not in factor_betas.index:
            # TODO: treat index underlyings as single-factor (beta=1 to own index)
            continue
        if trade_id not in greeks_by_id.index:
            continue

        S0 = float(today_prices[ticker])
        betas_row = factor_betas.loc[ticker]

        # Common factor columns
        common_factors = [
            f for f in factor_scenarios.columns
            if f in betas_row.index and f not in ("alpha", "r_squared")
        ]
        if not common_factors:
            continue

        beta_vec = betas_row[common_factors].values        # shape (k,)
        fac_returns = factor_scenarios[common_factors].values  # shape (n, k)

        # Log-return for equity implied by factor shocks
        log_ret_equity = fac_returns @ beta_vec            # shape (n,)
        dS = S0 * (np.exp(log_ret_equity) - 1.0)          # approximate spot change

        delta = float(greeks_by_id.loc[trade_id, "delta"])
        gamma = float(greeks_by_id.loc[trade_id, "gamma"])
        notional = float(trade["notional"])
        pos_sign = int(trade["position_sign"])

        approx_pnl = (delta * dS + 0.5 * gamma * dS**2) * notional * pos_sign
        portfolio_pnl += approx_pnl

    return portfolio_pnl


# ---------------------------------------------------------------------------
# Idiosyncratic component
# ---------------------------------------------------------------------------


def compute_residuals(
    equity_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_betas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute OLS residuals for each equity ticker.

    ε_i,t = r_i,t − (α_i + Σ_k β_i,k · F_k,t)

    Parameters
    ----------
    equity_returns : DataFrame of equity log-returns
    factor_returns : DataFrame of factor log-returns (same index)
    factor_betas   : output of fit_factor_model

    Returns
    -------
    DataFrame of residuals, same shape as equity_returns on the common index.
    """
    factor_cols = [c for c in factor_betas.columns if c not in ("alpha", "r_squared")]

    common_idx = equity_returns.index.intersection(factor_returns.index)
    eq = equity_returns.loc[common_idx]
    fac = factor_returns.loc[common_idx]

    residual_dict: dict[str, pd.Series] = {}
    for ticker in factor_betas.index:
        if ticker not in eq.columns:
            continue
        alpha_i = float(factor_betas.loc[ticker, "alpha"])
        avail = [f for f in factor_cols if f in fac.columns]
        beta_vec = factor_betas.loc[ticker, avail].values.astype(float)
        fitted = alpha_i + fac[avail].values @ beta_vec
        residual_dict[ticker] = pd.Series(
            eq[ticker].values - fitted, index=common_idx
        )

    return pd.DataFrame(residual_dict)


def compute_idio_pnl(
    trades: pd.DataFrame,
    today_prices: pd.Series,
    residual_scenarios: pd.DataFrame,
    today_greeks: pd.DataFrame,
) -> np.ndarray:
    """
    Estimate idiosyncratic portfolio P&L via delta-gamma on OLS residuals.

    For each scenario t:
      ΔS_i_idio = S_i · (exp(ε_i,t) − 1)
      ΔV_idio   ≈ (Δ · ΔS_idio + ½Γ · ΔS_idio²) · notional · pos_sign

    Parameters
    ----------
    trades             : trade DataFrame
    today_prices       : current spot prices (Series, index = tickers)
    residual_scenarios : OLS residuals DataFrame (n_scenarios × n_equities)
    today_greeks       : output of portfolio_greeks

    Returns
    -------
    1-D ndarray of shape (n_scenarios,).
    """
    n_scenarios = len(residual_scenarios)
    portfolio_pnl = np.zeros(n_scenarios)
    greeks_by_id = today_greeks.set_index("trade_id")

    for _, trade in trades.iterrows():
        ticker = trade["underlying"]
        trade_id = trade["trade_id"]

        if ticker not in residual_scenarios.columns:
            continue
        if trade_id not in greeks_by_id.index:
            continue

        S0 = float(today_prices[ticker])
        eps = residual_scenarios[ticker].values          # shape (n,)
        dS = S0 * (np.exp(eps) - 1.0)

        delta = float(greeks_by_id.loc[trade_id, "delta"])
        gamma = float(greeks_by_id.loc[trade_id, "gamma"])
        notional = float(trade["notional"])
        pos_sign = int(trade["position_sign"])

        portfolio_pnl += (delta * dS + 0.5 * gamma * dS**2) * notional * pos_sign

    return portfolio_pnl


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


def compute_factor_var(
    trades: pd.DataFrame,
    equity_prices: pd.DataFrame,
    index_prices: pd.DataFrame,
    r: float = 0.05,
    as_of: pd.Timestamp | None = None,
    lookback: int = 252,
    confidence_levels: list[float] | None = None,
    alpha_skew: float = -0.15,
    beta_skew: float = 0.05,
) -> dict:
    """
    End-to-end factor VAR pipeline.

    Parameters
    ----------
    trades         : options portfolio DataFrame
    equity_prices  : historical equity price DataFrame (n_days × n_equities)
    index_prices   : historical index price DataFrame (n_days × n_factors)
    r              : risk-free rate
    as_of          : valuation date; defaults to last equity price date
    lookback       : historical lookback window (scenarios)
    confidence_levels : defaults to [0.95, 0.99]
    alpha_skew, beta_skew : vol skew parameters

    Returns
    -------
    dict with method, pnl_vector, factor_betas, var/es figures.
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    if as_of is None:
        as_of = equity_prices.index[-1]
    as_of = pd.Timestamp(as_of)

    # 1. Build vol surface and price portfolio today
    rolling_vols = compute_rolling_vol(equity_prices)
    today_prices_eq = equity_prices.loc[as_of]
    today_prices_idx = index_prices.loc[as_of] if as_of in index_prices.index else index_prices.iloc[-1]
    today_prices = pd.concat([today_prices_eq, today_prices_idx])

    priced = price_portfolio(
        trades, today_prices_eq, rolling_vols, r, as_of, alpha_skew, beta_skew
    )
    # Compute Greeks for delta-gamma approximation
    greeks = portfolio_greeks(
        trades, today_prices_eq, rolling_vols, r, as_of, alpha_skew, beta_skew
    )

    # 2. Compute log-returns
    eq_returns = compute_log_returns(equity_prices)
    idx_returns = compute_log_returns(index_prices)

    # 3. Fit factor model
    factor_betas = fit_factor_model(eq_returns, idx_returns)

    # 4. Build factor scenario set (last 252 days of factor returns)
    factor_scenarios = build_scenario_set(idx_returns, lookback=lookback)

    # 5a. Compute factor-driven (systematic) P&L
    pnl_factor = compute_factor_pnl(
        trades, today_prices_eq, factor_betas, factor_scenarios, greeks
    )

    # 5b. Compute idiosyncratic P&L from OLS residuals
    eq_residuals = compute_residuals(eq_returns, idx_returns, factor_betas)
    # Align residual scenarios to the same date range as factor_scenarios
    common_dates = eq_residuals.index.intersection(factor_scenarios.index)
    if len(common_dates) > 0:
        idio_scenarios = eq_residuals.loc[common_dates]
        pnl_idio = compute_idio_pnl(trades, today_prices_eq, idio_scenarios, greeks)
        # Align lengths (should be equal, but guard for edge cases)
        n = min(len(pnl_factor), len(pnl_idio))
        pnl = pnl_factor[:n] + pnl_idio[:n]
    else:
        pnl = pnl_factor

    # 6. VAR and ES on combined (systematic + idiosyncratic) P&L
    results = compute_var_es(pnl, confidence_levels)
    results = scale_to_10d(results)
    results["method"] = "factor_var"
    results["pnl_vector"] = pnl
    results["factor_betas"] = factor_betas
    results["as_of"] = as_of
    results["n_scenarios"] = len(pnl)

    return results
