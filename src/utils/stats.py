"""
Statistical helpers: VAR/ES quantile computation, Kupiec and Christoffersen tests.
All VAR and ES values are returned as positive numbers (loss convention).
"""

import warnings
import numpy as np
from scipy.stats import chi2


SQRT_10 = np.sqrt(10)


def _cl_key(cl: float) -> str:
    """Return integer percentage string for a confidence level, e.g. 0.99 -> '99'."""
    return str(int(round(cl * 100)))


def compute_var_es(
    pnl: np.ndarray,
    confidence_levels: list[float],
) -> dict:
    """
    Compute historical-simulation VAR and ES for a P&L vector.

    Convention
    ----------
    - Losses are negative in pnl.
    - VAR and ES are returned as positive numbers.
    - VAR(α): the loss such that exactly floor(N*(1-α)) scenarios are worse.
    - ES(α):  mean loss of those floor(N*(1-α)) worst scenarios.

    Parameters
    ----------
    pnl : 1-D array of scenario P&L values
    confidence_levels : list of floats, e.g. [0.95, 0.99]

    Returns
    -------
    dict with keys "var_{cl}" and "es_{cl}" for each confidence level,
    e.g. {"var_95": 1200.0, "es_95": 1600.0, "var_99": 2300.0, "es_99": 2800.0}
    """
    pnl = np.asarray(pnl, dtype=float)
    n = len(pnl)
    sorted_pnl = np.sort(pnl)  # ascending: worst (most negative) first

    results: dict[str, float] = {}
    for cl in confidence_levels:
        n_tail = max(1, int(np.floor(n * (1.0 - cl))))
        key = _cl_key(cl)
        # VAR: the boundary loss (n_tail-th worst scenario, 0-indexed)
        results[f"var_{key}"] = float(-sorted_pnl[n_tail - 1])
        # ES: mean of the n_tail worst scenarios
        results[f"es_{key}"] = float(-np.mean(sorted_pnl[:n_tail]))

    return results


def scale_to_10d(results_1d: dict) -> dict:
    """
    Scale a 1-day VAR/ES results dict to 10-day using the sqrt(10) rule.
    Adds keys "var_{cl}_10d" and "es_{cl}_10d" alongside existing "var_{cl}" / "es_{cl}".

    Parameters
    ----------
    results_1d : output of compute_var_es

    Returns
    -------
    Extended dict with 10-day figures added.
    """
    out = dict(results_1d)
    for key, val in results_1d.items():
        out[f"{key}_10d"] = val * SQRT_10
    return out


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """
    Weighted quantile for age-weighted historical simulation.

    Parameters
    ----------
    values   : sorted ascending values (P&L scenarios)
    weights  : non-negative weights summing to 1, aligned to values
    quantile : e.g. 0.01 for the 1% quantile

    Returns
    -------
    Interpolated quantile value (NOT sign-flipped — caller converts to loss)
    """
    # Sort by value
    sort_idx = np.argsort(values)
    sorted_vals = values[sort_idx]
    sorted_wts = weights[sort_idx]
    cumulative = np.cumsum(sorted_wts)
    # Find the first index where cumulative weight reaches quantile
    idx = np.searchsorted(cumulative, quantile, side="left")
    idx = min(idx, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def compute_weighted_var_es(
    pnl: np.ndarray,
    weights: np.ndarray,
    confidence_levels: list[float],
) -> dict:
    """
    Age-weighted VAR and ES using a weighted P&L distribution.

    Parameters
    ----------
    pnl     : scenario P&L array (losses negative)
    weights : non-negative array summing to 1 (same length as pnl)
    confidence_levels : list of floats

    Returns
    -------
    Same dict format as compute_var_es.
    """
    pnl = np.asarray(pnl, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sort_idx = np.argsort(pnl)
    sorted_pnl = pnl[sort_idx]
    sorted_wts = weights[sort_idx]
    cumulative = np.cumsum(sorted_wts)

    results: dict[str, float] = {}
    for cl in confidence_levels:
        tail_prob = 1.0 - cl
        key = _cl_key(cl)
        # VAR: first scenario where cumulative weight >= tail_prob
        var_idx = np.searchsorted(cumulative, tail_prob, side="left")
        var_idx = min(var_idx, len(sorted_pnl) - 1)
        results[f"var_{key}"] = float(-sorted_pnl[var_idx])
        # ES: weighted mean of scenarios in the tail
        tail_mask = cumulative <= tail_prob
        if tail_mask.any():
            tail_pnl = sorted_pnl[tail_mask]
            tail_wts = sorted_wts[tail_mask]
            results[f"es_{key}"] = float(-np.average(tail_pnl, weights=tail_wts))
        else:
            # Edge case: single scenario in tail
            results[f"es_{key}"] = results[f"var_{key}"]

    return results


# ---------------------------------------------------------------------------
# Backtesting statistics
# ---------------------------------------------------------------------------


def kupiec_test(
    n_exceptions: int,
    n_obs: int,
    confidence_level: float,
) -> dict:
    """
    Kupiec Proportion of Failures (POF) test.

    Tests H0: true exception rate = 1 - confidence_level.

    Parameters
    ----------
    n_exceptions : number of VAR breaches observed
    n_obs        : total number of observations
    confidence_level : VAR confidence level (e.g. 0.99)

    Returns
    -------
    dict with keys: lr_stat, p_value, reject_h0 (at 5% significance)
    """
    p = 1.0 - confidence_level
    x = n_exceptions
    n = n_obs

    if x == 0:
        # Exact formula for x=0: MLE of p_hat is 0.
        # LR = 2·n·ln(1/(1-p)) — derived from the limit of the general formula.
        # Gives a valid chi²(1) p-value; correctly rejects an over-conservative
        # model where zero exceptions are observed but n*(1-CL) are expected.
        warnings.warn(
            "Kupiec: zero exceptions — model may be over-conservative.",
            UserWarning,
            stacklevel=2,
        )
        lr = 2.0 * n * np.log(1.0 / (1.0 - p))
        p_value = 1.0 - chi2.cdf(lr, df=1)
        return {
            "lr_stat": float(lr),
            "p_value": float(p_value),
            "reject_h0": bool(p_value < 0.05),
            "n_exceptions": x,
            "n_obs": n,
            "expected_exceptions": n * p,
        }
    elif x == n:
        warnings.warn(
            "Kupiec: all observations are exceptions — p-value unreliable.",
            UserWarning,
            stacklevel=2,
        )
        p_hat = 1.0 - 1e-10
    else:
        p_hat = x / n

    lr = 2.0 * (
        x * np.log(p_hat / p) + (n - x) * np.log((1.0 - p_hat) / (1.0 - p))
    )
    p_value = 1.0 - chi2.cdf(lr, df=1)

    return {
        "lr_stat": float(lr),
        "p_value": float(p_value),
        "reject_h0": bool(p_value < 0.05),
        "n_exceptions": n_exceptions,
        "n_obs": n_obs,
        "expected_exceptions": n * p,
    }


def christoffersen_test(exceptions: np.ndarray) -> dict:
    """
    Christoffersen conditional coverage / independence test.

    Tests H0: exception process is i.i.d. (no clustering).

    Parameters
    ----------
    exceptions : binary array (1 = exception on that day, 0 = no exception)

    Returns
    -------
    dict with keys: lr_stat, p_value, reject_h0
    """
    exc = np.asarray(exceptions, dtype=int)
    n = len(exc)

    # Transition counts
    n00 = np.sum((exc[:-1] == 0) & (exc[1:] == 0))
    n01 = np.sum((exc[:-1] == 0) & (exc[1:] == 1))
    n10 = np.sum((exc[:-1] == 1) & (exc[1:] == 0))
    n11 = np.sum((exc[:-1] == 1) & (exc[1:] == 1))

    # Transition probabilities
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    p_hat = (n01 + n11) / (n - 1)

    # Log-likelihood ratio
    def _ll(p_exc, n0, n1):
        if p_exc in (0.0, 1.0):
            return 0.0
        return n0 * np.log(1.0 - p_exc) + n1 * np.log(p_exc)

    ll_h0 = _ll(p_hat, n00 + n10, n01 + n11)
    ll_h1 = _ll(p01, n00, n01) + _ll(p11, n10, n11)
    lr = 2.0 * (ll_h1 - ll_h0)
    lr = max(lr, 0.0)  # numerical safety

    p_value = 1.0 - chi2.cdf(lr, df=1)

    return {
        "lr_stat": float(lr),
        "p_value": float(p_value),
        "reject_h0": bool(p_value < 0.05),
        "n00": int(n00), "n01": int(n01),
        "n10": int(n10), "n11": int(n11),
        "p01": float(p01), "p11": float(p11),
    }


def traffic_light(n_exceptions: int) -> str:
    """
    Basel traffic light classification for 250-day backtest.

    Green  : 0–4 exceptions
    Amber  : 5–9 exceptions
    Red    : 10+ exceptions
    """
    if n_exceptions <= 4:
        return "Green"
    elif n_exceptions <= 9:
        return "Amber"
    else:
        return "Red"
