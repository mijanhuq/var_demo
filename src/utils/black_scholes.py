"""
Black-Scholes pricer for European calls and puts.
All functions are vectorised over numpy arrays.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d1 and d2 for BS formula. T must be > 0."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    sigma: np.ndarray | float,
    option_type: str = "call",
) -> np.ndarray | float:
    """
    Black-Scholes price for a European option.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry in years (T <= 0 returns intrinsic value)
    r : continuously compounded risk-free rate
    sigma : implied volatility (annualised)
    option_type : 'call' or 'put'

    Returns
    -------
    Option price (same shape as inputs)
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    scalar_input = S.ndim == 0 and K.ndim == 0 and T.ndim == 0 and sigma.ndim == 0
    S, K, T, sigma = np.atleast_1d(S, K, T, sigma)
    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)
    price = np.zeros_like(S)

    expired = T <= 0
    live = ~expired

    if option_type == "call":
        price[expired] = np.maximum(S[expired] - K[expired], 0.0)
    else:
        price[expired] = np.maximum(K[expired] - S[expired], 0.0)

    if live.any():
        d1, d2 = _d1_d2(S[live], K[live], T[live], r, sigma[live])
        disc = np.exp(-r * T[live])
        if option_type == "call":
            price[live] = S[live] * norm.cdf(d1) - K[live] * disc * norm.cdf(d2)
        else:
            price[live] = K[live] * disc * norm.cdf(-d2) - S[live] * norm.cdf(-d1)

    return float(price[0]) if scalar_input else price


def bs_delta(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    sigma: np.ndarray | float,
    option_type: str = "call",
) -> np.ndarray | float:
    """First derivative of BS price w.r.t. spot (delta)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    scalar_input = S.ndim == 0 and K.ndim == 0 and T.ndim == 0 and sigma.ndim == 0
    S, K, T, sigma = np.atleast_1d(S, K, T, sigma)
    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)
    delta = np.zeros_like(S)

    expired = T <= 0
    live = ~expired

    if option_type == "call":
        delta[expired] = np.where(S[expired] > K[expired], 1.0, 0.0)
    else:
        delta[expired] = np.where(S[expired] < K[expired], -1.0, 0.0)

    if live.any():
        d1, _ = _d1_d2(S[live], K[live], T[live], r, sigma[live])
        if option_type == "call":
            delta[live] = norm.cdf(d1)
        else:
            delta[live] = norm.cdf(d1) - 1.0

    return float(delta[0]) if scalar_input else delta


def bs_gamma(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    sigma: np.ndarray | float,
) -> np.ndarray | float:
    """Second derivative of BS price w.r.t. spot (gamma). Same for calls and puts."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    scalar_input = S.ndim == 0 and K.ndim == 0 and T.ndim == 0 and sigma.ndim == 0
    S, K, T, sigma = np.atleast_1d(S, K, T, sigma)
    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)
    gamma = np.zeros_like(S)

    live = T > 0
    if live.any():
        d1, _ = _d1_d2(S[live], K[live], T[live], r, sigma[live])
        gamma[live] = norm.pdf(d1) / (S[live] * sigma[live] * np.sqrt(T[live]))

    return float(gamma[0]) if scalar_input else gamma


def bs_vega(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    sigma: np.ndarray | float,
) -> np.ndarray | float:
    """
    Derivative of BS price w.r.t. sigma (vega). Same for calls and puts.
    Returned per unit of vol (e.g. vega of 5 means +0.05 for +1pp vol move).
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    scalar_input = S.ndim == 0 and K.ndim == 0 and T.ndim == 0 and sigma.ndim == 0
    S, K, T, sigma = np.atleast_1d(S, K, T, sigma)
    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)
    vega = np.zeros_like(S)

    live = T > 0
    if live.any():
        d1, _ = _d1_d2(S[live], K[live], T[live], r, sigma[live])
        vega[live] = S[live] * norm.pdf(d1) * np.sqrt(T[live])

    return float(vega[0]) if scalar_input else vega


def bs_theta(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    sigma: np.ndarray | float,
    option_type: str = "call",
) -> np.ndarray | float:
    """
    Derivative of BS price w.r.t. time (theta), expressed per calendar day.
    Theta is negative for long options (time decay).
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    scalar_input = S.ndim == 0 and K.ndim == 0 and T.ndim == 0 and sigma.ndim == 0
    S, K, T, sigma = np.atleast_1d(S, K, T, sigma)
    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)
    theta = np.zeros_like(S)

    live = T > 0
    if live.any():
        d1, d2 = _d1_d2(S[live], K[live], T[live], r, sigma[live])
        disc = np.exp(-r * T[live])
        common = -(S[live] * norm.pdf(d1) * sigma[live]) / (2 * np.sqrt(T[live]))
        if option_type == "call":
            theta[live] = (common - r * K[live] * disc * norm.cdf(d2)) / 252
        else:
            theta[live] = (common + r * K[live] * disc * norm.cdf(-d2)) / 252

    return float(theta[0]) if scalar_input else theta


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """
    Compute all first-order Greeks for a single option.

    Returns
    -------
    dict with keys: delta, gamma, vega, theta
    """
    return {
        "delta": bs_delta(S, K, T, r, sigma, option_type),
        "gamma": bs_gamma(S, K, T, r, sigma),
        "vega": bs_vega(S, K, T, r, sigma),
        "theta": bs_theta(S, K, T, r, sigma, option_type),
    }
