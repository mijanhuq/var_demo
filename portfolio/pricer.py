"""
portfolio/pricer.py
-------------------
Analytical Black-Scholes pricer for European options.

All public functions accept both scalar and array inputs (numpy-compatible)
via numpy broadcasting.  Scalar inputs return Python floats; array inputs
return ndarrays.

Conventions
-----------
- All rates (r, q, sigma) are annualised decimals  (e.g. 0.05 = 5 %)
- T is time to expiry in years
- Theta = ∂V/∂t  (calendar time derivative, NEGATIVE for long options)
  Divide by 252 to convert to per-trading-day theta
- Vega = ∂V/∂σ   per unit vol change (multiply by 0.01 for per-1% move)

Edge cases handled
------------------
- T ≤ 0   → intrinsic value  max(S-K, 0) for call / max(K-S, 0) for put
- σ ≤ 0   → forward intrinsic  max(S·e^{-qT} - K·e^{-rT}, 0) etc.
- Very large σ → price bounded by S·e^{-qT}  (upper bound for calls)
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import norm as _norm_dist

ArrayLike = Union[float, np.ndarray]

# Standard-normal CDF and PDF (vectorised via scipy)
_N = _norm_dist.cdf
_n = _norm_dist.pdf

# Threshold below which sigma is treated as zero-vol
_SIGMA_FLOOR = 1e-8
# Threshold below which T is treated as at-expiry
_T_FLOOR = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(*args: ArrayLike) -> tuple[np.ndarray, ...]:
    """Convert all arguments to float64 ndarrays."""
    return tuple(np.asarray(a, dtype=float) for a in args)


def _scalar_wrap(result: np.ndarray, was_scalar: bool) -> ArrayLike:
    """Return a Python float when all original inputs were scalars."""
    if was_scalar:
        return float(result.item())
    return result


def _was_scalar(*args: ArrayLike) -> bool:
    """True if every argument was a Python scalar (not an ndarray)."""
    return all(np.ndim(a) == 0 for a in args)


def _d1_d2(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute d1 and d2 for the Black-Scholes formula.

    Inputs must already be safe (T > 0, sigma > 0).
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def _intrinsic(S: np.ndarray, K: np.ndarray, is_call: bool) -> np.ndarray:
    """Intrinsic value at expiry (T = 0)."""
    if is_call:
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


def _forward_intrinsic(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, q: float, is_call: bool
) -> np.ndarray:
    """Forward intrinsic value when sigma = 0."""
    fwd = S * np.exp(-q * T) - K * np.exp(-r * T)
    if is_call:
        return np.maximum(fwd, 0.0)
    return np.maximum(-fwd, 0.0)


# ---------------------------------------------------------------------------
# Public pricing function
# ---------------------------------------------------------------------------

def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    option_type: str = "call",
) -> ArrayLike:
    """
    Black-Scholes price for a European vanilla option.

    Parameters
    ----------
    S           : spot price (scalar or array)
    K           : strike price (scalar or array)
    T           : time to expiry in years (scalar or array)
    r           : risk-free rate (annualised decimal)
    sigma       : implied volatility (annualised decimal, scalar or array)
    q           : continuous dividend yield (annualised decimal, default 0)
    option_type : 'call' or 'put'

    Returns
    -------
    Option price — float if all inputs scalar, ndarray otherwise.
    Always ≥ 0.
    """
    scalar = _was_scalar(S, K, T, sigma)
    S, K, T, sigma = _to_array(S, K, T, sigma)
    r, q = float(r), float(q)
    is_call = option_type.lower() == "call"

    # --- Boolean masks for edge cases ---
    at_expiry = T <= _T_FLOOR
    zero_vol = sigma < _SIGMA_FLOOR
    normal = ~at_expiry & ~zero_vol

    # --- Safe inputs for the main formula (avoid division by zero) ---
    safe_T = np.where(at_expiry, 1.0, T)
    safe_sigma = np.where(zero_vol, 0.01, sigma)

    # --- Full Black-Scholes formula (computed for all cells) ---
    d1, d2 = _d1_d2(S, K, safe_T, r, safe_sigma, q)
    disc_q = np.exp(-q * safe_T)
    disc_r = np.exp(-r * safe_T)

    if is_call:
        bs_val = S * disc_q * _N(d1) - K * disc_r * _N(d2)
    else:
        bs_val = K * disc_r * _N(-d2) - S * disc_q * _N(-d1)

    # --- Override with edge-case results where appropriate ---
    price = np.where(
        at_expiry,
        _intrinsic(S, K, is_call),
        np.where(
            zero_vol,
            _forward_intrinsic(S, K, T, r, q, is_call),
            bs_val,
        ),
    )
    price = np.maximum(price, 0.0)  # floor at 0 (numerical safety)

    return _scalar_wrap(price, scalar)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def bs_delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    option_type: str = "call",
) -> ArrayLike:
    """
    Black-Scholes delta = ∂V/∂S.

    Call:  e^{-qT} · N(d1)       ∈ (0, 1)
    Put:   e^{-qT} · (N(d1)−1)   ∈ (−1, 0)

    At expiry: 1.0 if ITM, 0.0 if OTM (ATM returns 0.5 by convention).
    Zero-vol: 1.0 if forward ITM, 0.0 otherwise.
    """
    scalar = _was_scalar(S, K, T, sigma)
    S, K, T, sigma = _to_array(S, K, T, sigma)
    r, q = float(r), float(q)
    is_call = option_type.lower() == "call"

    at_expiry = T <= _T_FLOOR
    zero_vol = sigma < _SIGMA_FLOOR
    normal = ~at_expiry & ~zero_vol

    safe_T = np.where(at_expiry, 1.0, T)
    safe_sigma = np.where(zero_vol, 0.01, sigma)
    d1, _ = _d1_d2(S, K, safe_T, r, safe_sigma, q)

    disc_q = np.exp(-q * safe_T)

    if is_call:
        bs_delta_val = disc_q * _N(d1)
        expiry_val = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5))
        zero_vol_val = np.where(
            S * np.exp(-q * T) > K * np.exp(-r * T), 1.0, 0.0
        )
    else:
        bs_delta_val = disc_q * (_N(d1) - 1.0)
        expiry_val = np.where(S < K, -1.0, np.where(S > K, 0.0, -0.5))
        zero_vol_val = np.where(
            S * np.exp(-q * T) < K * np.exp(-r * T), -1.0, 0.0
        )

    result = np.where(
        at_expiry,
        expiry_val,
        np.where(zero_vol, zero_vol_val, bs_delta_val),
    )
    return _scalar_wrap(result, scalar)


def bs_gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
) -> ArrayLike:
    """
    Black-Scholes gamma = ∂²V/∂S²  (identical for calls and puts).

    = e^{-qT} · N'(d1) / (S · σ · √T)

    Always ≥ 0.  Returns 0 at expiry or zero vol (formula is undefined/∞
    at those limits; the VaR engine treats near-expiry positions specially).
    """
    scalar = _was_scalar(S, K, T, sigma)
    S, K, T, sigma = _to_array(S, K, T, sigma)
    r, q = float(r), float(q)

    at_expiry = T <= _T_FLOOR
    zero_vol = sigma < _SIGMA_FLOOR
    use_formula = ~at_expiry & ~zero_vol

    safe_T = np.where(at_expiry, 1.0, T)
    safe_sigma = np.where(zero_vol, 0.01, sigma)
    safe_S = np.where(S <= 0, 1.0, S)  # guard against S=0 in denominator

    d1, _ = _d1_d2(safe_S, K, safe_T, r, safe_sigma, q)
    disc_q = np.exp(-q * safe_T)
    sqrt_T = np.sqrt(safe_T)

    gamma_val = disc_q * _n(d1) / (safe_S * safe_sigma * sqrt_T)
    result = np.where(use_formula, gamma_val, 0.0)
    result = np.maximum(result, 0.0)

    return _scalar_wrap(result, scalar)


def bs_vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
) -> ArrayLike:
    """
    Black-Scholes vega = ∂V/∂σ  (identical for calls and puts).

    = S · e^{-qT} · N'(d1) · √T

    Units: per unit vol change (e.g., vega=37.5 means $37.5 per +1.0 vol move,
    or $0.375 per +0.01 = 1% vol move).

    Always ≥ 0.
    """
    scalar = _was_scalar(S, K, T, sigma)
    S, K, T, sigma = _to_array(S, K, T, sigma)
    r, q = float(r), float(q)

    at_expiry = T <= _T_FLOOR
    zero_vol = sigma < _SIGMA_FLOOR
    use_formula = ~at_expiry & ~zero_vol

    safe_T = np.where(at_expiry, 1.0, T)
    safe_sigma = np.where(zero_vol, 0.01, sigma)
    safe_S = np.where(S <= 0, 0.0, S)

    d1, _ = _d1_d2(safe_S, K, safe_T, r, safe_sigma, q)
    disc_q = np.exp(-q * safe_T)
    sqrt_T = np.sqrt(safe_T)

    vega_val = safe_S * disc_q * _n(d1) * sqrt_T
    result = np.where(use_formula, vega_val, 0.0)
    result = np.maximum(result, 0.0)

    return _scalar_wrap(result, scalar)


def bs_theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    option_type: str = "call",
) -> ArrayLike:
    """
    Black-Scholes theta = ∂V/∂t  (calendar time derivative).

    Negative for long options (option loses value as time passes).
    Annualised — divide by 252 to get per-trading-day theta.

    Call:
        −[S·e^{-qT}·N'(d1)·σ / (2√T)]  − r·K·e^{-rT}·N(d2)
                                         + q·S·e^{-qT}·N(d1)

    Put:
        −[S·e^{-qT}·N'(d1)·σ / (2√T)]  + r·K·e^{-rT}·N(−d2)
                                         − q·S·e^{-qT}·N(−d1)

    Returns 0.0 at expiry (no more time value to decay).
    """
    scalar = _was_scalar(S, K, T, sigma)
    S, K, T, sigma = _to_array(S, K, T, sigma)
    r, q = float(r), float(q)
    is_call = option_type.lower() == "call"

    at_expiry = T <= _T_FLOOR
    zero_vol = sigma < _SIGMA_FLOOR
    use_formula = ~at_expiry & ~zero_vol

    safe_T = np.where(at_expiry, 1.0, T)
    safe_sigma = np.where(zero_vol, 0.01, sigma)

    d1, d2 = _d1_d2(S, K, safe_T, r, safe_sigma, q)
    disc_q = np.exp(-q * safe_T)
    disc_r = np.exp(-r * safe_T)
    sqrt_T = np.sqrt(safe_T)
    nd1 = _n(d1)

    # Common term (time decay)
    time_decay = -S * disc_q * nd1 * safe_sigma / (2.0 * sqrt_T)

    if is_call:
        theta_val = (
            time_decay
            - r * K * disc_r * _N(d2)
            + q * S * disc_q * _N(d1)
        )
    else:
        theta_val = (
            time_decay
            + r * K * disc_r * _N(-d2)
            - q * S * disc_q * _N(-d1)
        )

    result = np.where(use_formula, theta_val, 0.0)
    return _scalar_wrap(result, scalar)


# ---------------------------------------------------------------------------
# Convenience: all Greeks in one call
# ---------------------------------------------------------------------------

def bs_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    option_type: str = "call",
) -> dict[str, ArrayLike]:
    """
    Compute option price and all first-order Greeks in a single call.

    Internally computes d1 / d2 once and derives all quantities from them,
    making this more efficient than calling each function separately when
    all metrics are needed.

    Returns
    -------
    dict with keys:
        'price'  — option price
        'delta'  — ∂V/∂S
        'gamma'  — ∂²V/∂S²
        'vega'   — ∂V/∂σ (per unit vol)
        'theta'  — ∂V/∂t (annualised, negative for long options)
    """
    scalar = _was_scalar(S, K, T, sigma)
    S, K, T, sigma = _to_array(S, K, T, sigma)
    r, q = float(r), float(q)
    is_call = option_type.lower() == "call"

    at_expiry = T <= _T_FLOOR
    zero_vol = sigma < _SIGMA_FLOOR
    use_formula = ~at_expiry & ~zero_vol

    safe_T = np.where(at_expiry, 1.0, T)
    safe_sigma = np.where(zero_vol, 0.01, sigma)
    safe_S = np.where(S <= 0, 1.0, S)

    d1, d2 = _d1_d2(safe_S, K, safe_T, r, safe_sigma, q)
    Nd1 = _N(d1)
    Nd2 = _N(d2)
    nd1 = _n(d1)
    disc_q = np.exp(-q * safe_T)
    disc_r = np.exp(-r * safe_T)
    sqrt_T = np.sqrt(safe_T)

    # --- Price ---
    if is_call:
        bs_price_val = safe_S * disc_q * Nd1 - K * disc_r * Nd2
    else:
        bs_price_val = K * disc_r * _N(-d2) - safe_S * disc_q * _N(-d1)

    price = np.where(
        at_expiry,
        _intrinsic(safe_S, K, is_call),
        np.where(zero_vol, _forward_intrinsic(safe_S, K, T, r, q, is_call), bs_price_val),
    )
    price = np.maximum(price, 0.0)

    # --- Delta ---
    if is_call:
        bs_delta_val = disc_q * Nd1
        expiry_delta = np.where(safe_S > K, 1.0, np.where(safe_S < K, 0.0, 0.5))
    else:
        bs_delta_val = disc_q * (Nd1 - 1.0)
        expiry_delta = np.where(safe_S < K, -1.0, np.where(safe_S > K, 0.0, -0.5))

    delta = np.where(at_expiry, expiry_delta, np.where(zero_vol, 0.0, bs_delta_val))

    # --- Gamma (same for call and put) ---
    gamma_val = disc_q * nd1 / (safe_S * safe_sigma * sqrt_T)
    gamma = np.where(use_formula, gamma_val, 0.0)
    gamma = np.maximum(gamma, 0.0)

    # --- Vega (same for call and put) ---
    vega_val = safe_S * disc_q * nd1 * sqrt_T
    vega = np.where(use_formula, vega_val, 0.0)
    vega = np.maximum(vega, 0.0)

    # --- Theta ---
    time_decay = -safe_S * disc_q * nd1 * safe_sigma / (2.0 * sqrt_T)
    if is_call:
        theta_val = time_decay - r * K * disc_r * Nd2 + q * safe_S * disc_q * Nd1
    else:
        theta_val = time_decay + r * K * disc_r * _N(-d2) - q * safe_S * disc_q * _N(-d1)

    theta = np.where(use_formula, theta_val, 0.0)

    wrap = _scalar_wrap
    return {
        "price": wrap(price, scalar),
        "delta": wrap(delta, scalar),
        "gamma": wrap(gamma, scalar),
        "vega":  wrap(vega, scalar),
        "theta": wrap(theta, scalar),
    }


# ---------------------------------------------------------------------------
# TODO (Phase 1 Step 4): reprice_position
# ---------------------------------------------------------------------------
# def reprice_position(
#     position: OptionPosition,
#     S: ArrayLike,          # spot price(s) for this position's ticker
#     sigma: ArrayLike,      # implied vol(s) from vol surface for this leg
#     r: float,
#     q: float,
# ) -> tuple[ArrayLike, dict[str, ArrayLike]]:
#     """
#     Price a single OptionPosition and return (price, greeks).
#     Wraps bs_greeks with the position's fixed K, T, and option_type.
#     Used by the portfolio valuation engine and the VaR scenario pricer.
#     """
#     raise NotImplementedError("Implement in Phase 1 Step 4 alongside portfolio.py")
