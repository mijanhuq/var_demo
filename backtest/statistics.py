"""
backtest/statistics.py
----------------------
Pure statistical functions for VaR backtesting.

No dependencies on data, portfolio, or var modules — operates only on
numpy arrays, making these functions independently testable.

Public API
----------
    count_exceptions            — number of VaR breaches in a P&L series
    exception_rate              — breach count / n_days
    traffic_light               — Basel III colour (green / yellow / red)
    KupiecResult                — Kupiec POF test result container
    kupiec_pof_test             — Proportion of Failures likelihood-ratio test
    ChristoffersenResult        — Christoffersen independence test result container
    christoffersen_independence_test — tests for exception clustering
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.stats


# ---------------------------------------------------------------------------
# Exception counting helpers
# ---------------------------------------------------------------------------

def count_exceptions(
    actual_pnl: np.ndarray,
    var_estimates: np.ndarray,
) -> int:
    """
    Count the number of VaR exceptions (breaches).

    A breach occurs on day i when actual_pnl[i] < -var_estimates[i],
    i.e. the realised loss exceeds the VaR forecast.

    Parameters
    ----------
    actual_pnl : np.ndarray
        Realised daily P&L, shape (n,).  Losses are negative.
    var_estimates : np.ndarray
        VaR forecasts, shape (n,).  Positive numbers (loss magnitudes).

    Returns
    -------
    int — number of days on which a breach occurred.
    """
    return int(np.sum(actual_pnl < -var_estimates))


def exception_rate(
    actual_pnl: np.ndarray,
    var_estimates: np.ndarray,
) -> float:
    """
    Fraction of days on which a VaR breach occurred.

    Returns count_exceptions / len(actual_pnl).
    """
    return count_exceptions(actual_pnl, var_estimates) / len(actual_pnl)


# ---------------------------------------------------------------------------
# Basel traffic light
# ---------------------------------------------------------------------------

def traffic_light(n_exceptions: int, n_days: int = 250) -> str:
    """
    Basel III supervisory traffic-light classification.

    Based on exceptions observed over a 250-trading-day (1-year) window
    at 99% confidence.  Thresholds:

        Green  : 0 – 4 exceptions  (model likely adequate)
        Yellow : 5 – 9 exceptions  (caution; may require investigation)
        Red    : ≥ 10 exceptions   (model likely inadequate)

    The ``n_days`` parameter is accepted for interface completeness but the
    Basel thresholds are defined for n_days = 250 and are not rescaled.

    Parameters
    ----------
    n_exceptions : int
        Number of observed VaR breaches.
    n_days : int
        Number of backtest days (informational only; default 250).

    Returns
    -------
    str — one of "green", "yellow", "red".
    """
    if n_exceptions < 5:
        return "green"
    if n_exceptions < 10:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# Kupiec Proportion of Failures (POF) test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KupiecResult:
    """
    Result of the Kupiec Proportion of Failures likelihood-ratio test.

    Attributes
    ----------
    statistic : float
        LR test statistic (non-negative).
    p_value : float
        p-value under chi²(1) null distribution.
    reject : bool
        True when p_value < significance (model is rejected).
    n_exceptions : int
        Observed number of exceptions.
    n_days : int
        Total number of backtest days.
    """
    statistic: float
    p_value: float
    reject: bool
    n_exceptions: int
    n_days: int


def kupiec_pof_test(
    n_exceptions: int,
    n_days: int,
    confidence: float,
    significance: float = 0.05,
) -> KupiecResult:
    """
    Kupiec (1995) Proportion of Failures likelihood-ratio test.

    Tests H0: the true exception probability equals (1 − confidence).

    LR = 2 · [ x·ln(p̂/p₀) + (n−x)·ln((1−p̂)/(1−p₀)) ]  ~ chi²(1)

    where:
        x    = n_exceptions
        n    = n_days
        p₀   = 1 − confidence   (expected exception rate)
        p̂   = x / n             (observed exception rate)

    Edge cases:
        x = 0  → LR = −2·n·ln(1−p₀)   (one-sided limit)
        x = n  → LR = −2·n·ln(p₀)     (one-sided limit)

    Parameters
    ----------
    n_exceptions : int
        Number of observed VaR breaches.
    n_days : int
        Total number of backtest days.
    confidence : float
        VaR confidence level (e.g. 0.99 for 99%).
    significance : float
        Test significance level (default 0.05).

    Returns
    -------
    KupiecResult
    """
    p0 = 1.0 - confidence
    x = n_exceptions
    n = n_days

    if x == 0:
        lr = -2.0 * n * math.log(1.0 - p0)
    elif x == n:
        lr = -2.0 * n * math.log(p0)
    else:
        p_hat = x / n
        lr = 2.0 * (
            x * math.log(p_hat / p0)
            + (n - x) * math.log((1.0 - p_hat) / (1.0 - p0))
        )

    p_value = float(scipy.stats.chi2.sf(lr, df=1))
    return KupiecResult(
        statistic=lr,
        p_value=p_value,
        reject=p_value < significance,
        n_exceptions=x,
        n_days=n,
    )


# ---------------------------------------------------------------------------
# Christoffersen Independence test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChristoffersenResult:
    """
    Result of the Christoffersen (1998) independence test.

    Tests H0: VaR exceptions are independent across consecutive days
    (i.e. today's breach does not predict tomorrow's breach).

    Attributes
    ----------
    statistic : float
        LR independence statistic (non-negative).
    p_value : float
        p-value under chi²(1) null distribution.
    reject : bool
        True when p_value < significance (clustering detected).
    n00, n01, n10, n11 : int
        Transition counts: n_ij = number of consecutive (i, j) pairs.
    """
    statistic: float
    p_value: float
    reject: bool
    n00: int
    n01: int
    n10: int
    n11: int


def christoffersen_independence_test(
    exceptions: np.ndarray,
    significance: float = 0.05,
) -> ChristoffersenResult:
    """
    Christoffersen (1998) independence likelihood-ratio test.

    Tests whether VaR exceptions cluster in time.  Under H0, the
    probability of an exception tomorrow is independent of whether one
    occurred today.

    Transition matrix counts from consecutive pairs:

        n_ij = #{t : I_{t-1} = i, I_t = j}

    Transition probabilities:

        π_01 = n01 / (n00 + n01)   (P(exception | no exception yesterday))
        π_11 = n11 / (n10 + n11)   (P(exception | exception yesterday))
        π    = (n01 + n11) / n     (unconditional exception rate)

    LR_ind = 2 · [n01·ln(π_01) + n00·ln(1−π_01)
                 + n11·ln(π_11) + n10·ln(1−π_11)
                 − (n01+n11)·ln(π) − (n00+n10)·ln(1−π)]

    Degenerate cases (π_01 = 0, π_11 = 0, π_11 = 1, or ≤ 1 total
    exception) cannot reject H0: LR = 0, p_value = 1, reject = False.

    Parameters
    ----------
    exceptions : np.ndarray
        1-D boolean (or 0/1 integer) array of length n_days.
        True / 1 indicates a VaR breach on that day.
    significance : float
        Test significance level (default 0.05).

    Returns
    -------
    ChristoffersenResult
    """
    exc = np.asarray(exceptions, dtype=int)
    n = len(exc)

    if n < 2:
        return ChristoffersenResult(
            statistic=0.0, p_value=1.0, reject=False,
            n00=0, n01=0, n10=0, n11=0,
        )

    # Count transitions
    prev = exc[:-1]
    curr = exc[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    total_exceptions = n01 + n11

    # Degenerate: fewer than 2 exceptions → no 1→x transitions possible
    if total_exceptions <= 1 or (n10 + n11) == 0 or (n00 + n01) == 0:
        return ChristoffersenResult(
            statistic=0.0, p_value=1.0, reject=False,
            n00=n00, n01=n01, n10=n10, n11=n11,
        )

    pi_01 = n01 / (n00 + n01)
    pi_11 = n11 / (n10 + n11)
    pi = total_exceptions / (n - 1)  # denominator = number of transitions

    # Degenerate: transition probability is 0 or 1 (log would be -inf)
    if pi_01 <= 0.0 or pi_01 >= 1.0 or pi_11 <= 0.0 or pi_11 >= 1.0:
        return ChristoffersenResult(
            statistic=0.0, p_value=1.0, reject=False,
            n00=n00, n01=n01, n10=n10, n11=n11,
        )
    if pi <= 0.0 or pi >= 1.0:
        return ChristoffersenResult(
            statistic=0.0, p_value=1.0, reject=False,
            n00=n00, n01=n01, n10=n10, n11=n11,
        )

    lr = 2.0 * (
        n01 * math.log(pi_01) + n00 * math.log(1.0 - pi_01)
        + n11 * math.log(pi_11) + n10 * math.log(1.0 - pi_11)
        - total_exceptions * math.log(pi)
        - (n - 1 - total_exceptions) * math.log(1.0 - pi)
    )

    # LR should be non-negative; clamp for numerical safety
    lr = max(0.0, lr)
    p_value = float(scipy.stats.chi2.sf(lr, df=1))
    return ChristoffersenResult(
        statistic=lr,
        p_value=p_value,
        reject=p_value < significance,
        n00=n00, n01=n01, n10=n10, n11=n11,
    )
