"""
Options portfolio: trade format definition, validation, loading, and pricing.
All options are European (calls and puts).
"""

from __future__ import annotations

from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

from utils.black_scholes import bs_price, bs_greeks
from data.vol_surface import surface_vol, compute_rolling_vol

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "trade_id",
    "underlying",
    "option_type",
    "strike",
    "expiry",
    "notional",
    "position_sign",
    "premium",
    "trade_date",
]

VALID_OPTION_TYPES = {"call", "put"}
VALID_POSITION_SIGNS = {1, -1}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_trades(df: pd.DataFrame) -> None:
    """
    Validate a trade DataFrame against the required schema.
    Raises ValueError for any constraint violation.

    Parameters
    ----------
    df : DataFrame with columns matching REQUIRED_COLUMNS
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Trade DataFrame missing columns: {missing}")

    invalid_type = ~df["option_type"].isin(VALID_OPTION_TYPES)
    if invalid_type.any():
        bad = df.loc[invalid_type, "option_type"].unique().tolist()
        raise ValueError(f"Invalid option_type values: {bad}. Must be 'call' or 'put'.")

    invalid_sign = ~df["position_sign"].isin(VALID_POSITION_SIGNS)
    if invalid_sign.any():
        bad = df.loc[invalid_sign, "position_sign"].unique().tolist()
        raise ValueError(f"Invalid position_sign values: {bad}. Must be +1 or -1.")

    non_pos_notional = df["notional"] <= 0
    if non_pos_notional.any():
        raise ValueError("All notional values must be positive.")

    non_pos_strike = df["strike"] <= 0
    if non_pos_strike.any():
        raise ValueError("All strike values must be positive.")

    # Check expiry is after trade_date
    for _, row in df.iterrows():
        expiry = pd.Timestamp(row["expiry"])
        trade_dt = pd.Timestamp(row["trade_date"])
        if expiry <= trade_dt:
            raise ValueError(
                f"Trade {row['trade_id']}: expiry {expiry.date()} must be after "
                f"trade_date {trade_dt.date()}."
            )


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_portfolio(path: str | Path) -> pd.DataFrame:
    """
    Load a portfolio from the CSV trade format.

    Returns
    -------
    Validated DataFrame of trades.
    """
    df = pd.read_csv(path)
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["position_sign"] = df["position_sign"].astype(int)
    validate_trades(df)
    return df


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def _time_to_expiry(as_of: pd.Timestamp, expiry: pd.Timestamp) -> float:
    """Compute T in years (business-day fraction of 252)."""
    delta_days = (expiry - as_of).days
    return max(delta_days / 365.25, 0.0)


def price_trade(
    row: pd.Series,
    spot_prices: pd.Series,
    rolling_vols: dict,
    r: float,
    as_of: pd.Timestamp,
    alpha: float = -0.15,
    beta: float = 0.05,
) -> float:
    """
    Price a single trade using Black-Scholes + synthetic vol surface.

    Parameters
    ----------
    row         : one row from the trades DataFrame
    spot_prices : Series of spot prices keyed by ticker (for as_of date)
    rolling_vols: output of compute_rolling_vol
    r           : risk-free rate
    as_of       : valuation date

    Returns
    -------
    Option value per unit of notional (not multiplied by notional or position_sign).
    Zero if option has expired.
    """
    ticker = row["underlying"]

    if ticker not in spot_prices.index:
        raise KeyError(f"Ticker {ticker!r} not found in spot prices.")

    S = float(spot_prices[ticker])
    K = float(row["strike"])
    T = _time_to_expiry(as_of, pd.Timestamp(row["expiry"]))
    opt_type = row["option_type"]

    if T <= 0:
        # Expired option
        return max(S - K, 0.0) if opt_type == "call" else max(K - S, 0.0)

    sigma = surface_vol(rolling_vols, ticker, as_of, K, S, r, T, alpha=alpha, beta=beta)
    return bs_price(S, K, T, r, sigma, option_type=opt_type)


def price_portfolio(
    trades: pd.DataFrame,
    spot_prices: pd.Series,
    rolling_vols: dict,
    r: float,
    as_of: pd.Timestamp,
    alpha: float = -0.15,
    beta: float = 0.05,
) -> pd.DataFrame:
    """
    Price all trades in the portfolio.

    Returns
    -------
    DataFrame with columns:
        trade_id, unit_price, position_value
        (position_value = unit_price × notional × position_sign)
    """
    rows = []
    for _, trade in trades.iterrows():
        unit_price = price_trade(trade, spot_prices, rolling_vols, r, as_of, alpha, beta)
        position_value = unit_price * trade["notional"] * trade["position_sign"]
        rows.append({
            "trade_id": trade["trade_id"],
            "unit_price": unit_price,
            "position_value": position_value,
        })

    return pd.DataFrame(rows)


def portfolio_greeks(
    trades: pd.DataFrame,
    spot_prices: pd.Series,
    rolling_vols: dict,
    r: float,
    as_of: pd.Timestamp,
    alpha: float = -0.15,
    beta: float = 0.05,
) -> pd.DataFrame:
    """
    Compute BS Greeks for each trade.

    Returns
    -------
    DataFrame with columns: trade_id, delta, gamma, vega, theta
    (each in unit terms, not notional-scaled).
    """
    rows = []
    for _, trade in trades.iterrows():
        ticker = trade["underlying"]
        S = float(spot_prices[ticker])
        K = float(trade["strike"])
        T = _time_to_expiry(as_of, pd.Timestamp(trade["expiry"]))
        opt_type = trade["option_type"]

        if T <= 0:
            greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        else:
            sigma = surface_vol(rolling_vols, ticker, as_of, K, S, r, T, alpha=alpha, beta=beta)
            greeks = bs_greeks(S, K, T, r, sigma, option_type=opt_type)

        greeks["trade_id"] = trade["trade_id"]
        rows.append(greeks)

    return pd.DataFrame(rows)[["trade_id", "delta", "gamma", "vega", "theta"]]


# ---------------------------------------------------------------------------
# Synthetic portfolio generator
# ---------------------------------------------------------------------------


_DEFAULT_VOL = 0.25  # flat fallback vol when no rolling_vols supplied


def generate_portfolio(
    spot_prices: pd.Series,
    n_trades: int = 50,
    r: float = 0.05,
    as_of: pd.Timestamp | None = None,
    seed: int = 42,
    rolling_vols: dict | None = None,
    alpha: float = -0.15,
    beta: float = 0.05,
) -> pd.DataFrame:
    """
    Generate a synthetic portfolio of European options.

    Parameters
    ----------
    spot_prices  : current spot prices (Series, index = tickers)
    n_trades     : total number of trades to generate
    r            : risk-free rate (for ATM forward calculation)
    as_of        : valuation date (defaults to today)
    seed         : random seed for reproducibility
    rolling_vols : output of compute_rolling_vol; when supplied the premium is
                   the true Black-Scholes price using the vol surface on as_of.
                   When None, a flat 25% vol is used as the ATM proxy.
    alpha        : skew coefficient passed to surface_vol (ignored if rolling_vols is None)
    beta         : smile curvature coefficient (ignored if rolling_vols is None)

    Returns
    -------
    Validated trades DataFrame.
    """
    rng = np.random.default_rng(seed)

    if as_of is None:
        as_of = pd.Timestamp(date.today())

    tickers = spot_prices.index.tolist()

    # Expiry dates: roughly 3, 6, 9, 12 months out
    expiry_offsets_days = [63, 126, 189, 252]

    rows = []
    for i in range(n_trades):
        ticker = rng.choice(tickers)
        S = float(spot_prices[ticker])
        opt_type = rng.choice(["call", "put"])
        position_sign = int(rng.choice([1, -1]))

        # Strike: within ±20% of spot (moneyness range typical for liquid options)
        moneyness = rng.uniform(0.80, 1.20)
        K = round(S * moneyness, 2)

        expiry_days = int(rng.choice(expiry_offsets_days))
        expiry = as_of + pd.offsets.BDay(expiry_days)
        T = _time_to_expiry(as_of, expiry)

        notional = int(rng.choice([100, 200, 500]))

        # Compute fair-value premium via Black-Scholes
        if rolling_vols is not None and T > 0:
            try:
                sigma = surface_vol(rolling_vols, ticker, as_of, K, S, r, T,
                                    alpha=alpha, beta=beta)
            except (ValueError, KeyError):
                sigma = _DEFAULT_VOL
        else:
            sigma = _DEFAULT_VOL
        premium = round(float(bs_price(S, K, T, r, sigma, option_type=opt_type)), 4)

        rows.append({
            "trade_id":      f"OPT-{i+1:03d}",
            "underlying":    ticker,
            "option_type":   opt_type,
            "strike":        K,
            "expiry":        expiry.date().isoformat(),
            "notional":      notional,
            "position_sign": position_sign,
            "premium":       premium,
            "trade_date":    as_of.date().isoformat(),
        })

    df = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    validate_trades(df)
    return df
