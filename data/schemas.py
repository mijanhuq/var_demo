"""
data/schemas.py
---------------
Validation functions for all market data DataFrames.

Each function raises DataValidationError (a subclass of ValueError) on failure
so callers can catch specifically or broadly.  All checks are fast (vectorised)
and intended to run at data ingestion / generation boundaries, not inside
hot loops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import MONEYNESS_LABELS, TENOR_LABELS


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class DataValidationError(ValueError):
    """Raised when a market data structure fails a validation check."""


# ---------------------------------------------------------------------------
# Individual DataFrame validators
# ---------------------------------------------------------------------------

def validate_prices(df: pd.DataFrame, allow_zero: bool = False) -> None:
    """
    Validate an equity prices DataFrame.

    Expected shape: (n_days, n_assets)
    All values must be positive finite floats.
    """
    if df.empty:
        raise DataValidationError("Prices DataFrame is empty.")
    if not np.issubdtype(df.values.dtype, np.floating):
        raise DataValidationError(
            f"Prices dtype must be float, got {df.values.dtype}."
        )
    if df.isnull().values.any():
        raise DataValidationError("Prices DataFrame contains NaN values.")
    if np.isinf(df.values).any():
        raise DataValidationError("Prices DataFrame contains Inf values.")
    if not allow_zero and (df.values <= 0).any():
        raise DataValidationError(
            "Prices DataFrame contains non-positive values (expected strictly positive)."
        )


def validate_log_returns(df: pd.DataFrame) -> None:
    """
    Validate a log-returns DataFrame.

    Expected shape: (n_days, n_assets)
    Values must be finite floats.  A daily log-return beyond ±2.0 would
    imply a price move of >±738% in one day, which is treated as a data
    error rather than an extreme event.
    """
    if df.empty:
        raise DataValidationError("Log returns DataFrame is empty.")
    if not np.issubdtype(df.values.dtype, np.floating):
        raise DataValidationError(
            f"Log returns dtype must be float, got {df.values.dtype}."
        )
    if df.isnull().values.any():
        raise DataValidationError("Log returns DataFrame contains NaN values.")
    if np.isinf(df.values).any():
        raise DataValidationError("Log returns DataFrame contains Inf values.")
    if (np.abs(df.values) > 2.0).any():
        raise DataValidationError(
            "Log returns contain unrealistically large values (|r| > 2.0 per day)."
        )


def validate_atm_vols(df: pd.DataFrame) -> None:
    """
    Validate an ATM implied-vol time series DataFrame.

    Expected shape: (n_days, n_assets)
    Values must be in (0, 2.0] — i.e., 0%–200% annualised vol.
    """
    if df.empty:
        raise DataValidationError("ATM vols DataFrame is empty.")
    if df.isnull().values.any():
        raise DataValidationError("ATM vols DataFrame contains NaN values.")
    if (df.values <= 0).any():
        raise DataValidationError("ATM vols contain non-positive values.")
    if (df.values > 2.0).any():
        raise DataValidationError(
            "ATM vols contain values > 200%, which is outside the expected range."
        )


def validate_vol_surface(df: pd.DataFrame) -> None:
    """
    Validate the full implied vol surface DataFrame.

    Expected columns: 3-level MultiIndex (ticker, moneyness_label, tenor_label)
    Expected shape: (n_days, n_tickers * n_moneyness * n_tenors)
    All values must be positive and below 5.0 (500% vol).
    """
    if df.empty:
        raise DataValidationError("Vol surface DataFrame is empty.")
    if not isinstance(df.columns, pd.MultiIndex):
        raise DataValidationError(
            "Vol surface columns must be a 3-level MultiIndex (ticker, moneyness, tenor)."
        )
    if df.columns.nlevels != 3:
        raise DataValidationError(
            f"Vol surface MultiIndex must have 3 levels, got {df.columns.nlevels}."
        )
    if df.isnull().values.any():
        raise DataValidationError("Vol surface DataFrame contains NaN values.")
    if (df.values <= 0).any():
        raise DataValidationError("Vol surface contains non-positive implied vol values.")
    if (df.values > 5.0).any():
        raise DataValidationError(
            "Vol surface contains values > 500%, which is outside the expected range."
        )

    # Check that moneyness and tenor labels match the known grid
    moneyness_in_cols = df.columns.get_level_values("moneyness").unique().tolist()
    tenor_in_cols = df.columns.get_level_values("tenor").unique().tolist()
    unknown_m = set(moneyness_in_cols) - set(MONEYNESS_LABELS)
    unknown_t = set(tenor_in_cols) - set(TENOR_LABELS)
    if unknown_m:
        raise DataValidationError(
            f"Vol surface contains unknown moneyness labels: {unknown_m}"
        )
    if unknown_t:
        raise DataValidationError(
            f"Vol surface contains unknown tenor labels: {unknown_t}"
        )


# ---------------------------------------------------------------------------
# Composite validator
# ---------------------------------------------------------------------------

def validate_market_data(market_data: object) -> None:
    """
    Run all validation checks on a MarketData instance.

    Raises DataValidationError on the first failure found.
    """
    validate_prices(market_data.prices)
    validate_log_returns(market_data.log_returns)
    validate_atm_vols(market_data.atm_vols)
    validate_vol_surface(market_data.vol_surface)

    n_days = market_data.n_days
    n_assets = market_data.n_assets

    # Shape consistency across all DataFrames
    for name, df in [
        ("log_returns", market_data.log_returns),
        ("atm_vols", market_data.atm_vols),
    ]:
        if len(df) != n_days:
            raise DataValidationError(
                f"{name} has {len(df)} rows; expected {n_days} (matching prices)."
            )
        if df.shape[1] != n_assets:
            raise DataValidationError(
                f"{name} has {df.shape[1]} columns; expected {n_assets}."
            )

    if len(market_data.vol_surface) != n_days:
        raise DataValidationError(
            f"vol_surface has {len(market_data.vol_surface)} rows; expected {n_days}."
        )

    if len(market_data.trading_dates) != n_days:
        raise DataValidationError(
            f"trading_dates has {len(market_data.trading_dates)} entries; expected {n_days}."
        )

    # Crisis index bounds
    if market_data.crisis_start_idx >= market_data.crisis_end_idx:
        raise DataValidationError(
            "crisis_start_idx must be strictly less than crisis_end_idx."
        )
    if market_data.crisis_end_idx > n_days:
        raise DataValidationError(
            f"crisis_end_idx ({market_data.crisis_end_idx}) exceeds n_days ({n_days})."
        )

    # Dividend yields aligned with tickers
    if list(market_data.dividend_yields.index) != market_data.tickers:
        raise DataValidationError(
            "dividend_yields index does not match tickers list."
        )
