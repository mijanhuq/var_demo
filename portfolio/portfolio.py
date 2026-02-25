"""
portfolio/portfolio.py
----------------------
Portfolio construction and aggregation layer.

This module provides three public objects:

MarketSnapshot
    An immutable slice of market data at a single point in time — spot prices,
    the implied vol surface row, risk-free rate, and dividend yields.  It is the
    sole market input the Portfolio needs to produce prices and Greeks.

Portfolio
    A list of OptionPosition legs with methods for aggregated valuation:
        .value(snapshot)           → total portfolio MTM (dollars)
        .greeks(snapshot)          → portfolio-level dollar Greeks
        .position_summary(snapshot)→ per-leg DataFrame

build_from_config
    Factory function that reads a portfolio config dict (loaded from
    config/portfolio.yaml) and a MarketData object, then instantiates a
    Portfolio by applying position rules and sector filters.

Internal helper
---------------
_get_vol(vol_surface_row, ticker, K, S, tenor_years) → float
    Looks up the implied vol from the surface via:
      - nearest-tenor matching (no tenor interpolation in Phase 1)
      - linear moneyness interpolation between bracketing grid points
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from data import MONEYNESS_LABELS, MONEYNESS_LEVELS, TENOR_LABELS, TENOR_YEARS
from portfolio.position import OptionPosition, OptionType
from portfolio.pricer import bs_greeks


# ---------------------------------------------------------------------------
# MarketSnapshot
# ---------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """
    Market state at a single point in time.

    Parameters
    ----------
    spot_prices : pd.Series
        ticker → spot price (dollars).
    vol_surface : pd.Series
        Implied vol surface row with 3-level MultiIndex
        (ticker, moneyness_label, tenor_label).  Values are annualised
        decimal vols (e.g. 0.25 = 25 %).
    risk_free_rate : float
        Annualised continuously-compounded risk-free rate (decimal).
    dividend_yields : pd.Series
        ticker → continuous dividend yield (annualised decimal).
    date : pd.Timestamp, optional
        Calendar date of this snapshot (informational).
    """

    spot_prices: pd.Series
    vol_surface: pd.Series
    risk_free_rate: float
    dividend_yields: pd.Series
    date: Optional[pd.Timestamp] = None

    @classmethod
    def from_market_data(cls, market_data, date_idx: int = -1) -> "MarketSnapshot":
        """
        Construct a MarketSnapshot by slicing MarketData at a given date index.

        Parameters
        ----------
        market_data : MarketData
            Full generated market dataset.
        date_idx : int
            Row index into market_data.prices (default -1 = last date).
        """
        return cls(
            spot_prices=market_data.prices.iloc[date_idx],
            vol_surface=market_data.vol_surface.iloc[date_idx],
            risk_free_rate=market_data.risk_free_rate,
            dividend_yields=market_data.dividend_yields,
            date=market_data.trading_dates[date_idx],
        )


# ---------------------------------------------------------------------------
# Internal vol-surface helper
# ---------------------------------------------------------------------------

def _get_vol(
    vol_surface: pd.Series,
    ticker: str,
    K: float,
    S: float,
    tenor_years: float,
) -> float:
    """
    Return implied vol for a given ticker / strike / maturity from a surface row.

    Tenor selection
    ---------------
    Nearest label in TENOR_LABELS by absolute distance (no interpolation).

    Moneyness handling
    ------------------
    moneyness = K / S.  If outside the grid, clamp to the boundary vol.
    Within the grid, linearly interpolate between bracketing grid points.

    Parameters
    ----------
    vol_surface : pd.Series
        Single-date row of the vol surface (MultiIndex: ticker, moneyness, tenor).
    ticker : str
        Underlying identifier.
    K : float
        Absolute strike price.
    S : float
        Current spot price of the underlying.
    tenor_years : float
        Time to expiry in years (used to select the nearest tenor label).

    Returns
    -------
    float
        Annualised implied vol (decimal).
    """
    # 1. Nearest tenor label
    tenor_label = min(TENOR_LABELS, key=lambda lbl: abs(TENOR_YEARS[lbl] - tenor_years))

    # 2. Moneyness
    m = K / S

    # 3. Boundary clamp
    if m <= MONEYNESS_LEVELS[0]:
        return float(vol_surface[ticker, MONEYNESS_LABELS[0], tenor_label])
    if m >= MONEYNESS_LEVELS[-1]:
        return float(vol_surface[ticker, MONEYNESS_LABELS[-1], tenor_label])

    # 4. Linear interpolation between bracketing moneyness grid points
    for i in range(len(MONEYNESS_LEVELS) - 1):
        lo, hi = MONEYNESS_LEVELS[i], MONEYNESS_LEVELS[i + 1]
        if lo <= m <= hi:
            lbl_lo = MONEYNESS_LABELS[i]
            lbl_hi = MONEYNESS_LABELS[i + 1]
            vol_lo = float(vol_surface[ticker, lbl_lo, tenor_label])
            vol_hi = float(vol_surface[ticker, lbl_hi, tenor_label])
            t = (m - lo) / (hi - lo)
            return vol_lo + t * (vol_hi - vol_lo)

    # Should never reach here given the clamp above
    raise ValueError(f"Cannot interpolate vol: ticker={ticker}, moneyness={m:.6f}")


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    "ticker", "option_type", "strike", "tenor_years",
    "quantity", "notional_per_contract",
    "spot", "moneyness", "iv",
    "price", "delta", "gamma", "vega", "theta",
    "position_value", "position_delta", "position_gamma",
    "position_vega", "position_theta",
]


class Portfolio:
    """
    A collection of OptionPosition legs with aggregated valuation.

    Parameters
    ----------
    positions : list[OptionPosition]
        Individual option legs.  May be empty (zero-value portfolio).
    """

    def __init__(self, positions: list[OptionPosition]) -> None:
        self.positions: list[OptionPosition] = list(positions)

    # ------------------------------------------------------------------
    # Public valuation methods
    # ------------------------------------------------------------------

    def value(self, snapshot: MarketSnapshot) -> float:
        """
        Total mark-to-market value of the portfolio in dollars.

        = Σ_leg  bs_price(leg) × notional_shares(leg)

        Long positions (positive notional_shares) add positive value;
        short positions (negative notional_shares) subtract.
        """
        return float(self.position_summary(snapshot)["position_value"].sum())

    def greeks(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """
        Portfolio-level dollar Greeks aggregated across all legs.

        Each Greek is scaled by the leg's signed notional_shares
        (= quantity × notional_per_contract), so the result is in
        dollar-equivalent units per unit of the underlying move.

        Returns
        -------
        dict with keys: 'value', 'delta', 'gamma', 'vega', 'theta'
        """
        df = self.position_summary(snapshot)
        return {
            "value": float(df["position_value"].sum()),
            "delta": float(df["position_delta"].sum()),
            "gamma": float(df["position_gamma"].sum()),
            "vega":  float(df["position_vega"].sum()),
            "theta": float(df["position_theta"].sum()),
        }

    def position_summary(self, snapshot: MarketSnapshot) -> pd.DataFrame:
        """
        Per-leg DataFrame with market data, per-share Greeks, and
        notional-scaled position Greeks.

        Columns
        -------
        ticker, option_type, strike, tenor_years, quantity,
        notional_per_contract, spot, moneyness, iv,
        price, delta, gamma, vega, theta,
        position_value, position_delta, position_gamma,
        position_vega, position_theta
        """
        if not self.positions:
            return pd.DataFrame(columns=_SUMMARY_COLS)

        rows = []
        for pos in self.positions:
            S = float(snapshot.spot_prices[pos.ticker])
            q = float(snapshot.dividend_yields[pos.ticker])
            sigma = _get_vol(
                snapshot.vol_surface, pos.ticker, pos.strike, S, pos.tenor_years
            )
            g = bs_greeks(
                S, pos.strike, pos.tenor_years,
                snapshot.risk_free_rate, sigma, q,
                pos.option_type.value,
            )
            n = pos.notional_shares  # signed: positive = long, negative = short
            rows.append({
                "ticker":               pos.ticker,
                "option_type":          pos.option_type.value,
                "strike":               pos.strike,
                "tenor_years":          pos.tenor_years,
                "quantity":             pos.quantity,
                "notional_per_contract": pos.notional_per_contract,
                "spot":                 S,
                "moneyness":            pos.strike / S,
                "iv":                   sigma,
                "price":                g["price"],
                "delta":                g["delta"],
                "gamma":                g["gamma"],
                "vega":                 g["vega"],
                "theta":                g["theta"],
                "position_value":       g["price"] * n,
                "position_delta":       g["delta"] * n,
                "position_gamma":       g["gamma"] * n,
                "position_vega":        g["vega"]  * n,
                "position_theta":       g["theta"] * n,
            })

        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.positions)

    def __repr__(self) -> str:
        return f"Portfolio({len(self.positions)} positions)"


# ---------------------------------------------------------------------------
# Factory: build from config
# ---------------------------------------------------------------------------

def build_from_config(
    portfolio_cfg: dict,
    market_data,
    date_idx: int = -1,
) -> Portfolio:
    """
    Instantiate a Portfolio from a portfolio config dict and MarketData.

    Position rules are applied to every underlying (or to a sector-filtered
    subset when a rule has a 'sectors' key).  The strike for each leg is set
    as K = moneyness × S_current at date_idx.

    Index overlay positions are skipped in Phase 1 (TODO Phase 2).

    Parameters
    ----------
    portfolio_cfg : dict
        Parsed contents of config/portfolio.yaml.
    market_data : MarketData
        Full generated market dataset.
    date_idx : int
        Row index used to read spot prices for strike computation (default −1).

    Returns
    -------
    Portfolio
    """
    spot_prices = market_data.prices.iloc[date_idx]

    # Build ticker → sector reverse mapping
    sector_of: dict[str, str] = {}
    for sector, tickers in market_data.sectors.items():
        for ticker in tickers:
            sector_of[ticker] = sector

    positions: list[OptionPosition] = []

    for rule in portfolio_cfg.get("position_rules", []):
        option_type = OptionType.parse(str(rule["option_type"]))
        moneyness = float(rule["moneyness"])
        tenor_label = str(rule["tenor"])
        tenor_years = TENOR_YEARS[tenor_label]
        quantity = int(rule["quantity"])
        allowed_sectors = rule.get("sectors", None)  # None → apply to all sectors

        for underlying in portfolio_cfg.get("underlyings", []):
            ticker = str(underlying["ticker"])
            notional_per_contract = int(underlying.get("notional_per_contract", 100))
            sector = sector_of.get(ticker)

            # Apply sector filter when the rule specifies one
            if allowed_sectors is not None and sector not in allowed_sectors:
                continue

            S = float(spot_prices[ticker])
            K = round(moneyness * S, 4)

            positions.append(
                OptionPosition(
                    ticker=ticker,
                    option_type=option_type,
                    strike=K,
                    tenor_years=tenor_years,
                    quantity=quantity,
                    notional_per_contract=notional_per_contract,
                )
            )

    # Index overlay — TODO Phase 2
    # if portfolio_cfg.get("index_overlay", {}).get("enabled", False):
    #     _build_index_overlay(portfolio_cfg["index_overlay"], market_data, date_idx, positions)

    return Portfolio(positions)
