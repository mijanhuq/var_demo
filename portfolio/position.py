"""
portfolio/position.py
---------------------
Data structures representing individual portfolio positions.

OptionPosition holds all static data for a single option leg — the fields
that are fixed at trade inception and do not change when the market moves.
The pricer receives these alongside live market data to produce prices and
Greeks at any point in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OptionType(str, Enum):
    """European option type.  Inherits from str so 'call' == OptionType.CALL."""
    CALL = "call"
    PUT = "put"

    @classmethod
    def parse(cls, value: str) -> "OptionType":
        """Accept 'call'/'put' (case-insensitive) and return the enum member."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid option type '{value}'. Expected 'call' or 'put'."
            )


# ---------------------------------------------------------------------------
# Position dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionPosition:
    """
    A single European vanilla option leg (call or put).

    Parameters
    ----------
    ticker : str
        Underlying equity identifier (must match a key in MarketData.prices).
    option_type : OptionType
        CALL or PUT.
    strike : float
        Absolute strike price K in the same currency as the underlying.
        Fixed at trade inception; does not change as the spot moves.
    tenor_years : float
        Time to expiry T in years at the time the position was entered.
        The VaR engine uses this directly without decrementing it on a
        daily basis (i.e. 1-day horizon approximation in Phase 1).
    quantity : int
        Number of contracts.  Positive = long, negative = short.
    notional_per_contract : int
        Number of shares (or index units) per contract.  Default 100.
    """

    ticker: str
    option_type: OptionType
    strike: float
    tenor_years: float
    quantity: int
    notional_per_contract: int = 100

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def notional_shares(self) -> int:
        """Total number of shares represented (signed: negative = short)."""
        return self.quantity * self.notional_per_contract

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        return self.option_type == OptionType.PUT

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        direction = "long" if self.quantity > 0 else "short"
        return (
            f"{direction} {abs(self.quantity)} × {self.option_type.value} "
            f"K={self.strike:.2f}  T={self.tenor_years:.3f}y  [{self.ticker}]"
        )
