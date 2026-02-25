"""
data/
-----
Shared constants for the market data layer.

These are imported by both data/generator.py and downstream modules
(portfolio, var) to ensure consistent labelling across the system.
"""

# --- Vol surface grid ---
MONEYNESS_LEVELS: list[float] = [0.85, 0.95, 1.00, 1.05, 1.15]
MONEYNESS_LABELS: list[str] = ["0.85", "0.95", "1.00", "1.05", "1.15"]

TENOR_LABELS: list[str] = ["1m", "3m", "6m", "12m"]
TENOR_YEARS: dict[str, float] = {
    "1m": 1 / 12,
    "3m": 3 / 12,
    "6m": 6 / 12,
    "12m": 1.0,
}

# --- Portfolio structure ---
SECTORS: list[str] = ["Technology", "Financials", "Healthcare", "Energy", "Consumer"]
N_PER_SECTOR: int = 5
N_ASSETS: int = len(SECTORS) * N_PER_SECTOR  # 25
