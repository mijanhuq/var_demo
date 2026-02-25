"""
data/generator.py
-----------------
Synthetic market data generator — Mode A (Phase 1).

Generates correlated equity price paths for 25 underlyings across 5 sectors
via Geometric Brownian Motion (GBM), plus a parametric implied volatility
surface with put skew and an upward-sloping term structure.

An optional crisis period is injected into the simulation to create a
realistic stress environment for Stress VaR exercises (Phase 2).

Public API
----------
    MarketData          — immutable container for all generated market data
    MarketDataGenerator — generates a MarketData instance from a config dict

Usage
-----
    import yaml
    from data.generator import MarketDataGenerator

    with open("config/model.yaml") as f:
        config = yaml.safe_load(f)

    md = MarketDataGenerator(config, seed=42).generate()

Mode B (yfinance real data ingestion) is deferred to Phase 2 and will live in
data/loader.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from data import (
    MONEYNESS_LABELS,
    MONEYNESS_LEVELS,
    N_ASSETS,
    N_PER_SECTOR,
    SECTORS,
    TENOR_LABELS,
    TENOR_YEARS,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Sector simulation parameters
SECTOR_PARAMS: dict[str, dict] = {
    "Technology": {"base_vol": 0.28, "mu": 0.12, "div_yield": 0.010},
    "Financials":  {"base_vol": 0.24, "mu": 0.10, "div_yield": 0.025},
    "Healthcare":  {"base_vol": 0.20, "mu": 0.09, "div_yield": 0.015},
    "Energy":      {"base_vol": 0.32, "mu": 0.07, "div_yield": 0.030},
    "Consumer":    {"base_vol": 0.18, "mu": 0.08, "div_yield": 0.020},
}

# Initial price sampling ranges per sector (uniform draw)
SECTOR_S0_RANGE: dict[str, tuple[float, float]] = {
    "Technology": (120.0, 300.0),
    "Financials":  (50.0, 150.0),
    "Healthcare":  (80.0, 200.0),
    "Energy":      (40.0, 120.0),
    "Consumer":    (100.0, 300.0),
}

# Sector ticker prefixes
SECTOR_PREFIX: dict[str, str] = {
    "Technology": "TECH",
    "Financials":  "FINL",
    "Healthcare":  "HLTH",
    "Energy":      "ENGY",
    "Consumer":    "CONS",
}

# Correlation structure
RHO_INTRA: float = 0.65   # within the same sector
RHO_INTER: float = 0.25   # across different sectors

# Vol surface shape parameters
SKEW_SLOPE: float = 0.30    # vol-point premium per unit moneyness below 1.0 (put wing)
CALL_SLOPE: float = -0.05   # vol-point discount per unit moneyness above 1.0 (call wing)
TERM_SLOPE: float = 0.08    # fraction of ATM vol added per year of tenor

# ATM vol dynamics (AR-1 mean-reverting process)
VOL_AR1_PHI: float = 0.95        # persistence
VOL_AR1_SHOCK_STD: float = 0.003  # daily vol-of-vol shock std

# Crisis regime
CRISIS_MU: float = -0.50             # annualised log drift during crisis (negative = drawdown)
CRISIS_VOL_MULTIPLIER: float = 2.50  # ATM vol spike factor at peak of crisis
CRISIS_RAMP_DAYS: int = 21           # smooth linear ramp in/out of crisis

# Idiosyncratic noise added to sector params to differentiate assets within a sector
IDIO_MU_STD: float = 0.010    # std of mu perturbation
IDIO_VOL_STD: float = 0.020   # std of sigma perturbation


# ---------------------------------------------------------------------------
# MarketData container
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    """
    Immutable container for all generated market data.

    Attributes
    ----------
    prices : DataFrame (n_days × n_assets)
        Daily closing equity prices.  Index: DatetimeIndex of trading dates.
        Columns: ticker names.

    log_returns : DataFrame (n_days × n_assets)
        Daily log returns: log_returns[t] = ln(prices[t] / prices[t-1]).
        log_returns[0] is the return relative to the (unobserved) S0.
        Index and columns match prices.

    atm_vols : DataFrame (n_days × n_assets)
        ATM implied volatility time series for each underlying.
        Mean-reverting AR(1) process in normal regime; amplified during crisis.

    vol_surface : DataFrame (n_days × n_tickers × n_moneyness × n_tenors)
        Full implied vol surface.
        Columns: 3-level MultiIndex (ticker, moneyness_label, tenor_label).
        Values: annualised implied vol (decimal, e.g. 0.25 = 25%).

    risk_free_rate : float
        Constant risk-free rate (annualised, continuously compounded).

    dividend_yields : Series (n_assets,)
        Constant dividend yield per ticker (annualised).

    tickers : list[str]
        Ordered list of 25 ticker names.

    sectors : dict[str, list[str]]
        Maps sector name → list of tickers in that sector.

    trading_dates : DatetimeIndex
        Business-day date index (length n_days).

    crisis_start_idx : int
        Integer index of the first day of the crisis regime (inclusive).

    crisis_end_idx : int
        Integer index of the first day after the crisis regime (exclusive).
        Set equal to crisis_start_idx (or n_days) when crisis is disabled.
    """

    prices: pd.DataFrame
    log_returns: pd.DataFrame
    atm_vols: pd.DataFrame
    vol_surface: pd.DataFrame
    risk_free_rate: float
    dividend_yields: pd.Series
    tickers: list[str]
    sectors: dict[str, list[str]]
    trading_dates: pd.DatetimeIndex
    crisis_start_idx: int
    crisis_end_idx: int

    @property
    def n_days(self) -> int:
        """Number of trading days in the dataset."""
        return len(self.prices)

    @property
    def n_assets(self) -> int:
        """Number of equity underlyings."""
        return len(self.tickers)

    def get_scenario_window(self, end_idx: int, window: int = 252) -> "ScenarioWindow":
        """
        Return log returns and vol changes for the lookback window ending at end_idx.

        Used by the VaR engine to extract the 252 historical scenarios.

        Args:
            end_idx: last day (exclusive) of the lookback window.
            window:  number of scenarios (default: 252 = 1 year).

        Returns:
            ScenarioWindow with equity_returns and vol_changes arrays.

        TODO: implement ScenarioWindow dataclass in var/historical.py and wire up.
        """
        raise NotImplementedError(
            "get_scenario_window will be implemented when var/historical.py is built."
        )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MarketDataGenerator:
    """
    Generates synthetic market data for the Historical VaR model (Mode A).

    Parameters
    ----------
    config : dict
        Parsed contents of config/model.yaml.
    seed : int
        Master random seed for reproducibility.  All internal RNG calls
        use numpy.random.default_rng(seed) to ensure isolation.
    """

    def __init__(self, config: dict, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self) -> MarketData:
        """
        Run the full synthetic data generation pipeline.

        Steps
        -----
        1. Build ticker names and sector mapping.
        2. Build block-structured correlation matrix.
        3. Sample initial prices and per-asset drift/vol parameters.
        4. Simulate correlated GBM equity paths with optional crisis regime.
        5. Simulate ATM implied vol time series (AR-1 + crisis spike).
        6. Build the full parametric vol surface from ATM vols.
        7. Align everything to a business-day DatetimeIndex.

        Returns
        -------
        MarketData
        """
        sim_cfg = self.config.get("simulation", {})
        n_days: int = sim_cfg.get("n_trading_days", 1510)
        start_date: str = sim_cfg.get("start_date", "2018-01-02")
        rf_rate: float = self.config.get("market", {}).get("risk_free_rate", 0.05)

        crisis_cfg = self.config.get("crisis", {})
        inject_crisis: bool = crisis_cfg.get("inject", True)
        if inject_crisis:
            crisis_start: int = int(crisis_cfg.get("start_day", 630))
            crisis_end: int = int(crisis_cfg.get("end_day", 882))
            crisis_end = min(crisis_end, n_days)
        else:
            # No crisis: set both to n_days so slice [n_days:n_days] is empty
            crisis_start = n_days
            crisis_end = n_days

        # Step 1: tickers and sector structure
        tickers, sectors, sectors_order = self._build_tickers()

        # Step 2: correlation matrix
        corr_matrix = self._build_correlation_matrix()

        # Step 3: initial prices and asset-level parameters
        S0 = self._sample_initial_prices(sectors_order)
        mu_vec, sigma_vec = self._build_asset_params(sectors_order)
        div_yields = self._build_dividend_yields(tickers, sectors_order)

        # Step 4: equity price simulation
        prices_df, log_returns_df = self._generate_equity_prices(
            n_days, tickers, mu_vec, sigma_vec, corr_matrix, S0, crisis_start, crisis_end
        )

        # Step 5: ATM vol time series
        atm_vols_df = self._generate_atm_vols(
            n_days, tickers, sigma_vec, crisis_start, crisis_end
        )

        # Step 6: full vol surface
        vol_surface_df = self._generate_vol_surface(atm_vols_df, tickers)

        # Step 7: business-day index
        trading_dates = pd.bdate_range(start=start_date, periods=n_days)
        for df in (prices_df, log_returns_df, atm_vols_df, vol_surface_df):
            df.index = trading_dates

        return MarketData(
            prices=prices_df,
            log_returns=log_returns_df,
            atm_vols=atm_vols_df,
            vol_surface=vol_surface_df,
            risk_free_rate=rf_rate,
            dividend_yields=div_yields,
            tickers=tickers,
            sectors=sectors,
            trading_dates=trading_dates,
            crisis_start_idx=crisis_start,
            crisis_end_idx=crisis_end,
        )

    # ------------------------------------------------------------------
    # Private helpers — ticker / parameter construction
    # ------------------------------------------------------------------

    def _build_tickers(self) -> tuple[list[str], dict[str, list[str]], list[str]]:
        """
        Build the flat ticker list, sector→tickers mapping, and
        the per-ticker sector label list (parallel to tickers).

        Returns
        -------
        tickers : flat list of 25 ticker strings (ordered by sector)
        sectors : dict mapping sector name → [tickers in that sector]
        sectors_order : list of 25 sector labels, one per ticker
        """
        tickers: list[str] = []
        sectors: dict[str, list[str]] = {}
        sectors_order: list[str] = []

        for sector in SECTORS:
            prefix = SECTOR_PREFIX[sector]
            sector_tickers = [f"{prefix}_{i + 1:02d}" for i in range(N_PER_SECTOR)]
            tickers.extend(sector_tickers)
            sectors[sector] = sector_tickers
            sectors_order.extend([sector] * N_PER_SECTOR)

        return tickers, sectors, sectors_order

    def _build_correlation_matrix(self) -> np.ndarray:
        """
        Build a block-structured (sector-based) correlation matrix.

        Within-sector pairs → RHO_INTRA (0.65)
        Cross-sector pairs  → RHO_INTER (0.25)
        Diagonal            → 1.0

        A small regularisation term is added if the matrix is not
        strictly positive definite (guards against floating-point issues).

        Returns
        -------
        corr : ndarray (N_ASSETS × N_ASSETS)
        """
        n = N_ASSETS
        corr = np.full((n, n), RHO_INTER)

        for s_idx in range(len(SECTORS)):
            start = s_idx * N_PER_SECTOR
            end = start + N_PER_SECTOR
            corr[start:end, start:end] = RHO_INTRA

        np.fill_diagonal(corr, 1.0)

        # Regularise to ensure positive definiteness for Cholesky decomposition
        min_eig = np.linalg.eigvalsh(corr).min()
        if min_eig < 1e-8:
            eps = abs(min_eig) + 1e-6
            corr += eps * np.eye(n)
            # Re-normalise so diagonal stays at 1
            d = np.sqrt(np.diag(corr))
            corr /= np.outer(d, d)

        return corr

    def _sample_initial_prices(self, sectors_order: list[str]) -> np.ndarray:
        """
        Sample initial equity prices uniformly from sector-specific ranges.

        Returns
        -------
        S0 : ndarray (N_ASSETS,)
        """
        S0 = np.zeros(N_ASSETS)
        for i, sector in enumerate(sectors_order):
            lo, hi = SECTOR_S0_RANGE[sector]
            S0[i] = self.rng.uniform(lo, hi)
        return np.round(S0, 2)

    def _build_asset_params(
        self, sectors_order: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build per-asset annualised drift (mu) and volatility (sigma) vectors.

        Adds small idiosyncratic noise within each sector so individual
        assets are distinguishable while remaining sector-coherent.

        Returns
        -------
        mu    : ndarray (N_ASSETS,)  — annualised log drift
        sigma : ndarray (N_ASSETS,)  — annualised volatility
        """
        mu = np.array(
            [SECTOR_PARAMS[s]["mu"] for s in sectors_order], dtype=float
        )
        sigma = np.array(
            [SECTOR_PARAMS[s]["base_vol"] for s in sectors_order], dtype=float
        )
        mu += self.rng.normal(0.0, IDIO_MU_STD, N_ASSETS)
        sigma += self.rng.normal(0.0, IDIO_VOL_STD, N_ASSETS)
        sigma = np.clip(sigma, 0.10, 0.60)
        return mu, sigma

    def _build_dividend_yields(
        self, tickers: list[str], sectors_order: list[str]
    ) -> pd.Series:
        """
        Build per-ticker dividend yield Series from sector defaults.

        Returns
        -------
        pd.Series indexed by ticker
        """
        yields = [SECTOR_PARAMS[s]["div_yield"] for s in sectors_order]
        return pd.Series(yields, index=tickers, name="dividend_yield", dtype=float)

    # ------------------------------------------------------------------
    # Private helpers — equity price simulation
    # ------------------------------------------------------------------

    def _generate_equity_prices(
        self,
        n_days: int,
        tickers: list[str],
        mu: np.ndarray,
        sigma: np.ndarray,
        corr_matrix: np.ndarray,
        S0: np.ndarray,
        crisis_start: int,
        crisis_end: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate correlated equity log returns and prices via GBM.

        Normal regime
        -------------
        Each asset follows:  dS/S = mu·dt + sigma·dW
        → daily log return = (mu - ½σ²)·dt + σ·√dt·Z_corr

        Crisis regime  [crisis_start, crisis_end)
        ----------------------------------------
        Negative drift (CRISIS_MU) and amplified vol (sigma × CRISIS_VOL_MULTIPLIER).
        A linear ramp of CRISIS_RAMP_DAYS is applied at the start and end of the
        crisis window to avoid an artificial discontinuity.

        Returns
        -------
        prices      : DataFrame (n_days × n_assets)
        log_returns : DataFrame (n_days × n_assets)
        """
        dt = 1.0 / 252
        L = np.linalg.cholesky(corr_matrix)

        # --- Normal regime log returns ---
        Z = self.rng.standard_normal((n_days, N_ASSETS))
        Z_corr = Z @ L.T
        drift_normal = (mu - 0.5 * sigma ** 2) * dt
        log_returns = drift_normal + sigma * np.sqrt(dt) * Z_corr  # (n_days, N_ASSETS)

        # --- Crisis regime override ---
        if crisis_start < crisis_end:
            crisis_len = crisis_end - crisis_start
            crisis_sigma = sigma * CRISIS_VOL_MULTIPLIER
            crisis_mu_vec = np.full(N_ASSETS, CRISIS_MU)

            Z_c = self.rng.standard_normal((crisis_len, N_ASSETS))
            Z_c_corr = Z_c @ L.T
            drift_crisis = (crisis_mu_vec - 0.5 * crisis_sigma ** 2) * dt
            log_returns_crisis = drift_crisis + crisis_sigma * np.sqrt(dt) * Z_c_corr

            # Smooth blending factor: ramp from 0→1 at entry, 1→0 at exit
            ramp = min(CRISIS_RAMP_DAYS, max(1, crisis_len // 4))
            alphas = np.ones(crisis_len)
            alphas[:ramp] = np.linspace(0.0, 1.0, ramp)
            alphas[-ramp:] = np.linspace(1.0, 0.0, ramp)
            alphas = alphas[:, np.newaxis]  # (crisis_len, 1) for broadcasting

            log_returns[crisis_start:crisis_end] = (
                (1.0 - alphas) * log_returns[crisis_start:crisis_end]
                + alphas * log_returns_crisis
            )

        # --- Reconstruct price path from cumulative log returns ---
        log_prices = np.empty((n_days + 1, N_ASSETS))
        log_prices[0] = np.log(S0)
        log_prices[1:] = log_prices[0] + np.cumsum(log_returns, axis=0)
        prices = np.exp(log_prices[1:])

        return (
            pd.DataFrame(prices, columns=tickers, dtype=float),
            pd.DataFrame(log_returns, columns=tickers, dtype=float),
        )

    # ------------------------------------------------------------------
    # Private helpers — implied vol surface
    # ------------------------------------------------------------------

    def _generate_atm_vols(
        self,
        n_days: int,
        tickers: list[str],
        base_sigma: np.ndarray,
        crisis_start: int,
        crisis_end: int,
    ) -> pd.DataFrame:
        """
        Generate an ATM implied vol time series for each ticker.

        Models vol as a mean-reverting AR(1) process:
            vol[t] = φ·vol[t-1] + (1-φ)·base_vol + shock[t]

        During the crisis period the vols are multiplied by
        CRISIS_VOL_MULTIPLIER with the same smooth ramp used for prices.

        Returns
        -------
        DataFrame (n_days × n_assets), values in (0, 2.0]
        """
        atm_vols = np.zeros((n_days, N_ASSETS))
        atm_vols[0] = base_sigma.copy()

        shocks = self.rng.normal(0.0, VOL_AR1_SHOCK_STD, (n_days, N_ASSETS))
        for t in range(1, n_days):
            atm_vols[t] = (
                VOL_AR1_PHI * atm_vols[t - 1]
                + (1.0 - VOL_AR1_PHI) * base_sigma
                + shocks[t]
            )

        atm_vols = np.clip(atm_vols, 0.05, 1.00)

        # Crisis vol spike with smooth ramp
        if crisis_start < crisis_end:
            crisis_len = crisis_end - crisis_start
            ramp = min(CRISIS_RAMP_DAYS, max(1, crisis_len // 4))
            factors = np.ones(crisis_len)
            factors[:ramp] = np.linspace(1.0, CRISIS_VOL_MULTIPLIER, ramp)
            factors[-ramp:] = np.linspace(CRISIS_VOL_MULTIPLIER, 1.0, ramp)
            # Middle: constant at CRISIS_VOL_MULTIPLIER
            factors[ramp : crisis_len - ramp] = CRISIS_VOL_MULTIPLIER
            atm_vols[crisis_start:crisis_end] *= factors[:, np.newaxis]

        atm_vols = np.clip(atm_vols, 0.05, 2.00)

        return pd.DataFrame(atm_vols, columns=tickers, dtype=float)

    @staticmethod
    def _compute_iv(
        atm_vol: np.ndarray, moneyness: float, tenor_years: float
    ) -> np.ndarray:
        """
        Compute implied vol for a given (moneyness, tenor) node.

        Put wing  (moneyness < 1.0):  iv = ATM + SKEW_SLOPE × (1 - m)
        Call wing (moneyness > 1.0):  iv = ATM + CALL_SLOPE × (m - 1)
        Term structure:               iv += ATM × TERM_SLOPE × T

        All adjustments are additive in vol-point space (decimal vol).

        Parameters
        ----------
        atm_vol     : ndarray (n_days,) — ATM implied vol for one ticker
        moneyness   : K/S ratio
        tenor_years : option tenor in years

        Returns
        -------
        ndarray (n_days,) — implied vol, clipped to [0.01, ∞)
        """
        if moneyness < 1.0:
            skew_adj = SKEW_SLOPE * (1.0 - moneyness)
        else:
            skew_adj = CALL_SLOPE * (moneyness - 1.0)

        term_adj = atm_vol * TERM_SLOPE * tenor_years
        iv = atm_vol + skew_adj + term_adj
        return np.maximum(0.01, iv)

    def _generate_vol_surface(
        self, atm_vols_df: pd.DataFrame, tickers: list[str]
    ) -> pd.DataFrame:
        """
        Build the full parametric implied vol surface.

        Iterates over all (ticker, moneyness, tenor) combinations and calls
        _compute_iv for each, storing the result in a wide DataFrame with a
        3-level MultiIndex on columns.

        Returns
        -------
        DataFrame shape: (n_days, N_ASSETS × n_moneyness × n_tenors)
                       = (n_days, 25 × 5 × 4) = (n_days, 500)
        Column MultiIndex names: ['ticker', 'moneyness', 'tenor']
        """
        n_days = len(atm_vols_df)
        n_cols = N_ASSETS * len(MONEYNESS_LABELS) * len(TENOR_LABELS)
        vol_data = np.empty((n_days, n_cols))

        col_tuples: list[tuple[str, str, str]] = []
        col_idx = 0

        for ticker in tickers:
            atm = atm_vols_df[ticker].values  # (n_days,)
            for m_label, moneyness in zip(MONEYNESS_LABELS, MONEYNESS_LEVELS):
                for t_label, tenor_years in TENOR_YEARS.items():
                    vol_data[:, col_idx] = self._compute_iv(atm, moneyness, tenor_years)
                    col_tuples.append((ticker, m_label, t_label))
                    col_idx += 1

        columns = pd.MultiIndex.from_tuples(
            col_tuples, names=["ticker", "moneyness", "tenor"]
        )
        return pd.DataFrame(vol_data, columns=columns, dtype=float)
