"""
tests/conftest.py
-----------------
Shared pytest fixtures for all test modules.

Fixtures are session-scoped where possible so expensive data generation
runs only once per test session.  Tests that need to mutate data should
copy the fixture value rather than modify it in place.
"""

from pathlib import Path

import pytest
import yaml

from data.generator import MarketDataGenerator
from portfolio.portfolio import MarketSnapshot, Portfolio, build_from_config
from portfolio.position import OptionPosition, OptionType
from backtest.engine import BacktestResult, run_backtest
from var.historical import VaRResult, compute_historical_var, compute_pnl_scenarios

_CONFIG_DIR = Path(__file__).parent.parent / "config"


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def minimal_config() -> dict:
    """
    Minimal configuration for fast unit tests.
    Uses 300 trading days (~14 months) with 25 underlyings.
    Crisis period: days 100–200 (100-day window).
    """
    return {
        "simulation": {
            "seed": 42,
            "n_trading_days": 300,
            "start_date": "2020-01-02",
        },
        "market": {"risk_free_rate": 0.05},
        "crisis": {
            "inject": True,
            "start_day": 100,
            "end_day": 200,
        },
    }


@pytest.fixture(scope="session")
def standard_config() -> dict:
    """
    Full-size configuration matching production settings.
    1510 trading days (~6 years).  Crisis: days 630–882 (252-day window).
    Used for statistical tests (GEN-03, GEN-04) and integration tests.
    """
    return {
        "simulation": {
            "seed": 42,
            "n_trading_days": 1510,
            "start_date": "2018-01-02",
        },
        "market": {"risk_free_rate": 0.05},
        "crisis": {
            "inject": True,
            "start_day": 630,
            "end_day": 882,
        },
    }


@pytest.fixture(scope="session")
def no_crisis_config() -> dict:
    """Configuration with crisis injection disabled."""
    return {
        "simulation": {
            "seed": 42,
            "n_trading_days": 300,
            "start_date": "2020-01-02",
        },
        "market": {"risk_free_rate": 0.05},
        "crisis": {"inject": False},
    }


# ---------------------------------------------------------------------------
# Generator fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def market_data_small(minimal_config):
    """
    Pre-generated MarketData from minimal_config (300 days).
    Fast to generate; used for structural and edge-case tests.
    """
    gen = MarketDataGenerator(config=minimal_config, seed=42)
    return gen.generate()


@pytest.fixture(scope="session")
def market_data_standard(standard_config):
    """
    Pre-generated MarketData from standard_config (1510 days).
    Slower to generate; session-scoped so it runs only once.
    Used for statistical tests and integration tests.
    """
    gen = MarketDataGenerator(config=standard_config, seed=42)
    return gen.generate()


@pytest.fixture(scope="session")
def market_data_no_crisis(no_crisis_config):
    """Pre-generated MarketData with crisis injection disabled."""
    gen = MarketDataGenerator(config=no_crisis_config, seed=42)
    return gen.generate()


# ---------------------------------------------------------------------------
# Portfolio fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def portfolio_config() -> dict:
    """Parsed config/portfolio.yaml — 25 underlyings + 6 position rules."""
    with open(_CONFIG_DIR / "portfolio.yaml") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def snapshot(market_data_small) -> MarketSnapshot:
    """MarketSnapshot at the last date of market_data_small (300 days)."""
    return MarketSnapshot.from_market_data(market_data_small, date_idx=-1)


@pytest.fixture(scope="session")
def full_portfolio(portfolio_config, market_data_small) -> Portfolio:
    """
    Full portfolio built from config/portfolio.yaml against market_data_small.
    120 single-name option legs (6 rules × 25 underlyings, with sector filters).
    """
    return build_from_config(portfolio_config, market_data_small, date_idx=-1)


# ---------------------------------------------------------------------------
# VaR fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def var_result_small(full_portfolio, snapshot, market_data_small) -> VaRResult:
    """
    VaR result for full_portfolio at the last date of market_data_small.
    Uses 252-scenario lookback (market_data_small has 300 days so this fits).
    """
    return compute_historical_var(
        full_portfolio, snapshot, market_data_small, lookback_window=252
    )


# ---------------------------------------------------------------------------
# Backtest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def backtest_result_small(full_portfolio, market_data_small) -> BacktestResult:
    """
    46-day rolling backtest on market_data_small (300 days, lookback=252).
    backtest_window=47 → 46 (VaR, P&L) pairs.
    """
    return run_backtest(
        full_portfolio, market_data_small,
        lookback_window=252, backtest_window=47,
    )
