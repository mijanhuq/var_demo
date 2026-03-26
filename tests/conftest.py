"""
Shared pytest fixtures for all tests.
All fixtures use a fixed random seed (42) for reproducibility.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

_EQUITY_TICKERS_SMALL = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ"]
_INDEX_TICKERS_SMALL  = ["^GSPC", "^NDX", "^RUT", "^VIX"]
_N_DAYS  = 1_300  # ~5 years
_SEED    = 42


def _make_prices(tickers, start_values, vol, n_days, seed_offset=0):
    rng = np.random.default_rng(_SEED + seed_offset)
    log_rets = rng.normal(0.0003, vol, (n_days, len(tickers)))
    price_paths = np.cumprod(np.exp(log_rets), axis=0)
    base = np.array(start_values, dtype=float)
    prices = price_paths * base[np.newaxis, :]
    dates = pd.bdate_range(end="2024-12-31", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture(scope="session")
def sample_prices() -> pd.DataFrame:
    """5 equity tickers, 5-year daily history, seed=42."""
    return _make_prices(
        _EQUITY_TICKERS_SMALL,
        [170.0, 380.0, 160.0, 195.0, 155.0],
        vol=0.015,
        n_days=_N_DAYS,
    )


@pytest.fixture(scope="session")
def sample_index_prices() -> pd.DataFrame:
    """4 index tickers, 5-year daily history."""
    return _make_prices(
        _INDEX_TICKERS_SMALL,
        [4500.0, 15000.0, 2000.0, 20.0],
        vol=0.010,
        n_days=_N_DAYS,
        seed_offset=10,
    )


@pytest.fixture(scope="session")
def sample_rolling_vols(sample_prices):
    from data.vol_surface import compute_rolling_vol
    return compute_rolling_vol(sample_prices)


@pytest.fixture(scope="session")
def today_prices(sample_prices) -> pd.Series:
    return sample_prices.iloc[-1]


@pytest.fixture(scope="session")
def today_index_prices(sample_index_prices) -> pd.Series:
    return sample_index_prices.iloc[-1]


@pytest.fixture(scope="session")
def as_of(sample_prices) -> pd.Timestamp:
    return sample_prices.index[-1]


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_portfolio(sample_prices, sample_rolling_vols) -> pd.DataFrame:
    """Small synthetic portfolio (8 trades) for fast unit tests."""
    today_prices = sample_prices.iloc[-1]
    as_of = sample_prices.index[-1]
    from data.portfolio import generate_portfolio
    return generate_portfolio(
        today_prices, n_trades=8, seed=42, as_of=as_of,
        rolling_vols=sample_rolling_vols,
    )


@pytest.fixture(scope="session")
def large_portfolio(sample_prices, sample_rolling_vols) -> pd.DataFrame:
    """Larger portfolio (50 trades) for integration tests."""
    today_prices = sample_prices.iloc[-1]
    as_of = sample_prices.index[-1]
    from data.portfolio import generate_portfolio
    return generate_portfolio(
        today_prices, n_trades=50, seed=42, as_of=as_of,
        rolling_vols=sample_rolling_vols,
    )


# ---------------------------------------------------------------------------
# Scenario building helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_log_returns(sample_prices):
    from var.scenarios import compute_log_returns
    return compute_log_returns(sample_prices)


@pytest.fixture(scope="session")
def sample_scenario_set(sample_log_returns):
    from var.scenarios import build_scenario_set
    return build_scenario_set(sample_log_returns, lookback=252)


@pytest.fixture(scope="session")
def sample_pnl_vector(
    sample_portfolio,
    sample_prices,
    sample_rolling_vols,
    today_prices,
    as_of,
):
    """Pre-computed P&L vector for use in VAR/ES tests."""
    from data.portfolio import price_portfolio
    from var.scenarios import compute_log_returns, build_scenario_set, build_stressed_price_matrix
    from var.revaluation import full_reprice_pnl

    priced = price_portfolio(sample_portfolio, today_prices, sample_rolling_vols, 0.05, as_of)
    today_values = priced.set_index("trade_id")["position_value"]
    log_returns = compute_log_returns(sample_prices)
    scenario_set = build_scenario_set(log_returns, lookback=252)
    scenario_prices = build_stressed_price_matrix(today_prices, scenario_set)
    return full_reprice_pnl(
        sample_portfolio, today_prices, today_values, scenario_prices,
        sample_rolling_vols, 0.05, as_of,
    )
