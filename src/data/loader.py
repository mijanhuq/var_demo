"""
Market data loader: downloads historical adjusted close prices from yfinance
and caches to disk. Supports equities and indices.
"""

import time
import warnings
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0   # seconds; doubles on each attempt (1 s, 2 s, 4 s)


# ---------------------------------------------------------------------------
# Default universes (requirements §2.1 and §2.2)
# ---------------------------------------------------------------------------

EQUITY_TICKERS: list[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    # Financials
    "JPM", "GS", "BAC", "MS", "BLK",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    # Consumer
    "AMZN", "TSLA", "HD", "NKE", "MCD",
    # Energy / Industrials
    "XOM", "CVX", "CAT", "BA", "GE",
]

INDEX_TICKERS: list[str] = [
    "^GSPC",   # S&P 500
    "^NDX",    # NASDAQ-100
    "^RUT",    # Russell 2000
    "^VIX",    # CBOE VIX
    "XLK",     # Tech sector ETF
    "XLF",     # Financials sector ETF
    "XLE",     # Energy sector ETF
]

ALL_TICKERS: list[str] = EQUITY_TICKERS + INDEX_TICKERS

# Threshold above which a ticker's missing data fraction triggers a warning
_MISSING_DATA_WARN_THRESHOLD = 0.05

# Default history: 5 years + small buffer for rolling vol warm-up
_DEFAULT_YEARS = 5


# ---------------------------------------------------------------------------
# Core download and cache
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_")
    return cache_dir / f"{safe}.csv"


def _download_single(ticker: str, start: str, end: str) -> pd.Series:
    """
    Download adjusted close for one ticker from yfinance.

    Retries up to _MAX_RETRIES times with exponential backoff on transient
    network errors (ConnectionError, TimeoutError, OSError).  Other exceptions
    (e.g. ValueError for empty data) are raised immediately.
    """
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                show_errors=False,
            )
            if raw.empty:
                raise ValueError(
                    f"yfinance returned no data for {ticker!r} ({start} – {end})"
                )
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = ticker
            return close

        except ValueError:
            raise   # empty-data errors are not retryable
        except (ConnectionError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                warnings.warn(
                    f"Network error downloading {ticker!r} "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES}): {exc}. "
                    f"Retrying in {delay:.0f}s.",
                    UserWarning,
                    stacklevel=3,
                )
                time.sleep(delay)

    raise ConnectionError(
        f"Failed to download {ticker!r} after {_MAX_RETRIES} attempts."
    ) from last_exc


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download (or load from cache) adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers   : list of yfinance ticker strings
    start     : ISO date string "YYYY-MM-DD"
    end       : ISO date string "YYYY-MM-DD"
    cache_dir : directory to read/write CSV caches; None = no caching

    Returns
    -------
    DataFrame with DatetimeIndex and tickers as columns; no NaN.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    series_list: list[pd.Series] = []

    for ticker in tickers:
        cached_file = _cache_path(cache_dir, ticker) if cache_dir else None

        if cached_file is not None and cached_file.exists():
            s = pd.read_csv(cached_file, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = ticker
        else:
            s = _download_single(ticker, start, end)
            if cached_file is not None:
                s.to_frame().to_csv(cached_file)

        series_list.append(s)

    df = pd.concat(series_list, axis=1)
    df = df.sort_index()

    # Check and fill missing data
    for ticker in df.columns:
        missing_frac = df[ticker].isna().mean()
        if missing_frac > _MISSING_DATA_WARN_THRESHOLD:
            warnings.warn(
                f"{ticker}: {missing_frac:.1%} of values are missing — "
                "check if this ticker was listed for the full period.",
                UserWarning,
                stacklevel=2,
            )

    # Forward-fill then back-fill (handles non-trading days and short gaps)
    df = df.ffill().bfill()

    if df.isna().any(axis=None):
        raise ValueError("NaN values remain after filling — check input data.")

    return df


def load_prices(
    tickers: list[str] | None = None,
    n_years: int = _DEFAULT_YEARS,
    cache_dir: Path | str | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Main entry point: load n_years of adjusted close prices.

    Parameters
    ----------
    tickers   : list of tickers; defaults to ALL_TICKERS (25 equities + 7 indices)
    n_years   : number of years of history to fetch
    cache_dir : path to local cache directory; None = no caching
    end_date  : last date to fetch (defaults to yesterday)

    Returns
    -------
    DataFrame, DatetimeIndex, columns = tickers, no NaN, all prices > 0.
    """
    if tickers is None:
        tickers = ALL_TICKERS

    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    start_date = end_date - timedelta(days=int(n_years * 365.25) + 10)  # small buffer

    if cache_dir is None:
        # Default cache relative to this file's location
        # TODO: make configurable via environment variable or config file
        cache_dir = Path(__file__).parents[2] / "data" / "raw"

    prices = download_prices(
        tickers=tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        cache_dir=Path(cache_dir),
    )

    if len(prices) < 252:
        raise ValueError(
            f"Only {len(prices)} trading days loaded — need at least 252 "
            "for a 1-year lookback. Check your date range."
        )

    return prices
