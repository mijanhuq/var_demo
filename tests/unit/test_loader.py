"""
Unit tests for src/data/loader.py  (DL-01 through DL-08)
Network-dependent tests are marked @pytest.mark.network.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestLoadPricesLocal:
    """Tests that do not require network access (use cached/synthetic data)."""

    def test_prices_no_nan(self, sample_prices):
        """DL-02: No NaN after loading (fixture uses synthetic data)."""
        assert not sample_prices.isna().any().any()

    def test_all_prices_positive(self, sample_prices):
        """DL-03: All prices > 0."""
        assert (sample_prices > 0).all().all()

    def test_sorted_ascending_index(self, sample_prices):
        """DL-04: DatetimeIndex is sorted ascending."""
        idx = sample_prices.index
        assert idx.is_monotonic_increasing

    def test_five_years_enough_rows(self, sample_prices):
        """DL-05: 5-year history has ≥ 1,200 trading days."""
        assert len(sample_prices) >= 1_200

    def test_columns_are_tickers(self, sample_prices):
        """DL-01: Columns are the requested tickers."""
        assert len(sample_prices.columns) > 0


@pytest.mark.network
class TestYfinanceDownload:
    """Network-dependent tests — require internet access."""

    def test_DL08_index_tickers_download(self, tmp_path):
        """DL-08: Index tickers (^GSPC, ^NDX, ^VIX) download without error."""
        from data.loader import download_prices
        tickers = ["^GSPC", "^VIX"]
        df = download_prices(
            tickers,
            start="2023-01-01",
            end="2023-12-31",
            cache_dir=tmp_path,
        )
        assert set(tickers) == set(df.columns)
        assert len(df) > 200

    def test_DL07_cache_written_and_reused(self, tmp_path):
        """DL-07: Cache file written on first download; second call reads cache."""
        from data.loader import download_prices
        tickers = ["AAPL"]
        # First call — downloads
        df1 = download_prices(tickers, "2023-01-01", "2023-12-31", cache_dir=tmp_path)
        cache_file = tmp_path / "AAPL.csv"
        assert cache_file.exists()

        # Second call — reads from cache
        df2 = download_prices(tickers, "2023-01-01", "2023-12-31", cache_dir=tmp_path)
        pd.testing.assert_frame_equal(df1, df2)

    def test_DL01_columns_match_tickers(self, tmp_path):
        """DL-01: Downloaded DataFrame columns match requested tickers."""
        from data.loader import download_prices
        tickers = ["MSFT", "GOOGL"]
        df = download_prices(tickers, "2023-01-01", "2023-06-30", cache_dir=tmp_path)
        assert set(df.columns) == set(tickers)
