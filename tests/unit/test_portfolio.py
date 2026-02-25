"""
tests/unit/test_portfolio.py
-----------------------------
Unit tests for portfolio/portfolio.py.

Test IDs (from docs/test_plan.md §4.3)
---------------------------------------
PFL-01  Portfolio value equals sum of individual position values
PFL-02  Portfolio delta equals sum of position deltas × signed notional
PFL-03  Portfolio gamma equals sum of position gammas × signed notional
PFL-04  Short position has negated Greeks relative to identical long position
PFL-05  Empty portfolio has zero value and zero Greeks
PFL-06  ATM straddle delta is smaller in magnitude than a single call leg's delta
PFL-07  build_from_config loads config correctly: position count, strikes,
        sector filters, and sign of short positions

Additional tests cover:
  - MarketSnapshot construction and slicing
  - _get_vol moneyness interpolation and boundary clamping
  - position_summary shape and column content
  - performance: 120-position portfolio priced in < 10 s
"""

import time

import pytest

from portfolio.portfolio import (
    MarketSnapshot,
    Portfolio,
    _get_vol,
    build_from_config,
)
from portfolio.position import OptionPosition, OptionType
from portfolio.pricer import bs_delta, bs_gamma, bs_greeks, bs_price


# ===========================================================================
# Helpers
# ===========================================================================

def _make_position(
    ticker: str,
    option_type: str,
    strike: float,
    tenor_years: float,
    quantity: int = 1,
    notional_per_contract: int = 100,
) -> OptionPosition:
    return OptionPosition(
        ticker=ticker,
        option_type=OptionType.parse(option_type),
        strike=strike,
        tenor_years=tenor_years,
        quantity=quantity,
        notional_per_contract=notional_per_contract,
    )


def _price_position(pos: OptionPosition, snapshot: MarketSnapshot) -> float:
    """Manual leg price × signed notional."""
    S = float(snapshot.spot_prices[pos.ticker])
    q = float(snapshot.dividend_yields[pos.ticker])
    sigma = _get_vol(snapshot.vol_surface, pos.ticker, pos.strike, S, pos.tenor_years)
    price = bs_price(
        S, pos.strike, pos.tenor_years,
        snapshot.risk_free_rate, sigma, q,
        pos.option_type.value,
    )
    return float(price) * pos.notional_shares


def _delta_position(pos: OptionPosition, snapshot: MarketSnapshot) -> float:
    """Manual leg delta × signed notional."""
    S = float(snapshot.spot_prices[pos.ticker])
    q = float(snapshot.dividend_yields[pos.ticker])
    sigma = _get_vol(snapshot.vol_surface, pos.ticker, pos.strike, S, pos.tenor_years)
    d = bs_delta(
        S, pos.strike, pos.tenor_years,
        snapshot.risk_free_rate, sigma, q,
        pos.option_type.value,
    )
    return float(d) * pos.notional_shares


def _gamma_position(pos: OptionPosition, snapshot: MarketSnapshot) -> float:
    """Manual leg gamma × signed notional."""
    S = float(snapshot.spot_prices[pos.ticker])
    q = float(snapshot.dividend_yields[pos.ticker])
    sigma = _get_vol(snapshot.vol_surface, pos.ticker, pos.strike, S, pos.tenor_years)
    g = bs_gamma(
        S, pos.strike, pos.tenor_years,
        snapshot.risk_free_rate, sigma, q,
    )
    return float(g) * pos.notional_shares


# ===========================================================================
# MarketSnapshot tests
# ===========================================================================

class TestMarketSnapshot:
    """Tests for MarketSnapshot.from_market_data."""

    def test_creates_snapshot_instance(self, market_data_small):
        snap = MarketSnapshot.from_market_data(market_data_small, date_idx=-1)
        assert isinstance(snap, MarketSnapshot)

    def test_spot_prices_match_market_data_at_last_date(self, market_data_small):
        snap = MarketSnapshot.from_market_data(market_data_small, date_idx=-1)
        for ticker in market_data_small.tickers:
            expected = float(market_data_small.prices.iloc[-1][ticker])
            assert snap.spot_prices[ticker] == pytest.approx(expected, rel=1e-9)

    def test_spot_prices_at_first_date(self, market_data_small):
        snap = MarketSnapshot.from_market_data(market_data_small, date_idx=0)
        for ticker in market_data_small.tickers:
            expected = float(market_data_small.prices.iloc[0][ticker])
            assert snap.spot_prices[ticker] == pytest.approx(expected, rel=1e-9)

    def test_all_tickers_present_in_spot_prices(self, market_data_small):
        snap = MarketSnapshot.from_market_data(market_data_small, date_idx=-1)
        assert set(snap.spot_prices.index) == set(market_data_small.tickers)

    def test_vol_surface_row_length(self, market_data_small):
        snap = MarketSnapshot.from_market_data(market_data_small, date_idx=-1)
        # 25 tickers × 5 moneyness × 4 tenors = 500
        assert len(snap.vol_surface) == 500

    def test_vol_surface_values_positive(self, market_data_small):
        snap = MarketSnapshot.from_market_data(market_data_small, date_idx=-1)
        assert (snap.vol_surface > 0).all()

    def test_date_recorded_correctly(self, market_data_small):
        snap = MarketSnapshot.from_market_data(market_data_small, date_idx=-1)
        assert snap.date == market_data_small.trading_dates[-1]

    def test_risk_free_rate_matches(self, market_data_small):
        snap = MarketSnapshot.from_market_data(market_data_small, date_idx=-1)
        assert snap.risk_free_rate == pytest.approx(market_data_small.risk_free_rate)

    def test_different_date_idx_gives_different_prices(self, market_data_small):
        snap_first = MarketSnapshot.from_market_data(market_data_small, date_idx=0)
        snap_last = MarketSnapshot.from_market_data(market_data_small, date_idx=-1)
        # 300-day GBM path — prices almost certainly differ
        assert not snap_first.spot_prices.equals(snap_last.spot_prices)


# ===========================================================================
# _get_vol tests
# ===========================================================================

class TestGetVol:
    """Tests for the internal vol-surface interpolation helper."""

    def test_exact_moneyness_match_atm(self, snapshot, market_data_small):
        """ATM moneyness 1.00 maps exactly to a grid point."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        vol = _get_vol(snapshot.vol_surface, ticker, S, S, 0.25)  # K=S → moneyness=1
        assert vol > 0.0
        assert vol < 2.0  # sanity: vol should be reasonable

    def test_vol_between_grid_points_interpolated(self, snapshot, market_data_small):
        """Moneyness 0.90 lies between 0.85 and 0.95 grid points."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        K_lo = 0.85 * S  # maps to MONEYNESS_LABELS[0]
        K_hi = 0.95 * S  # maps to MONEYNESS_LABELS[1]
        K_mid = 0.90 * S
        vol_lo = _get_vol(snapshot.vol_surface, ticker, K_lo, S, 0.25)
        vol_hi = _get_vol(snapshot.vol_surface, ticker, K_hi, S, 0.25)
        vol_mid = _get_vol(snapshot.vol_surface, ticker, K_mid, S, 0.25)
        # Interpolated value must lie between the two boundary vols
        assert min(vol_lo, vol_hi) <= vol_mid <= max(vol_lo, vol_hi)

    def test_vol_below_grid_clamped_to_lower_boundary(self, snapshot, market_data_small):
        """Moneyness 0.50 (deep OTM put) is clamped to the 0.85 grid vol."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        vol_boundary = _get_vol(snapshot.vol_surface, ticker, 0.85 * S, S, 0.25)
        vol_extreme = _get_vol(snapshot.vol_surface, ticker, 0.50 * S, S, 0.25)
        assert vol_extreme == pytest.approx(vol_boundary)

    def test_vol_above_grid_clamped_to_upper_boundary(self, snapshot, market_data_small):
        """Moneyness 1.50 (deep OTM call) is clamped to the 1.15 grid vol."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        vol_boundary = _get_vol(snapshot.vol_surface, ticker, 1.15 * S, S, 0.25)
        vol_extreme = _get_vol(snapshot.vol_surface, ticker, 1.50 * S, S, 0.25)
        assert vol_extreme == pytest.approx(vol_boundary)

    def test_nearest_tenor_selected(self, snapshot, market_data_small):
        """Tenor 0.10 years is nearest to 1m (0.0833y) not 3m (0.25y)."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        vol_1m = _get_vol(snapshot.vol_surface, ticker, S, S, 1 / 12)
        vol_near_1m = _get_vol(snapshot.vol_surface, ticker, S, S, 0.10)
        vol_3m = _get_vol(snapshot.vol_surface, ticker, S, S, 0.25)
        assert vol_near_1m == pytest.approx(vol_1m)
        assert vol_near_1m != pytest.approx(vol_3m)

    def test_vol_positive_for_all_tickers(self, snapshot, market_data_small):
        """All tickers return a positive vol for an ATM 3m lookup."""
        for ticker in market_data_small.tickers:
            S = float(snapshot.spot_prices[ticker])
            vol = _get_vol(snapshot.vol_surface, ticker, S, S, 0.25)
            assert vol > 0.0, f"Non-positive vol for {ticker}"


# ===========================================================================
# PFL-01: Portfolio value
# ===========================================================================

class TestPortfolioValue:
    """PFL-01: Portfolio value equals sum of individual position values."""

    def test_single_long_call_value_positive(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        pos = _make_position(ticker, "call", S * 1.05, 0.25, quantity=1)
        port = Portfolio([pos])
        assert port.value(snapshot) > 0.0

    def test_single_long_put_value_positive(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        pos = _make_position(ticker, "put", S * 0.95, 0.50, quantity=1)
        port = Portfolio([pos])
        assert port.value(snapshot) > 0.0

    def test_portfolio_value_equals_sum_of_leg_values(self, snapshot, market_data_small):
        """PFL-01: Σ(leg values) == portfolio.value()."""
        tickers = market_data_small.tickers[:2]
        positions = [
            _make_position(tickers[0], "call", float(snapshot.spot_prices[tickers[0]]) * 1.00, 0.25, 2),
            _make_position(tickers[1], "put",  float(snapshot.spot_prices[tickers[1]]) * 0.95, 0.50, 3),
            _make_position(tickers[0], "call", float(snapshot.spot_prices[tickers[0]]) * 1.05, 0.25, -1),
        ]
        port = Portfolio(positions)
        expected = sum(_price_position(p, snapshot) for p in positions)
        assert port.value(snapshot) == pytest.approx(expected, rel=1e-9)

    def test_value_from_position_summary_consistent(self, snapshot, market_data_small):
        """portfolio.value() == position_summary['position_value'].sum()."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        positions = [
            _make_position(ticker, "call", S * 1.00, 0.25, 1),
            _make_position(ticker, "put",  S * 0.95, 0.50, 2),
        ]
        port = Portfolio(positions)
        df = port.position_summary(snapshot)
        assert port.value(snapshot) == pytest.approx(df["position_value"].sum(), rel=1e-9)


# ===========================================================================
# PFL-02 / PFL-03: Portfolio Greeks aggregation
# ===========================================================================

class TestPortfolioGreeks:
    """
    PFL-02: Portfolio delta = Σ(per-share delta × signed notional_shares).
    PFL-03: Portfolio gamma = Σ(per-share gamma × signed notional_shares).
    """

    def _make_two_leg_portfolio(self, snapshot, market_data_small):
        tickers = market_data_small.tickers[:2]
        positions = [
            _make_position(tickers[0], "call", float(snapshot.spot_prices[tickers[0]]) * 1.00, 0.25, 2),
            _make_position(tickers[1], "put",  float(snapshot.spot_prices[tickers[1]]) * 0.95, 0.50, 3),
        ]
        return Portfolio(positions), positions

    def test_portfolio_delta_equals_sum_of_position_deltas(self, snapshot, market_data_small):
        """PFL-02."""
        port, positions = self._make_two_leg_portfolio(snapshot, market_data_small)
        expected_delta = sum(_delta_position(p, snapshot) for p in positions)
        result = port.greeks(snapshot)
        assert result["delta"] == pytest.approx(expected_delta, rel=1e-9)

    def test_portfolio_gamma_equals_sum_of_position_gammas(self, snapshot, market_data_small):
        """PFL-03."""
        port, positions = self._make_two_leg_portfolio(snapshot, market_data_small)
        expected_gamma = sum(_gamma_position(p, snapshot) for p in positions)
        result = port.greeks(snapshot)
        assert result["gamma"] == pytest.approx(expected_gamma, rel=1e-9)

    def test_greeks_dict_has_all_keys(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        pos = _make_position(ticker, "call", S, 0.25, 1)
        port = Portfolio([pos])
        g = port.greeks(snapshot)
        assert set(g.keys()) == {"value", "delta", "gamma", "vega", "theta"}

    def test_long_call_delta_positive(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        pos = _make_position(ticker, "call", S, 0.25, 1)
        g = Portfolio([pos]).greeks(snapshot)
        assert g["delta"] > 0.0

    def test_long_put_delta_negative(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        pos = _make_position(ticker, "put", S, 0.25, 1)
        g = Portfolio([pos]).greeks(snapshot)
        assert g["delta"] < 0.0

    def test_gamma_always_positive(self, snapshot, market_data_small):
        """Gamma is positive for both long calls and long puts."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        for opt_type in ("call", "put"):
            pos = _make_position(ticker, opt_type, S, 0.25, 1)
            g = Portfolio([pos]).greeks(snapshot)
            assert g["gamma"] > 0.0, f"Gamma not positive for long {opt_type}"


# ===========================================================================
# PFL-04: Short position negates Greeks
# ===========================================================================

class TestShortPositionNegation:
    """PFL-04: Short position has exactly negated Greeks vs identical long."""

    def test_short_call_value_negates_long_call_value(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        K, T = S * 1.05, 0.25
        long_port  = Portfolio([_make_position(ticker, "call", K, T,  1)])
        short_port = Portfolio([_make_position(ticker, "call", K, T, -1)])
        assert short_port.value(snapshot) == pytest.approx(-long_port.value(snapshot), rel=1e-9)

    def test_short_call_delta_negates_long_call_delta(self, snapshot, market_data_small):
        """PFL-04: short call delta = −long call delta."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        K, T = S * 1.05, 0.25
        long_g  = Portfolio([_make_position(ticker, "call", K, T,  1)]).greeks(snapshot)
        short_g = Portfolio([_make_position(ticker, "call", K, T, -1)]).greeks(snapshot)
        assert short_g["delta"] == pytest.approx(-long_g["delta"], rel=1e-9)

    def test_short_call_gamma_negates_long_call_gamma(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        K, T = S * 1.05, 0.25
        long_g  = Portfolio([_make_position(ticker, "call", K, T,  1)]).greeks(snapshot)
        short_g = Portfolio([_make_position(ticker, "call", K, T, -1)]).greeks(snapshot)
        assert short_g["gamma"] == pytest.approx(-long_g["gamma"], rel=1e-9)

    def test_short_call_vega_negates_long_call_vega(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        K, T = S * 1.05, 0.25
        long_g  = Portfolio([_make_position(ticker, "call", K, T,  1)]).greeks(snapshot)
        short_g = Portfolio([_make_position(ticker, "call", K, T, -1)]).greeks(snapshot)
        assert short_g["vega"] == pytest.approx(-long_g["vega"], rel=1e-9)

    def test_long_short_same_option_portfolio_value_zero(self, snapshot, market_data_small):
        """Long and short of the same option cancel to zero."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        K, T = S * 1.05, 0.25
        positions = [
            _make_position(ticker, "call", K, T,  1),
            _make_position(ticker, "call", K, T, -1),
        ]
        assert Portfolio(positions).value(snapshot) == pytest.approx(0.0, abs=1e-10)


# ===========================================================================
# PFL-05: Empty portfolio
# ===========================================================================

class TestEmptyPortfolio:
    """PFL-05: Empty portfolio has zero value and zero Greeks."""

    def test_empty_portfolio_value_is_zero(self, snapshot):
        assert Portfolio([]).value(snapshot) == 0.0

    def test_empty_portfolio_greeks_all_zero(self, snapshot):
        g = Portfolio([]).greeks(snapshot)
        for key, val in g.items():
            assert val == 0.0, f"Expected 0 for '{key}', got {val}"

    def test_empty_portfolio_summary_is_empty_dataframe(self, snapshot):
        df = Portfolio([]).position_summary(snapshot)
        assert len(df) == 0
        assert "ticker" in df.columns


# ===========================================================================
# PFL-06: Delta-neutral portfolio
# ===========================================================================

class TestDeltaNeutralPortfolio:
    """
    PFL-06: An ATM straddle (long call + long put at same K/T) has a smaller
    portfolio delta magnitude than a single call leg.

    For S = K, the call and put deltas partially cancel:
        Δ_straddle = Δ_call + Δ_put = e^{-qT}(2·N(d1) − 1)
    Since N(d1) < 1, |Δ_straddle| < Δ_call always holds.
    """

    def test_atm_straddle_delta_smaller_than_individual_call(
        self, snapshot, market_data_small
    ):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        K, T = S, 0.25  # ATM

        call_pos = _make_position(ticker, "call", K, T, 1)
        put_pos  = _make_position(ticker, "put",  K, T, 1)

        straddle = Portfolio([call_pos, put_pos])
        call_only = Portfolio([call_pos])

        straddle_delta = abs(straddle.greeks(snapshot)["delta"])
        call_delta = abs(call_only.greeks(snapshot)["delta"])

        assert straddle_delta < call_delta

    def test_atm_straddle_gamma_exceeds_individual_call(
        self, snapshot, market_data_small
    ):
        """Straddle gamma is approximately double a single-leg's gamma."""
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        K, T = S, 0.25

        call_pos = _make_position(ticker, "call", K, T, 1)
        put_pos  = _make_position(ticker, "put",  K, T, 1)

        straddle = Portfolio([call_pos, put_pos])
        call_only = Portfolio([call_pos])

        straddle_gamma = straddle.greeks(snapshot)["gamma"]
        call_gamma = call_only.greeks(snapshot)["gamma"]
        assert straddle_gamma == pytest.approx(2.0 * call_gamma, rel=1e-6)


# ===========================================================================
# PFL-07: build_from_config
# ===========================================================================

class TestBuildFromConfig:
    """PFL-07: Loading portfolio from config file produces correct positions."""

    # Expected position count from config/portfolio.yaml:
    #   Rule 1: put  0.95 3m  → 25 (all)
    #   Rule 2: put  0.85 6m  → 25 (all)
    #   Rule 3: call 1.05 3m  → 25 (all)
    #   Rule 4: call 1.00 1m  → 10 (Technology + Energy only, 5+5)
    #   Rule 5: put  1.00 1m  → 10 (Technology + Energy only)
    #   Rule 6: call 1.05 12m → 25 (all)
    #   Total                  120
    EXPECTED_POSITIONS = 120

    def test_position_count(self, full_portfolio):
        """PFL-07: build_from_config creates exactly 120 positions."""
        assert len(full_portfolio.positions) == self.EXPECTED_POSITIONS

    def test_strikes_set_as_moneyness_times_spot(self, full_portfolio, snapshot):
        """PFL-07: Each strike equals the position rule's moneyness × spot."""
        expected_moneyness = {0.85, 0.95, 1.00, 1.05}
        for pos in full_portfolio.positions:
            S = float(snapshot.spot_prices[pos.ticker])
            m = round(pos.strike / S, 2)
            assert m in expected_moneyness, (
                f"{pos.ticker}: unexpected moneyness {m:.4f} (strike={pos.strike:.2f}, S={S:.2f})"
            )

    def test_sector_filter_atm_straddle_positions(self, full_portfolio, market_data_small):
        """
        PFL-07: The ATM 1m straddle rule applies only to Technology and Energy
        (5 + 5 = 10 names × 2 option types = 20 positions).
        """
        atm_1m = [
            p for p in full_portfolio.positions
            if abs(p.tenor_years - 1 / 12) < 0.002  # 1m tenor
        ]
        assert len(atm_1m) == 20

        # Build ticker → sector map
        sector_of = {
            ticker: sector
            for sector, tickers in market_data_small.sectors.items()
            for ticker in tickers
        }
        allowed = {"Technology", "Energy"}
        for pos in atm_1m:
            sector = sector_of[pos.ticker]
            assert sector in allowed, (
                f"{pos.ticker} (sector={sector}) found in 1m ATM positions; "
                f"expected only {allowed}"
            )

    def test_all_25_underlyings_have_positions(self, full_portfolio, market_data_small):
        """Every underlying in market_data_small has at least one position."""
        port_tickers = {p.ticker for p in full_portfolio.positions}
        assert port_tickers == set(market_data_small.tickers)

    def test_short_call_overlay_positions_exist(self, full_portfolio):
        """Rule 3 (short call 1.05 3m, qty=−30) must produce negative-quantity positions."""
        short_positions = [p for p in full_portfolio.positions if p.quantity < 0]
        assert len(short_positions) > 0

    def test_short_call_delta_is_negative(self, full_portfolio, snapshot):
        """Short call positions contribute negative delta to the portfolio."""
        short_calls = [
            p for p in full_portfolio.positions
            if p.quantity < 0 and p.option_type == OptionType.CALL
        ]
        assert short_calls, "No short calls found in full_portfolio"
        sample = short_calls[0]
        port = Portfolio([sample])
        g = port.greeks(snapshot)
        assert g["delta"] < 0.0

    def test_portfolio_has_positive_total_value(self, full_portfolio, snapshot):
        """Long-biased portfolio (net long puts + calls) should have positive MTM."""
        # Net quantity across all positions is positive (more longs than shorts)
        total_value = full_portfolio.value(snapshot)
        assert total_value > 0.0

    def test_portfolio_has_negative_theta(self, full_portfolio, snapshot):
        """Long option positions decay; net-long portfolio has negative theta."""
        g = full_portfolio.greeks(snapshot)
        assert g["theta"] < 0.0


# ===========================================================================
# position_summary tests
# ===========================================================================

class TestPositionSummary:
    """Tests for Portfolio.position_summary()."""

    def test_summary_row_count_matches_position_count(self, full_portfolio, snapshot):
        df = full_portfolio.position_summary(snapshot)
        assert len(df) == len(full_portfolio.positions)

    def test_summary_contains_required_columns(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        port = Portfolio([_make_position(ticker, "call", S, 0.25, 1)])
        df = port.position_summary(snapshot)
        required = {
            "ticker", "option_type", "strike", "tenor_years",
            "quantity", "notional_per_contract",
            "spot", "moneyness", "iv",
            "price", "delta", "gamma", "vega", "theta",
            "position_value", "position_delta", "position_gamma",
            "position_vega", "position_theta",
        }
        assert required.issubset(set(df.columns))

    def test_long_position_value_positive(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        port = Portfolio([_make_position(ticker, "call", S, 0.25, 1)])
        df = port.position_summary(snapshot)
        assert df["position_value"].iloc[0] > 0.0

    def test_short_position_value_negative(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        port = Portfolio([_make_position(ticker, "call", S, 0.25, -1)])
        df = port.position_summary(snapshot)
        assert df["position_value"].iloc[0] < 0.0

    def test_iv_column_is_positive(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        port = Portfolio([_make_position(ticker, "call", S, 0.25, 1)])
        df = port.position_summary(snapshot)
        assert (df["iv"] > 0).all()

    def test_moneyness_column_correct(self, snapshot, market_data_small):
        ticker = market_data_small.tickers[0]
        S = float(snapshot.spot_prices[ticker])
        K = S * 1.05
        port = Portfolio([_make_position(ticker, "call", K, 0.25, 1)])
        df = port.position_summary(snapshot)
        assert df["moneyness"].iloc[0] == pytest.approx(K / S, rel=1e-6)


# ===========================================================================
# Performance
# ===========================================================================

class TestPerformance:
    """EDG-04: 120-position portfolio priced in < 10 s."""

    def test_full_portfolio_prices_in_time(self, full_portfolio, snapshot):
        start = time.perf_counter()
        _ = full_portfolio.position_summary(snapshot)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"position_summary took {elapsed:.2f}s (limit 10s)"
