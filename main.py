"""
main.py
-------
CLI entry point for the Historical VaR model.

Phase 1 pipeline:
    1. Generate synthetic market data (Mode A)
    2. Build equity options portfolio from config
    3. Compute HVaR and ES at 95% and 99%
    4. Run backtesting (Kupiec, Christoffersen, traffic light)
    5. Generate reports (P&L distribution, VaR time series, exceptions, Greeks)

Usage:
    python main.py
    python main.py --config config/model.yaml --portfolio config/portfolio.yaml

TODO (Phase 2): add --mode real flag to switch to yfinance data ingestion
TODO (Phase 2): add --stress flag to run Stress VaR
TODO (Phase 2): add --factor flag to run factor VaR decomposition
"""

from __future__ import annotations

import argparse
import os
import time

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Historical VaR model for an equities options portfolio."
    )
    parser.add_argument(
        "--config",
        default="config/model.yaml",
        help="Path to model configuration YAML (default: config/model.yaml)",
    )
    parser.add_argument(
        "--portfolio",
        default="config/portfolio.yaml",
        help="Path to portfolio configuration YAML (default: config/portfolio.yaml)",
    )
    args = parser.parse_args()

    # --- Load configuration ---
    config = load_config(args.config)
    portfolio_config = load_config(args.portfolio)

    seed = config["simulation"].get("seed", 42)
    print(f"[1/5] Generating synthetic market data  (seed={seed}) ...")

    # --- Step 1: Generate market data ---
    t0 = time.perf_counter()
    from data.generator import MarketDataGenerator
    from data.schemas import validate_market_data

    generator = MarketDataGenerator(config=config, seed=seed)
    market_data = generator.generate()
    validate_market_data(market_data)

    elapsed = time.perf_counter() - t0
    print(
        f"    Generated {market_data.n_days} trading days × "
        f"{market_data.n_assets} underlyings  ({elapsed:.2f}s)"
    )
    print(
        f"    Crisis window: days {market_data.crisis_start_idx}–"
        f"{market_data.crisis_end_idx}"
    )

    # --- Step 2: Build portfolio ---
    print("[2/5] Building portfolio from config ...")
    t0 = time.perf_counter()

    from portfolio.portfolio import MarketSnapshot, build_from_config

    portfolio = build_from_config(portfolio_config, market_data, date_idx=-1)
    snapshot = MarketSnapshot.from_market_data(market_data, date_idx=-1)

    elapsed = time.perf_counter() - t0
    print(
        f"    {len(portfolio.positions)} positions across "
        f"{market_data.n_assets} underlyings  ({elapsed:.2f}s)"
    )
    print(f"    Portfolio value: {portfolio.value(snapshot):,.0f}")

    # --- Step 3: Compute VaR and ES ---
    print("[3/5] Computing Historical VaR and ES ...")
    t0 = time.perf_counter()

    from var.historical import compute_historical_var

    lookback = config["var"]["lookback_window"]
    var_result = compute_historical_var(portfolio, snapshot, market_data, lookback)

    elapsed = time.perf_counter() - t0
    print(
        f"    VaR 99%: {var_result.var_99:>12,.0f}  |  "
        f"VaR 95%: {var_result.var_95:>12,.0f}  ({elapsed:.2f}s)"
    )
    print(
        f"    ES  99%: {var_result.es_99:>12,.0f}  |  "
        f"ES  95%: {var_result.es_95:>12,.0f}"
    )

    # --- Step 4: Run backtesting ---
    print("[4/5] Running backtesting ...")
    t0 = time.perf_counter()

    from backtest.engine import run_backtest
    from backtest.statistics import traffic_light

    backtest_window = config["backtest"]["window_days"]
    bt_result = run_backtest(portfolio, market_data, lookback, backtest_window)

    elapsed = time.perf_counter() - t0
    light99 = traffic_light(bt_result.n_exceptions_99)
    light95 = traffic_light(bt_result.n_exceptions_95)
    print(
        f"    {bt_result.n_backtest_days}-day backtest  ({elapsed:.1f}s)"
    )
    print(
        f"    Exceptions 99%: {bt_result.n_exceptions_99}  [{light99.upper()}]  |  "
        f"Exceptions 95%: {bt_result.n_exceptions_95}  [{light95.upper()}]"
    )

    # --- Step 5: Generate reports ---
    output_dir = config.get("reporting", {}).get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[5/5] Generating reports to {output_dir}/ ...")

    from reporting.report import generate_report

    generate_report(
        var_result, bt_result, portfolio, snapshot,
        output_dir, config.get("reporting", {}),
    )
    print(f"    Charts and summary saved to {output_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
