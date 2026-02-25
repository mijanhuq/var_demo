"""
reporting/report.py
--------------------
Matplotlib charts and formatted summary table for the Historical VaR model.

Public API
----------
    generate_report         — orchestrates all 4 charts + summary table
    plot_pnl_distribution   — histogram of VaR scenario P&Ls
    plot_var_timeseries     — rolling VaR vs realised P&L over backtest window
    plot_exceptions         — scatter of exception days
    plot_greeks             — horizontal bar chart of portfolio Greeks
    print_summary_table     — formatted text summary to stdout + file
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for CLI)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from backtest.engine import BacktestResult
from backtest.statistics import (
    christoffersen_independence_test,
    exception_rate,
    kupiec_pof_test,
    traffic_light,
)
from portfolio.portfolio import MarketSnapshot, Portfolio
from var.historical import VaRResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(
    fig: plt.Figure,
    path: str,
    reporting_cfg: dict,
) -> None:
    """Save figure to disk using reporting config, then close."""
    if reporting_cfg.get("save_figures", True):
        fmt = reporting_cfg.get("figure_format", "png")
        dpi = reporting_cfg.get("dpi", 150)
        fig.savefig(path + "." + fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 1: P&L Distribution
# ---------------------------------------------------------------------------

def plot_pnl_distribution(
    var_result: VaRResult,
    output_dir: str,
    reporting_cfg: dict,
) -> None:
    """
    Histogram of the 252 scenario P&Ls with VaR/ES marker lines.

    Saved as ``pnl_distribution.<fmt>`` in ``output_dir``.
    """
    pnl = var_result.pnl_vector

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pnl, bins=40, color="#5b9bd5", edgecolor="white", linewidth=0.4,
            label="Scenario P&L")

    ax.axvline(-var_result.var_95, color="#e6a817", linestyle="--", linewidth=1.5,
               label=f"−VaR 95%  {var_result.var_95:,.0f}")
    ax.axvline(-var_result.var_99, color="#c00000", linestyle="--", linewidth=1.5,
               label=f"−VaR 99%  {var_result.var_99:,.0f}")
    ax.axvline(-var_result.es_99, color="#7b0000", linestyle=":",  linewidth=1.5,
               label=f"−ES  99%  {var_result.es_99:,.0f}")

    ax.set_xlabel("1-Day P&L ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Scenario P&L Distribution  (252 historical scenarios)")
    ax.legend(fontsize=9)
    fig.tight_layout()

    _save_or_show(fig, os.path.join(output_dir, "pnl_distribution"), reporting_cfg)


# ---------------------------------------------------------------------------
# Chart 2: VaR Time Series
# ---------------------------------------------------------------------------

def plot_var_timeseries(
    backtest_result: BacktestResult,
    output_dir: str,
    reporting_cfg: dict,
) -> None:
    """
    Rolling 1-day VaR (99% and 95%) vs realised P&L over the backtest window.

    Exception days are marked with red triangles.
    Saved as ``var_timeseries.<fmt>`` in ``output_dir``.
    """
    dates = backtest_result.dates
    pnl = backtest_result.actual_pnl
    var99 = backtest_result.var_estimates_99
    var95 = backtest_result.var_estimates_95
    exc99 = backtest_result.exceptions_99

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dates, pnl, color="#888888", linewidth=0.8, label="Realised P&L", zorder=2)
    ax.plot(dates, -var95, color="#e6a817", linestyle="--", linewidth=1.2,
            label="−VaR 95%", zorder=3)
    ax.plot(dates, -var99, color="#c00000", linestyle="--", linewidth=1.2,
            label="−VaR 99%", zorder=3)

    # Mark 99% exception days
    exc_dates = [d for d, e in zip(dates, exc99) if e]
    exc_pnl   = pnl[exc99]
    if len(exc_dates) > 0:
        ax.scatter(exc_dates, exc_pnl, color="#c00000", marker="v", s=40,
                   zorder=5, label=f"Exception 99% ({int(exc99.sum())})")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    ax.set_xlabel("Date")
    ax.set_ylabel("P&L ($)")
    ax.set_title("Rolling 1-Day VaR vs Realised P&L")
    ax.legend(fontsize=9)
    fig.tight_layout()

    _save_or_show(fig, os.path.join(output_dir, "var_timeseries"), reporting_cfg)


# ---------------------------------------------------------------------------
# Chart 3: Exception Scatter
# ---------------------------------------------------------------------------

def plot_exceptions(
    backtest_result: BacktestResult,
    output_dir: str,
    reporting_cfg: dict,
) -> None:
    """
    Scatter plot of realised P&L with exception days highlighted.

    99% exceptions: red.  95%-only exceptions (not breaching 99%): amber.
    Saved as ``exceptions.<fmt>`` in ``output_dir``.
    """
    dates = backtest_result.dates
    pnl = backtest_result.actual_pnl
    exc99 = backtest_result.exceptions_99
    exc95 = backtest_result.exceptions_95

    exc95_only = exc95 & ~exc99          # breaches 95% but not 99%
    normal = ~exc95                      # no breach

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter([d for d, m in zip(dates, normal) if m],
               pnl[normal],
               color="#aaaaaa", s=8, zorder=2, label="No exception")

    if exc95_only.any():
        ax.scatter([d for d, m in zip(dates, exc95_only) if m],
                   pnl[exc95_only],
                   color="#e6a817", s=20, zorder=3,
                   label=f"95% exception ({int(exc95_only.sum())})")

    if exc99.any():
        ax.scatter([d for d, m in zip(dates, exc99) if m],
                   pnl[exc99],
                   color="#c00000", s=25, marker="v", zorder=4,
                   label=f"99% exception ({int(exc99.sum())})")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(pnl.mean(), color="#5b9bd5", linewidth=0.8, linestyle=":",
               label=f"Mean P&L {pnl.mean():,.0f}")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    ax.set_xlabel("Date")
    ax.set_ylabel("Realised P&L ($)")
    ax.set_title("VaR Exception Days")
    ax.legend(fontsize=9)
    fig.tight_layout()

    _save_or_show(fig, os.path.join(output_dir, "exceptions"), reporting_cfg)


# ---------------------------------------------------------------------------
# Chart 4: Portfolio Greeks
# ---------------------------------------------------------------------------

def plot_greeks(
    portfolio: Portfolio,
    snapshot: MarketSnapshot,
    output_dir: str,
    reporting_cfg: dict,
) -> None:
    """
    Horizontal bar chart of portfolio-level Greeks at the valuation date.

    Greeks are scaled to human-readable units:
        Delta — as-is ($ per $1 spot move)
        Gamma — ×1000 ($ per $1000 spot move²)
        Vega  — ÷100  ($ per 1% vol move)
        Theta — as-is, annualised ($ per year)

    Saved as ``greeks.<fmt>`` in ``output_dir``.
    """
    g = portfolio.greeks(snapshot)

    labels = ["Delta", "Gamma ×1000", "Vega ÷100", "Theta"]
    values = [
        g["delta"],
        g["gamma"] * 1000.0,
        g["vega"]  / 100.0,
        g["theta"],
    ]
    colours = ["#5b9bd5" if v >= 0 else "#c00000" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=colours, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.bar_label(bars, fmt="{:,.1f}", padding=4, fontsize=9)

    ax.set_xlabel("Scaled value ($)")
    ax.set_title(f"Portfolio Greeks  (valuation: {snapshot.date})")
    fig.tight_layout()

    _save_or_show(fig, os.path.join(output_dir, "greeks"), reporting_cfg)


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------

def print_summary_table(
    var_result: VaRResult,
    backtest_result: BacktestResult,
    portfolio: Portfolio,
    snapshot: MarketSnapshot,
    output_dir: str,
) -> None:
    """
    Print a formatted VaR summary to stdout and write it to
    ``output_dir/summary.txt``.
    """
    g = portfolio.greeks(snapshot)

    n99 = backtest_result.n_exceptions_99
    n95 = backtest_result.n_exceptions_95
    n_days = backtest_result.n_backtest_days
    rate99 = exception_rate(backtest_result.actual_pnl, backtest_result.var_estimates_99)
    rate95 = exception_rate(backtest_result.actual_pnl, backtest_result.var_estimates_95)
    light99 = traffic_light(n99)
    light95 = traffic_light(n95)

    kup99 = kupiec_pof_test(n99, n_days, confidence=0.99)
    kup95 = kupiec_pof_test(n95, n_days, confidence=0.95)
    chr99 = christoffersen_independence_test(backtest_result.exceptions_99.astype(int))

    lines = [
        "=" * 55,
        "  HISTORICAL VAR SUMMARY",
        "=" * 55,
        f"  Valuation date   : {snapshot.date}",
        f"  Portfolio legs   : {len(portfolio.positions)}",
        f"  Portfolio value  : {var_result.base_value:>12,.0f}",
        "",
        "  --- VaR / ES  (1-day, full revaluation, 252 scenarios) ---",
        f"  VaR  95%  :  {var_result.var_95:>12,.0f}",
        f"  VaR  99%  :  {var_result.var_99:>12,.0f}",
        f"  ES   95%  :  {var_result.es_95:>12,.0f}",
        f"  ES   99%  :  {var_result.es_99:>12,.0f}",
        "",
        f"  --- Backtest  ({n_days} days) ---",
        f"  Exceptions 99%  :  {n99:>4d}  ({rate99:.1%})  [{light99.upper()}]",
        f"  Exceptions 95%  :  {n95:>4d}  ({rate95:.1%})  [{light95.upper()}]",
        f"  Kupiec  99%  p  :  {kup99.p_value:.4f}  "
        f"{'[REJECT]' if kup99.reject else '[pass]  '}",
        f"  Kupiec  95%  p  :  {kup95.p_value:.4f}  "
        f"{'[REJECT]' if kup95.reject else '[pass]  '}",
        f"  Christoffersen 99% p : {chr99.p_value:.4f}  "
        f"{'[REJECT]' if chr99.reject else '[pass]  '}",
        "",
        "  --- Portfolio Greeks ---",
        f"  Delta  :  {g['delta']:>14,.1f}",
        f"  Gamma  :  {g['gamma']:>14.5f}",
        f"  Vega   :  {g['vega'] / 100.0:>14,.1f}  (per 1% vol move)",
        f"  Theta  :  {g['theta']:>14,.1f}  (annualised)",
        "=" * 55,
    ]

    text = "\n".join(lines)
    print(text)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(text + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    var_result: VaRResult,
    backtest_result: BacktestResult,
    portfolio: Portfolio,
    snapshot: MarketSnapshot,
    output_dir: str,
    reporting_cfg: Optional[dict] = None,
) -> None:
    """
    Generate all Phase 1 output: 4 charts + 1 summary table.

    Output files written to ``output_dir/``:
        pnl_distribution.<fmt>
        var_timeseries.<fmt>
        exceptions.<fmt>
        greeks.<fmt>
        summary.txt

    Parameters
    ----------
    var_result : VaRResult
        Output of compute_historical_var.
    backtest_result : BacktestResult
        Output of run_backtest.
    portfolio : Portfolio
        The option portfolio.
    snapshot : MarketSnapshot
        Current market snapshot (valuation date).
    output_dir : str
        Directory to write output files.  Created if it does not exist.
    reporting_cfg : dict, optional
        Sub-dict from model.yaml ``reporting`` section.
        Keys: save_figures, figure_format, dpi.
    """
    if reporting_cfg is None:
        reporting_cfg = {}

    os.makedirs(output_dir, exist_ok=True)

    plot_pnl_distribution(var_result, output_dir, reporting_cfg)
    plot_var_timeseries(backtest_result, output_dir, reporting_cfg)
    plot_exceptions(backtest_result, output_dir, reporting_cfg)
    plot_greeks(portfolio, snapshot, output_dir, reporting_cfg)
    print_summary_table(var_result, backtest_result, portfolio, snapshot, output_dir)
