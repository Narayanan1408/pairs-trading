"""
Charts that tell the story of the strategy. Used by main.py and the dashboard.
"""
import matplotlib

matplotlib.use("Agg")  # draw to files/memory instead of opening windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

from pairs_trading.config import Settings

COLOR_A = "#2563eb"      # blue
COLOR_B = "#f59e0b"      # amber
COLOR_LONG = "#16a34a"   # green
COLOR_SHORT = "#dc2626"  # red
COLOR_GREY = "#6b7280"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def plot_pair_scan(scan: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Bar chart of cointegration p-values: shorter bar = more tightly linked pair."""
    top = scan.head(top_n).iloc[::-1]  # reverse so the best pair is drawn at the top
    labels = top["stock_a"] + " / " + top["stock_b"]
    colors = [COLOR_LONG if ok else COLOR_GREY for ok in top["cointegrated"]]

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(top) + 1.2))
    ax.barh(labels, top["p_value"], color=colors)
    ax.axvline(0.05, color=COLOR_SHORT, linestyle="--", label="0.05 significance line")
    ax.set_xlabel("Cointegration p-value (lower = stronger link)")
    ax.set_title("Which pairs move together? (formation period)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_hedge_fit(price_a: pd.Series, price_b: pd.Series, name_a: str, name_b: str,
                   intercept: float, hedge_ratio: float) -> plt.Figure:
    """Scatter of log prices with the fitted regression line (where the hedge ratio comes from)."""
    log_a, log_b = np.log(price_a), np.log(price_b)
    line_x = np.linspace(log_b.min(), log_b.max(), 100)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(log_b, log_a, s=6, alpha=0.4, color=COLOR_A, label="One trading day")
    ax.plot(line_x, intercept + hedge_ratio * line_x, color=COLOR_SHORT, linewidth=2,
            label=f"log({name_a}) = {intercept:.2f} + {hedge_ratio:.2f} x log({name_b})")
    ax.set_xlabel(f"log price of {name_b}")
    ax.set_ylabel(f"log price of {name_a}")
    ax.set_title("The 'normal' relationship between the two stocks")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_strategy(df: pd.DataFrame, trades: pd.DataFrame, name_a: str, name_b: str,
                  settings: Settings) -> plt.Figure:
    """
    Three stacked panels that read top to bottom like a story:
      1. the two prices          (do they move together?)
      2. the z-score + trades    (when did we buy/sell and why?)
      3. the account balance     (did it make money?)
    """
    fig, (ax_price, ax_z, ax_eq) = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1.3, 1]})

    # --- Panel 1: both prices rescaled to start at 100 so they're comparable
    ax_price.plot(df.index, 100 * df["price_a"] / df["price_a"].iloc[0], color=COLOR_A, label=name_a)
    ax_price.plot(df.index, 100 * df["price_b"] / df["price_b"].iloc[0], color=COLOR_B, label=name_b)
    ax_price.set_title(f"1. Prices of {name_a} and {name_b} (both start at 100)", loc="left")

    # --- Panel 2: z-score with the trading bands and shaded trades
    ax_z.plot(df.index, df["zscore"], color="black", linewidth=0.9, label="Spread z-score")
    for level, style, label in [(settings.entry_z, "--", "Entry"),
                                (settings.exit_z, ":", "Exit"),
                                (settings.stop_z, "-.", "Stop loss")]:
        ax_z.axhline(level, color=COLOR_GREY, linestyle=style, linewidth=1, label=f"{label} (+/-{level})")
        ax_z.axhline(-level, color=COLOR_GREY, linestyle=style, linewidth=1)

    # Shade the days we were in a trade. The x-axis "transform" makes 0..1 mean
    # "bottom to top of the panel", so the shading always fills the full height.
    full_height = ax_z.get_xaxis_transform()
    ax_z.fill_between(df.index, 0, 1, where=df["position"] == 1, transform=full_height,
                      color=COLOR_LONG, alpha=0.15, label=f"Long spread\n(buy {name_a}, sell {name_b})")
    ax_z.fill_between(df.index, 0, 1, where=df["position"] == -1, transform=full_height,
                      color=COLOR_SHORT, alpha=0.15, label=f"Short spread\n(sell {name_a}, buy {name_b})")

    if len(trades):
        ax_z.scatter(trades["entry_date"], df.loc[trades["entry_date"], "zscore"],
                     marker="o", color="black", zorder=3, s=25, label="Trade opened")
        ax_z.scatter(trades["exit_date"], df.loc[trades["exit_date"], "zscore"],
                     marker="x", color="black", zorder=3, s=35, label="Trade closed")
    ax_z.set_title("2. How stretched is the spread? (we trade when it crosses the bands)", loc="left")

    # --- Panel 3: account value of the strategy vs simply holding both stocks
    ax_eq.plot(df.index, df["equity"], color=COLOR_LONG, linewidth=1.8, label="Pairs strategy")
    ax_eq.plot(df.index, df["benchmark_equity"], color=COLOR_GREY, linewidth=1,
               linestyle="--", label=f"Buy & hold 50% {name_a} / 50% {name_b}")
    ax_eq.axhline(settings.initial_capital, color="black", linewidth=0.6)
    ax_eq.set_title(f"3. Account value (started with ${settings.initial_capital:,.0f})", loc="left")
    ax_eq.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))

    # Legends go outside the plots, on the right, so they never hide the data.
    for ax in (ax_price, ax_z, ax_eq):
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)

    fig.suptitle(f"Pairs trading {name_a} vs {name_b}: out-of-sample backtest",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_robustness(grid: pd.DataFrame, by_window: pd.Series, stress: pd.DataFrame,
                    pairs: pd.DataFrame, walk_daily: pd.Series, settings: Settings) -> plt.Figure:
    """One page summarising every robustness test (see robustness.py)."""
    fig = plt.figure(figsize=(15, 9.5))
    layout = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

    # --- Test 1a: Sharpe for each entry/exit combination (a "heatmap")
    ax = fig.add_subplot(layout[0, 0])
    image = ax.imshow(grid.values.astype(float), cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(grid.columns)), [f"{c:g}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{r:g}" for r in grid.index])
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            ax.text(col, row, f"{grid.iat[row, col]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("Exit band (z)")
    ax.set_ylabel("Entry band (z)")
    ax.set_title("1. Sharpe ratio for different bands (V/MA)", loc="left")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046)

    # --- Test 1b: Sharpe for each z-score window
    ax = fig.add_subplot(layout[0, 1])
    colors = [COLOR_LONG if v > 0 else COLOR_SHORT for v in by_window]
    ax.bar([str(w) for w in by_window.index], by_window.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Z-score window (days)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("1. Sharpe for different look-back windows", loc="left")

    # --- Test 2: realism stress test
    ax = fig.add_subplot(layout[0, 2])
    colors = [COLOR_LONG if v > 0 else COLOR_SHORT for v in stress["total_return"]]
    ax.barh(stress.index[::-1], 100 * stress["total_return"].values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Total return 2022-2025 (%)")
    ax.set_title("2. Higher costs, slower orders (V/MA)", loc="left")

    # --- Test 3: does the cointegration p-value predict later returns?
    ax = fig.add_subplot(layout[1, 0])
    colors = [COLOR_LONG if ok else COLOR_GREY for ok in pairs["cointegrated"]]
    ax.scatter(pairs["p_value"], 100 * pairs["total_return"], c=colors, s=40, zorder=3)
    for row in pairs.itertuples():
        ax.annotate(f"{row.stock_a}/{row.stock_b}", (row.p_value, 100 * row.total_return),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.axvline(0.05, color=COLOR_SHORT, linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cointegration p-value, 2018-2021 (left of red line = passed)")
    ax.set_ylabel("Return 2022-2025 (%)")
    ax.set_title("3. Every pair: test result vs later profit", loc="left")

    # --- Test 4: walk-forward account value
    ax = fig.add_subplot(layout[1, 1:])
    equity = settings.initial_capital * (1 + walk_daily).cumprod()
    ax.plot(equity.index, equity, color=COLOR_LONG, linewidth=1.6)
    ax.axhline(settings.initial_capital, color="black", linewidth=0.6)
    for year in sorted(set(walk_daily.index.year))[1:]:
        ax.axvline(pd.Timestamp(f"{year}-01-01"), color=COLOR_GREY, linewidth=0.4, alpha=0.5)
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.set_title("4. Walk-forward: re-pick pairs every year from the previous 3 years, then trade them",
                 loc="left")

    fig.suptitle("Robustness checks", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
