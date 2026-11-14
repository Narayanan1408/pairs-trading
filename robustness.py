"""
Robustness checks: did the strategy work because the idea is good, or because
we got lucky with one pair, one set of settings and one time period?

    python robustness.py

Runs five tests, prints the results and saves images/robustness.png.
  1. Settings sensitivity  - change entry / exit bands and the z-score window
  2. Realism stress test   - higher trading costs and a slower (1-day late) execution
  3. Every pair            - does passing the cointegration test predict profits?
  4. Walk-forward          - repeat the WHOLE process every year from 2013 to 2025
  5. Statistical luck      - is the Sharpe ratio distinguishable from zero?
"""
from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd

from pairs_trading import plots
from pairs_trading.backtest import TRADING_DAYS_PER_YEAR, run_backtest
from pairs_trading.config import DEFAULT_UNIVERSE, Settings
from pairs_trading.data import download_prices
from pairs_trading.pair_selection import find_cointegrated_pairs
from pairs_trading.pipeline import run_pair

PAIR = ("V", "MA")
FIRST_YEAR, LAST_YEAR = 2010, 2025


def sharpe(daily_returns: pd.Series) -> float:
    """Yearly average return divided by yearly volatility."""
    volatility = daily_returns.std()
    return daily_returns.mean() / volatility * np.sqrt(TRADING_DAYS_PER_YEAR) if volatility > 0 else 0.0


def total_return(daily_returns: pd.Series) -> float:
    return (1 + daily_returns).prod() - 1


def run(prices: pd.DataFrame, a: str, b: str, settings: Settings) -> pd.Series:
    """Daily strategy returns for one pair and one set of settings."""
    return run_pair(prices[a], prices[b], settings).results["strategy_return"]


# ---------------------------------------------------------------------------
# Test 1: settings sensitivity
# ---------------------------------------------------------------------------
def test_settings(prices: pd.DataFrame, base: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """A robust strategy should work for a RANGE of settings, not one lucky combination."""
    entries, exits = [1.5, 2.0, 2.5, 3.0], [0.0, 0.25, 0.5, 1.0]
    grid = pd.DataFrame(index=entries, columns=exits, dtype=float)
    for entry_z in entries:
        for exit_z in exits:
            grid.loc[entry_z, exit_z] = sharpe(run(prices, *PAIR, replace(base, entry_z=entry_z, exit_z=exit_z)))

    windows = [10, 20, 30, 60, 90]
    by_window = pd.Series({w: sharpe(run(prices, *PAIR, replace(base, zscore_window=w))) for w in windows})
    return grid, by_window


# ---------------------------------------------------------------------------
# Test 2: realism stress test
# ---------------------------------------------------------------------------
def test_stress(prices: pd.DataFrame, base: Settings) -> pd.DataFrame:
    """Make trading harder (more costs, slower orders) and see if profits survive."""
    rows = {}
    for cost in [0, 5, 10, 20]:
        returns = run(prices, *PAIR, replace(base, cost_bps=cost))
        rows[f"{cost} bps costs"] = (total_return(returns), sharpe(returns))

    # Act one day LATE: shift the decisions one extra day before backtesting.
    result = run_pair(prices[PAIR[0]], prices[PAIR[1]], base)
    late = result.results[["price_a", "price_b", "position"]].copy()
    late["position"] = late["position"].shift(1).fillna(0)
    returns = run_backtest(late, result.hedge_ratio, base.initial_capital, base.cost_bps)["strategy_return"]
    rows["1 day late"] = (total_return(returns), sharpe(returns))

    return pd.DataFrame(rows, index=["total_return", "sharpe"]).T


# ---------------------------------------------------------------------------
# Test 3: every pair
# ---------------------------------------------------------------------------
def test_all_pairs(prices: pd.DataFrame, base: Settings) -> pd.DataFrame:
    """If cointegration really matters, pairs that pass the test should do better later."""
    formation = prices.loc[base.formation_start:base.formation_end]
    scan = find_cointegrated_pairs(formation, DEFAULT_UNIVERSE)
    returns = [run(prices, r.stock_a, r.stock_b, base) for r in scan.itertuples()]
    scan["total_return"] = [total_return(r) for r in returns]
    scan["sharpe"] = [sharpe(r) for r in returns]
    return scan


# ---------------------------------------------------------------------------
# Test 4: walk-forward
# ---------------------------------------------------------------------------
def test_walk_forward(prices: pd.DataFrame, base: Settings, formation_years: int = 3) -> pd.DataFrame:
    """
    Repeat the whole recipe every year, like a real trader would:
      - look back 3 years, run the cointegration scan
      - trade EVERY pair that passes (p < 0.05) for the next year, money split equally
      - if no pair passes, sit in cash for that year
    """
    yearly_returns, log = [], []
    for year in range(FIRST_YEAR + formation_years, LAST_YEAR + 1):
        settings = replace(base,
                           formation_start=f"{year - formation_years}-01-01",
                           formation_end=f"{year - 1}-12-31",
                           trading_start=f"{year}-01-01",
                           trading_end=f"{year}-12-31")
        formation = prices.loc[settings.formation_start:settings.formation_end]
        chosen = find_cointegrated_pairs(formation, DEFAULT_UNIVERSE).query("cointegrated")

        trading_days = prices.loc[settings.trading_start:settings.trading_end].index
        if chosen.empty:
            daily = pd.Series(0.0, index=trading_days)
        else:
            pair_returns = [run(prices, r.stock_a, r.stock_b, settings) for r in chosen.itertuples()]
            daily = pd.concat(pair_returns, axis=1).mean(axis=1).reindex(trading_days).fillna(0)

        yearly_returns.append(daily)
        log.append({"year": year,
                    "pairs traded": ", ".join(chosen["stock_a"] + "/" + chosen["stock_b"]) or "(cash)",
                    "return": total_return(daily)})

    daily = pd.concat(yearly_returns)
    return pd.DataFrame(log), daily


# ---------------------------------------------------------------------------
# Test 5: could the result be luck?
# ---------------------------------------------------------------------------
def t_statistic(daily_returns: pd.Series) -> float:
    """
    t-stat of the average daily return. Roughly: Sharpe x sqrt(years).
    Above ~2 means "unlikely to be luck" at the usual 95% confidence.
    """
    return daily_returns.mean() / daily_returns.std() * np.sqrt(len(daily_returns))


def main() -> None:
    base = Settings()
    tickers = sorted({t for group in DEFAULT_UNIVERSE.values() for t in group})
    print(f"Downloading {len(tickers)} tickers, {FIRST_YEAR}-{LAST_YEAR}...")
    prices = download_prices(tickers, f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31")
    pd.set_option("display.width", 120)

    grid, by_window = test_settings(prices, base)
    print("\n1. SETTINGS SENSITIVITY: Sharpe ratio of V/MA (rows = entry z, columns = exit z)")
    print(grid.round(2).to_string())
    print("\n   Sharpe by z-score window (days):")
    print(by_window.round(2).to_string())

    stress = test_stress(prices, base)
    print("\n2. REALISM STRESS TEST (V/MA)")
    print(stress.to_string(formatters={"total_return": "{:.1%}".format, "sharpe": "{:.2f}".format}))

    pairs = test_all_pairs(prices, base)
    print("\n3. EVERY PAIR: cointegrated in 2018-2021 -> traded 2022-2025")
    print(pairs[["stock_a", "stock_b", "p_value", "cointegrated", "total_return", "sharpe"]]
          .to_string(index=False, float_format="{:.3f}".format))
    print("\n   Average Sharpe, passed the test:", round(pairs.query("cointegrated")["sharpe"].mean(), 2),
          "| failed the test:", round(pairs.query("not cointegrated")["sharpe"].mean(), 2))

    walk_log, walk_daily = test_walk_forward(prices, base)
    print("\n4. WALK-FORWARD: re-select pairs every year (3-year look-back)")
    print(walk_log.to_string(index=False, formatters={"return": "{:.1%}".format}))
    print(f"   Overall: total {total_return(walk_daily):.1%}, Sharpe {sharpe(walk_daily):.2f}")

    headline = run(prices, *PAIR, base)
    print("\n5. LUCK CHECK (t-statistic, above ~2 = probably not luck)")
    print(f"   V/MA 2022-2025:       {t_statistic(headline):.2f}")
    print(f"   Walk-forward 2013-25: {t_statistic(walk_daily):.2f}")

    fig = plots.plot_robustness(grid, by_window, stress, pairs, walk_daily, base)
    fig.savefig("images/robustness.png", dpi=150)
    print("\nChart saved to images/robustness.png")


if __name__ == "__main__":
    main()
