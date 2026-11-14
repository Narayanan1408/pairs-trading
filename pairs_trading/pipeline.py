"""
Glue that runs every step for one pair, in the right order.
Both main.py (command line) and app.py (dashboard) call this.
"""
from dataclasses import dataclass

import pandas as pd

from pairs_trading.backtest import list_trades, performance_summary, run_backtest
from pairs_trading.config import Settings
from pairs_trading.strategy import (compute_spread, compute_zscore,
                                    estimate_hedge_ratio, generate_positions)


@dataclass
class PairResult:
    intercept: float
    hedge_ratio: float
    results: pd.DataFrame   # day-by-day table: prices, spread, z-score, position, equity
    trades: pd.DataFrame    # one row per round-trip trade
    summary: dict           # headline performance numbers


def run_pair(price_a: pd.Series, price_b: pd.Series, settings: Settings) -> PairResult:
    # Keep only days where BOTH stocks have a price.
    both = pd.concat([price_a, price_b], axis=1).dropna()
    price_a, price_b = both.iloc[:, 0], both.iloc[:, 1]

    # 1. Learn the relationship using ONLY the formation period.
    formation = slice(settings.formation_start, settings.formation_end)
    intercept, hedge_ratio = estimate_hedge_ratio(price_a[formation], price_b[formation])

    # 2. Spread and z-score over all dates (the rolling window needs some history
    #    before the first trading day).
    spread = compute_spread(price_a, price_b, intercept, hedge_ratio)
    zscore = compute_zscore(spread, settings.zscore_window)

    # 3. Keep only the trading period, which the model has never seen, and
    #    decide our positions in it (we start with no open trades).
    signals = pd.DataFrame({"price_a": price_a, "price_b": price_b,
                            "spread": spread, "zscore": zscore})
    signals = signals.loc[settings.trading_start:settings.trading_end].copy()
    signals["position"] = generate_positions(signals["zscore"], settings.entry_z,
                                             settings.exit_z, settings.stop_z)

    # 4. Simulate the trades and measure the result.
    results = run_backtest(signals, hedge_ratio, settings.initial_capital, settings.cost_bps)
    trades = list_trades(results)
    summary = performance_summary(results, trades)

    return PairResult(intercept, hedge_ratio, results, trades, summary)
