"""
Tests that run on made-up (synthetic) prices, so they need no internet.

Run them with:   pytest
"""
import numpy as np
import pandas as pd

from pairs_trading.backtest import list_trades, run_backtest
from pairs_trading.pair_selection import find_cointegrated_pairs
from pairs_trading.strategy import estimate_hedge_ratio, generate_positions


def make_fake_pair(hedge_ratio: float = 1.5, days: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Two fake stocks that are cointegrated by construction."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=days)

    log_b = 4 + np.cumsum(rng.normal(0, 0.01, days))   # B wanders randomly
    noise = np.zeros(days)
    for t in range(1, days):                           # noise that keeps pulling back to 0
        noise[t] = 0.9 * noise[t - 1] + rng.normal(0, 0.01)
    log_a = 0.5 + hedge_ratio * log_b + noise          # A follows B, plus that noise

    return pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=dates)


def test_cointegration_test_finds_the_linked_pair():
    prices = make_fake_pair()
    scan = find_cointegrated_pairs(prices, {"fake": ["A", "B"]})
    assert scan.loc[0, "p_value"] < 0.05


def test_hedge_ratio_is_recovered():
    prices = make_fake_pair(hedge_ratio=1.5)
    _, hedge_ratio = estimate_hedge_ratio(prices["A"], prices["B"])
    assert abs(hedge_ratio - 1.5) < 0.1


def test_position_rules():
    z = pd.Series([0.0, -2.5, -1.0, -0.2, 0.0, 2.5, 1.0, 0.2, 2.5, 4.5])
    positions = generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    #           flat, open long, hold, take profit, flat, open short, hold, take profit, short, stop loss
    expected = [0,    1,         1,    0,           0,    -1,         -1,   0,           -1,    0]
    assert positions.tolist() == expected


def test_no_look_ahead_and_costs_are_charged():
    prices = make_fake_pair(days=5)
    signals = pd.DataFrame({"price_a": prices["A"], "price_b": prices["B"],
                            "position": [0, 1, 1, 0, 0]})
    result = run_backtest(signals, hedge_ratio=1.5, initial_capital=100, cost_bps=10)

    # Day we open the trade: we have not earned anything yet, we only pay the cost.
    assert np.isclose(result["strategy_return"].iloc[1], -0.001)
    # Next day: we earn that day's spread return.
    assert np.isclose(result["strategy_return"].iloc[2], result["spread_return"].iloc[2])


def test_trades_are_listed():
    prices = make_fake_pair(days=6)
    signals = pd.DataFrame({"price_a": prices["A"], "price_b": prices["B"],
                            "position": [0, 1, 1, 0, -1, 0]})
    trades = list_trades(run_backtest(signals, 1.5, 100, 0))
    assert trades["direction"].tolist() == ["Long spread", "Short spread"]
    assert trades["days_held"].tolist() == [2, 1]
