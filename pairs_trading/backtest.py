"""
Step 3 of the strategy: pretend we traded the signals and measure the result.
"""
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def run_backtest(signals: pd.DataFrame, hedge_ratio: float,
                 initial_capital: float, cost_bps: float) -> pd.DataFrame:
    """
    Add daily returns and an account balance ("equity") to the signals table.

    Holding the spread long means: put money into A and short-sell
    hedge_ratio times as much of B, scaled so the trade uses $1 in total:

        spread_return = (return_A - hedge_ratio * return_B) / (1 + |hedge_ratio|)
    """
    df = signals.copy()
    return_a = df["price_a"].pct_change()
    return_b = df["price_b"].pct_change()

    df["spread_return"] = (return_a - hedge_ratio * return_b) / (1 + abs(hedge_ratio))

    # We decide at today's close, so today's decision only earns TOMORROW's move.
    # shift(1) pushes positions forward one day to avoid "look-ahead bias".
    held = df["position"].shift(1).fillna(0)

    # Every change in position costs money (broker fees + bid/ask spread).
    traded = df["position"].diff().abs().fillna(0)
    costs = traded * cost_bps / 10_000

    df["strategy_return"] = (held * df["spread_return"]).fillna(0) - costs
    df["equity"] = initial_capital * (1 + df["strategy_return"]).cumprod()

    # Benchmark: simply buy both stocks 50/50 and hold them.
    benchmark_return = (0.5 * return_a + 0.5 * return_b).fillna(0)
    df["benchmark_equity"] = initial_capital * (1 + benchmark_return).cumprod()
    return df


def list_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the day-by-day positions into a list of round-trip trades."""
    trades = []
    entry_date = None
    previous = 0

    for date, position in df["position"].items():
        if position != previous:
            if previous != 0:
                trades.append(_describe_trade(df, entry_date, date, previous))
            entry_date = date if position != 0 else None
        previous = position

    if previous != 0:  # a trade is still open on the last day
        trades.append(_describe_trade(df, entry_date, df.index[-1], previous))

    return pd.DataFrame(trades, columns=["entry_date", "exit_date", "direction",
                                         "days_held", "return"])


def _describe_trade(df: pd.DataFrame, entry_date, exit_date, direction: int) -> dict:
    """Summarise one trade: when it opened/closed and how much it made."""
    daily_returns = df.loc[entry_date:exit_date, "strategy_return"]
    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "direction": "Long spread" if direction == 1 else "Short spread",
        "days_held": len(daily_returns) - 1,
        "return": (1 + daily_returns).prod() - 1,
    }


def performance_summary(df: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """The headline numbers people look at when judging a strategy."""
    returns = df["strategy_return"]
    years = len(returns) / TRADING_DAYS_PER_YEAR

    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
    annual_return = (1 + total_return) ** (1 / years) - 1
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Sharpe ratio = average yearly return per unit of risk (higher is better, >1 is good).
    sharpe = returns.mean() * TRADING_DAYS_PER_YEAR / annual_volatility if annual_volatility > 0 else 0.0

    drawdown = df["equity"] / df["equity"].cummax() - 1

    return {
        "Total return": total_return,
        "Annual return": annual_return,
        "Annual volatility": annual_volatility,
        "Sharpe ratio": sharpe,
        "Max drawdown": drawdown.min(),
        "Number of trades": len(trades),
        "Win rate": (trades["return"] > 0).mean() if len(trades) else 0.0,
        "Time in market": (df["position"] != 0).mean(),
        "Buy & hold return": df["benchmark_equity"].iloc[-1] / df["benchmark_equity"].iloc[0] - 1,
    }
