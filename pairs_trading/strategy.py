"""
Step 2 of the strategy: turn two price series into trading signals.

    prices  ->  hedge ratio  ->  spread  ->  z-score  ->  positions
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def estimate_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> tuple[float, float]:
    """
    Fit the straight line   log(A) = intercept + hedge_ratio * log(B)
    with ordinary least squares (OLS) regression.

    hedge_ratio says how much A tends to move when B moves 1%.
    """
    x = sm.add_constant(np.log(price_b))
    model = sm.OLS(np.log(price_a), x).fit()
    intercept, hedge_ratio = model.params.iloc[0], model.params.iloc[1]
    return intercept, hedge_ratio


def compute_spread(price_a: pd.Series, price_b: pd.Series,
                   intercept: float, hedge_ratio: float) -> pd.Series:
    """
    The spread is whatever is left after removing the "normal" relationship:

        spread = log(A) - (intercept + hedge_ratio * log(B))

    When the two stocks are in sync, the spread sits near 0.
    """
    return np.log(price_a) - (intercept + hedge_ratio * np.log(price_b))


def compute_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    How unusual is today's spread?

        z = (spread - recent average) / recent standard deviation

    z = +2 means the spread is 2 standard deviations ABOVE normal (A looks
    expensive compared to B). Only past days are used (a rolling window),
    so the strategy never peeks into the future.
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    return (spread - rolling_mean) / rolling_std


def generate_positions(zscore: pd.Series, entry_z: float,
                       exit_z: float, stop_z: float) -> pd.Series:
    """
    Decide, day by day, what we hold:

      +1  long the spread  = buy A, sell B  (spread unusually LOW,  bet it rises)
      -1  short the spread = sell A, buy B  (spread unusually HIGH, bet it falls)
       0  no position

    Rules
      flat  and  -stop_z < z < -entry_z   ->  open long  (+1)
      flat  and   entry_z < z <  stop_z   ->  open short (-1)
      long  and  z > -exit_z              ->  close: spread is back to normal (take profit)
      short and  z <  exit_z              ->  close: spread is back to normal (take profit)
      long  and  z < -stop_z              ->  close: spread kept stretching (stop loss)
      short and  z >  stop_z              ->  close: spread kept stretching (stop loss)
    """
    positions = []
    current = 0

    for z in zscore:
        if np.isnan(z):
            current = 0
        elif current == 0:
            if -stop_z < z < -entry_z:
                current = 1
            elif entry_z < z < stop_z:
                current = -1
        elif current == 1 and (z > -exit_z or z < -stop_z):
            current = 0
        elif current == -1 and (z < exit_z or z > stop_z):
            current = 0
        positions.append(current)

    return pd.Series(positions, index=zscore.index, name="position")
