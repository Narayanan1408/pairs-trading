"""
Step 1 of the strategy: find two stocks whose prices are "tied together".

We use the Engle-Granger cointegration test. In plain English it asks:
"If I combine these two prices in the right ratio, does the result keep
snapping back to the same average level?"  A p-value below 0.05 means yes,
with 95% confidence.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(prices: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """
    Test every pair of stocks inside each industry group and rank them.

    Returns a table with one row per pair, best (lowest p-value) first.
    """
    rows = []
    for group_name, tickers in groups.items():
        available = [t for t in tickers if t in prices.columns]

        for stock_a, stock_b in combinations(available, 2):
            pair = np.log(prices[[stock_a, stock_b]].dropna())
            _, p_value, _ = coint(pair[stock_a], pair[stock_b])

            rows.append({
                "group": group_name,
                "stock_a": stock_a,
                "stock_b": stock_b,
                "p_value": p_value,
                "correlation": pair[stock_a].corr(pair[stock_b]),
                "cointegrated": p_value < 0.05,
            })

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
