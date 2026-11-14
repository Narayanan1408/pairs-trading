"""
All the settings ("knobs") of the project in one place.

Change a number here and both main.py and the dashboard pick it up.
"""
from dataclasses import dataclass

# Candidate stocks, grouped by industry. Pairs trading works best when two
# companies are driven by the same business forces, so we only test pairs
# from inside the same group (Coke vs Pepsi, never Coke vs Exxon).
DEFAULT_UNIVERSE = {
    "Beverages": ["KO", "PEP"],
    "Oil & Gas": ["XOM", "CVX", "COP"],
    "Payments": ["V", "MA", "AXP"],
    "Home Improvement": ["HD", "LOW"],
    "Banks": ["JPM", "BAC", "C", "WFC"],
    "Country ETFs": ["EWA", "EWC"],
}


@dataclass
class Settings:
    # --- Time periods ------------------------------------------------------
    # We LEARN the relationship between the two stocks in the formation period
    # and then TRADE it in a later period the model has never seen.
    formation_start: str = "2018-01-01"
    formation_end: str = "2021-12-31"
    trading_start: str = "2022-01-01"
    trading_end: str = "2025-12-31"

    # --- Signal parameters -------------------------------------------------
    zscore_window: int = 30  # days of history used for the spread's "normal" level
    entry_z: float = 2.0     # open a trade when the spread is 2 std devs from normal
    exit_z: float = 0.5      # close it when the spread is back within 0.5 std devs
    stop_z: float = 4.0      # give up if it stretches past 4 std devs (pair may be broken)

    # --- Money -------------------------------------------------------------
    initial_capital: float = 10_000.0
    cost_bps: float = 5.0    # trading cost: 5 basis points = 0.05% of each dollar traded
