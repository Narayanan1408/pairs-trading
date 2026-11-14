"""Download daily stock prices from Yahoo Finance (free, no account needed)."""
import pandas as pd
import yfinance as yf


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Return a table of daily closing prices: one row per day, one column per ticker.

    Prices are "adjusted" for dividends and stock splits, so a 2-for-1 split
    doesn't show up as a fake 50% crash.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]

    # Remove days where every price is missing (e.g. a market holiday glitch).
    prices = prices.dropna(how="all")

    if prices.empty:
        raise ValueError(f"No price data found for {tickers} between {start} and {end}.")
    return prices
