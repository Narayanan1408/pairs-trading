"""
Run the whole pairs-trading pipeline from the command line.

    python main.py                  # scan all pairs, then trade the best one
    python main.py --pair KO PEP    # skip the scan and trade a pair you choose

Charts are saved to the images/ folder.
"""
import argparse
from pathlib import Path

from pairs_trading import plots
from pairs_trading.config import DEFAULT_UNIVERSE, Settings
from pairs_trading.data import download_prices
from pairs_trading.pair_selection import find_cointegrated_pairs
from pairs_trading.pipeline import run_pair

IMAGES_DIR = Path("images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a pairs-trading strategy.")
    parser.add_argument("--pair", nargs=2, metavar=("STOCK_A", "STOCK_B"),
                        help="trade this pair instead of the best one found by the scan")
    args = parser.parse_args()

    settings = Settings()
    IMAGES_DIR.mkdir(exist_ok=True)

    # 1. Download prices for every stock we might need.
    tickers = args.pair or sorted({t for group in DEFAULT_UNIVERSE.values() for t in group})
    print(f"Downloading prices for {len(tickers)} tickers...")
    prices = download_prices(tickers, settings.formation_start, settings.trading_end)

    # 2. Choose the pair: either the user's pick or the best one from the scan.
    if args.pair:
        stock_a, stock_b = args.pair
    else:
        formation_prices = prices.loc[settings.formation_start:settings.formation_end]
        scan = find_cointegrated_pairs(formation_prices, DEFAULT_UNIVERSE)
        print("\nPair scan (formation period, lowest p-value = strongest link):")
        print(scan.head(10).to_string(index=False, float_format="{:.4f}".format))
        plots.plot_pair_scan(scan).savefig(IMAGES_DIR / "pair_scan.png", dpi=150)
        stock_a, stock_b = scan.loc[0, "stock_a"], scan.loc[0, "stock_b"]

    # 3. Run the strategy on that pair.
    print(f"\nTrading {stock_a} vs {stock_b}...")
    result = run_pair(prices[stock_a], prices[stock_b], settings)
    print(f"Hedge ratio: {result.hedge_ratio:.3f}  "
          f"(for every $1 of {stock_a} we hold ${result.hedge_ratio:.2f} of {stock_b} the other way)")

    # 4. Print the results.
    print(f"\nPerformance {settings.trading_start} -> {settings.trading_end}")
    for name, value in result.summary.items():
        shown = f"{value:.2f}" if name == "Sharpe ratio" else (
            f"{value}" if name == "Number of trades" else f"{value:.1%}")
        print(f"  {name:<20} {shown:>8}")

    # 5. Save charts for the README.
    formation = slice(settings.formation_start, settings.formation_end)
    plots.plot_hedge_fit(prices[stock_a][formation], prices[stock_b][formation], stock_a, stock_b,
                         result.intercept, result.hedge_ratio).savefig(IMAGES_DIR / "hedge_fit.png", dpi=150)
    plots.plot_strategy(result.results, result.trades, stock_a, stock_b,
                        settings).savefig(IMAGES_DIR / "backtest.png", dpi=150)
    print(f"\nCharts saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
