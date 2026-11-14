"""
Interactive dashboard for the pairs-trading strategy.

Start it with:   streamlit run app.py
"""
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from pairs_trading import plots
from pairs_trading.backtest import list_trades
from pairs_trading.config import DEFAULT_UNIVERSE, Settings
from pairs_trading.data import download_prices
from pairs_trading.pair_selection import find_cointegrated_pairs
from pairs_trading.pipeline import run_pair

st.set_page_config(page_title="Pairs Trading Lab", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Downloading prices from Yahoo Finance...")
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Download once, then reuse the result while the sliders move."""
    return download_prices(list(tickers), start, end)


def show(fig: plt.Figure) -> None:
    """Draw a matplotlib figure in the page, then free its memory."""
    st.pyplot(fig)
    plt.close(fig)


def explain_day(row: pd.Series, previous: int, a: str, b: str, s: Settings) -> str:
    """Describe in plain English what the strategy did on one day, and why."""
    z, position = row["zscore"], row["position"]
    if position == 1 and previous == 0:
        return (f"🟢 **OPEN LONG SPREAD**: z = {z:.2f} dropped below -{s.entry_z}. "
                f"{a} looks unusually cheap compared with {b}, so we **buy {a} and short-sell {b}**.")
    if position == -1 and previous == 0:
        return (f"🔴 **OPEN SHORT SPREAD**: z = {z:.2f} rose above +{s.entry_z}. "
                f"{a} looks unusually expensive compared with {b}, so we **short-sell {a} and buy {b}**.")
    if position == 0 and previous != 0:
        reason = "it stretched past the stop-loss band" if abs(z) > s.stop_z else "it is back to normal"
        return f"⚪ **CLOSE TRADE**: z = {z:.2f}, so {reason}."
    if position == 1:
        return f"⏳ Holding long spread. Waiting for z (now {z:.2f}) to climb back above -{s.exit_z}."
    if position == -1:
        return f"⏳ Holding short spread. Waiting for z (now {z:.2f}) to fall back below +{s.exit_z}."
    return f"💤 No trade. z = {z:.2f} is inside the ±{s.entry_z} bands, so nothing unusual is happening."


# ----------------------------------------------------------------------------
# Sidebar: every knob the user can turn
# ----------------------------------------------------------------------------
default = Settings()

st.sidebar.header("1. Pick two stocks")
stock_a = st.sidebar.text_input("Stock A", "V").strip().upper()
stock_b = st.sidebar.text_input("Stock B", "MA").strip().upper()
st.sidebar.caption("Ideas: V/MA, KO/PEP, XOM/CVX, HD/LOW, EWA/EWC")

st.sidebar.header("2. Time periods")
learn_years = st.sidebar.slider("Learn the relationship (formation)", 2010, 2025, (2018, 2021))
trade_years = st.sidebar.slider("Trade it (out-of-sample)", 2010, 2025, (2022, 2025))

st.sidebar.header("3. Trading rules")
settings = Settings(
    formation_start=f"{learn_years[0]}-01-01",
    formation_end=f"{learn_years[1]}-12-31",
    trading_start=f"{trade_years[0]}-01-01",
    trading_end=f"{trade_years[1]}-12-31",
    zscore_window=st.sidebar.slider("Z-score window (days)", 10, 120, default.zscore_window),
    entry_z=st.sidebar.slider("Entry band (z)", 1.0, 3.5, default.entry_z, 0.1),
    exit_z=st.sidebar.slider("Exit band (z)", 0.0, 1.5, default.exit_z, 0.1),
    stop_z=st.sidebar.slider("Stop-loss band (z)", 2.5, 6.0, default.stop_z, 0.1),
    cost_bps=st.sidebar.slider("Trading cost (basis points)", 0.0, 20.0, default.cost_bps, 0.5),
)

if learn_years[1] >= trade_years[0]:
    st.sidebar.warning("The trading period should start after the formation period ends, "
                       "otherwise the model is tested on data it already saw.")

# ----------------------------------------------------------------------------
# Run the strategy
# ----------------------------------------------------------------------------
st.title("📈 Pairs Trading Lab")
st.write("Two stocks that normally move together sometimes drift apart. This strategy bets "
         "that they will snap back together, and this page lets you watch it happen.")

try:
    prices = load_prices((stock_a, stock_b), settings.formation_start, settings.trading_end)
    result = run_pair(prices[stock_a], prices[stock_b], settings)
except Exception as error:  # bad ticker, no data for those years, etc.
    st.error(f"Could not run the strategy for {stock_a} / {stock_b}: {error}")
    st.stop()

df, trades, summary = result.results, result.trades, result.summary

tab_backtest, tab_replay, tab_scan, tab_help = st.tabs(
    ["📊 Backtest", "▶️ Step-by-step replay", "🔍 Pair scanner", "📘 How it works"])

# --- Tab 1: full backtest ----------------------------------------------------
with tab_backtest:
    cols = st.columns(5)
    cols[0].metric("Total return", f"{summary['Total return']:.1%}",
                   f"{summary['Buy & hold return']:.1%} buy & hold", delta_color="off")
    cols[1].metric("Sharpe ratio", f"{summary['Sharpe ratio']:.2f}")
    cols[2].metric("Max drawdown", f"{summary['Max drawdown']:.1%}")
    cols[3].metric("Trades", summary["Number of trades"])
    cols[4].metric("Win rate", f"{summary['Win rate']:.0%}")

    st.caption(f"Hedge ratio learned in {learn_years[0]}-{learn_years[1]}: "
               f"**{result.hedge_ratio:.2f}**. For every $1 of {stock_a}, "
               f"we hold ${result.hedge_ratio:.2f} of {stock_b} in the opposite direction.")
    show(plots.plot_strategy(df, trades, stock_a, stock_b, settings))

    with st.expander("See every trade"):
        st.dataframe(trades.style.format({"return": "{:.2%}"}), use_container_width=True)

# --- Tab 2: replay the strategy one day at a time ------------------------------
with tab_replay:
    st.write("Drag the slider to move through time and see what the strategy decides each day.")
    first_day, last_day = df.index[0].date(), df.index[-1].date()
    chosen = st.slider("Date", first_day, last_day, first_day + timedelta(days=120), format="YYYY-MM-DD")

    so_far = df.loc[:pd.Timestamp(chosen)]
    today = so_far.iloc[-1]
    previous = so_far["position"].iloc[-2] if len(so_far) > 1 else 0

    left, right = st.columns([3, 1])
    left.info(f"**{so_far.index[-1]:%d %b %Y}**: " + explain_day(today, previous, stock_a, stock_b, settings))
    right.metric("Account value", f"${today['equity']:,.0f}",
                 f"{today['equity'] / settings.initial_capital - 1:.1%}")

    trades_so_far = list_trades(so_far)
    if today["position"] != 0:   # the latest trade is still open, so it has no exit yet
        trades_so_far = trades_so_far.iloc[:-1]
    show(plots.plot_strategy(so_far, trades_so_far, stock_a, stock_b, settings))

# --- Tab 3: scan the default universe for the best pairs ------------------------
with tab_scan:
    st.write("The cointegration test checks, for every pair inside each industry, whether the two "
             "prices are statistically tied together. **p-value < 0.05 = tied together.**")
    universe = tuple(sorted({t for group in DEFAULT_UNIVERSE.values() for t in group}))
    formation_prices = load_prices(universe, settings.formation_start, settings.formation_end)
    scan = find_cointegrated_pairs(formation_prices, DEFAULT_UNIVERSE)

    st.dataframe(scan.style.format({"p_value": "{:.4f}", "correlation": "{:.3f}"}),
                 use_container_width=True)
    show(plots.plot_pair_scan(scan))

# --- Tab 4: plain-English explanation ------------------------------------------
with tab_help:
    st.markdown(f"""
### The idea in one sentence
Find two stocks that move together, wait until they drift unusually far apart,
then bet on them coming back together.

### The four steps
1. **Find a pair.** The *cointegration test* checks whether some mix of the two prices
   always drifts back to the same level. Correlation alone isn't enough, because two
   prices can both rise and still drift apart forever.
2. **Measure the normal relationship.** A regression on the formation period gives the
   *hedge ratio*: `log({stock_a}) ≈ intercept + hedge_ratio × log({stock_b})`.
3. **Measure how unusual today is.** The *spread* is how far today's prices are from that
   line. The *z-score* rescales it: z = 2 means "2 standard deviations away from normal".
4. **Trade.** If z < −{settings.entry_z}, buy {stock_a} and short {stock_b}. If z > +{settings.entry_z},
   do the opposite. Close when |z| gets back under {settings.exit_z}, or bail out
   if it passes {settings.stop_z}.

### Why the result is "market neutral"
We are always long one stock and short the other. If the whole market crashes, both
legs fall and the losses largely cancel. The strategy only cares about the *gap*
between the two stocks, so don't expect it to beat buy & hold in a bull market.
Its goal is small, steady returns that don't depend on the market's direction.
""")
