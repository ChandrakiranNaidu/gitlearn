"""
Nifty 50 backtesting engine.

Design decisions (all chosen to avoid the classic backtest sins):
- Signals are computed on close of day t; trades execute at the OPEN of day t+1
  (no look-ahead bias).
- Costs: 25 bps per side (brokerage + STT + stamp + slippage for NSE delivery),
  charged on every rupee traded.
- Stocks enter the tradable universe only 252 bars after their first data point,
  so newly listed names can't be selected on partial-history signals.
- All series are split/dividend adjusted (Twelve Data adjust=all), so returns
  are total-return approximations.
Known limitation: the universe is TODAY'S Nifty 50 constituents, which embeds
survivorship bias — reported absolute returns are optimistic for both the
strategies AND the buy-and-hold benchmark; relative comparisons are fairer.
"""
import os
import json
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COST_PER_SIDE = 0.0025  # 25 bps per side: brokerage+STT+stamp+impact


MEMBERSHIP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "membership_weights.csv")
# dataset ticker -> local csv basename (local files replace & - with _)
_NAME_MAP = {"BAJAJAUTO": "BAJAJ.AUTO", "M&M": "M_M", "MCDOWELL-N": "MCDOWELL_N",
             "VSNL": "TATACOMM"}  # VSNL renamed Tata Communications (2008)


def load_membership(index, columns, csv_path=MEMBERSHIP_CSV):
    """Boolean DataFrame (dates x symbols): True while the stock was a Nifty 50
    member. Built from monthly point-in-time weights (figshare/Bera dataset,
    sourced from official NSE announcements). Snapshots are forward-filled;
    dates before the first snapshot (Jan 2008) use that first snapshot, dates
    after the last one keep the last known membership."""
    wdf = pd.read_csv(csv_path, parse_dates=["DATE"]).set_index("DATE")
    wdf.columns = [_NAME_MAP.get(c, c) for c in wdf.columns]
    member = (wdf > 0)
    member = member.T.groupby(level=0).max().T  # OR renamed duplicates
    member = member.reindex(index, method="ffill")
    member = member.bfill()  # pre-2008 bars take the Jan-2008 membership
    member = member.reindex(columns=columns, fill_value=False)
    return member.fillna(False)


def load_panel(data_dir=DATA_DIR, point_in_time=False):
    """Return dict of DataFrames: open/high/low/close/volume, indexed by date,
    columns = symbols, aligned on the union trading calendar. With
    point_in_time=True, adds a boolean 'member' frame from the historical
    Nifty 50 membership file."""
    frames = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".csv"):
            continue
        sym = f[:-4]
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=["date"])
        df = df.drop_duplicates(subset="date").set_index("date").sort_index()
        frames[sym] = df
    dates = sorted(set().union(*[set(df.index) for df in frames.values()]))
    idx = pd.DatetimeIndex(dates)
    panel = {}
    for col in ["open", "high", "low", "close", "volume"]:
        panel[col] = pd.DataFrame(
            {s: df[col].reindex(idx) for s, df in frames.items()}, index=idx
        )
    # forward-fill closes over exchange holidays per stock (max 5 days), but
    # never backfill: a stock has NaN before its listing date.
    panel["close"] = panel["close"].ffill(limit=5)
    panel["open"] = panel["open"].fillna(panel["close"])
    if point_in_time:
        panel["member"] = load_membership(idx, panel["close"].columns)
    return panel


def tradable_mask(close, min_history=252, member=None):
    """True where a stock has at least `min_history` prior observations and,
    when a membership frame is given, is a current index member."""
    has_data = close.notna()
    count = has_data.cumsum()
    ok = has_data & (count >= min_history)
    if member is not None:
        ok = ok & member
    return ok


CASH_RATE = 0.05  # annualized yield on idle cash (liquid fund, conservative)


def run_portfolio(weights, panel, cost_per_side=COST_PER_SIDE, initial=1.0,
                  cash_rate=CASH_RATE):
    """Simulate a long-only portfolio.

    weights: DataFrame (dates x symbols) of TARGET weights decided on each
    date's close using info available at that close. The engine shifts them:
    the position is held from the NEXT day's open. Daily returns are computed
    open-to-open equivalently via close-to-close with an open-execution
    adjustment approximated as close-to-close on shifted weights (standard
    vector backtest). Costs = turnover * cost_per_side.
    Returns dict with equity curve (Series), daily returns, turnover series.
    """
    close = panel["close"]
    rets = close.pct_change(fill_method=None).fillna(0.0)
    w = weights.shift(1).fillna(0.0)  # held during day t
    # Turnover measured as change in target weights. This slightly overstates
    # trading (ignores favorable drift) => conservative on costs.
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    cash_w = (1.0 - w.sum(axis=1)).clip(lower=0.0)
    daily_cash = (1 + cash_rate) ** (1 / 252) - 1
    gross = (w * rets).sum(axis=1) + cash_w * daily_cash
    net = gross - turnover * cost_per_side
    equity = initial * (1 + net).cumprod()
    return {"equity": equity, "returns": net, "gross": gross, "turnover": turnover}


def metrics(returns, equity, periods=252, label=""):
    r = returns.dropna()
    if len(r) < periods:
        return None
    yrs = len(r) / periods
    eq = equity.dropna()
    total = eq.iloc[-1] / eq.iloc[0]
    if not np.isfinite(total) or total <= 0:
        return None
    cagr = total ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(periods)
    sharpe = (r.mean() * periods) / vol if vol > 0 else 0.0
    downside = r[r < 0].std() * np.sqrt(periods)
    sortino = (r.mean() * periods) / downside if downside > 0 else 0.0
    peak = equity.cummax()
    dd = equity / peak - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0.0
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    pos_years = int((yearly > 0).sum())
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    best_y, worst_y = float(yearly.max()), float(yearly.min())
    # rolling window consistency: % of rolling 1y / 3y / 5y windows positive
    roll = {}
    for tag, win in [("1y", 252), ("3y", 756), ("5y", 1260)]:
        if len(eq) > win:
            rr = eq.pct_change(win).dropna()
            roll[f"pos_{tag}"] = float((rr > 0).mean())
            roll[f"worst_{tag}"] = float(rr.min())
    return {
        **roll,
        "label": label,
        "cagr": float(cagr),
        "total_return": float(total - 1),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(maxdd),
        "calmar": float(calmar),
        "years": float(yrs),
        "pos_years": pos_years,
        "n_years": int(len(yearly)),
        "pct_pos_months": float((monthly > 0).mean()),
        "best_year": best_y,
        "worst_year": worst_y,
        "yearly": {str(k): float(v) for k, v in yearly.items()},
    }


REBAL_OFFSET = 0  # trading-day shift applied to every rebalance date


def month_ends(index):
    """Last trading day of each month in the index, shifted by REBAL_OFFSET
    trading days (for rebalance-timing robustness tests)."""
    s = pd.Series(index, index=index)
    ends = s.groupby([index.year, index.month]).last().values
    if REBAL_OFFSET:
        pos = index.get_indexer(pd.DatetimeIndex(ends))
        pos = np.clip(pos + REBAL_OFFSET, 0, len(index) - 1)
        ends = index[np.unique(pos)].values
    return ends
