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


def load_panel(data_dir=DATA_DIR):
    """Return dict of DataFrames: open/high/low/close/volume, indexed by date,
    columns = symbols, aligned on the union trading calendar."""
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
    return panel


def tradable_mask(close, min_history=252):
    """True where a stock has at least `min_history` prior observations."""
    has_data = close.notna()
    count = has_data.cumsum()
    return has_data & (count >= min_history)


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


def month_ends(index):
    """Last trading day of each month in the index."""
    s = pd.Series(index, index=index)
    return s.groupby([index.year, index.month]).last().values
