"""
Paper-trading engine for the All-Weather 40/60 strategy (champion of the
point-in-time backtest). Runs once per market day after the close.

Usage:
    python3 run_paper.py STATE_IN.json OUT.json

STATE_IN.json: {"config": {...}|null, "state": {...}|null}
  config: {"capital": 500000, "started": true, "start_date": "YYYY-MM-DD"}
  state:  engine-owned document (positions, cash, nav, mom_held, ...)

OUT.json: {"action": "none"|"skip_holiday"|"init"|"mark"|"rebalance",
           "docs": {"state": {...}, "nav": {...}, "trades": [...], "log": {...}},
           "summary": "one line"}

Design notes:
- Fractional shares allowed (paper money), 25 bps cost per side on traded value.
- Cash accrues 5% p.a. daily (liquid fund), matching the backtest.
- Rebalances on the FIRST run of each calendar month (robust to holidays).
- Momentum holding band needs memory: state.mom_held carries the sleeve's names.
"""
import json
import sys
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

COST = 0.0025
CASH_RATE_DAILY = (1 + 0.05) ** (1 / 252) - 1
MOM_SHARE, LV_SHARE = 0.4, 0.6
TOP_N, HOLD_BAND = 10, 18

UNIVERSE = {  # internal symbol -> Yahoo ticker (current Nifty 50 members we track)
 "ADANIENT":"ADANIENT.NS","ADANIPORTS":"ADANIPORTS.NS","APOLLOHOSP":"APOLLOHOSP.NS",
 "ASIANPAINT":"ASIANPAINT.NS","AXISBANK":"AXISBANK.NS","BAJAJ.AUTO":"BAJAJ-AUTO.NS",
 "BAJAJFINSV":"BAJAJFINSV.NS","BAJFINANCE":"BAJFINANCE.NS","BEL":"BEL.NS",
 "BHARTIARTL":"BHARTIARTL.NS","CIPLA":"CIPLA.NS","COALINDIA":"COALINDIA.NS",
 "DRREDDY":"DRREDDY.NS","EICHERMOT":"EICHERMOT.NS","GRASIM":"GRASIM.NS",
 "HCLTECH":"HCLTECH.NS","HDFCBANK":"HDFCBANK.NS","HDFCLIFE":"HDFCLIFE.NS",
 "HEROMOTOCO":"HEROMOTOCO.NS","HINDALCO":"HINDALCO.NS","HINDUNILVR":"HINDUNILVR.NS",
 "ICICIBANK":"ICICIBANK.NS","INDUSINDBK":"INDUSINDBK.NS","INFY":"INFY.NS","ITC":"ITC.NS",
 "JIOFIN":"JIOFIN.NS","JSWSTEEL":"JSWSTEEL.NS","KOTAKBANK":"KOTAKBANK.NS","LT":"LT.NS",
 "M&M":"M&M.NS","MARUTI":"MARUTI.NS","NESTLEIND":"NESTLEIND.NS","NTPC":"NTPC.NS",
 "ONGC":"ONGC.NS","POWERGRID":"POWERGRID.NS","RELIANCE":"RELIANCE.NS",
 "SBILIFE":"SBILIFE.NS","SBIN":"SBIN.NS","SHRIRAMFIN":"SHRIRAMFIN.NS",
 "SUNPHARMA":"SUNPHARMA.NS","TATACONSUM":"TATACONSUM.NS","TATAMOTORS":"TMPV.NS",
 "TATASTEEL":"TATASTEEL.NS","TCS":"TCS.NS","TECHM":"TECHM.NS","TITAN":"TITAN.NS",
 "TRENT":"TRENT.NS","ULTRACEMCO":"ULTRACEMCO.NS","WIPRO":"WIPRO.NS",
}


def fetch_prices():
    tickers = list(UNIVERSE.values())
    start = (dt.date.today() - dt.timedelta(days=560)).isoformat()
    raw = yf.download(tickers, start=start, progress=False, auto_adjust=True)
    close = raw["Close"]
    inv = {v: k for k, v in UNIVERSE.items()}
    close.columns = [inv[c] for c in close.columns]
    close = close.dropna(how="all").ffill(limit=5)
    return close


def target_weights(close, mom_held_prev):
    """All-Weather 40/60 target weights from the latest close."""
    last = close.index[-1]
    ok = close.notna().sum() >= 260  # enough history
    cols = [c for c in close.columns if ok[c]]
    c = close[cols]
    mom_scores = []
    for lb in (63, 126, 252):
        m = c.iloc[-1 - 21] / c.iloc[-1 - lb - 21] - 1
        mom_scores.append(m.rank(pct=True))
    score = sum(mom_scores) / 3
    abs_mom = c.iloc[-1 - 21] / c.iloc[-1 - 273] - 1
    score = score[abs_mom > 0].dropna()
    order = score.sort_values(ascending=False)
    ranks = pd.Series(range(1, len(order) + 1), index=order.index)
    keep = {s for s in (mom_held_prev or []) if s in ranks.index and ranks[s] <= HOLD_BAND}
    for s in order.index:
        if len(keep) >= TOP_N:
            break
        keep.add(s)
    sma200 = c.rolling(200).mean().iloc[-1]
    breadth = float((c.iloc[-1] > sma200).mean())
    scale = float(np.clip((breadth - 0.2) / 0.3, 0, 1))
    w = pd.Series(0.0, index=cols)
    if keep:
        w[list(keep)] += (MOM_SHARE / TOP_N) * scale
    vol = c.pct_change().iloc[-252:].std()
    low = vol.nsmallest(TOP_N)
    w[low.index] += LV_SHARE / TOP_N
    return w[w > 0], sorted(keep), sorted(low.index), breadth, scale


def main():
    inp = json.load(open(sys.argv[1]))
    config, state = inp.get("config"), inp.get("state")
    out = {"action": "none", "docs": {}, "summary": ""}
    if not config or not config.get("started"):
        out["summary"] = "no active paper portfolio; nothing to do"
        json.dump(out, open(sys.argv[2], "w"))
        return

    close = fetch_prices()
    today = str(close.index[-1].date())
    px = close.iloc[-1]
    stale = (dt.date.today() - close.index[-1].date()).days
    if state and state.get("last_mark") == today:
        out["action"] = "skip_holiday"
        out["summary"] = f"no new bar (latest {today}, {stale}d old) — market holiday or data lag"
        json.dump(out, open(sys.argv[2], "w"))
        return

    # benchmark: equal-weight the whole universe (index of our own NAV chart)
    uni = [c for c in close.columns if not np.isnan(px.get(c, np.nan))]
    bench_ret = float(close[uni].pct_change().iloc[-1].mean())

    if not state:  # ---- first run: build the book ----
        w, mom_held, lv_held, breadth, scale = target_weights(close, None)
        capital = float(config["capital"])
        positions, trades, spent = {}, [], 0.0
        for sym, wt in w.items():
            value = capital * wt / (1 + COST)  # reserve room for entry costs
            shares = value / float(px[sym])
            cost = value * COST
            positions[sym] = {"shares": round(shares, 4), "avg_price": round(float(px[sym]), 2)}
            trades.append({"date": today, "sym": sym, "side": "BUY",
                           "shares": round(shares, 4), "price": round(float(px[sym]), 2),
                           "value": round(value, 2), "cost": round(cost, 2)})
            spent += value + cost
        cash = capital - spent
        nav = cash + sum(p["shares"] * float(px[s]) for s, p in positions.items())
        state = {"positions": positions, "cash": round(cash, 2), "nav": round(nav, 2),
                 "mom_held": mom_held, "lv_held": lv_held,
                 "last_mark": today, "last_rebalance_month": today[:7],
                 "bench_nav": float(config["capital"]), "breadth": round(breadth, 3),
                 "invested": round(1 - cash / nav, 4)}
        out["action"] = "init"
        out["docs"]["trades"] = trades
        out["summary"] = (f"initialized ₹{capital:,.0f} into {len(positions)} positions "
                          f"(breadth {breadth:.0%}, momentum sleeve scale {scale:.0%})")
    else:  # ---- daily mark, monthly rebalance ----
        positions = state["positions"]
        cash = float(state["cash"]) * (1 + CASH_RATE_DAILY)
        state["bench_nav"] = round(float(state.get("bench_nav", config["capital"])) * (1 + bench_ret), 2)
        trades = []
        this_month = today[:7]
        if this_month != state.get("last_rebalance_month"):
            w, mom_held, lv_held, breadth, scale = target_weights(close, state.get("mom_held"))
            nav_now = cash + sum(p["shares"] * float(px[s]) for s, p in positions.items()
                                 if not np.isnan(px.get(s, np.nan)))
            new_pos = {}
            for sym, wt in w.items():
                tgt_val = nav_now * wt
                cur_sh = positions.get(sym, {}).get("shares", 0.0)
                cur_val = cur_sh * float(px[sym])
                delta = tgt_val - cur_val
                if abs(delta) > nav_now * 0.002:  # ignore dust
                    side = "BUY" if delta > 0 else "SELL"
                    sh = delta / float(px[sym])
                    cost = abs(delta) * COST
                    cash -= delta + cost
                    trades.append({"date": today, "sym": sym, "side": side,
                                   "shares": round(abs(sh), 4), "price": round(float(px[sym]), 2),
                                   "value": round(abs(delta), 2), "cost": round(cost, 2)})
                    cur_sh += sh
                new_pos[sym] = {"shares": round(cur_sh, 4),
                                "avg_price": positions.get(sym, {}).get("avg_price", round(float(px[sym]), 2))}
            for sym, p in positions.items():  # full exits
                if sym not in new_pos and p["shares"] > 0 and not np.isnan(px.get(sym, np.nan)):
                    val = p["shares"] * float(px[sym]); cost = val * COST
                    cash += val - cost
                    trades.append({"date": today, "sym": sym, "side": "SELL",
                                   "shares": p["shares"], "price": round(float(px[sym]), 2),
                                   "value": round(val, 2), "cost": round(cost, 2)})
            positions = {s: p for s, p in new_pos.items() if p["shares"] > 1e-6}
            state["mom_held"] = mom_held
            state["lv_held"] = lv_held
            state["last_rebalance_month"] = this_month
            state["breadth"] = round(breadth, 3)
            out["action"] = "rebalance"
            out["summary"] = f"monthly rebalance: {len(trades)} trades, breadth {breadth:.0%}"
        else:
            out["action"] = "mark"
        nav = cash + sum(p["shares"] * float(px[s]) for s, p in positions.items()
                         if not np.isnan(px.get(s, np.nan)))
        state.update({"positions": positions, "cash": round(cash, 2),
                      "nav": round(nav, 2), "last_mark": today,
                      "invested": round(1 - cash / nav, 4)})
        if trades:
            out["docs"]["trades"] = trades
        if out["action"] == "mark":
            ret = nav / float(config["capital"]) - 1
            out["summary"] = f"marked {today}: NAV ₹{nav:,.0f} ({ret:+.2%} since start)"

    # live per-position marks for the dashboard
    marks = {s: round(float(px[s]), 2) for s in state["positions"]
             if not np.isnan(px.get(s, np.nan))}
    state["last_prices"] = marks
    out["docs"]["state"] = state
    out["docs"]["nav"] = {"date": today, "nav": state["nav"],
                          "bench_nav": state["bench_nav"],
                          "invested": state["invested"]}
    out["docs"]["log"] = {"date": today, "action": out["action"], "msg": out["summary"]}
    json.dump(out, open(sys.argv[2], "w"))
    print(out["summary"])


if __name__ == "__main__":
    main()
