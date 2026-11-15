"""
Strategy research lab: backtests every strategy over the full history,
splits results into in-sample (2008-2017) and out-of-sample (2018+),
and writes results.json for the dashboard.

"Consistent profitability" bar a strategy must clear:
  1. Net CAGR > 0 in BOTH the in-sample and out-of-sample windows.
  2. Out-of-sample Sharpe >= 0.8 and beats buy & hold Sharpe.
  3. Positive calendar years >= 70% of all years.
  4. Max drawdown shallower than buy & hold.
"""
import json
import sys
import numpy as np
import pandas as pd
from engine import load_panel, run_portfolio, metrics
from strategies import STRATEGIES

IS_END = "2017-12-31"
OOS_START = "2018-01-01"
WARMUP_END = "2008-06-30"  # ignore first ~18 months (indicator warm-up)


def evaluate(key, name, fn, kwargs, panel):
    w = fn(panel, **kwargs)
    res = run_portfolio(w, panel)
    r, eq = res["returns"], res["equity"]
    r = r[r.index > WARMUP_END]
    eq = eq[eq.index > WARMUP_END]
    eq = eq / eq.iloc[0]
    full = metrics(r, eq, label=name)
    r_is, eq_is = r[r.index <= IS_END], eq[eq.index <= IS_END]
    r_oos, eq_oos = r[r.index >= OOS_START], eq[eq.index >= OOS_START]
    ins = metrics(r_is, eq_is / eq_is.iloc[0], label=name) if len(r_is) else None
    oos = metrics(r_oos, eq_oos / eq_oos.iloc[0], label=name) if len(r_oos) else None
    avg_turnover = float(res["turnover"][res["turnover"].index > WARMUP_END]
                         .resample("YE").sum().mean())
    # monthly equity samples for charts
    monthly_eq = eq.resample("ME").last().dropna()
    peak = eq.cummax()
    dd = (eq / peak - 1).resample("ME").min().dropna()
    return {
        "key": key, "name": name,
        "full": full, "insample": ins, "oos": oos,
        "annual_turnover": avg_turnover,
        "equity": {str(k.date()): round(float(v), 4) for k, v in monthly_eq.items()},
        "drawdown": {str(k.date()): round(float(v), 4) for k, v in dd.items()},
    }


def main():
    panel = load_panel()
    out = {"strategies": [], "meta": {
        "universe": list(panel["close"].columns),
        "start": str(panel["close"].index[0].date()),
        "end": str(panel["close"].index[-1].date()),
        "n_stocks": panel["close"].shape[1],
        "cost_bps_per_side": 25,
        "is_window": f"{WARMUP_END}..{IS_END}",
        "oos_window": f"{OOS_START}..{panel['close'].index[-1].date()}",
    }}
    only = sys.argv[1:] or None
    for key, (name, fn, kwargs) in STRATEGIES.items():
        if only and key not in only:
            continue
        print(f"running {key}...", flush=True)
        out["strategies"].append(evaluate(key, name, fn, kwargs, panel))
    with open("results.json", "w") as f:
        json.dump(out, f)
    # console summary
    hdr = f"{'strategy':28} {'CAGR':>7} {'Shrp':>5} {'MaxDD':>7} | {'IS CAGR':>8} {'OOS CAGR':>8} {'OOS Shrp':>8} {'+yrs':>5}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for s in out["strategies"]:
        f_, i_, o_ = s["full"], s["insample"], s["oos"]
        print(f"{s['name']:28} {f_['cagr']*100:6.1f}% {f_['sharpe']:5.2f} "
              f"{f_['max_dd']*100:6.1f}% | {i_['cagr']*100:7.1f}% "
              f"{o_['cagr']*100:7.1f}% {o_['sharpe']:8.2f} "
              f"{f_['pos_years']}/{f_['n_years']}")


if __name__ == "__main__":
    main()
