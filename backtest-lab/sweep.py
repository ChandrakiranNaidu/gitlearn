"""Parameter sweep with honest IS/OOS discipline: rank configurations by
IN-SAMPLE Sharpe only, then report how the top picks did OUT-OF-SAMPLE."""
import itertools
import numpy as np
import pandas as pd
from engine import load_panel, run_portfolio, metrics
from strategies import mom_blend_bands, low_vol, mom_lowvol, vol_targeted

IS_END = "2017-12-31"
OOS_START = "2018-01-01"
WARMUP_END = "2008-06-30"


def eval_weights(w, panel):
    res = run_portfolio(w, panel)
    r, eq = res["returns"], res["equity"]
    r = r[r.index > WARMUP_END]
    eq = eq[eq.index > WARMUP_END]
    eq = eq / eq.iloc[0]
    r_is, eq_is = r[r.index <= IS_END], eq[eq.index <= IS_END]
    r_oos, eq_oos = r[r.index >= OOS_START], eq[eq.index >= OOS_START]
    mi = metrics(r_is, eq_is / eq_is.iloc[0])
    mo = metrics(r_oos, eq_oos / eq_oos.iloc[0])
    mf = metrics(r, eq)
    return mi, mo, mf


def main():
    panel = load_panel(point_in_time="--pit" in __import__("sys").argv)
    results = []

    grid = list(itertools.product(
        [8, 10, 12, 15],        # top_n
        [None, 14, 18, 22],     # hold_band (None => top_n, hard cutoff)
        [True, False],          # trend gate
        [200, None],            # regime
    ))
    for top_n, band, gate, regime in grid:
        hb = band or top_n
        w = mom_blend_bands(panel, top_n=top_n, hold_band=hb,
                            trend_gate=gate, regime_sma=regime)
        mi, mo, mf = eval_weights(w, panel)
        if mi and mo:
            results.append(dict(
                name=f"momblend n={top_n} band={hb} gate={gate} regime={regime}",
                is_sharpe=mi["sharpe"], is_cagr=mi["cagr"],
                oos_sharpe=mo["sharpe"], oos_cagr=mo["cagr"],
                full_sharpe=mf["sharpe"], full_cagr=mf["cagr"],
                full_dd=mf["max_dd"], pos_years=mf["pos_years"],
                n_years=mf["n_years"],
            ))
        print(".", end="", flush=True)

    # combos: momentum sleeve + low-vol sleeve
    for mshare in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        wm = mom_blend_bands(panel, top_n=10, hold_band=18)
        wl = low_vol(panel, top_n=10)
        w = wm * mshare + wl * (1 - mshare)
        mi, mo, mf = eval_weights(w, panel)
        results.append(dict(
            name=f"combo mom{int(mshare*100)}/lowvol{int((1-mshare)*100)}",
            is_sharpe=mi["sharpe"], is_cagr=mi["cagr"],
            oos_sharpe=mo["sharpe"], oos_cagr=mo["cagr"],
            full_sharpe=mf["sharpe"], full_cagr=mf["cagr"],
            full_dd=mf["max_dd"], pos_years=mf["pos_years"],
            n_years=mf["n_years"],
        ))
        print("+", end="", flush=True)

    df = pd.DataFrame(results).sort_values("is_sharpe", ascending=False)
    pd.set_option("display.width", 200)
    print("\n\nTop 15 by IN-SAMPLE Sharpe (OOS shown for validation):")
    cols = ["name", "is_sharpe", "is_cagr", "oos_sharpe", "oos_cagr",
            "full_sharpe", "full_cagr", "full_dd", "pos_years", "n_years"]
    print(df[cols].head(15).to_string(
        index=False,
        formatters={c: (lambda v: f"{v*100:5.1f}%") for c in
                    ["is_cagr", "oos_cagr", "full_cagr", "full_dd"]}))
    df.to_csv("sweep_results_pit.csv" if "--pit" in __import__("sys").argv else "sweep_results.csv", index=False)


if __name__ == "__main__":
    main()
