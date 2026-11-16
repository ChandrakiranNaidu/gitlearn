"""
Robustness suite for the point-in-time champion (All-Weather 40/60):
 1. Parameter plateau — perturb every knob, confirm performance is flat.
 2. Block bootstrap — confidence intervals for Sharpe and CAGR.
 3. Rebalance-day shift — move the monthly trade date +/- 3 trading days.
Writes robustness.json for the dashboard.
"""
import json
import numpy as np
import engine
from engine import load_panel, run_portfolio, metrics
from strategies import mom_blend_bands, low_vol

WARMUP_END = "2008-06-30"
CHAMP = dict(mshare=0.4, top_n=10, band=18)


def champion_weights(panel, mshare=0.4, top_n=10, band=18):
    wm = mom_blend_bands(panel, top_n=top_n, hold_band=band,
                         trend_gate=False, regime_sma=200)
    wl = low_vol(panel, top_n=top_n)
    return wm * mshare + wl * (1 - mshare)


def full_metrics(w, panel):
    res = run_portfolio(w, panel)
    r = res["returns"][res["returns"].index > WARMUP_END]
    eq = (1 + r).cumprod()
    return metrics(r, eq), r


def main():
    panel = load_panel(point_in_time=True)
    out = {}

    # ---- 1. parameter plateau --------------------------------------------
    print("plateau...", flush=True)
    plateau = []
    base_m, _ = full_metrics(champion_weights(panel, **CHAMP), panel)
    for name, kwargs in [
        ("champion 40/60 n10 b18", CHAMP),
        ("mom share 30%", dict(CHAMP, mshare=0.3)),
        ("mom share 50%", dict(CHAMP, mshare=0.5)),
        ("mom share 60%", dict(CHAMP, mshare=0.6)),
        ("top 8 names/sleeve", dict(CHAMP, top_n=8)),
        ("top 12 names/sleeve", dict(CHAMP, top_n=12)),
        ("band 14", dict(CHAMP, band=14)),
        ("band 22", dict(CHAMP, band=22)),
        ("band 26", dict(CHAMP, band=26)),
    ]:
        m, _ = full_metrics(champion_weights(panel, **kwargs), panel)
        plateau.append({"config": name, "cagr": m["cagr"], "sharpe": m["sharpe"],
                        "max_dd": m["max_dd"], "pos_years": m["pos_years"],
                        "n_years": m["n_years"]})
        print(f"  {name:24} cagr={m['cagr']*100:5.1f}%  sharpe={m['sharpe']:.2f}  dd={m['max_dd']*100:5.1f}%")
    out["plateau"] = plateau

    # ---- 2. block bootstrap ----------------------------------------------
    print("bootstrap...", flush=True)
    _, r = full_metrics(champion_weights(panel, **CHAMP), panel)
    rng = np.random.default_rng(42)
    vals = r.to_numpy()
    n, block = len(vals), 21
    n_blocks = n // block
    sharpes, cagrs, maxdds = [], [], []
    for _ in range(2000):
        starts = rng.integers(0, n - block, n_blocks)
        sample = np.concatenate([vals[s:s + block] for s in starts])
        mu, sd = sample.mean(), sample.std()
        sharpes.append(mu * 252 / (sd * np.sqrt(252)))
        eq = np.cumprod(1 + sample)
        cagrs.append(eq[-1] ** (252 / len(sample)) - 1)
        peak = np.maximum.accumulate(eq)
        maxdds.append(((eq / peak) - 1).min())
    def ci(a):
        return [float(np.percentile(a, 5)), float(np.percentile(a, 50)),
                float(np.percentile(a, 95))]
    out["bootstrap"] = {
        "n_draws": 2000, "block_days": block,
        "sharpe_ci": ci(sharpes), "cagr_ci": ci(cagrs), "maxdd_ci": ci(maxdds),
        "p_sharpe_gt0": float(np.mean(np.array(sharpes) > 0)),
        "p_cagr_gt0": float(np.mean(np.array(cagrs) > 0)),
    }
    print(f"  sharpe 90% CI: [{out['bootstrap']['sharpe_ci'][0]:.2f}, "
          f"{out['bootstrap']['sharpe_ci'][2]:.2f}]  "
          f"cagr 90% CI: [{out['bootstrap']['cagr_ci'][0]*100:.1f}%, "
          f"{out['bootstrap']['cagr_ci'][2]*100:.1f}%]  "
          f"P(cagr>0)={out['bootstrap']['p_cagr_gt0']*100:.1f}%")

    # ---- 3. rebalance-day shift ------------------------------------------
    print("rebalance shift...", flush=True)
    shifts = []
    for off in [-3, -2, -1, 0, 1, 2, 3]:
        engine.REBAL_OFFSET = off
        try:
            m, _ = full_metrics(champion_weights(panel, **CHAMP), panel)
        finally:
            engine.REBAL_OFFSET = 0
        shifts.append({"offset": off, "cagr": m["cagr"], "sharpe": m["sharpe"],
                       "max_dd": m["max_dd"]})
        print(f"  offset {off:+d}d  cagr={m['cagr']*100:5.1f}%  sharpe={m['sharpe']:.2f}")
    out["rebal_shift"] = shifts

    with open("robustness.json", "w") as f:
        json.dump(out, f)
    print("written robustness.json")


if __name__ == "__main__":
    main()
