"""
Strategy library. Every function returns a DataFrame of target weights
(dates x symbols), computed only from information available at each date's
close. The engine shifts execution to the next bar.
"""
import numpy as np
import pandas as pd
from engine import tradable_mask, month_ends


def _monthly(weights_at_rebal, index):
    """Expand weights set on rebalance dates to a daily frame (held constant)."""
    w = pd.DataFrame(np.nan, index=index, columns=weights_at_rebal.columns)
    w.loc[weights_at_rebal.index] = weights_at_rebal.values
    return w.ffill().fillna(0.0)


def buy_hold_equal_weight(panel):
    """Benchmark: equal weight across tradable stocks, rebalanced monthly."""
    close = panel["close"]
    ok = tradable_mask(close, 60)
    rebs = month_ends(close.index)
    rows = {}
    for d in rebs:
        names = ok.loc[d]
        n = names.sum()
        rows[d] = (names / n).astype(float) if n else names.astype(float) * 0.0
    wr = pd.DataFrame(rows).T
    wr.index = pd.DatetimeIndex(wr.index)
    return _monthly(wr, close.index)


def sma_crossover(panel, fast=50, slow=200):
    """Per-stock trend following: hold (equal weight) stocks whose fast SMA is
    above slow SMA; otherwise that slot sits in cash."""
    close = panel["close"]
    ok = tradable_mask(close, slow + 10)
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    sig = ((f > s) & ok).astype(float)
    n_universe = ok.sum(axis=1).replace(0, np.nan)
    w = sig.div(n_universe, axis=0).fillna(0.0)  # cash for non-trending slots
    return w


def xs_momentum(panel, lookback=252, skip=21, top_n=10, regime_sma=200,
                vol_weight=False):
    """Cross-sectional momentum: each month, rank tradable stocks by
    (lookback - skip) return (classic 12-1), buy the top_n equally (or
    inverse-vol) weighted. Regime filter: scale into cash by the fraction of
    universe above its own regime_sma (breadth), if regime_sma given."""
    close = panel["close"]
    ok = tradable_mask(close, lookback + skip + 10)
    mom = close.shift(skip) / close.shift(lookback + skip) - 1
    vol = close.pct_change(fill_method=None).rolling(63).std()
    if regime_sma:
        above = (close > close.rolling(regime_sma).mean()) & ok
        breadth = above.sum(axis=1) / ok.sum(axis=1).replace(0, np.nan)
    rebs = month_ends(close.index)
    rows = {}
    for d in rebs:
        m = mom.loc[d].where(ok.loc[d])
        m = m[m > 0]  # absolute-momentum gate: only names with positive momentum
        top = m.nlargest(top_n)
        w = pd.Series(0.0, index=close.columns)
        if len(top):
            if vol_weight:
                iv = 1.0 / vol.loc[d, top.index].replace(0, np.nan)
                iv = iv.fillna(0.0)
                w[top.index] = iv / iv.sum() if iv.sum() > 0 else 1.0 / len(top)
            else:
                w[top.index] = 1.0 / top_n  # unfilled slots stay in cash
        if regime_sma and not np.isnan(breadth.loc[d]):
            w = w * float(np.clip(breadth.loc[d] / 0.5, 0, 1))
        rows[d] = w
    wr = pd.DataFrame(rows).T
    wr.index = pd.DatetimeIndex(wr.index)
    return _monthly(wr, close.index)


def high_52w(panel, top_n=10, regime_sma=200):
    """Buy stocks closest to their 52-week high (George-Hwang), monthly."""
    close = panel["close"]
    ok = tradable_mask(close, 262)
    ratio = close / close.rolling(252).max()
    if regime_sma:
        above = (close > close.rolling(regime_sma).mean()) & ok
        breadth = above.sum(axis=1) / ok.sum(axis=1).replace(0, np.nan)
    rebs = month_ends(close.index)
    rows = {}
    for d in rebs:
        r = ratio.loc[d].where(ok.loc[d]).dropna()
        top = r.nlargest(top_n)
        w = pd.Series(0.0, index=close.columns)
        if len(top):
            w[top.index] = 1.0 / top_n
        if regime_sma and not np.isnan(breadth.loc[d]):
            w = w * float(np.clip(breadth.loc[d] / 0.5, 0, 1))
        rows[d] = w
    wr = pd.DataFrame(rows).T
    wr.index = pd.DatetimeIndex(wr.index)
    return _monthly(wr, close.index)


def mean_reversion_rsi(panel, rsi_len=2, buy_below=10, exit_above=70,
                       sma_filter=200, max_hold=10, top_n=10):
    """Short-term dip buying: RSI(2) < buy_below while above 200 SMA; exit on
    RSI > exit_above or after max_hold days. Daily, equal weight across
    active signals capped at top_n slots."""
    close = panel["close"]
    ok = tradable_mask(close, sma_filter + 10)
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / rsi_len, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_len, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    trend = close > close.rolling(sma_filter).mean()
    entry = (rsi < buy_below) & trend & ok
    exit_ = rsi > exit_above
    arr = np.zeros(close.shape)
    state = np.zeros(close.shape[1])
    age = np.zeros(close.shape[1])
    ent = entry.to_numpy()
    exi = exit_.fillna(False).to_numpy()
    for i in range(close.shape[0]):
        state = np.where(exi[i] | (age >= max_hold), 0, state)
        age = np.where(state > 0, age + 1, 0)
        newly = ent[i] & (state == 0)
        state = np.where(newly, 1, state)
        arr[i] = state
    pos = pd.DataFrame(arr, index=close.index, columns=close.columns)
    slots = pos.sum(axis=1).clip(lower=top_n)
    return pos.div(slots, axis=0).fillna(0.0)


def donchian_breakout(panel, entry_len=55, exit_len=20, top_n=10):
    """Classic turtle-style breakout with ATR-free equal weighting."""
    close = panel["close"]
    high, low = panel["high"], panel["low"]
    ok = tradable_mask(close, entry_len + 10)
    upper = high.rolling(entry_len).max().shift(1)
    lower = low.rolling(exit_len).min().shift(1)
    arr = np.zeros(close.shape)
    state = np.zeros(close.shape[1])
    cu, cl, c, okv = (upper.to_numpy(), lower.to_numpy(),
                      close.to_numpy(), ok.to_numpy())
    for i in range(close.shape[0]):
        with np.errstate(invalid="ignore"):
            brk = (c[i] > cu[i]) & okv[i]
            ext = c[i] < cl[i]
        state = np.where(ext, 0, state)
        state = np.where(brk & (state == 0), 1, state)
        arr[i] = state
    pos = pd.DataFrame(arr, index=close.index, columns=close.columns)
    slots = pos.sum(axis=1).clip(lower=top_n)
    return pos.div(slots, axis=0).fillna(0.0)


STRATEGIES = {
    "buy_hold": ("Buy & Hold (Equal Weight)", buy_hold_equal_weight, {}),
    "sma_50_200": ("Golden Cross 50/200", sma_crossover, {}),
    "mom_12_1_top10": ("XS Momentum 12-1 Top 10", xs_momentum,
                       dict(lookback=252, skip=21, top_n=10)),
    "mom_6_1_top10": ("XS Momentum 6-1 Top 10", xs_momentum,
                      dict(lookback=126, skip=21, top_n=10)),
    "mom_12_1_top5": ("XS Momentum 12-1 Top 5", xs_momentum,
                      dict(lookback=252, skip=21, top_n=5)),
    "mom_12_1_iv": ("XS Momentum 12-1 InvVol", xs_momentum,
                    dict(lookback=252, skip=21, top_n=10, vol_weight=True)),
    "high52": ("52-Week High Top 10", high_52w, {}),
    "rsi2_dip": ("RSI-2 Dip Buyer", mean_reversion_rsi, {}),
    "donchian": ("Donchian 55/20 Breakout", donchian_breakout, {}),
}


def _rank_pct(s):
    return s.rank(pct=True)


def mom_blend_bands(panel, top_n=10, hold_band=18, trend_gate=True,
                    regime_sma=200, lbs=((63, 21), (126, 21), (252, 21))):
    """Blended-horizon momentum with holding bands.

    Rank = average percentile rank of 3m, 6m, 12m returns (1m skip each).
    Buy top_n; keep a holding until it drops below rank hold_band (reduces
    turnover vs hard top_n cutoff). Optional per-stock trend gate
    (close > 200 SMA) and index-breadth regime scaling.
    """
    close = panel["close"]
    ok = tradable_mask(close, max(lb + sk for lb, sk in lbs) + 10)
    moms = [close.shift(sk) / close.shift(lb + sk) - 1 for lb, sk in lbs]
    trend = close > close.rolling(200).mean()
    above = trend & ok
    breadth = above.sum(axis=1) / ok.sum(axis=1).replace(0, np.nan)
    rebs = month_ends(close.index)
    rows = {}
    held = set()
    for d in rebs:
        score = sum(_rank_pct(m.loc[d].where(ok.loc[d])) for m in moms) / len(lbs)
        if trend_gate:
            score = score.where(trend.loc[d])
        score = score.dropna()
        # absolute momentum gate: 12-1 return must be positive
        abs_mom = moms[-1].loc[d]
        score = score[abs_mom.reindex(score.index) > 0]
        order = score.sort_values(ascending=False)
        ranks = pd.Series(range(1, len(order) + 1), index=order.index)
        keep = {s for s in held if s in ranks.index and ranks[s] <= hold_band}
        for s in order.index:
            if len(keep) >= top_n:
                break
            keep.add(s)
        held = set(keep)
        w = pd.Series(0.0, index=close.columns)
        if keep:
            w[list(keep)] = 1.0 / top_n
        b = breadth.loc[d]
        if regime_sma and not np.isnan(b):
            w = w * float(np.clip((b - 0.2) / 0.3, 0, 1))
        rows[d] = w
    wr = pd.DataFrame(rows).T
    wr.index = pd.DatetimeIndex(wr.index)
    return _monthly(wr, close.index)


def low_vol(panel, top_n=10, vol_len=252):
    """Betting-against-volatility: hold the top_n least volatile stocks."""
    close = panel["close"]
    ok = tradable_mask(close, vol_len + 10)
    vol = close.pct_change(fill_method=None).rolling(vol_len).std()
    rebs = month_ends(close.index)
    rows = {}
    for d in rebs:
        v = vol.loc[d].where(ok.loc[d]).dropna()
        low = v.nsmallest(top_n)
        w = pd.Series(0.0, index=close.columns)
        if len(low):
            w[low.index] = 1.0 / len(low)
        rows[d] = w
    wr = pd.DataFrame(rows).T
    wr.index = pd.DatetimeIndex(wr.index)
    return _monthly(wr, close.index)


def mom_lowvol(panel, top_n=10, hold_band=18):
    """Blend: average of momentum rank and low-volatility rank."""
    close = panel["close"]
    ok = tradable_mask(close, 300)
    mom = close.shift(21) / close.shift(273) - 1
    vol = close.pct_change(fill_method=None).rolling(252).std()
    trend = close > close.rolling(200).mean()
    rebs = month_ends(close.index)
    rows = {}
    held = set()
    for d in rebs:
        m = _rank_pct(mom.loc[d].where(ok.loc[d]))
        v = _rank_pct((-vol.loc[d]).where(ok.loc[d]))
        score = (0.6 * m + 0.4 * v).where(trend.loc[d]).dropna()
        order = score.sort_values(ascending=False)
        ranks = pd.Series(range(1, len(order) + 1), index=order.index)
        keep = {s for s in held if s in ranks.index and ranks[s] <= hold_band}
        for s in order.index:
            if len(keep) >= top_n:
                break
            keep.add(s)
        held = set(keep)
        w = pd.Series(0.0, index=close.columns)
        if keep:
            w[list(keep)] = 1.0 / top_n
        rows[d] = w
    wr = pd.DataFrame(rows).T
    wr.index = pd.DatetimeIndex(wr.index)
    return _monthly(wr, close.index)


def vol_targeted(panel, base_fn=None, target=0.15, cap=1.0, **kwargs):
    """Overlay: scale base strategy weights so trailing 63-day realized
    portfolio vol ~= target (never levered above cap)."""
    w = base_fn(panel, **kwargs)
    close = panel["close"]
    rets = close.pct_change(fill_method=None).fillna(0.0)
    port = (w.shift(1).fillna(0.0) * rets).sum(axis=1)
    realized = port.rolling(63).std() * np.sqrt(252)
    scale = (target / realized).clip(upper=cap).shift(1).fillna(0.0)
    return w.mul(scale, axis=0)


STRATEGIES.update({
    "mom_blend": ("Momentum Blend + Bands", mom_blend_bands, {}),
    "low_vol": ("Low Volatility Top 10", low_vol, {}),
    "mom_lowvol": ("Momentum x Low-Vol 60/40", mom_lowvol, {}),
    "mom_blend_vt": ("Momentum Blend VolTarget 15%", vol_targeted,
                     dict(base_fn=mom_blend_bands, target=0.15)),
    "mom_lowvol_vt": ("Mom x LowVol VolTarget 15%", vol_targeted,
                      dict(base_fn=mom_lowvol, target=0.15)),
})


def combo_mom_lowvol(panel, mshare=0.5, top_n=10, hold_band=22):
    """FINAL CANDIDATE 'All-Weather Nifty': 50% blended-momentum sleeve
    (with breadth regime -> cash) + 50% low-volatility sleeve. The two sleeves
    are weakly correlated: momentum wins in trends, low-vol cushions crashes;
    idle cash earns the liquid-fund rate."""
    wm = mom_blend_bands(panel, top_n=top_n, hold_band=hold_band,
                         trend_gate=False, regime_sma=200)
    wl = low_vol(panel, top_n=top_n)
    return wm * mshare + wl * (1 - mshare)


STRATEGIES.update({
    "mom_blend22": ("Momentum Blend Band-22", mom_blend_bands,
                    dict(top_n=10, hold_band=22, trend_gate=False)),
    "allweather": ("All-Weather 50/50 Mom+LowVol", combo_mom_lowvol, {}),
})
