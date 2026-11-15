# Nifty 50 Backtest Lab

A self-contained backtesting environment + strategy research lab for Nifty 50
stocks, with ~20 years (2007–2026) of split/dividend-adjusted daily OHLCV data
for 49 current constituents (source: Twelve Data, NSE).

## Layout

| File | Purpose |
|---|---|
| `data/*.csv` | Daily adjusted OHLCV per stock, 2007 → 2026 |
| `engine.py` | Panel loader, portfolio simulator (costs, cash yield), metrics |
| `strategies.py` | 16 strategy implementations (weights generators) |
| `run_lab.py` | Backtests every strategy, IS/OOS split, writes `results.json` |
| `sweep.py` | Parameter sweep — ranks by in-sample Sharpe, validates out-of-sample |

Run: `python3 run_lab.py` (needs pandas + numpy only).

## Methodology guardrails

- **No look-ahead**: signals use day *t* close, execution next bar.
- **Costs**: 25 bps per side (brokerage + STT + stamp + impact), charged on turnover.
- **Cash yield**: idle cash earns 5% p.a. (liquid fund).
- **Listing-date aware**: stocks become selectable only after 252 bars of history.
- **IS/OOS discipline**: parameters chosen on 2008–2017 only; 2018–2026 is
  untouched validation.
- **Known bias**: universe = today's Nifty 50 (survivorship). Absolute returns
  are optimistic for strategies *and* the buy & hold benchmark alike; relative
  comparisons are the fair read.

## Winner: "Momentum Blend Band-22"

Multi-horizon cross-sectional momentum (3/6/12-month returns, 1-month skip,
averaged percentile ranks), long the top 10 equal-weighted, with a rank-22
holding band to cut turnover, a positive 12-1 momentum gate, and a market
breadth regime filter (scales to cash as fewer stocks trade above their
200-day average). Monthly rebalance.

| Metric | Strategy | Buy & Hold EW |
|---|---|---|
| CAGR (2008–2026, net) | **21.9%** | 21.5% |
| Sharpe | **1.25** | 1.15 |
| Max drawdown | **−28.3%** | −43.1% |
| Positive years | **16 / 19** | 17 / 19 |
| Out-of-sample CAGR (2018+) | **16.8%** | 18.0% |
| Out-of-sample Sharpe | 0.96 | 1.04 |
| Rolling 3-year windows positive | **100%** (worst +13%) | 99.9% (worst −7%) |
| Rolling 5-year windows positive | **100%** (worst +56%) | 100% |
| 2008 crash year | **+2%** | −27% |

Runner-up: **All-Weather 50/50** (half momentum sleeve, half low-volatility
sleeve) — highest Sharpe of all (1.30), OOS Sharpe 1.00, max DD −27.4%.

> This is research, not investment advice. Past performance (especially on a
> survivorship-biased universe) does not guarantee future results.
