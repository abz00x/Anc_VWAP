#!/usr/bin/env python3
"""Leverage / liquidation analysis for AVWAP band bounce trades.

For leveraged swing trades (e.g. Hyperliquid perps): when price reaches a band,
what's the chance it reverts to the mean (AVWAP) BEFORE a tail move liquidates
the position?  This is a *first-passage* test at each band -- target (AVWAP) vs
liquidation, whichever is hit first -- reporting win / liquidation / timeout
rates and the expected return per trade IN % OF MARGIN at the chosen leverage.

Trade side is mean-reversion: long at the lower bands (bounce up), short at the
upper bands (fade down).  Liquidation distance is approximated as
(100/leverage - maint%) against the position; real Hyperliquid liquidation is a
little tighter (maintenance margin + fees + funding are only partly modelled).

NOT ADVICE.  At high leverage a single wick = -100% of margin.

Examples
--------
    python3 leverage.py SNDK --anchor 20d --interval 30m --leverage 10
    python3 leverage.py SNDK --anchor 2026-04-01 --interval 4h --leverage 10 --horizon 30
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from anchored_vwap import analyze

LONG_BANDS = [("-1σ", "lower1"), ("-2σ", "lower2"), ("-3σ", "lower3")]
SHORT_BANDS = [("+1σ", "upper1"), ("+2σ", "upper2"), ("+3σ", "upper3")]
TARGET_ALIASES = {"avwap": "avwap", "+1": "upper1", "-1": "lower1",
                  "+2": "upper2", "-2": "lower2", "+3": "upper3", "-3": "lower3"}
ENTRY_COLS = {"-1": "lower1", "-2": "lower2", "-3": "lower3",
              "+1": "upper1", "+2": "upper2", "+3": "upper3"}


def _touches(res, col, warmup):
    lvl = res[col].to_numpy(float)
    hi = res["High"].to_numpy(float)
    lo = res["Low"].to_numpy(float)
    ev, prev = [], False
    for i in range(max(warmup, 1), len(res)):
        if np.isnan(lvl[i]):
            prev = False
            continue
        touch = lo[i] <= lvl[i] <= hi[i]
        if touch and not prev:
            ev.append(i)
        prev = touch
    return ev


def first_passage(res, entry_col, side, target_col, lev, liq_pct, horizon,
                  warmup, fee_frac, trend_lookback=0):
    o = res["Open"].to_numpy(float)
    h = res["High"].to_numpy(float)
    lo = res["Low"].to_numpy(float)
    c = res["Close"].to_numpy(float)
    tgt = res[target_col].to_numpy(float)
    av = res["avwap"].to_numpy(float)
    n = len(res)
    rows = []
    for i in _touches(res, entry_col, warmup):
        if trend_lookback > 0:                    # regime gate: AVWAP slope must agree
            k = i - trend_lookback
            if k < 0:
                continue
            if (side == "long") != (av[i] > av[k]):
                continue
        e = i + 1
        if e >= n:
            break
        entry = o[e]
        liq = entry * (1 - liq_pct) if side == "long" else entry * (1 + liq_pct)
        jmax = min(e + horizon, n - 1)
        kind, move = None, 0.0
        for j in range(e, jmax + 1):
            tp = tgt[j]
            if side == "long":
                if lo[j] <= liq:                      # liquidation wick (checked first)
                    kind, move = "liq", -liq_pct; break
                if not np.isnan(tp) and tp > entry and h[j] >= tp:
                    kind, move = "win", tp / entry - 1.0; break
            else:
                if h[j] >= liq:
                    kind, move = "liq", -liq_pct; break
                if not np.isnan(tp) and tp < entry and lo[j] <= tp:
                    kind, move = "win", 1.0 - tp / entry; break
        if kind is None:                              # timeout -> mark to market
            kind = "timeout"
            move = (c[jmax] / entry - 1.0) if side == "long" else (entry / c[jmax] - 1.0)
        acct = -1.0 if kind == "liq" else lev * move - 2 * lev * fee_frac
        rows.append(dict(kind=kind, move=move, acct=acct, bars=jmax - e))
    return pd.DataFrame(rows)


def summarize(df):
    if df.empty:
        return None
    return dict(
        n=len(df),
        win=float((df["kind"] == "win").mean()),
        liq=float((df["kind"] == "liq").mean()),
        timeout=float((df["kind"] == "timeout").mean()),
        ev=float(df["acct"].mean()),
        median_acct=float(df["acct"].median()),
    )


def consolidated(ticker, anchor, intervals, band, side, target_col, lev,
                 liq_pct, horizon, warmup, fee_frac, window):
    """One row per timeframe for a single (band, side) -- cross-TF robustness."""
    col = ENTRY_COLS.get(band.lower().replace("σ", ""), band)
    print(f"\n{ticker}  —  {side} {band} -> {target_col}   {lev:g}x "
          f"(liq {liq_pct * 100:.1f}%)   anchor {anchor}   horizon {horizon}")
    print(f"{'TF':<6}{'N':>4}{'WIN%':>6}{'LIQ%':>6}{'TIME%':>7}{'EV/margin':>11}{'MEDIAN':>9}")
    print("-" * 49)
    for iv in intervals:
        try:
            res, _ = analyze(ticker, anchor=anchor, interval=iv, swing_window_days=window)
            s = summarize(first_passage(res, col, side, target_col, lev,
                                        liq_pct, horizon, warmup, fee_frac))
            if not s:
                print(f"{iv:<6}   no touches")
                continue
            flag = "  <-- +EV" if s["ev"] > 0 else ""
            print(f"{iv:<6}{s['n']:>4}{s['win'] * 100:>5.0f}%{s['liq'] * 100:>5.0f}%"
                  f"{s['timeout'] * 100:>6.0f}%{s['ev'] * 100:>+10.1f}%"
                  f"{s['median_acct'] * 100:>+8.1f}%{flag}")
        except Exception as e:
            print(f"{iv:<6}   {e}")
    print("\nNote: timeframes of the same window overlap -> NOT independent samples.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Leverage/liquidation first-passage analysis at AVWAP bands",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("ticker")
    p.add_argument("-a", "--anchor", default="ytd")
    p.add_argument("-i", "--interval", default="1d")
    p.add_argument("-I", "--intervals", nargs="+",
                   help="compare these timeframes for one band/side (see --band/--side)")
    p.add_argument("--band", default="-1",
                   help="entry band for --intervals compare (default: -1)")
    p.add_argument("--side", choices=["long", "short"], default="long",
                   help="side for --intervals compare (default: long)")
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--maint-pct", type=float, default=0.5,
                   help="maintenance buffer %% subtracted from liq distance (default: 0.5)")
    p.add_argument("--target-band", default="avwap",
                   help="reversion target: avwap (default), +1/-1, ...")
    p.add_argument("--horizon", type=int, default=20,
                   help="max bars to reach target before timeout (default: 20)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--fee-bps", type=float, default=4.0,
                   help="taker fee per side in bps on notional (default: 4)")
    p.add_argument("--window", type=int, default=365)
    args = p.parse_args(argv)

    liq_pct = (100.0 / args.leverage - args.maint_pct) / 100.0
    if liq_pct <= 0:
        p.error("leverage too high for the maintenance buffer (liq distance <= 0)")
    fee_frac = args.fee_bps / 1e4
    tgt_col = TARGET_ALIASES.get(args.target_band.lower().replace("σ", ""), args.target_band)

    if args.intervals:
        consolidated(args.ticker.upper(), args.anchor, args.intervals, args.band,
                     args.side, tgt_col, args.leverage, liq_pct, args.horizon,
                     args.warmup, fee_frac, args.window)
        return 0

    res, ats = analyze(args.ticker.upper(), anchor=args.anchor,
                       interval=args.interval, swing_window_days=args.window)

    print(f"\n{args.ticker.upper()} {args.interval}  anchor {ats.date()}  ({len(res)} bars)")
    print(f"{args.leverage:g}x  ->  liquidation at {liq_pct * 100:.1f}% against you   |   "
          f"target {tgt_col}   horizon {args.horizon}   fee {args.fee_bps}bps/side")
    print(f"EV is per-trade return as % of margin (win = +{args.leverage:g}x move, "
          f"liq = -100%)")
    print(f"\n{'BAND':<6}{'SIDE':<7}{'N':>4}{'WIN%':>6}{'LIQ%':>6}{'TIME%':>7}"
          f"{'EV/margin':>11}{'MEDIAN':>9}")
    print("-" * 60)
    any_row = False
    for name, col in LONG_BANDS + SHORT_BANDS:
        side = "long" if (name, col) in LONG_BANDS else "short"
        s = summarize(first_passage(res, col, side, tgt_col, args.leverage,
                                    liq_pct, args.horizon, args.warmup, fee_frac))
        if not s:
            continue
        any_row = True
        flag = "  <-- +EV" if s["ev"] > 0 else ""
        print(f"{name:<6}{side:<7}{s['n']:>4}{s['win'] * 100:>5.0f}%{s['liq'] * 100:>5.0f}%"
              f"{s['timeout'] * 100:>6.0f}%{s['ev'] * 100:>+10.1f}%{s['median_acct'] * 100:>+8.1f}%{flag}")
    if not any_row:
        print("  no band touches in this window")
    else:
        print("\nWIN = reverted to target before liq.  EV>0 means +EV at this leverage "
              "ON THIS SAMPLE (small N -> low confidence).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
