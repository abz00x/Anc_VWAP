#!/usr/bin/env python3
"""Quick 'is it actionable right now' readout for AVWAP band setups.

For each timeframe, prints current price, AVWAP, z-score, the nearest band and
% distance to it, plus a transparent tag based on the z-zone and band proximity.
Run it on a loop to use as a poor-man's alert, e.g.:

    while true; do clear; python3 monitor.py SNDK --anchor low --window 10 \
        --intervals 30m 15m; sleep 300; done

Examples
--------
    python3 monitor.py SNDK --anchor 2026-04-01 --intervals 1d 4h 1h
    python3 monitor.py SNDK --anchor low --window 10 --intervals 30m 15m --near 1.0
"""
from __future__ import annotations

import argparse

from anchored_vwap import analyze

BANDS = [("+3σ", "upper3"), ("+2σ", "upper2"), ("+1σ", "upper1"),
         ("AVWAP", "avwap"),
         ("-1σ", "lower1"), ("-2σ", "lower2"), ("-3σ", "lower3")]

# +2σ/+3σ are the exhaustion caps (fade/trim); AVWAP and below are the
# value/dip side where longs set up.  +1σ is treated as a buy band (pullback).
CAP_BANDS = {"+3σ", "+2σ"}
BUY_BANDS = {"+1σ", "AVWAP", "-1σ", "-2σ", "-3σ"}


def nearest_band(last):
    """(band name, signed % distance) for the closest band; + = price above it."""
    px = float(last["Close"])
    best = None
    for name, col in BANDS:
        lvl = float(last[col])
        if lvl == 0 or lvl != lvl:  # zero / NaN guard
            continue
        dist = (px / lvl - 1.0) * 100.0
        if best is None or abs(dist) < abs(best[1]):
            best = (name, dist)
    return best


def tag(last, near_pct: float) -> str:
    """Transparent state label from z-zone + nearest-band proximity."""
    z = float(last["z"])
    name, dist = nearest_band(last)
    near = abs(dist) <= near_pct
    if z >= 2.5:
        return f"🔴 EXTENDED (z {z:+.1f}) — trim / no new longs"
    if z <= -2.5:
        return f"🔥 deep oversold (z {z:+.1f}) — snapback watch"
    if near and name in CAP_BANDS:
        return f"🔴 at {name} cap ({dist:+.1f}%) — fade/trim, don't chase"
    if near and name in BUY_BANDS:
        return f"👀 at {name} ({dist:+.1f}%) — potential long, watch trigger"
    if z >= 2.0:
        return f"🟠 stretched (z {z:+.1f}) — hold, don't chase"
    if z <= -2.0:
        return f"🟢 oversold (z {z:+.1f}) — pullback buy zone"
    return f"· neutral (z {z:+.1f}); nearest {name} {dist:+.1f}%"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Live AVWAP band proximity / state monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("ticker")
    p.add_argument("-a", "--anchor", default="ytd",
                   help="ytd | high | low | YYYY-MM-DD | NNd/w/m/y  (default: ytd)")
    p.add_argument("-I", "--intervals", nargs="+", default=["1d", "4h", "1h"],
                   help="timeframes to check (default: 1d 4h 1h)")
    p.add_argument("--near", type=float, default=1.5,
                   help="%% distance that counts as 'at' a band (default: 1.5)")
    p.add_argument("--window", type=int, default=365,
                   help="lookback days for high/low swing anchor (default: 365)")
    args = p.parse_args(argv)

    t = args.ticker.upper()
    print(f"\n{t}   anchor {args.anchor}")
    for iv in args.intervals:
        try:
            res, ats = analyze(t, anchor=args.anchor, interval=iv,
                               swing_window_days=args.window)
            last = res.iloc[-1]
            print(f"  {iv:>4}  px {float(last['Close']):>9.2f}  "
                  f"AVWAP {float(last['avwap']):>9.2f}  z {float(last['z']):+5.2f}   "
                  f"{tag(last, args.near)}")
        except Exception as e:
            print(f"  {iv:>4}  {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
