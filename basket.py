#!/usr/bin/env python3
"""Validate a band setup across many tickers (out-of-sample-ish).

Runs the leverage first-passage test (default: long -1σ -> AVWAP) on every
ticker and POOLS the events, so you get a real sample size instead of one
ticker's N.  The pooled WIN/LIQ/EV and the share of tickers that come out +EV
tell you whether the edge generalises or was a single-name story.

Note: timeframes of one ticker overlap, but *different tickers* are largely
independent -- so pooling across names is the closest thing to out-of-sample
you get without waiting for new data.

Examples
--------
    python3 basket.py -w basket.txt --anchor 20d --interval 30m --leverage 10
    python3 basket.py AAPL NVDA TSLA BTC-USD ETH-USD --interval 1d --anchor ytd
    python3 basket.py -w basket.txt --interval 5m --anchor 10d   # stricter liq estimate
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from anchored_vwap import analyze
from leverage import ENTRY_COLS, TARGET_ALIASES, first_passage, summarize


def load_tickers(args) -> list[str]:
    tickers = list(args.tickers)
    if args.watchlist:
        with open(args.watchlist) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if line:
                    tickers += line.replace(",", " ").split()
    seen: list[str] = []
    for t in tickers:
        t = t.upper().strip()
        if t and t not in seen:
            seen.append(t)
    return seen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Validate a band/leverage setup across a basket of tickers",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("tickers", nargs="*", help="tickers, e.g. AAPL NVDA BTC-USD")
    p.add_argument("-w", "--watchlist", help="file with tickers (# comments ok)")
    p.add_argument("-a", "--anchor", default="20d")
    p.add_argument("-i", "--interval", default="30m")
    p.add_argument("--band", default="-1", help="entry band (default: -1)")
    p.add_argument("--side", choices=["long", "short"], default="long")
    p.add_argument("--target-band", default="avwap")
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--maint-pct", type=float, default=0.5)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--window", type=int, default=365)
    args = p.parse_args(argv)

    tickers = load_tickers(args)
    if not tickers:
        p.error("no tickers (pass tickers or -w file)")

    entry_col = ENTRY_COLS.get(args.band.lower().replace("σ", ""), args.band)
    tgt_col = TARGET_ALIASES.get(args.target_band.lower().replace("σ", ""), args.target_band)
    liq_pct = (100.0 / args.leverage - args.maint_pct) / 100.0
    if liq_pct <= 0:
        p.error("leverage too high for the maintenance buffer (liq distance <= 0)")
    fee_frac = args.fee_bps / 1e4

    print(f"\n{args.side} {args.band} -> {tgt_col}   {args.leverage:g}x (liq {liq_pct * 100:.1f}%)   "
          f"anchor {args.anchor}  {args.interval}  horizon {args.horizon}")
    print(f"{'TICKER':<10}{'N':>4}{'WIN%':>6}{'LIQ%':>6}{'TIME%':>7}{'EV/margin':>11}")
    print("-" * 44)

    pool, n_pos, n_neg = [], 0, 0
    for t in tickers:
        try:
            res, _ = analyze(t, anchor=args.anchor, interval=args.interval,
                             swing_window_days=args.window)
            df = first_passage(res, entry_col, args.side, tgt_col, args.leverage,
                               liq_pct, args.horizon, args.warmup, fee_frac)
            s = summarize(df)
            if not s:
                print(f"{t:<10}   no touches")
                continue
            pool.append(df)
            n_pos, n_neg = (n_pos + 1, n_neg) if s["ev"] > 0 else (n_pos, n_neg + 1)
            print(f"{t:<10}{s['n']:>4}{s['win'] * 100:>5.0f}%{s['liq'] * 100:>5.0f}%"
                  f"{s['timeout'] * 100:>6.0f}%{s['ev'] * 100:>+10.1f}%"
                  f"{'  +EV' if s['ev'] > 0 else ''}")
        except Exception as e:
            print(f"{t:<10}   {e}", file=sys.stderr)

    if not pool:
        print("\nno band touches across the basket")
        return 0

    ps = summarize(pd.concat(pool, ignore_index=True))
    print("-" * 44)
    print(f"{'POOLED':<10}{ps['n']:>4}{ps['win'] * 100:>5.0f}%{ps['liq'] * 100:>5.0f}%"
          f"{ps['timeout'] * 100:>6.0f}%{ps['ev'] * 100:>+10.1f}%")
    print(f"\ntickers +EV: {n_pos}/{n_pos + n_neg}    pooled EV/trade: "
          f"{ps['ev'] * 100:+.1f}% of margin    pooled liq-rate: {ps['liq'] * 100:.0f}%")
    print("If POOLED is +EV with a low liq-rate AND most tickers are +EV, the edge "
          "generalises.\nIf it's carried by one or two names, it doesn't.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
