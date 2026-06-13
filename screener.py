#!/usr/bin/env python3
"""Anchored VWAP screener.

Scan one or more tickers, compute each one's Anchored VWAP from a chosen anchor,
and report how stretched price is from the AVWAP as a z-score (volume-weighted
std-dev units).  Mirrors the snap-back zones of the reference z-score indicator,
but anchored instead of a rolling 20d SMA.

Examples
--------
    python screener.py AAPL MSFT NVDA
    python screener.py -w watchlist.txt --anchor ytd
    python screener.py TSLA --anchor low --window 365      # anchor at 1y low
    python screener.py SPY  --anchor 2026-01-02            # anchor at a date
    python screener.py -w watchlist.txt --min-abs-z 2 --csv out.csv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from anchored_vwap import analyze, zone_label


def load_tickers(args) -> list[str]:
    tickers = list(args.tickers)
    if args.watchlist:
        with open(args.watchlist) as fh:
            for line in fh:
                line = line.split("#")[0].strip()          # strip comments
                if line:
                    tickers += line.replace(",", " ").split()
    seen: list[str] = []
    for t in tickers:
        t = t.upper().strip()
        if t and t not in seen:
            seen.append(t)
    return seen


def _print_table(df: pd.DataFrame, anchor: str) -> None:
    print(f"\nAnchored VWAP screener  —  anchor: {anchor}   ({len(df)} ticker(s))")
    print(f"{'TICKER':<8}{'CLOSE':>10}{'AVWAP':>10}{'DEV%':>9}{'Z':>8}   ZONE")
    print("-" * 62)
    for _, r in df.iterrows():
        print(f"{r['ticker']:<8}{r['close']:>10.2f}{r['avwap']:>10.2f}"
              f"{r['dev_pct']:>7.1f}%{r['z']:>8.2f}   {r['zone']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Anchored VWAP screener (z-score of price deviation from AVWAP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("tickers", nargs="*", help="tickers, e.g. AAPL MSFT NVDA")
    p.add_argument("-w", "--watchlist",
                   help="file with tickers (one or many per line, # comments ok)")
    p.add_argument("-a", "--anchor", default="ytd",
                   help="ytd | high | low | YYYY-MM-DD | NNd/w/m/y  (default: ytd)")
    p.add_argument("-i", "--interval", default="1d",
                   help="bar interval, e.g. 1d 1h 1wk  (default: 1d)")
    p.add_argument("--window", type=int, default=365,
                   help="lookback days for high/low swing anchor (default: 365)")
    p.add_argument("--min-abs-z", type=float, default=0.0,
                   help="only show rows with |z| >= this (default: 0 = show all)")
    p.add_argument("--sort", choices=["absz", "z", "dev", "ticker"], default="absz",
                   help="sort order (default: absz = most stretched first)")
    p.add_argument("--csv", help="also write the results table to this CSV path")
    args = p.parse_args(argv)

    tickers = load_tickers(args)
    if not tickers:
        p.error("no tickers given (pass tickers or use --watchlist)")

    rows = []
    for t in tickers:
        try:
            res, anchor_ts = analyze(t, anchor=args.anchor, interval=args.interval,
                                     swing_window_days=args.window)
            last = res.iloc[-1]
            z = float(last["z"])
            rows.append(dict(
                ticker=t,
                close=float(last["Close"]),
                avwap=float(last["avwap"]),
                z=z,
                dev_pct=float(last["dev_pct"]),
                bars=len(res),
                anchor=anchor_ts.date().isoformat(),
                zone=zone_label(z),
            ))
        except Exception as e:  # keep scanning the rest of the list
            print(f"  ! {t}: {e}", file=sys.stderr)

    if not rows:
        print("no results")
        return 1

    df = pd.DataFrame(rows)
    df = df[df["z"].abs() >= args.min_abs_z]
    if df.empty:
        print(f"no tickers with |z| >= {args.min_abs_z}")
        return 0

    if args.sort == "absz":
        df = df.reindex(df["z"].abs().sort_values(ascending=False).index)
    elif args.sort == "z":
        df = df.sort_values("z")
    elif args.sort == "dev":
        df = df.reindex(df["dev_pct"].abs().sort_values(ascending=False).index)
    else:
        df = df.sort_values("ticker")

    _print_table(df, args.anchor)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
