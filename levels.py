#!/usr/bin/env python3
"""Show the Anchored VWAP + ±1/±2/±3σ bands for a ticker across timeframes.

Mirrors the TradingView "Anchored VWAP (hlc3)" header readout, on each of the
requested intervals (default 1h / 4h / 1d).

Examples
--------
    python levels.py SNDK --anchor 2026-04-01
    python levels.py SNDK --anchor low --intervals 1h 4h 1d
"""
from __future__ import annotations

import argparse

from anchored_vwap import analyze, zone_label


def show(ticker: str, anchor: str, intervals: list[str], window: int) -> None:
    for iv in intervals:
        try:
            res, ats = analyze(ticker, anchor=anchor, interval=iv, swing_window_days=window)
            last = res.iloc[-1]
            z = float(last["z"])
            print(f"\n{ticker}  {iv}   anchor {ats.date()}   ({len(res)} bars)")
            print(f"  close   {last['Close']:>11.2f}    z {z:+.2f}    "
                  f"dev {last['dev_pct']:+.1f}%   {zone_label(z)}")
            print(f"  +3σ     {last['upper3']:>11.2f}")
            print(f"  +2σ     {last['upper2']:>11.2f}")
            print(f"  +1σ     {last['upper1']:>11.2f}    (blue line)")
            print(f"  AVWAP   {last['avwap']:>11.2f}")
            print(f"  -1σ     {last['lower1']:>11.2f}    (blue line)")
            print(f"  -2σ     {last['lower2']:>11.2f}")
            print(f"  -3σ     {last['lower3']:>11.2f}")
        except Exception as e:
            print(f"\n{ticker}  {iv}: {e}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Anchored VWAP levels across timeframes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("ticker")
    p.add_argument("-a", "--anchor", default="ytd",
                   help="ytd | high | low | YYYY-MM-DD | NNd/w/m/y  (default: ytd)")
    p.add_argument("-I", "--intervals", nargs="+", default=["1h", "4h", "1d"],
                   help="timeframes to show (default: 1h 4h 1d)")
    p.add_argument("--window", type=int, default=365,
                   help="lookback days for high/low swing anchor (default: 365)")
    args = p.parse_args(argv)
    show(args.ticker.upper(), args.anchor, args.intervals, args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
