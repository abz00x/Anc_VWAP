#!/usr/bin/env python3
"""Grid-sweep the AVWAP strategy to see if ANY stop/target/exit combo works.

Fetches once, then runs strategy.simulate across signals × stops × exits and
ranks by expectancy, flagging combos whose return beats buy & hold. One run
tells you whether the edge survives as a mechanical system.

Examples
--------
    python3 sweep.py SNDK --anchor 20d --interval 30m
    python3 sweep.py SNDK --anchor 2026-04-01 --interval 4h --time-stop 30
"""
from __future__ import annotations

import argparse

import pandas as pd

import strategy as st
from anchored_vwap import analyze

SIGNALS = ["reclaim", "dip", "pullback"]
STOPS = [2.0, 3.0, 5.0]
# label, exit_mode, param
EXITS = [("tgt1.5R", "target", 1.5), ("tgt2R", "target", 2.0), ("tgt3R", "target", 3.0),
         ("trail5%", "trail", 5.0), ("trail8%", "trail", 8.0), ("band", "band", None)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Grid sweep of AVWAP strategy parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("ticker")
    p.add_argument("-a", "--anchor", default="ytd")
    p.add_argument("-i", "--interval", default="1d")
    p.add_argument("--time-stop", type=int, default=20)
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--window", type=int, default=365)
    p.add_argument("--top", type=int, default=20, help="rows to show (default: 20)")
    args = p.parse_args(argv)

    res, ats = analyze(args.ticker.upper(), anchor=args.anchor,
                       interval=args.interval, swing_window_days=args.window)
    bh = float(res["Close"].iloc[-1] / res["Close"].iloc[0] - 1.0)

    rows = []
    for sig in SIGNALS:
        band, mode = st.PRESETS[sig]
        for stop in STOPS:
            for label, xm, param in EXITS:
                tr = st.simulate(res, band, mode, stop,
                                 target_r=param if xm == "target" else 2.0,
                                 time_stop=args.time_stop, exit_mode=xm,
                                 trail_pct=param if xm == "trail" else 5.0,
                                 exit_band="avwap")
                m = st.metrics(tr, res, args.risk_pct)
                if m["n"] == 0:
                    continue
                rows.append(dict(signal=sig, stop=stop, exit=label, n=m["n"],
                                 win=m["win_rate"], expR=m["expectancy_r"],
                                 pf=m["profit_factor"], strat=m["strat_return"],
                                 dd=m["max_dd"]))

    print(f"\n{args.ticker.upper()} {args.interval}  anchor {ats.date()}  ({len(res)} bars)")
    print(f"buy & hold over window: {bh * 100:+.1f}%   "
          f"(time-stop {args.time_stop}, risk {args.risk_pct}%/trade)")
    if not rows:
        print("no trades in any combo")
        return 0

    df = pd.DataFrame(rows).sort_values("expR", ascending=False).head(args.top)
    print(f"\n{'SIGNAL':<9}{'STOP':>5}{'EXIT':>9}{'N':>4}{'WIN':>6}{'EXP_R':>7}"
          f"{'PF':>6}{'STRAT':>8}{'vsB&H':>7}{'MAXDD':>7}")
    print("-" * 78)
    for _, r in df.iterrows():
        beat = " *" if r["strat"] > bh else ""
        pf = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
        print(f"{r['signal']:<9}{r['stop']:>5.1f}{r['exit']:>9}{int(r['n']):>4}"
              f"{r['win'] * 100:>5.0f}%{r['expR']:>+7.2f}{pf:>6}"
              f"{r['strat'] * 100:>+7.1f}%{(r['strat'] - bh) * 100:>+6.1f}%"
              f"{r['dd'] * 100:>6.1f}%{beat}")
    print("\n  * = beat buy & hold   |   EXP_R = expectancy (R) per trade   "
          "|   sorted by EXP_R")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
