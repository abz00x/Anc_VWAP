#!/usr/bin/env python3
"""Back-solve the best exit for a band entry.

For each historical entry (a touch of the entry band), race candidate exits
(AVWAP, the bands beyond it, and a few fixed %s) against your stop as a
first-passage test, and report P(hit), stop-out%, timeout%, average R and EV at
leverage.  Nearer exits hit more often but pay less -- this finds the sweet spot
between "higher chance" and "bigger win".

Examples
--------
    python3 exits.py SNDK --anchor 20d --interval 30m --band -1 --side long --stop-pct 3
    python3 exits.py SNDK --anchor 20d --interval 15m --band -1 --side long --leverage 10
"""
from __future__ import annotations

import argparse

import numpy as np

from anchored_vwap import analyze
from leverage import ENTRY_COLS, _touches


def candidate_targets(side):
    """(label, kind, value) exits to test.  band -> column; pct -> fraction."""
    if side == "long":
        bands = [("AVWAP", "avwap"), ("+1σ", "upper1"), ("+2σ", "upper2"), ("+3σ", "upper3")]
    else:
        bands = [("AVWAP", "avwap"), ("-1σ", "lower1"), ("-2σ", "lower2"), ("-3σ", "lower3")]
    pcts = [("+2%", 0.02), ("+4%", 0.04), ("+6%", 0.06), ("+8%", 0.08)]
    return [(lbl, "band", col) for lbl, col in bands] + [(lbl, "pct", v) for lbl, v in pcts]


def eval_target(res, entries, side, kind, tval, stop_pct, lev, horizon, fee_frac):
    o = res["Open"].to_numpy(float)
    h = res["High"].to_numpy(float)
    lo = res["Low"].to_numpy(float)
    c = res["Close"].to_numpy(float)
    band = res[tval].to_numpy(float) if kind == "band" else None
    n = len(res)
    moves, kinds, bars = [], [], []
    for e in entries:
        entry = o[e]
        if kind == "band":                       # only valid if target is the right side of entry
            t0 = band[e]
            if np.isnan(t0) or (side == "long" and t0 <= entry) or (side == "short" and t0 >= entry):
                continue
        stop = entry * (1 - stop_pct / 100) if side == "long" else entry * (1 + stop_pct / 100)
        jmax = min(e + horizon, n - 1)
        exit_px = res_kind = None
        exit_j = jmax
        for j in range(e, jmax + 1):
            tp = band[j] if kind == "band" else (entry * (1 + tval) if side == "long"
                                                 else entry * (1 - tval))
            if side == "long":
                if lo[j] <= stop:
                    exit_px, res_kind, exit_j = stop, "stop", j; break
                if (kind != "band" or (not np.isnan(tp) and tp > entry)) and h[j] >= tp:
                    exit_px, res_kind, exit_j = tp, "win", j; break
            else:
                if h[j] >= stop:
                    exit_px, res_kind, exit_j = stop, "stop", j; break
                if (kind != "band" or (not np.isnan(tp) and tp < entry)) and lo[j] <= tp:
                    exit_px, res_kind, exit_j = tp, "win", j; break
        if exit_px is None:
            exit_px, res_kind = c[jmax], "timeout"
        move = (exit_px / entry - 1) if side == "long" else (1 - exit_px / entry)
        moves.append(move)
        kinds.append(res_kind)
        bars.append(exit_j - e)
    if not moves:
        return None
    moves = np.array(moves)
    kinds = np.array(kinds)
    risk = stop_pct / 100.0
    return dict(
        n=len(moves),
        hit=float((kinds == "win").mean()),
        stop=float((kinds == "stop").mean()),
        timeout=float((kinds == "timeout").mean()),
        avg_r=float(moves.mean() / risk),
        ev=float(lev * moves.mean() - 2 * lev * fee_frac),
        med_bars=int(np.median(bars)),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Back-solve the highest-EV / highest-probability exit for a band entry",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("ticker")
    p.add_argument("-a", "--anchor", default="ytd")
    p.add_argument("-i", "--interval", default="1d")
    p.add_argument("--band", default="-1", help="entry band (default: -1)")
    p.add_argument("--side", choices=["long", "short"], default="long")
    p.add_argument("--stop-pct", type=float, default=3.0, help="stop %% (default: 3.0)")
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--window", type=int, default=365)
    args = p.parse_args(argv)

    entry_col = ENTRY_COLS.get(args.band.lower().replace("σ", ""), args.band)
    fee_frac = args.fee_bps / 1e4
    res, ats = analyze(args.ticker.upper(), anchor=args.anchor,
                       interval=args.interval, swing_window_days=args.window)
    entries = [i + 1 for i in _touches(res, entry_col, args.warmup) if i + 1 < len(res)]

    print(f"\n{args.ticker.upper()} {args.interval}  anchor {ats.date()}  ({len(res)} bars)")
    print(f"entry: {args.side} {args.band}  ({len(entries)} signals)   stop {args.stop_pct}%  "
          f"{args.leverage:g}x   horizon {args.horizon}   fee {args.fee_bps}bps")
    print(f"\n{'EXIT':<7}{'N':>4}{'HIT%':>6}{'STOP%':>6}{'TIME%':>7}{'AVG_R':>7}"
          f"{'EV/margin':>11}{'BARS':>6}")
    print("-" * 54)
    rows = []
    for label, kind, tval in candidate_targets(args.side):
        s = eval_target(res, entries, args.side, kind, tval, args.stop_pct,
                        args.leverage, args.horizon, fee_frac)
        if s:
            rows.append((label, s))
    rows.sort(key=lambda kv: kv[1]["ev"], reverse=True)
    for label, s in rows:
        print(f"{label:<7}{s['n']:>4}{s['hit'] * 100:>5.0f}%{s['stop'] * 100:>5.0f}%"
              f"{s['timeout'] * 100:>6.0f}%{s['avg_r']:>+7.2f}{s['ev'] * 100:>+10.1f}%"
              f"{s['med_bars']:>6}")
    if rows:
        best_ev = max(rows, key=lambda kv: kv[1]["ev"])
        best_hit = max(rows, key=lambda kv: kv[1]["hit"])
        print(f"\nbest EV:  {best_ev[0]} ({best_ev[1]['ev'] * 100:+.1f}%/margin)   |   "
              f"highest hit-rate: {best_hit[0]} ({best_hit[1]['hit'] * 100:.0f}%)")
        print("sorted by EV.  Small N -> low confidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
