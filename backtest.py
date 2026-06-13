#!/usr/bin/env python3
"""Backtest price behaviour at the Anchored VWAP bands.

For a ticker / anchor / interval, find every bar where price *touches* (or
crosses) each band, split them into support vs resistance tests, and measure
the forward return over a horizon -- i.e. does price bounce?

  * support test    = price approached the level from above (level was below
                      the prior close).  A "bounce" = positive forward return.
  * resistance test = price approached from below.  A "rejection" = negative
                      forward return.

BOUNCE% is the share of events that resolved in the bounce/rejection direction.

Examples
--------
    python backtest.py SNDK --anchor 2026-04-01 --interval 4h --horizon 10
    python backtest.py SNDK --anchor low --interval 1d --mode cross_down
    python backtest.py SNDK --anchor 2026-04-01 --interval 4h --chart sndk.png --csv ev.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from anchored_vwap import analyze

# label, column  (top -> bottom)
LEVELS = [("+3σ", "upper3"), ("+2σ", "upper2"), ("+1σ", "upper1"),
          ("AVWAP", "avwap"),
          ("-1σ", "lower1"), ("-2σ", "lower2"), ("-3σ", "lower3")]


def detect_events(res: pd.DataFrame, col: str, mode: str) -> list[int]:
    """Integer positions where price touches / crosses the ``col`` level."""
    lvl = res[col].to_numpy(float)
    hi = res["High"].to_numpy(float)
    lo = res["Low"].to_numpy(float)
    cl = res["Close"].to_numpy(float)
    events: list[int] = []
    prev_touch = False
    for i in range(1, len(res)):
        if np.isnan(lvl[i]) or np.isnan(lvl[i - 1]):
            prev_touch = False
            continue
        if mode == "touch":
            touch = lo[i] <= lvl[i] <= hi[i]
            if touch and not prev_touch:      # only the bar that enters the band
                events.append(i)
            prev_touch = touch
        elif mode == "cross_up":
            if cl[i - 1] < lvl[i - 1] and cl[i] >= lvl[i]:
                events.append(i)
        elif mode == "cross_down":
            if cl[i - 1] > lvl[i - 1] and cl[i] <= lvl[i]:
                events.append(i)
    return events


def forward(res: pd.DataFrame, i: int, h: int):
    """(close-to-close return, max favourable excursion, max adverse) over h bars."""
    cl = res["Close"].to_numpy(float)
    hi = res["High"].to_numpy(float)
    lo = res["Low"].to_numpy(float)
    j = min(i + h, len(res) - 1)
    if j <= i:
        return None
    base = cl[i]
    return (cl[j] / base - 1.0,
            hi[i + 1:j + 1].max() / base - 1.0,
            lo[i + 1:j + 1].min() / base - 1.0)


def role_of(res: pd.DataFrame, i: int, col: str) -> str:
    prior_close = res["Close"].to_numpy(float)[i - 1]
    level = res[col].to_numpy(float)[i]
    return "support" if prior_close > level else "resistance"


def baseline_forward(res: pd.DataFrame, horizon: int, warmup: int):
    """Unconditional forward-return stats over all bars -- the drift to beat.

    In a strong trend every level looks like it 'bounces', so the only honest
    read is each level's return *relative to* this baseline.
    """
    cl = res["Close"].to_numpy(float)
    n = len(res)
    rets = []
    for i in range(max(warmup, 0), n - 1):
        j = min(i + horizon, n - 1)
        if j > i:
            rets.append(cl[j] / cl[i] - 1.0)
    if not rets:
        return None
    r = np.array(rets)
    return dict(n=len(r), mean=float(r.mean()), median=float(np.median(r)),
                pos=float((r > 0).mean()))


def run(res: pd.DataFrame, mode: str, horizon: int, warmup: int = 5):
    base = baseline_forward(res, horizon, warmup)
    rows = []
    events_by_level: dict[str, list[int]] = {}
    for name, col in LEVELS:
        evs = [i for i in detect_events(res, col, mode) if i >= warmup]
        events_by_level[name] = evs
        for role in ("support", "resistance"):
            sel = [i for i in evs if role_of(res, i, col) == role]
            fs = [f for f in (forward(res, i, horizon) for i in sel) if f]
            if not fs:
                continue
            fwd = np.array([f[0] for f in fs])
            mfe = np.array([f[1] for f in fs])
            mae = np.array([f[2] for f in fs])
            pos = float((fwd > 0).mean())
            excess = float(fwd.mean() - base["mean"]) if base else float("nan")
            rows.append(dict(level=name, role=role, n=len(fwd),
                             mean_fwd=float(fwd.mean()), excess=excess,
                             med_fwd=float(np.median(fwd)), pos=pos,
                             mfe=float(mfe.mean()), mae=float(mae.mean())))
    return pd.DataFrame(rows), events_by_level, base


def events_table(res: pd.DataFrame, mode: str, horizon: int, warmup: int = 5) -> pd.DataFrame:
    rows = []
    for name, col in LEVELS:
        for i in detect_events(res, col, mode):
            if i < warmup:
                continue
            f = forward(res, i, horizon)
            rows.append(dict(time=res.index[i], level=name, role=role_of(res, i, col),
                             close=float(res["Close"].iloc[i]),
                             level_val=float(res[col].iloc[i]),
                             fwd=f[0] if f else np.nan,
                             mfe=f[1] if f else np.nan,
                             mae=f[2] if f else np.nan))
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True) if rows else pd.DataFrame()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Backtest Anchored VWAP band touches / crosses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("ticker")
    p.add_argument("-a", "--anchor", default="ytd",
                   help="ytd | high | low | YYYY-MM-DD | NNd/w/m/y  (default: ytd)")
    p.add_argument("-i", "--interval", default="1d", help="1h | 4h | 1d | 1wk  (default: 1d)")
    p.add_argument("--mode", choices=["touch", "cross_up", "cross_down"], default="touch",
                   help="event type (default: touch)")
    p.add_argument("--horizon", type=int, default=10,
                   help="forward bars to measure (default: 10)")
    p.add_argument("--warmup", type=int, default=5,
                   help="skip events in the first N bars after the anchor, where the "
                        "bands are still degenerate / stacked (default: 5)")
    p.add_argument("--window", type=int, default=365,
                   help="lookback days for high/low swing anchor (default: 365)")
    p.add_argument("--csv", help="write per-event detail to this CSV")
    p.add_argument("--chart", help="save a PNG chart with bands + touch markers")
    args = p.parse_args(argv)

    res, anchor_ts = analyze(args.ticker.upper(), anchor=args.anchor,
                             interval=args.interval, swing_window_days=args.window)
    summary, events_by_level, base = run(res, args.mode, args.horizon, args.warmup)

    title = (f"{args.ticker.upper()} {args.interval}  anchor {anchor_ts.date()}  "
             f"— {args.mode}, fwd {args.horizon} bars, warmup {args.warmup}  ({len(res)} bars)")
    print("\n" + title)
    if base:
        print(f"baseline (all bars): mean {base['mean'] * 100:+.1f}%   "
              f"median {base['median'] * 100:+.1f}%   positive {base['pos'] * 100:.0f}%   "
              f"n={base['n']}")
        print("EXCESS = level's mean forward return minus this baseline (the real edge)")
    if summary.empty:
        print("  no events detected")
    else:
        print(f"\n{'LEVEL':<7}{'ROLE':<12}{'N':>3}{'MEAN_FWD':>10}{'EXCESS':>9}"
              f"{'MEDIAN':>9}{'POS%':>7}{'AVG_MFE':>9}{'AVG_MAE':>9}")
        print("-" * 75)
        for _, r in summary.iterrows():
            print(f"{r['level']:<7}{r['role']:<12}{int(r['n']):>3}"
                  f"{r['mean_fwd'] * 100:>9.1f}%{r['excess'] * 100:>8.1f}%"
                  f"{r['med_fwd'] * 100:>8.1f}%{r['pos'] * 100:>6.0f}%"
                  f"{r['mfe'] * 100:>8.1f}%{r['mae'] * 100:>8.1f}%")

    if args.csv:
        ev = events_table(res, args.mode, args.horizon, args.warmup)
        ev.to_csv(args.csv, index=False)
        print(f"\nwrote {len(ev)} events -> {args.csv}")

    if args.chart:
        from plotting import plot_avwap
        plot_avwap(res, title, args.chart, events=events_by_level)
        print(f"wrote chart -> {args.chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
