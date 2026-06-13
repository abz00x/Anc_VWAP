#!/usr/bin/env python3
"""Event-driven backtest of an actual AVWAP trading rule (long-only).

Unlike backtest.py (which only measures what happens after a band is touched),
this simulates real trades: entry signal -> fill next bar's open -> exit on stop,
target (R multiple), or time-stop -> track win rate, expectancy and an equity
curve, benchmarked against buy & hold over the same window.

Signals (long):
  reclaim   close crosses UP through a band (default AVWAP)  -> momentum/continuation
  dip       bar touches a band from above (default -1σ)      -> pullback bounce
  pullback  bar touches +1σ from above                       -> trend pullback

Examples
--------
    python3 strategy.py SNDK --anchor 2026-04-01 --interval 4h --signal pullback
    python3 strategy.py SNDK --anchor 20d --interval 30m --signal reclaim --target-r 2
    python3 strategy.py SNDK --anchor 20d --interval 30m --signal dip --chart eq.png --csv trades.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from anchored_vwap import analyze

# preset -> (band column, entry mode)
PRESETS = {"reclaim": ("avwap", "reclaim"),
           "dip": ("lower1", "dip"),
           "pullback": ("upper1", "dip")}
BAND_ALIASES = {"avwap": "avwap", "+1": "upper1", "+2": "upper2", "+3": "upper3",
                "-1": "lower1", "-2": "lower2", "-3": "lower3",
                "upper1": "upper1", "upper2": "upper2", "upper3": "upper3",
                "lower1": "lower1", "lower2": "lower2", "lower3": "lower3"}


def simulate(res, band_col, mode, stop_pct, target_r, time_stop,
             warmup=5, slippage_bps=0.0, exit_mode="target",
             trail_pct=5.0, exit_band="avwap"):
    """Long-only event-driven sim. exit_mode:
       target  fixed R-multiple target + hard stop + time stop
       trail   trailing % stop (ratchets up from the run-up high) + hard floor
       band    exit when close loses ``exit_band`` (ride the line) + hard stop
    """
    o = res["Open"].to_numpy(float)
    h = res["High"].to_numpy(float)
    lo = res["Low"].to_numpy(float)
    c = res["Close"].to_numpy(float)
    lvl = res[band_col].to_numpy(float)
    xb = res[exit_band].to_numpy(float) if exit_band in res.columns else None
    idx = res.index
    n = len(res)
    slip = slippage_bps / 1e4
    trades = []

    i = max(warmup, 1)
    while i < n - 1:
        if np.isnan(lvl[i]) or np.isnan(lvl[i - 1]):
            i += 1
            continue
        if mode == "reclaim":
            sig = c[i - 1] < lvl[i - 1] and c[i] >= lvl[i]
        elif mode == "dip":
            sig = (lo[i] <= lvl[i] <= h[i]) and c[i - 1] > lvl[i - 1]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        if not sig:
            i += 1
            continue

        e = i + 1                                   # fill at next bar's open
        if e >= n:
            break
        entry = o[e] * (1 + slip)
        hard_stop = entry * (1 - stop_pct / 100.0)
        risk = entry - hard_stop
        target = entry + target_r * risk
        run_peak = entry                            # highest high since entry

        exit_price = exit_i = reason = None
        jmax = min(e + time_stop, n - 1)
        for j in range(e, jmax + 1):
            eff_stop = hard_stop
            if exit_mode == "trail":
                eff_stop = max(hard_stop, run_peak * (1 - trail_pct / 100.0))
            stop_reason = "trail" if eff_stop > hard_stop else "stop"

            if o[j] <= eff_stop:                     # gap through stop
                exit_price, exit_i, reason = o[j] * (1 - slip), j, stop_reason; break
            if exit_mode == "target" and o[j] >= target:
                exit_price, exit_i, reason = o[j] * (1 - slip), j, "target"; break
            if lo[j] <= eff_stop:                    # intrabar stop (checked first)
                exit_price, exit_i, reason = eff_stop * (1 - slip), j, stop_reason; break
            if exit_mode == "target" and h[j] >= target:
                exit_price, exit_i, reason = target * (1 - slip), j, "target"; break
            if exit_mode == "band" and xb is not None and not np.isnan(xb[j]) \
                    and c[j] < xb[j]:               # lost the line
                exit_price, exit_i, reason = c[j] * (1 - slip), j, "band"; break
            run_peak = max(run_peak, h[j])

        if exit_price is None:                      # time stop at close
            exit_price, exit_i, reason = c[jmax] * (1 - slip), jmax, "time"

        trades.append(dict(
            entry_time=idx[e], exit_time=idx[exit_i],
            entry=entry, exit=exit_price,
            ret=exit_price / entry - 1.0,
            r=(exit_price - entry) / risk if risk > 0 else 0.0,
            bars=exit_i - e, reason=reason,
        ))
        i = exit_i + 1                              # no overlapping trades
    return pd.DataFrame(trades)


def metrics(trades, res, risk_pct):
    bh = float(res["Close"].iloc[-1] / res["Close"].iloc[0] - 1.0)
    if trades.empty:
        return dict(n=0, buy_hold=bh)
    ret = trades["ret"].to_numpy()
    r = trades["r"].to_numpy()
    wins = ret > 0
    gross_win = ret[wins].sum()
    gross_loss = -ret[~wins].sum()

    rf = risk_pct / 100.0
    eq = np.concatenate([[1.0], np.cumprod(1 + rf * r)])
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1).min())

    return dict(
        n=len(trades),
        win_rate=float(wins.mean()),
        avg_win=float(ret[wins].mean()) if wins.any() else 0.0,
        avg_loss=float(ret[~wins].mean()) if (~wins).any() else 0.0,
        expectancy_r=float(r.mean()),
        total_r=float(r.sum()),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        avg_bars=float(trades["bars"].mean()),
        exposure=float(trades["bars"].sum() / len(res)),
        strat_return=float(eq[-1] - 1.0),
        max_dd=max_dd,
        buy_hold=bh,
        equity=eq,
    )


def _chart(res, trades, eq, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg = "#0e0e12"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1],
                                   sharex=False)
    for ax in (ax1, ax2):
        ax.set_facecolor(bg)
        ax.tick_params(colors="#999")
        for s in ax.spines.values():
            s.set_color("#333")
        ax.grid(color="#1c1c22", lw=0.5)
    fig.patch.set_facecolor(bg)

    ax1.plot(res.index, res["Close"], color="#d8d8d8", lw=0.9, label="Close")
    ax1.plot(res.index, res["avwap"], color="white", lw=1.3, label="AVWAP")
    if not trades.empty:
        wins = trades["ret"] > 0
        ax1.scatter(trades["entry_time"], trades["entry"], marker="^", s=40,
                    color="#00e5ff", zorder=5, label="entry")
        ax1.scatter(trades["exit_time"][wins], trades["exit"][wins], marker="v",
                    s=40, color="#26a269", zorder=5, label="exit win")
        ax1.scatter(trades["exit_time"][~wins], trades["exit"][~wins], marker="v",
                    s=40, color="#e01b24", zorder=5, label="exit loss")
    ax1.set_title(title, color="white", fontsize=11)
    ax1.legend(loc="upper left", facecolor=bg, edgecolor="#333",
               labelcolor="#ccc", fontsize=8)

    ax2.plot(range(len(eq)), (eq - 1) * 100, color="#00e5ff", lw=1.3)
    ax2.axhline(0, color="#555", lw=0.6)
    ax2.set_title("equity curve (% , compounded at fixed risk/trade)",
                  color="#bbb", fontsize=9)
    ax2.set_xlabel("trade #", color="#999")

    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=bg)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Event-driven AVWAP strategy backtest (long-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("ticker")
    p.add_argument("-a", "--anchor", default="ytd",
                   help="ytd | high | low | YYYY-MM-DD | NNd/w/m/y (default: ytd)")
    p.add_argument("-i", "--interval", default="1d", help="1h | 4h | 1d (default: 1d)")
    p.add_argument("--signal", choices=list(PRESETS), default="reclaim",
                   help="reclaim (AVWAP) | dip (-1σ) | pullback (+1σ)  (default: reclaim)")
    p.add_argument("--entry-band", help="override band: avwap, +1/-1/+2/-2/+3/-3")
    p.add_argument("--entry-mode", choices=["reclaim", "dip"],
                   help="override mode (reclaim = cross up, dip = touch from above)")
    p.add_argument("--stop-pct", type=float, default=3.0, help="stop %% (default: 3.0)")
    p.add_argument("--target-r", type=float, default=2.0,
                   help="target as a multiple of risk (default: 2.0)")
    p.add_argument("--time-stop", type=int, default=20,
                   help="exit at market after N bars (default: 20)")
    p.add_argument("--exit", dest="exit_mode", choices=["target", "trail", "band"],
                   default="target", help="exit style (default: target)")
    p.add_argument("--trail-pct", type=float, default=5.0,
                   help="trailing stop %% for --exit trail (default: 5.0)")
    p.add_argument("--exit-band", default="avwap",
                   help="band to lose for --exit band (default: avwap)")
    p.add_argument("--risk-pct", type=float, default=1.0,
                   help="account %% risked per trade for the equity curve (default: 1.0)")
    p.add_argument("--warmup", type=int, default=5, help="skip first N bars (default: 5)")
    p.add_argument("--slippage-bps", type=float, default=0.0,
                   help="round-trip slippage in bps per side (default: 0)")
    p.add_argument("--window", type=int, default=365,
                   help="lookback days for high/low anchor (default: 365)")
    p.add_argument("--csv", help="write the trade list to this CSV")
    p.add_argument("--chart", help="save price + equity-curve PNG")
    args = p.parse_args(argv)

    band, mode = PRESETS[args.signal]
    if args.entry_band:
        band = BAND_ALIASES.get(args.entry_band.lower().replace("σ", ""), args.entry_band)
    if args.entry_mode:
        mode = args.entry_mode

    exit_band = BAND_ALIASES.get(args.exit_band.lower().replace("σ", ""), args.exit_band)
    res, anchor_ts = analyze(args.ticker.upper(), anchor=args.anchor,
                             interval=args.interval, swing_window_days=args.window)
    trades = simulate(res, band, mode, args.stop_pct, args.target_r,
                      args.time_stop, args.warmup, args.slippage_bps,
                      exit_mode=args.exit_mode, trail_pct=args.trail_pct,
                      exit_band=exit_band)
    m = metrics(trades, res, args.risk_pct)

    if args.exit_mode == "target":
        exit_desc = f"target {args.target_r}R"
    elif args.exit_mode == "trail":
        exit_desc = f"trail {args.trail_pct}%"
    else:
        exit_desc = f"exit<{exit_band}"
    print(f"\n{args.ticker.upper()} {args.interval}  anchor {anchor_ts.date()}  "
          f"({len(res)} bars)")
    print(f"signal: {mode} of {band}   |   stop {args.stop_pct}%  {exit_desc}  "
          f"time-stop {args.time_stop}  risk {args.risk_pct}%/trade")
    print("-" * 60)
    if m["n"] == 0:
        print(f"no trades.   buy & hold over window: {m['buy_hold'] * 100:+.1f}%")
        return 0
    print(f"trades         {m['n']}")
    print(f"win rate       {m['win_rate'] * 100:.0f}%")
    print(f"avg win        {m['avg_win'] * 100:+.1f}%      avg loss {m['avg_loss'] * 100:+.1f}%")
    print(f"expectancy     {m['expectancy_r']:+.2f}R / trade   (total {m['total_r']:+.1f}R)")
    print(f"profit factor  {m['profit_factor']:.2f}")
    print(f"avg hold       {m['avg_bars']:.0f} bars       exposure {m['exposure'] * 100:.0f}% of bars")
    print("-" * 60)
    print(f"strategy       {m['strat_return'] * 100:+.1f}%   (compounding {args.risk_pct}% risk/trade)")
    print(f"buy & hold     {m['buy_hold'] * 100:+.1f}%")
    print(f"max drawdown   {m['max_dd'] * 100:.1f}%")
    edge = "BEATS" if m["strat_return"] > m["buy_hold"] else "TRAILS"
    print(f"--> strategy {edge} buy & hold")

    if args.csv:
        trades.to_csv(args.csv, index=False)
        print(f"\nwrote {len(trades)} trades -> {args.csv}")
    if args.chart:
        _chart(res, trades, m["equity"], f"{args.ticker.upper()} {args.interval} "
               f"{mode}/{band}", args.chart)
        print(f"wrote chart -> {args.chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
