#!/usr/bin/env python3
"""Offline self-test -- no network required.

Builds a synthetic OHLCV frame and checks the anchored VWAP / z-score math, the
anchor-resolution helpers, interval normalisation, and the backtest event/forward
logic.  Run:  python selftest.py
"""
import numpy as np
import pandas as pd

import backtest as bt
import basket as bask
import exits as ex
import leverage as lv
import monitor as mon
import strategy as st
from anchored_vwap import (anchored_vwap, normalize_interval, resolve_anchor,
                           typical_price, zone_label)


def _frame():
    idx = pd.date_range("2026-01-02", periods=8, freq="B")
    return pd.DataFrame(
        {
            "High":   [11, 12, 13, 12, 15, 14, 16, 13],
            "Low":    [ 9, 10, 11, 10, 13, 12, 14, 11],
            "Close":  [10, 11, 12, 11, 14, 13, 15, 12],
            "Volume": [100, 200, 150, 300, 250, 400, 220, 180],
        },
        index=idx,
    )


def test_avwap_matches_manual():
    df = _frame()
    res = anchored_vwap(df)
    tp = typical_price(df)
    expected = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    assert np.allclose(res["avwap"], expected), "avwap != manual cumulative VWAP"
    assert np.isnan(res["z"].iloc[0]), "first-bar z should be NaN (no spread yet)"


def test_bands_are_symmetric_multiples():
    df = _frame()
    res = anchored_vwap(df).iloc[-1]
    one = res["upper1"] - res["avwap"]
    assert np.isclose(res["avwap"] - res["lower1"], one)
    assert np.isclose(res["upper2"] - res["avwap"], 2 * one)
    assert np.isclose(res["upper3"] - res["avwap"], 3 * one)


def test_resolve_anchor():
    df = _frame()
    assert resolve_anchor(df, "high") == df["High"].idxmax()
    assert resolve_anchor(df, "low") == df["Low"].idxmin()
    assert resolve_anchor(df, "ytd") == df.index[0]
    assert resolve_anchor(df, "2026-01-06") >= pd.Timestamp("2026-01-06")


def test_normalize_interval():
    assert normalize_interval("1h") == "60m"
    assert normalize_interval("4H") == "4h"
    assert normalize_interval("1d") == "1d"
    assert normalize_interval("weekly") == "1wk"


def test_zone_label():
    assert zone_label(3.1) == "🔥 EXTREME OVERBOUGHT"
    assert zone_label(2.6) == "🔴 strong overbought"
    assert zone_label(-2.1) == "🟢 oversold"
    assert zone_label(-3.1) == "🔥 EXTREME OVERSOLD"
    assert zone_label(0.5) == ""
    assert zone_label(float("nan")) == ""


def test_detect_events_and_forward():
    res = anchored_vwap(_frame())
    evs = bt.detect_events(res, "avwap", "touch")
    assert isinstance(evs, list) and all(isinstance(i, int) for i in evs)
    f = bt.forward(res, 1, 3)
    assert f is not None and len(f) == 3
    assert bt.forward(res, len(res) - 1, 3) is None        # no room past last bar
    assert bt.role_of(res, 3, "avwap") in ("support", "resistance")


def test_backtest_run_smoke():
    res = anchored_vwap(_frame())
    summary, events, base = bt.run(res, "touch", horizon=2, warmup=1)
    assert isinstance(summary, pd.DataFrame)
    assert set(events) == {name for name, _ in bt.LEVELS}
    assert base is None or {"n", "mean", "median", "pos"} <= set(base)
    if not summary.empty:
        assert {"level", "role", "n", "mean_fwd", "excess", "pos"} <= set(summary.columns)


def _trend_frame(n=120, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-04-01", periods=n, freq="h", tz="America/New_York")
    ret = rng.normal(0.002, 0.02, n)
    close = 700 * np.cumprod(1 + ret)
    high = close * (1 + abs(rng.normal(0, 0.01, n)))
    low = close * (1 - abs(rng.normal(0, 0.01, n)))
    vol = rng.integers(1000, 5000, n).astype(float)
    df = pd.DataFrame({"Open": close, "High": high, "Low": low,
                       "Close": close, "Volume": vol}, index=idx)
    return anchored_vwap(df)


def test_strategy_simulate_and_metrics():
    res = _trend_frame()
    trades = st.simulate(res, "avwap", "reclaim", stop_pct=3.0, target_r=2.0,
                         time_stop=10, warmup=5)
    assert isinstance(trades, pd.DataFrame)
    if not trades.empty:
        assert {"entry", "exit", "ret", "r", "reason"} <= set(trades.columns)
        # exits must come at or after entries; no overlap
        assert (trades["exit_time"] >= trades["entry_time"]).all()
    m = st.metrics(trades, res, risk_pct=1.0)
    assert "buy_hold" in m
    if m["n"]:
        assert -1.0 <= m["max_dd"] <= 0.0


def test_strategy_exit_modes():
    res = _trend_frame(seed=5)
    for xm in ("target", "trail", "band"):
        tr = st.simulate(res, "avwap", "reclaim", 3.0, 2.0, 12, warmup=5,
                         exit_mode=xm, trail_pct=5.0, exit_band="avwap")
        assert isinstance(tr, pd.DataFrame)
        if not tr.empty:
            assert (tr["exit_time"] >= tr["entry_time"]).all()
            assert st.metrics(tr, res, 1.0)["n"] == len(tr)


def test_leverage_first_passage():
    res = _trend_frame(seed=9)
    df = lv.first_passage(res, "lower1", "long", "avwap", lev=10.0,
                          liq_pct=0.095, horizon=20, warmup=5, fee_frac=0.0004)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert set(df["kind"].unique()) <= {"win", "liq", "timeout"}
        # liquidation always costs exactly one margin unit
        liqs = df[df["kind"] == "liq"]
        assert (liqs["acct"] == -1.0).all()
    s = lv.summarize(df)
    if s:
        assert 0.0 <= s["win"] <= 1.0 and 0.0 <= s["liq"] <= 1.0


def test_exits_eval_target():
    res = _trend_frame(seed=9)
    entries = [i + 1 for i in lv._touches(res, "lower1", 5) if i + 1 < len(res)]
    s = ex.eval_target(res, entries, "long", "band", "avwap",
                       stop_pct=3.0, lev=10.0, horizon=20, fee_frac=0.0004)
    assert s is None or {"hit", "stop", "timeout", "avg_r", "ev", "n"} <= set(s)
    if s:
        assert abs(s["hit"] + s["stop"] + s["timeout"] - 1.0) < 1e-9
    # candidate_targets returns band + pct entries
    cands = ex.candidate_targets("long")
    assert any(k == "band" for _, k, _ in cands) and any(k == "pct" for _, k, _ in cands)


def test_basket_load_tickers():
    import types
    ns = types.SimpleNamespace(tickers=["aapl", "MSFT", "aapl"], watchlist=None)
    assert bask.load_tickers(ns) == ["AAPL", "MSFT"]


def test_monitor_tag():
    res = anchored_vwap(_frame())
    last = res.iloc[-1]
    name, dist = mon.nearest_band(last)
    assert name in {n for n, _ in mon.BANDS}
    assert isinstance(dist, float)
    assert isinstance(mon.tag(last, 1.5), str)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
