#!/usr/bin/env python3
"""Offline self-test -- no network required.

Builds a synthetic OHLCV frame and checks the anchored VWAP / z-score math, the
anchor-resolution helpers, interval normalisation, and the backtest event/forward
logic.  Run:  python selftest.py
"""
import numpy as np
import pandas as pd

import backtest as bt
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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
