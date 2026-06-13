#!/usr/bin/env python3
"""Offline self-test for the AVWAP math -- no network required.

Builds a synthetic OHLCV frame and checks the anchored VWAP / z-score against
hand-computed values, plus the anchor-resolution helpers.  Run:  python selftest.py
"""
import numpy as np
import pandas as pd

from anchored_vwap import anchored_vwap, resolve_anchor, typical_price, zone_label


def _frame():
    idx = pd.date_range("2026-01-02", periods=6, freq="B")
    return pd.DataFrame(
        {
            "High":   [11, 12, 13, 12, 15, 14],
            "Low":    [ 9, 10, 11, 10, 13, 12],
            "Close":  [10, 11, 12, 11, 14, 13],
            "Volume": [100, 200, 150, 300, 250, 400],
        },
        index=idx,
    )


def test_avwap_matches_manual():
    df = _frame()
    res = anchored_vwap(df)
    tp = typical_price(df)
    cum_pv = (tp * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum()
    expected = cum_pv / cum_vol
    assert np.allclose(res["avwap"], expected), "avwap != manual cumulative VWAP"
    # first bar: price == typical price family, std ~ 0 -> z is NaN (no spread yet)
    assert np.isnan(res["z"].iloc[0]), "first-bar z should be NaN"


def test_zscore_sign():
    df = _frame()
    res = anchored_vwap(df)
    # last close (13) sits above the running avwap -> positive z
    assert res["z"].iloc[-1] > 0, "expected positive z when close > avwap"
    assert res["dev_pct"].iloc[-1] > 0


def test_resolve_anchor():
    df = _frame()
    assert resolve_anchor(df, "high") == df["High"].idxmax()
    assert resolve_anchor(df, "low") == df["Low"].idxmin()
    assert resolve_anchor(df, "ytd") == df.index[0]
    assert resolve_anchor(df, "2026-01-05") >= pd.Timestamp("2026-01-05")
    assert resolve_anchor(df, "2d") >= df.index[-1] - pd.Timedelta(days=2)


def test_zone_label():
    assert zone_label(3.1) == "🔥 EXTREME OVERBOUGHT"
    assert zone_label(2.6) == "🔴 strong overbought"
    assert zone_label(2.1) == "🔴 overbought"
    assert zone_label(0.5) == ""
    assert zone_label(-2.1) == "🟢 oversold"
    assert zone_label(-3.1) == "🔥 EXTREME OVERSOLD"
    assert zone_label(float("nan")) == ""


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
