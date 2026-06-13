"""Anchored VWAP calculations.

Pick an *anchor* bar, then accumulate volume-weighted price from that bar
forward.  We also compute volume-weighted standard-deviation bands around the
AVWAP and a z-score that measures how stretched price is from the AVWAP in
std-dev units -- the anchored analog of the classic (close - SMA) / stdev
z-score from the reference Pine indicator.
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

# NNd / NNw / NNm / NNy  ->  lookback anchor
_LOOKBACK_RE = re.compile(r"^(\d+)\s*([dwmy])$", re.IGNORECASE)
_UNIT_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}


def typical_price(df: pd.DataFrame) -> pd.Series:
    """(H + L + C) / 3 -- the price each bar's volume trades 'at' for VWAP."""
    return (df["High"] + df["Low"] + df["Close"]) / 3.0


def anchored_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AVWAP + volume-weighted std bands + z-score.

    ``df`` is assumed to already start at the anchor bar (slice it first).
    Returns a copy with added columns:
        avwap, vwstd, z, dev_pct, upper{1,2,3}, lower{1,2,3}
    """
    out = df.copy()
    tp = typical_price(out)
    vol = out["Volume"].astype(float)

    cum_vol = vol.cumsum()
    cum_pv = (tp * vol).cumsum()
    avwap = cum_pv / cum_vol

    # Volume-weighted variance of price around the running VWAP:
    #   E[tp^2] - E[tp]^2, with E weighted by volume.
    cum_pv2 = (tp * tp * vol).cumsum()
    var = (cum_pv2 / cum_vol) - avwap ** 2
    var = var.clip(lower=0)  # guard tiny negatives from float error
    vwstd = np.sqrt(var)

    out["avwap"] = avwap
    out["vwstd"] = vwstd
    out["z"] = (out["Close"] - avwap) / vwstd.replace(0, np.nan)
    out["dev_pct"] = (out["Close"] - avwap) / avwap * 100.0
    for k in (1, 2, 3):
        out[f"upper{k}"] = avwap + k * vwstd
        out[f"lower{k}"] = avwap - k * vwstd
    return out


def zone_label(z: float) -> str:
    """Map a z-score to the snap-back zone label (mirrors the Pine script)."""
    if z is None or z != z:  # None or NaN
        return ""
    if z >= 3.0:
        return "🔥 EXTREME OVERBOUGHT"
    if z >= 2.5:
        return "🔴 strong overbought"
    if z >= 2.0:
        return "🔴 overbought"
    if z <= -3.0:
        return "🔥 EXTREME OVERSOLD"
    if z <= -2.5:
        return "🟢 strong oversold"
    if z <= -2.0:
        return "🟢 oversold"
    return ""


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a (field, ticker) column MultiIndex -- drop the ticker."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch(ticker: str, start=None, period=None, interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV for one ticker via yfinance (split/div adjusted)."""
    import yfinance as yf

    kwargs = dict(interval=interval, auto_adjust=True, progress=False)
    if start is not None:
        kwargs["start"] = start
    elif period is not None:
        kwargs["period"] = period
    df = yf.download(ticker, **kwargs)
    if df is None or len(df) == 0:
        raise ValueError("no data returned (bad ticker, or no network?)")
    df = _flatten(df)
    needed = {"High", "Low", "Close", "Volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {missing} -- got {list(df.columns)}")
    return df


# Friendly interval names -> yfinance intervals.  yfinance has no native 4h
# (or 2h) bar, so those are built by resampling 60m data.
_INTERVAL_ALIASES = {
    "1h": "60m", "h": "60m", "1hr": "60m", "60min": "60m", "hourly": "60m",
    "2h": "2h", "4h": "4h", "4hr": "4h", "240m": "4h",
    "1d": "1d", "d": "1d", "day": "1d", "daily": "1d", "1day": "1d",
    "1w": "1wk", "w": "1wk", "1week": "1wk", "weekly": "1wk", "1wk": "1wk",
}
_RESAMPLE_FROM_60M = {"2h": "2h", "4h": "4h"}
_OHLCV_AGG = {"Open": "first", "High": "max", "Low": "min",
              "Close": "last", "Volume": "sum"}


def normalize_interval(interval: str) -> str:
    """Map a friendly interval name (e.g. '4h') to a yfinance/internal one."""
    return _INTERVAL_ALIASES.get(interval.strip().lower(), interval.strip().lower())


def fetch_interval(ticker: str, interval: str, start=None, period=None) -> pd.DataFrame:
    """Fetch OHLCV at ``interval``, resampling 60m -> 2h/4h when yfinance lacks it."""
    norm = normalize_interval(interval)
    if norm in _RESAMPLE_FROM_60M:
        base = fetch(ticker, start=start, period=period, interval="60m")
        rule = _RESAMPLE_FROM_60M[norm]
        agg = {c: _OHLCV_AGG[c] for c in _OHLCV_AGG if c in base.columns}
        try:
            out = base.resample(rule, label="left", closed="left").agg(agg)
        except ValueError:  # older pandas wants the uppercase offset alias ('4H')
            out = base.resample(rule.upper(), label="left", closed="left").agg(agg)
        return out.dropna(how="any")
    return fetch(ticker, start=start, period=period, interval=norm)


def _start_for_anchor(anchor: str, swing_window_days: int) -> dt.date:
    """Pick a fetch start date that comfortably includes the anchor bar."""
    today = pd.Timestamp.today().normalize()
    a = anchor.strip().lower()
    if a == "ytd":
        return dt.date(today.year, 1, 1)
    if a in ("high", "hi", "low", "lo"):
        return (today - pd.Timedelta(days=swing_window_days)).date()
    m = _LOOKBACK_RE.match(a)
    if m:
        days = _UNIT_DAYS[m.group(2).lower()] * int(m.group(1))
        return (today - pd.Timedelta(days=int(days * 1.6) + 10)).date()
    try:  # explicit YYYY-MM-DD
        return (pd.Timestamp(anchor) - pd.Timedelta(days=7)).date()
    except (ValueError, TypeError):
        return dt.date(today.year, 1, 1)


def resolve_anchor(df: pd.DataFrame, anchor: str) -> pd.Timestamp:
    """Map an anchor spec to an actual timestamp in ``df.index``.

    Supported specs:
      * ``YYYY-MM-DD`` -- first bar on/after that date
      * ``ytd``        -- first bar of the current calendar year
      * ``high``/``low`` -- bar of the highest High / lowest Low in df
      * ``NNd/w/m/y``  -- NN periods back from the last bar
    """
    idx = df.index
    a = anchor.strip().lower()

    if a == "ytd":
        target = pd.Timestamp(year=idx[-1].year, month=1, day=1, tz=idx.tz)
        sel = idx[idx >= target]
        return sel[0] if len(sel) else idx[0]
    if a in ("high", "hi"):
        return df["High"].idxmax()
    if a in ("low", "lo"):
        return df["Low"].idxmin()

    m = _LOOKBACK_RE.match(a)
    if m:
        days = _UNIT_DAYS[m.group(2).lower()] * int(m.group(1))
        target = idx[-1] - pd.Timedelta(days=days)
        sel = idx[idx >= target]
        return sel[0] if len(sel) else idx[0]

    try:  # explicit date
        target = pd.Timestamp(anchor)
        if idx.tz is not None and target.tz is None:
            target = target.tz_localize(idx.tz)
    except (ValueError, TypeError) as e:
        raise ValueError(f"could not parse anchor {anchor!r}: {e}")
    sel = idx[idx >= target]
    if len(sel) == 0:
        raise ValueError(f"anchor {anchor!r} is after the last available bar")
    return sel[0]


def analyze(ticker: str, anchor: str = "ytd", interval: str = "1d",
            swing_window_days: int = 365):
    """Fetch ``ticker``, resolve the anchor, and return (result_df, anchor_ts).

    ``result_df`` starts at the anchor bar and carries the AVWAP columns.
    """
    start = _start_for_anchor(anchor, swing_window_days)
    df = fetch_interval(ticker, interval, start=start)
    anchor_ts = resolve_anchor(df, anchor)
    sliced = df.loc[anchor_ts:]
    if len(sliced) < 2:
        raise ValueError(f"only {len(sliced)} bar(s) from anchor -- pick an earlier anchor")
    return anchored_vwap(sliced), anchor_ts
