# Anchored VWAP Screener

Take a ticker (or a whole watchlist), compute its **Anchored VWAP** from a chosen
anchor point, and flag how far price has stretched from that AVWAP — measured as a
**z-score** in volume-weighted standard-deviation units.

It's the anchored cousin of the reference *Snapback Z-Score* indicator: instead of
`(close − 20d SMA) / stdev`, it uses `(close − AVWAP) / volume-weighted stdev`, and
re-uses the same ±2 / ±2.5 / ±3 snap-back zones.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Use

```bash
# default anchor = year-to-date
python screener.py AAPL MSFT NVDA

# scan a watchlist file, only show stretched names, save a CSV
python screener.py -w watchlist.txt --min-abs-z 2 --csv out.csv

# anchor at the 1-year swing low (classic AVWAP-from-the-bottom)
python screener.py TSLA --anchor low --window 365

# anchor at a specific date (e.g. an earnings gap)
python screener.py SPY --anchor 2026-01-02
```

### Anchor options (`-a/--anchor`)

| Spec          | Meaning                                            |
|---------------|----------------------------------------------------|
| `ytd`         | first bar of the current calendar year (default)   |
| `high`        | bar of the highest high in the lookback window     |
| `low`         | bar of the lowest low in the lookback window       |
| `YYYY-MM-DD`  | first bar on/after that date                       |
| `NNd/w/m/y`   | NN days / weeks / months / years back              |

`--window` (days) sets the lookback used to find the `high`/`low` swing (default 365).

### Output

```
TICKER     CLOSE     AVWAP     DEV%       Z   ZONE
--------------------------------------------------------------
NVDA      131.20    118.40    10.8%    2.71   🔴 strong overbought
TSLA      210.50    248.90   -15.4%   -2.93   🟢 strong oversold
AAPL      291.13    285.10     2.1%    0.74
```

* **DEV%** — percent distance of close from the AVWAP.
* **Z** — deviation in volume-weighted std-dev units. Sort is by `|Z|` so the most
  stretched names float to the top.
* **ZONE** — snap-back label: `|z|≥2` over/oversold, `≥2.5` strong, `≥3` extreme.

## How it works

For every bar from the anchor forward, with typical price `tp = (H+L+C)/3`:

```
AVWAP   = Σ(tp · vol) / Σ(vol)
vw_std  = sqrt( Σ(tp² · vol)/Σ(vol) − AVWAP² )      # volume-weighted std
z       = (close − AVWAP) / vw_std
bands   = AVWAP ± {1,2,3} · vw_std
```

Data is split/dividend-adjusted (`yfinance`, `auto_adjust=True`) so multi-month
anchors don't jump on corporate actions.

## Test the math (no network)

```bash
python selftest.py
```

## Files

| File                | Purpose                                             |
|---------------------|-----------------------------------------------------|
| `anchored_vwap.py`  | core: fetch, anchor resolution, AVWAP/z-score math  |
| `screener.py`       | CLI that scans tickers and prints the table         |
| `selftest.py`       | offline unit checks for the math                    |
| `watchlist.txt`     | sample watchlist                                    |
