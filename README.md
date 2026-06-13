# Anchored VWAP Screener + Band Backtester

Tools for working with **Anchored VWAP** (hlc3 typical price) and its
**±1 / ±2 / ±3 standard-deviation bands** — the same indicator as the
TradingView "Anchored VWAP (hlc3)" overlay (`AVWAP`, blue ±1σ, orange ±2σ,
red ±3σ).

| Tool            | What it does                                                        |
|-----------------|---------------------------------------------------------------------|
| `levels.py`     | print AVWAP + all 6 bands for a ticker across 1h / 4h / 1d          |
| `backtest.py`   | how price behaved the last times it touched/crossed each band       |
| `strategy.py`   | event-driven backtest of the actual entry/stop/target rules + equity |
| `sweep.py`      | grid-sweep stop/target/exit combos; rank by expectancy vs buy & hold |
| `leverage.py`   | leverage/liquidation first-passage at bands (perps): WIN/LIQ%/EV    |
| `monitor.py`    | live "is it actionable now" readout: z + nearest band + state tag   |
| `screener.py`   | scan a watchlist, rank by z-score stretch from the AVWAP            |
| `selftest.py`   | offline math checks (no network)                                    |

## Install

```bash
python3 -m pip install -r requirements.txt
```
(`matplotlib` is only needed for `backtest.py --chart`.)

## 1. Multi-timeframe levels — `levels.py`

```bash
python levels.py SNDK --anchor 2026-04-01            # match the chart's swing-low anchor
python levels.py SNDK --anchor low --intervals 1h 4h 1d
```

```
SNDK  4h   anchor 2026-04-01   (212 bars)
  close       1994.99    z +2.15    dev +58.5%   🔴 overbought
  +3σ         2288.42
  +2σ         1945.26
  +1σ         1602.09    (blue line)
  AVWAP       1258.93
  -1σ          915.77    (blue line)
  -2σ          572.60
  -3σ          229.44
```

That's the TradingView header readout, reproduced per timeframe.

## 2. Band backtest — `backtest.py`

Finds every bar where price **touches** (or crosses) a band, splits the events
into **support** vs **resistance** tests, and measures the **forward return**:

```bash
python backtest.py SNDK --anchor 2026-04-01 --interval 4h --horizon 10
python backtest.py SNDK --anchor 2026-04-01 --interval 4h --chart sndk.png --csv events.csv
python backtest.py SNDK --anchor low --interval 1d --mode cross_down
```

```
SNDK 4h  anchor 2026-04-01  — touch, fwd 10 bars  (212 bars)
LEVEL  ROLE         N  MEAN_FWD   MEDIAN  BOUNCE%  AVG_MFE  AVG_MAE
--------------------------------------------------------------------
+1σ    resistance   5     -1.4%    -0.9%      60%     +1.8%    -4.1%
AVWAP  support      3     +2.9%    +3.1%      67%     +5.0%    -2.2%
-1σ    support      6     +4.3%    +4.0%      83%     +6.6%    -1.9%   <- "the blue line"
```

* **support** = price came *down* to the level (level was below the prior close);
  a bounce = positive forward return.
* **resistance** = price came *up* to the level; a rejection = negative return.
* **BOUNCE%** = share of events that resolved in the bounce/rejection direction.
* **MFE / MAE** = average best / worst excursion within the horizon.

`--mode`: `touch` (range straddles the level, default), `cross_up`, `cross_down`.
`--horizon`: number of forward bars to measure (default 10).
`--csv`: dump every individual event (time, level, role, fwd, mfe, mae).
`--chart`: save a dark, TradingView-style PNG with the bands and touch markers.

> ⚠️ Small samples: a band may only be touched a handful of times. Treat
> BOUNCE% as a hint, not a guarantee — read N alongside it.

## 3. Live monitor — `monitor.py`

A compact "where are we right now" readout per timeframe — current price, AVWAP,
z-score, the nearest band + % distance, and a plain-English state tag
(extended / stretched / at a band / oversold / neutral).

```bash
python3 monitor.py SNDK --anchor 2026-04-01 --intervals 1d 4h 1h
python3 monitor.py SNDK --anchor low --window 10 --intervals 30m 15m --near 1.0
```

```
SNDK   anchor low
  30m  px   1979.04  AVWAP   1743.81  z +1.83   🟠 stretched (z +1.8) — hold, don't chase
  15m  px   1979.04  AVWAP   1745.03  z +1.83   🟠 stretched (z +1.8) — hold, don't chase
```

Use it as a poor-man's alert by looping it:

```bash
while true; do clear; python3 monitor.py SNDK --anchor low --window 10 \
    --intervals 30m 15m; sleep 300; done
```

## 4. Strategy backtest — `strategy.py`

`backtest.py` measures what happens *after* a band touch; `strategy.py` trades
it: entry signal → fill the **next bar's open** → exit on stop / target (R
multiple) / time-stop, then reports win rate, expectancy and an equity curve —
benchmarked against **buy & hold** over the same window (the bar to beat).

```bash
python3 strategy.py SNDK --anchor 2026-04-01 --interval 4h --signal pullback
python3 strategy.py SNDK --anchor 20d --interval 30m --signal reclaim --chart eq.png
python3 strategy.py SNDK --anchor 20d --interval 30m --signal dip --csv trades.csv
```

```
SNDK 30m  anchor 2026-05-26  (182 bars)
signal: reclaim of avwap   |   stop 3.0%  target 2.0R  time-stop 20  risk 1.0%/trade
------------------------------------------------------------
trades         14
win rate       57%
avg win        +4.8%      avg loss -2.9%
expectancy     +0.42R / trade   (total +5.9R)
profit factor  1.90
avg hold       9 bars       exposure 61% of bars
------------------------------------------------------------
strategy       +38.2%   (compounding 1.0% risk/trade)
buy & hold     +21.4%
max drawdown   -7.5%
--> strategy BEATS buy & hold
```

Signals: `reclaim` (close crosses up through AVWAP), `dip` (touch −1σ from above),
`pullback` (touch +1σ from above). Tunables: `--stop-pct`, `--target-r`,
`--time-stop`, `--risk-pct`, `--slippage-bps`, plus `--entry-band`/`--entry-mode`
to build your own. **Read the last line** — if it doesn't *BEAT buy & hold*, the
timing isn't adding value over just holding.

Exits (`--exit`): `target` (fixed R, default), `trail` (`--trail-pct`, lets
winners run), `band` (`--exit-band`, ride until the line is lost). In a strong
trend `trail`/`band` usually beat a fixed target.

### Find what works — `sweep.py`

Runs the whole grid (signals × stops × exits) in one shot and ranks by
expectancy, flagging combos whose return beats buy & hold:

```bash
python3 sweep.py SNDK --anchor 20d --interval 30m
python3 sweep.py SNDK --anchor 2026-04-01 --interval 4h --time-stop 30
```

### Leverage / liquidation — `leverage.py`

For leveraged perps (Hyperliquid etc.): a *first-passage* test at each band —
does price revert to the mean (AVWAP) **before** a wick liquidates you? Long at
the lower bands (bounce up), short at the upper bands (fade down).

```bash
python3 leverage.py SNDK --anchor 20d --interval 30m --leverage 10
python3 leverage.py SNDK --anchor 2026-04-01 --interval 4h --leverage 10 --horizon 30
```

Reports per band: `WIN%` (reverted to target first), **`LIQ%`** (liquidated first
— the number that matters), `TIME%` (neither), and **`EV/margin`** per trade (a
win pays ~leverage × move; a liq is −100%). At 10x, liquidation is ~−9.5% away,
so one liq erases ~10 wins — watch `LIQ%`, not `WIN%`. Models a maintenance
buffer + taker fees; funding is not modelled. Small N = low confidence.

## 5. Watchlist screener — `screener.py`

```bash
python screener.py AAPL MSFT NVDA
python screener.py -w watchlist.txt --anchor ytd --min-abs-z 2 --csv out.csv
```
Ranks tickers by how stretched price is from the AVWAP (z-score in
volume-weighted std-dev units), with the ±2/2.5/3 snap-back zones.

## Anchor options (`-a/--anchor`)

| Spec          | Meaning                                            |
|---------------|----------------------------------------------------|
| `ytd`         | first bar of the current calendar year (default)   |
| `high`        | bar of the highest high in the lookback window     |
| `low`         | bar of the lowest low in the lookback window       |
| `YYYY-MM-DD`  | first bar on/after that date                       |
| `NNd/w/m/y`   | NN days / weeks / months / years back              |

`--window` (days) sets the lookback used to find the `high`/`low` swing.

## How it works

For every bar from the anchor forward, with typical price `tp = (H+L+C)/3`:

```
AVWAP   = Σ(tp · vol) / Σ(vol)
vw_std  = sqrt( Σ(tp² · vol)/Σ(vol) − AVWAP² )      # volume-weighted std
z       = (close − AVWAP) / vw_std
bands   = AVWAP ± {1, 2, 3} · vw_std
```

This matches TradingView's Anchored VWAP standard-deviation bands.

### Notes on timeframes & data

* **4h** is not a native yfinance interval — it's built by resampling **60m**
  bars, so bucket boundaries (and thus 4h values) won't be penny-identical to
  TradingView. The cumulative AVWAP is robust to this; backtest event timing is
  a little more sensitive.
* Intraday history from yfinance is limited: ~730 days for `1h`/`4h`, but only
  ~60 days for sub-hour (`5m`/`15m`/`30m`) and ~8 days for `1m`. A months-old
  anchor therefore only works on `1h`/`4h`/`1d`; on lower timeframes use a recent
  anchor (e.g. `--anchor 5d`, or `--anchor low --window 10`). The tool warns if a
  date anchor predates the data it could actually fetch.
* Daily data is split/dividend-adjusted (`auto_adjust=True`) so multi-month
  anchors don't jump on corporate actions.
* Values are close to, but won't exactly equal, TradingView (different data
  feed + session alignment). Tune `--anchor` to line the AVWAP up with the chart.

## Test the math (no network)

```bash
python selftest.py
```

## Files

| File                | Purpose                                                  |
|---------------------|----------------------------------------------------------|
| `anchored_vwap.py`  | core: fetch, interval resampling, anchor resolution, math |
| `levels.py`         | multi-timeframe AVWAP + bands readout                    |
| `backtest.py`       | band touch/cross forward-return backtest (+ optional chart) |
| `strategy.py`       | event-driven entry/stop/target backtest + equity curve   |
| `sweep.py`          | parameter grid sweep, ranked vs buy & hold               |
| `leverage.py`       | leverage/liquidation first-passage at bands (perps)      |
| `monitor.py`        | live band proximity / state readout                      |
| `plotting.py`       | optional matplotlib chart helper                         |
| `screener.py`       | watchlist z-score screener                               |
| `selftest.py`       | offline unit checks                                      |
| `watchlist.txt`     | sample watchlist                                         |
