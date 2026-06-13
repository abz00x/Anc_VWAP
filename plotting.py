"""Optional matplotlib chart of price + Anchored VWAP bands (TradingView-ish).

Imported lazily by backtest.py only when --chart is passed, so matplotlib stays
an optional dependency.
"""
from __future__ import annotations

import pandas as pd


def plot_avwap(res: pd.DataFrame, title: str, path: str,
               events: dict[str, list[int]] | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")          # headless / no display needed
    import matplotlib.pyplot as plt

    bg = "#0e0e12"
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    x = res.index

    ax.plot(x, res["Close"], color="#d8d8d8", lw=1.0, label="Close")
    ax.plot(x, res["avwap"], color="#ffffff", lw=1.7, label="AVWAP")
    ax.plot(x, res["upper1"], color="#3aa0ff", lw=1.0)
    ax.plot(x, res["lower1"], color="#3aa0ff", lw=1.0, label="±1σ (blue)")
    ax.plot(x, res["upper2"], color="#ffb000", lw=1.0)
    ax.plot(x, res["lower2"], color="#ffb000", lw=1.0, label="±2σ")
    ax.plot(x, res["upper3"], color="#ff4d4d", lw=1.0)
    ax.plot(x, res["lower3"], color="#ff4d4d", lw=1.0, label="±3σ")
    ax.fill_between(x, res["lower1"], res["upper1"], color="#3aa0ff", alpha=0.05)

    if events:
        marked = False
        for idxs in events.values():
            if not idxs:
                continue
            ax.scatter([res.index[i] for i in idxs],
                       [res["Close"].iloc[i] for i in idxs],
                       s=28, color="#00e5ff", edgecolor="black", linewidth=0.4,
                       zorder=5, label=None if marked else "touch")
            marked = True

    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="#999")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(color="#1c1c22", lw=0.5)
    ax.legend(loc="upper left", facecolor=bg, edgecolor="#333",
              labelcolor="#cccccc", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=bg)
    plt.close(fig)
    return path
