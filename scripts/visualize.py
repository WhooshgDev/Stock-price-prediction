import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

from typing import List, Tuple, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "axes.titlecolor": "#fff",
    "text.color": "#ccc",
    "xtick.color": "#888",
    "ytick.color": "#888",
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": "#333",
    "legend.labelcolor": "#ccc",
    "grid.color": "#2a2a2a",
    "grid.alpha": 0.5,
})

OUTPUT_DIR = "outputs"
DATA_PATH = "data/World-Stock-Prices-Dataset.csv"

EVENTS: List[Tuple[str, str, str, str]] = [
    ("Dot-com Crash", "2000-03-10", "2001-09-10", "#ff6b6b"),
    ("9/11 Attacks", "2001-09-11", "2001-09-21", "#ff4444"),
    ("Dot-com Aftermath", "2001-09-22", "2002-10-09", "#e57373"),
    ("Financial Crisis", "2007-12-01", "2009-06-30", "#ffa726"),
    ("COVID Crash", "2020-02-19", "2020-03-23", "#ef5350"),
    ("COVID Recovery", "2020-03-24", "2021-12-31", "#66bb6a"),
    ("AI Explosion", "2023-01-01", "2025-07-03", "#42a5f5"),
]

FOCUS_TICKERS = {
    "Tech: AI/Cloud": ["NVDA", "AMD", "MSFT", "GOOGL", "CRM"],
    "Tech: FAANG": ["AAPL", "AMZN", "NFLX", "GOOGL"],
    "Retail & Consumer": ["COST", "HD", "TGT", "JWN"],
    "Automotive & Logistics": ["UBER", "TSLA"],
    "Finance": ["JPM", "GS", "V", "MA"],
    "Entertainment & Media": ["DIS", "SPOT", "ZM", "NFLX"],
    "Healthcare": ["UNH", "JNJ", "PFE", "ABBV"],
}

HIGHLIGHT_TICKER = "AAPL"
HIGHLIGHT_LW = 2.5
HIGHLIGHT_ALPHA = 1.0
DEFAULT_LW = 0.8
DEFAULT_ALPHA = 0.4


def _line_style(ticker: str) -> Tuple[float, float]:
    if ticker == HIGHLIGHT_TICKER:
        return HIGHLIGHT_LW, HIGHLIGHT_ALPHA
    return DEFAULT_LW, DEFAULT_ALPHA


def add_event_spans(ax: Axes, alpha: float = 0.15) -> None:
    ylim = ax.get_ylim()
    if ylim == (0.0, 1.0):
        ax.autoscale_view()
        ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]

    spans = [pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
             for _, start, end, _ in EVENTS]

    levels = []
    for mid in spans:
        level = 0
        for j, prev_mid in enumerate(spans[:len(levels)]):
            if abs((mid - prev_mid).days) < 730 and levels[j] == level:
                level += 1
        levels.append(level)

    for i, (name, start, end, color) in enumerate(EVENTS):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=alpha, color=color, zorder=2)
        mid = spans[i]
        y_pos = ylim[1] - (yrange * 0.06 * levels[i])
        ax.text(mid, y_pos, name, ha="center", va="top",
                fontsize=7, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor=color, alpha=0.8))


def _plot_multiticker(
    ax: Axes, df: pd.DataFrame, tickers: List[str],
    value_col: str = "norm", title: str = "",
    ylabel: str = "", highlight: Optional[str] = HIGHLIGHT_TICKER,
) -> None:
    for ticker in tickers:
        t = df[df["Ticker"] == ticker]
        if t.empty:
            continue
        lw, alpha = _line_style(ticker)
        ax.plot(t["Date"], t[value_col], label=ticker, linewidth=lw, alpha=alpha)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def _normalize(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    result = df.copy()
    result["norm"] = result.groupby("Ticker")[col].transform(lambda x: x / x.iloc[0] * 100)
    return result


def _format_xaxis(ax: Axes, loc_years: int = 3) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(loc_years))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# ─── 1. AAPL full history ──────────────────────────────────────────────
def plot_aapl_history(df: pd.DataFrame) -> None:
    aapl = df[df["Ticker"] == "AAPL"].copy()
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    ax = axes[0]
    ax.plot(aapl["Date"], aapl["Close"], color="#42a5f5", linewidth=1)
    ax.fill_between(aapl["Date"], aapl["Close"], alpha=0.1, color="#42a5f5")
    ax.set_ylabel("Price ($)")
    ax.set_title("AAPL — Full History (2000–2025)", fontsize=14, fontweight="bold")
    add_event_spans(ax)
    ax.legend(["Close"])

    ax = axes[1]
    cumret = (1 + aapl["ret"]).cumprod()
    ax.plot(aapl["Date"], cumret, color="#66bb6a", linewidth=1)
    ax.fill_between(aapl["Date"], cumret, alpha=0.1, color="#66bb6a")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("AAPL — Cumulative Return (log scale)", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    add_event_spans(ax)

    ax = axes[2]
    vol = aapl["ret"].rolling(21).std() * np.sqrt(252) * 100
    ax.plot(aapl["Date"], vol, color="#ffa726", linewidth=0.8)
    ax.fill_between(aapl["Date"], vol, alpha=0.15, color="#ffa726")
    ax.set_ylabel("Volatility (%)")
    ax.set_title("AAPL — 21-Day Rolling Volatility (annualized)", fontsize=14, fontweight="bold")
    add_event_spans(ax)
    med_vol = vol.median()
    ax.axhline(y=med_vol, color="#888", linestyle="--", linewidth=0.5, label=f"Median: {med_vol:.1f}%")
    ax.legend()

    for a in axes:
        _format_xaxis(a)
        a.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_01_aapl_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_01_aapl_history.png")


# ─── 2. Sector performance through events ───────────────────────────────
def plot_sector_performance(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    sectors = df.groupby(["Date", "Industry_Tag"])["ret"].mean().reset_index()
    pivot = sectors.pivot(index="Date", columns="Industry_Tag", values="ret").fillna(0)
    cum_ret = (1 + pivot).cumprod()

    ax = axes[0]
    for col in cum_ret.columns:
        ax.plot(cum_ret.index, cum_ret[col], label=col, linewidth=1)
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Sector Performance (Equal-Weight)", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    add_event_spans(ax)
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    big_sectors = ["technology", "e-commerce", "finance", "retail", "healthcare", "automotive"]
    available = [s for s in big_sectors if s in cum_ret.columns]
    ax = axes[1]
    for col in available:
        ret_5y = cum_ret[col].pct_change(252) * 100
        ax.plot(cum_ret.index, ret_5y, label=col, linewidth=0.8)
    ax.set_ylabel("YoY Return (%)")
    ax.set_title("Sector YoY Return (252-day change)", fontsize=14, fontweight="bold")
    add_event_spans(ax)
    ax.axhline(y=0, color="#555", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for a in axes:
        _format_xaxis(a)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_02_sector_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_02_sector_performance.png")


# ─── 3. Tech tickers around key events ──────────────────────────────────
def plot_event_closeup(df: pd.DataFrame) -> None:
    tech_tickers = ["NVDA", "AMD", "MSFT", "GOOGL", "AAPL", "CRM"]
    tech = df[df["Ticker"].isin(tech_tickers)].copy()
    events_closeup = [
        ("Dot-com Crash", "1999-06-01", "2003-12-31"),
        ("Financial Crisis", "2007-06-01", "2009-12-31"),
        ("COVID Crash & AI", "2019-06-01", "2025-07-03"),
    ]

    fig, axes = plt.subplots(len(events_closeup), 1, figsize=(16, 14))

    for idx, (title, start, end) in enumerate(events_closeup):
        ax = axes[idx]
        mask = (tech["Date"] >= start) & (tech["Date"] <= end)
        subset = _normalize(tech[mask])

        _plot_multiticker(
            ax, subset, tech_tickers,
            title=title, ylabel="Normalized Price (start=100)",
        )
        ax.axhline(y=100, color="#555", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_03_event_closeup.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_03_event_closeup.png")


# ─── 4. Correlation heatmap during key periods ──────────────────────────
def plot_correlation_heatmaps(df: pd.DataFrame) -> None:
    periods = [
        ("Full History", "2000-01-01", "2025-07-03"),
        ("Dot-com Era", "2000-01-01", "2003-12-31"),
        ("Pre-COVID Normal", "2016-01-01", "2019-12-31"),
        ("COVID & AI Era", "2020-01-01", "2025-07-03"),
    ]

    core_tickers = ["NVDA", "AMD", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
                    "CRM", "CSCO", "NFLX", "DIS", "COST", "HD", "UBER", "SPOT", "ZM"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for ax, (period_name, start, end) in zip(axes.flat, periods):
        mask = (df["Ticker"].isin(core_tickers)) & (df["Date"] >= start) & (df["Date"] <= end)
        pivot = df[mask].pivot_table(index="Date", columns="Ticker", values="ret")
        corr = pivot.corr()

        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=7)
        ax.set_title(period_name, fontsize=12, fontweight="bold")

        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.5 else "#ccc"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color=color)

    fig.colorbar(im, ax=axes, shrink=0.4, label="Pearson Correlation")
    plt.suptitle("Stock Return Correlations Across Market Regimes", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_04_correlation_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_04_correlation_heatmaps.png")


# ─── 5. Tech stocks during COVID crash (Feb-Mar 2020) ────────────────────
def plot_covid_crash(df: pd.DataFrame) -> None:
    tech_tickers = ["NVDA", "AMD", "MSFT", "AAPL", "GOOGL", "AMZN",
                    "CRM", "CSCO", "INTC", "NFLX"]
    tech = df[df["Ticker"].isin(tech_tickers)].copy()
    covid = tech[(tech["Date"] >= "2020-01-01") & (tech["Date"] <= "2021-06-30")].copy()
    covid = _normalize(covid)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    _plot_multiticker(
        ax, covid, tech_tickers,
        title="COVID Crash & Recovery — Tech Stocks Normalized",
        ylabel="Normalized Price (Jan 1 2020 = 100)",
    )
    ax.axvline(pd.Timestamp("2020-02-19"), color="#ef5350", linestyle="--", linewidth=0.8, label="Peak")
    ax.axvline(pd.Timestamp("2020-03-23"), color="#66bb6a", linestyle="--", linewidth=0.8, label="Trough")
    ax.axhline(y=100, color="#555", linewidth=0.5)

    ax = axes[1]
    covid["drawdown"] = covid.groupby("Ticker")["norm"].transform(
        lambda x: (x / x.cummax() - 1) * 100
    )
    _plot_multiticker(
        ax, covid, tech_tickers, value_col="drawdown",
        title="Drawdown from Peak During COVID",
        ylabel="Drawdown (%)",
    )
    ax.axvline(pd.Timestamp("2020-02-19"), color="#ef5350", linestyle="--", linewidth=0.8)
    ax.axvline(pd.Timestamp("2020-03-23"), color="#66bb6a", linestyle="--", linewidth=0.8)
    ax.axhline(y=0, color="#555", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_05_covid_crash.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_05_covid_crash.png")


# ─── 6. AI Explosion (2023-2025) ────────────────────────────────────────
def plot_ai_explosion(df: pd.DataFrame) -> None:
    ai_tickers = ["NVDA", "AMD", "MSFT", "GOOGL", "CRM", "META", "AMZN", "AAPL", "UBER", "PLTR"]
    ai_tickers = [t for t in ai_tickers if t in df["Ticker"].unique()]

    ai = df[df["Ticker"].isin(ai_tickers)].copy()
    ai = ai[(ai["Date"] >= "2022-06-01") & (ai["Date"] <= "2025-07-03")].copy()
    ai = _normalize(ai)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    _plot_multiticker(
        ax, ai, ai_tickers,
        title="AI Era (2022–2025): Tech Stocks Normalized",
        ylabel="Normalized Price (Jun 2022 = 100)",
    )

    milestones = [
        ("ChatGPT Launch", "2022-11-30"),
        ("GPT-4 Launch", "2023-03-14"),
        ("NVDA Q1 2023 Earnings", "2023-05-24"),
    ]
    for label, date in milestones:
        ax.axvline(pd.Timestamp(date), color="#42a5f5", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1] * 0.95, label,
                rotation=90, fontsize=7, color="#42a5f5", alpha=0.8)

    ax = axes[1]
    ema_tickers = ["NVDA", "AMD", "GOOGL", "MSFT", "CRM"]
    for ticker in ema_tickers:
        t = df[(df["Ticker"] == ticker) & (df["Date"] >= "2022-06-01")].copy()
        t["ema_50"] = t["Close"].ewm(span=50).mean()
        lw, alpha = _line_style(ticker)
        ax.plot(t["Date"], t["Close"], alpha=0.3, linewidth=0.5, color="gray")
        ax.plot(t["Date"], t["ema_50"], label=ticker, linewidth=lw)
    ax.set_title("50-Day EMA — Key AI Stocks", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_06_ai_explosion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_06_ai_explosion.png")


# ─── 7. Volume & volatility signatures during crashes ───────────────────
def plot_volatility_signatures(df: pd.DataFrame) -> None:
    ticker = HIGHLIGHT_TICKER
    t = df[df["Ticker"] == ticker].copy()
    t["vol_21d"] = t["ret"].rolling(21).std() * np.sqrt(252) * 100
    t["volume_ma"] = t["Volume"].rolling(21).mean()
    t["rel_vol"] = t["Volume"] / t["volume_ma"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    ax.plot(t["Date"], t["Close"], color="#42a5f5", linewidth=1)
    ax.fill_between(t["Date"], t["Close"], alpha=0.1, color="#42a5f5")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{ticker} — Volatility & Volume Signatures During Market Events", fontsize=14, fontweight="bold")
    add_event_spans(ax, alpha=0.1)

    ax = axes[1]
    ax.plot(t["Date"], t["vol_21d"], color="#ffa726", linewidth=0.8)
    ax.fill_between(t["Date"], t["vol_21d"], alpha=0.2, color="#ffa726")
    ax.set_ylabel("Volatility (%)")
    threshold = t["vol_21d"].quantile(0.9)
    ax.axhline(y=threshold, color="#ef5350", linestyle="--", linewidth=0.5,
               label=f"90th pctile: {threshold:.0f}%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t["Date"], t["rel_vol"], color="#ab47bc", linewidth=0.8)
    ax.fill_between(t["Date"], t["rel_vol"], alpha=0.2, color="#ab47bc")
    ax.set_ylabel("Relative Volume")
    ax.axhline(y=1.0, color="#555", linestyle="--", linewidth=0.5)
    ax.axhline(y=2.0, color="#ef5350", linestyle="--", linewidth=0.5,
               label="2x normal volume")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for a in axes:
        _format_xaxis(a)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_07_volatility_signatures.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_07_volatility_signatures.png")


# ─── 8. Summary statistics table ────────────────────────────────────────
def plot_summary_table(df: pd.DataFrame) -> None:
    top_tickers = ["AAPL", "NVDA", "AMZN", "MSFT", "GOOGL", "AMD", "META",
                   "NFLX", "DIS", "COST", "HD", "CSCO", "CRM", "SPOT", "ZM", "UBER"]
    top_tickers = [t for t in top_tickers if t in df["Ticker"].unique()]

    rows = []
    for ticker in top_tickers:
        t = df[df["Ticker"] == ticker].copy()
        ann_ret = (1 + t["ret"]).prod() ** (252 / len(t)) - 1
        ann_vol = t["ret"].std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd = (t["Close"] / t["Close"].cummax() - 1).min()
        rows.append({
            "Ticker": ticker, "Industry": t["Industry_Tag"].iloc[0],
            "Country": t["Country"].iloc[0],
            "Ann. Return": f"{ann_ret*100:.1f}%",
            "Ann. Vol": f"{ann_vol*100:.1f}%",
            "Sharpe": f"{sharpe:.2f}",
            "Max DD": f"{max_dd*100:.1f}%",
            "Rows": len(t),
        })

    fig, ax = plt.subplots(figsize=(16, len(rows) * 0.45 + 1))
    ax.axis("off")
    col_labels = ["Ticker", "Industry", "Country", "Ann. Return", "Ann. Vol", "Sharpe", "Max DD", "Rows"]
    cell_data = [[r[c] for c in col_labels] for r in rows]
    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     loc="center", cellLoc="center", colWidths=[0.08, 0.15, 0.08, 0.1, 0.1, 0.08, 0.08, 0.06])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if key[0] == 0:
            cell.set_text_props(weight="bold", color="#fff")
            cell.set_facecolor("#2a2a2a")
        elif key[1] == 0:
            is_hl = cell.get_text().get_text() == HIGHLIGHT_TICKER
            cell.set_text_props(weight="bold", color="#42a5f5" if is_hl else "#ccc")
            cell.set_facecolor("#1f1f1f")
        else:
            cell.set_facecolor("#1a1a1a")
            cell.set_text_props(color="#ccc")

    ax.set_title("Key Ticker Summary Statistics", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_08_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_08_summary_table.png")


# ─── 9. Predicted vs Actual scatter ─────────────────────────────────────
def plot_pred_vs_actual(pred_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    a, p = pred_df["Actual_denorm"].values, pred_df["Predicted_denorm"].values
    ax.scatter(a, p, alpha=0.3, s=10, color="#42a5f5", edgecolors="none")
    lims = [
        min(a.min(), p.min()),
        max(a.max(), p.max()),
    ]
    ax.plot(lims, lims, "--", color="#66bb6a", linewidth=1, label="Perfect Prediction")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Predicted vs Actual (Validation Set)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    mape_avg = np.mean(np.abs((a - p) / (a + 1e-8))) * 100
    ax.text(0.05, 0.95, f"MAPE: {mape_avg:.2f}%", transform=ax.transAxes,
            fontsize=12, color="#fff", va="top",
            bbox=dict(boxstyle="round", facecolor="#2a2a2a", edgecolor="#333"))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_09_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_09_pred_vs_actual.png")


# ─── 10. Per-ticker MAPE bar chart ──────────────────────────────────────
def plot_ticker_mape(tm_df: pd.DataFrame, top_n: int = 30) -> None:
    tm = tm_df.sort_values("MAPE", ascending=True).head(top_n)
    colors = ["#42a5f5" if t != HIGHLIGHT_TICKER else "#ffa726" for t in tm["Ticker"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(tm)), tm["MAPE"].values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(tm)))
    ax.set_yticklabels(tm["Ticker"].values, fontsize=8)
    ax.set_xlabel("MAPE (%)")
    ax.set_title(f"Per-Ticker Forecast Error (Top {top_n})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    for bar, val in zip(bars, tm["MAPE"].values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=7, color="#ccc")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_10_ticker_mape.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_10_ticker_mape.png")


# ─── 11. Error distribution ────────────────────────────────────────────
def plot_error_distribution(pred_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    errors = pred_df["Actual_denorm"] - pred_df["Predicted_denorm"]
    pct_errors = (errors / (pred_df["Actual_denorm"] + 1e-8)) * 100

    ax = axes[0]
    ax.hist(errors, bins=60, color="#42a5f5", alpha=0.8, edgecolor="none")
    ax.axvline(x=0, color="#66bb6a", linestyle="--", linewidth=1)
    ax.set_xlabel("Absolute Error ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution (Actual - Predicted)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(pct_errors, bins=60, color="#ab47bc", alpha=0.8, edgecolor="none")
    ax.axvline(x=0, color="#66bb6a", linestyle="--", linewidth=1)
    ax.set_xlabel("Percentage Error (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Percentage Error Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_11_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_11_error_distribution.png")


# ─── 12. Prediction time series for key tickers ─────────────────────────
def plot_prediction_timeseries(pred_df: pd.DataFrame, df: pd.DataFrame) -> None:
    key_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    fig, axes = plt.subplots(len(key_tickers), 1, figsize=(16, 12), sharex=True)

    for idx, ticker in enumerate(key_tickers):
        ax = axes[idx]
        sub = pred_df[pred_df["Ticker"] == ticker].copy()
        if sub.empty:
            ax.set_title(f"{ticker} — No predictions in validation set", fontsize=11, fontweight="bold")
            continue

        sub = sub.sort_values("sample_idx")
        steps = np.arange(len(sub))
        ax.plot(steps, sub["Actual_denorm"].values, color="#42a5f5", linewidth=1, label="Actual")
        ax.plot(steps, sub["Predicted_denorm"].values, color="#ffa726", linewidth=1, alpha=0.8, label="Predicted")
        ax.fill_between(steps, sub["Actual_denorm"].values, sub["Predicted_denorm"].values,
                        alpha=0.1, color="#ef5350")
        ticker_mape = np.mean(np.abs((sub["Actual_denorm"] - sub["Predicted_denorm"]) / (sub["Actual_denorm"] + 1e-8))) * 100
        ax.set_title(f"{ticker} — Actual vs Predicted (MAPE: {ticker_mape:.2f}%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Price ($)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Prediction Step")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_12_prediction_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {OUTPUT_DIR}/viz_12_prediction_timeseries.png")


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["ret"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change())
    print(f"  {len(df)} rows, {df['Ticker'].nunique()} tickers, {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")

    print("\nLoading model predictions...")
    pred_df = pd.read_csv(f"{OUTPUT_DIR}/predictions.csv")
    tm_df = pd.read_csv(f"{OUTPUT_DIR}/ticker_metrics.csv")
    print(f"  {len(pred_df)} prediction rows, {tm_df['Ticker'].nunique()} tickers")

    print("\nGenerating visualizations...")
    plot_aapl_history(df)
    plot_sector_performance(df)
    plot_event_closeup(df)
    plot_correlation_heatmaps(df)
    plot_covid_crash(df)
    plot_ai_explosion(df)
    plot_volatility_signatures(df)
    plot_summary_table(df)
    print("\n--- Accuracy Visualizations ---")
    plot_pred_vs_actual(pred_df)
    plot_ticker_mape(tm_df, top_n=30)
    plot_error_distribution(pred_df)
    plot_prediction_timeseries(pred_df, df)
    print("\nAll 12 visualizations saved to current directory!")
