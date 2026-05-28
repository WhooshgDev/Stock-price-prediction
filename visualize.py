import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

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

DATA_PATH = "World-Stock-Prices-Dataset.csv"

EVENTS = [
    ("Dot-com Crash", "2000-03-10", "2002-10-09", "#ff6b6b"),
    ("9/11 Attacks", "2001-09-11", "2001-09-21", "#ff4444"),
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

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["ret"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change())
    return df

def add_event_spans(ax, alpha=0.15):
    for name, start, end, color in EVENTS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=alpha, color=color, zorder=2)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(mid, ax.get_ylim()[1], name, ha="center", va="top",
                fontsize=7, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor=color, alpha=0.8))


# ─── 1. NVDA full history ───────────────────────────────────────────────
def plot_nvda_history(df):
    nvda = df[df["Ticker"] == "NVDA"].copy()
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    ax.plot(nvda["Date"], nvda["Close"], color="#42a5f5", linewidth=1, label="Close")
    ax.fill_between(nvda["Date"], nvda["Close"], alpha=0.1, color="#42a5f5")
    ax.set_ylabel("Price ($)")
    ax.set_title("NVDA — Full History (2000–2025)", fontsize=14, fontweight="bold")
    add_event_spans(ax)
    ax.legend()

    ax = axes[1]
    cumret = (1 + nvda["ret"]).cumprod()
    ax.plot(nvda["Date"], cumret, color="#66bb6a", linewidth=1)
    ax.fill_between(nvda["Date"], cumret, alpha=0.1, color="#66bb6a")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("NVDA — Cumulative Return (log scale)", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    add_event_spans(ax)

    ax = axes[2]
    vol = nvda["ret"].rolling(21).std() * np.sqrt(252) * 100
    ax.plot(nvda["Date"], vol, color="#ffa726", linewidth=0.8)
    ax.fill_between(nvda["Date"], vol, alpha=0.15, color="#ffa726")
    ax.set_ylabel("Volatility (%)")
    ax.set_title("NVDA — 21-Day Rolling Volatility (annualized)", fontsize=14, fontweight="bold")
    add_event_spans(ax)
    ax.axhline(y=vol.median(), color="#888", linestyle="--", linewidth=0.5, label=f"Median: {vol.median():.1f}%")
    ax.legend()

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("viz_01_nvda_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_01_nvda_history.png")


# ─── 2. Sector performance through events ───────────────────────────────
def plot_sector_performance(df):
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

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig("viz_02_sector_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_02_sector_performance.png")


# ─── 3. Tech tickers around key events ──────────────────────────────────
def plot_event_closeup(df):
    tech = df[df["Ticker"].isin(["NVDA", "AMD", "MSFT", "GOOGL", "AAPL", "CRM"])].copy()
    events_closeup = [
        ("Dot-com Crash", "1999-06-01", "2003-12-31"),
        ("Financial Crisis", "2007-06-01", "2009-12-31"),
        ("COVID Crash & AI", "2019-06-01", "2025-07-03"),
    ]

    fig, axes = plt.subplots(len(events_closeup), 1, figsize=(16, 14))

    for idx, (title, start, end) in enumerate(events_closeup):
        ax = axes[idx]
        mask = (tech["Date"] >= start) & (tech["Date"] <= end)
        subset = tech[mask].copy()
        subset["norm"] = subset.groupby("Ticker")["Close"].transform(
            lambda x: x / x.iloc[0] * 100
        )

        for ticker in ["NVDA", "AMD", "MSFT", "GOOGL", "AAPL", "CRM"]:
            t = subset[subset["Ticker"] == ticker]
            if len(t) == 0:
                continue
            lw = 2.5 if ticker == "NVDA" else 0.8
            alpha = 1.0 if ticker == "NVDA" else 0.5
            ax.plot(t["Date"], t["norm"], label=ticker, linewidth=lw, alpha=alpha)

        ax.set_title(f"{title}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Normalized Price (start=100)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color="#555", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig("viz_03_event_closeup.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_03_event_closeup.png")


# ─── 4. Correlation heatmap during key periods ──────────────────────────
def plot_correlation_heatmaps(df):
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
        sub = df[mask].copy()
        pivot = sub.pivot_table(index="Date", columns="Ticker", values="ret")
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
    plt.savefig("viz_04_correlation_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_04_correlation_heatmaps.png")


# ─── 5. NVDA vs peers during COVID crash (Feb-Mar 2020) ─────────────────
def plot_covid_crash(df):
    tech = df[df["Ticker"].isin(["NVDA", "AMD", "MSFT", "AAPL", "GOOGL", "AMZN",
                                  "CRM", "CSCO", "INTC", "NFLX"])].copy()
    covid = tech[(tech["Date"] >= "2020-01-01") & (tech["Date"] <= "2021-06-30")].copy()
    covid["norm"] = covid.groupby("Ticker")["Close"].transform(lambda x: x / x.iloc[0] * 100)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    for ticker in covid["Ticker"].unique():
        t = covid[covid["Ticker"] == ticker]
        lw = 2.5 if ticker == "NVDA" else 0.8
        alpha = 1.0 if ticker == "NVDA" else 0.4
        ax.plot(t["Date"], t["norm"], label=ticker, linewidth=lw, alpha=alpha)
    ax.set_title("COVID Crash & Recovery — Tech Stocks Normalized", fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalized Price (Jan 1 2020 = 100)")
    ax.axvline(pd.Timestamp("2020-02-19"), color="#ef5350", linestyle="--", linewidth=0.8, label="Peak")
    ax.axvline(pd.Timestamp("2020-03-23"), color="#66bb6a", linestyle="--", linewidth=0.8, label="Trough")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="#555", linewidth=0.5)

    ax = axes[1]
    covid["drawdown"] = covid.groupby("Ticker")["norm"].transform(
        lambda x: (x / x.cummax() - 1) * 100
    )
    for ticker in covid["Ticker"].unique():
        t = covid[covid["Ticker"] == ticker]
        lw = 2.5 if ticker == "NVDA" else 0.8
        alpha = 1.0 if ticker == "NVDA" else 0.4
        ax.plot(t["Date"], t["drawdown"], label=ticker, linewidth=lw, alpha=alpha)
    ax.set_title("Drawdown from Peak During COVID", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.axvline(pd.Timestamp("2020-02-19"), color="#ef5350", linestyle="--", linewidth=0.8)
    ax.axvline(pd.Timestamp("2020-03-23"), color="#66bb6a", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="#555", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("viz_05_covid_crash.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_05_covid_crash.png")


# ─── 6. AI Explosion (2023-2025) ────────────────────────────────────────
def plot_ai_explosion(df):
    ai_related = ["NVDA", "AMD", "MSFT", "GOOGL", "CRM", "META", "AMZN", "AAPL", "UBER", "PLTR"]
    # only include tickers that exist
    ai_related = [t for t in ai_related if t in df["Ticker"].unique()]

    ai = df[df["Ticker"].isin(ai_related)].copy()
    ai = ai[(ai["Date"] >= "2022-06-01") & (ai["Date"] <= "2025-07-03")].copy()
    ai["norm"] = ai.groupby("Ticker")["Close"].transform(lambda x: x / x.iloc[0] * 100)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    for ticker in ai["Ticker"].unique():
        t = ai[ai["Ticker"] == ticker]
        lw = 2.5 if ticker == "NVDA" else 0.8
        alpha = 1.0 if ticker == "NVDA" else 0.4
        ax.plot(t["Date"], t["norm"], label=ticker, linewidth=lw, alpha=alpha)

    milestones = [
        ("ChatGPT Launch", "2022-11-30"),
        ("GPT-4 Launch", "2023-03-14"),
        ("NVDA Q1 2023 Earnings", "2023-05-24"),
    ]
    for label, date in milestones:
        ax.axvline(pd.Timestamp(date), color="#42a5f5", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1] * 0.95, label,
                rotation=90, fontsize=7, color="#42a5f5", alpha=0.8)

    ax.set_title("AI Era (2022–2025): Tech Stocks Normalized", fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalized Price (Jun 2022 = 100)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for ticker in ["NVDA", "AMD", "GOOGL", "MSFT", "CRM"]:
        t = df[(df["Ticker"] == ticker) & (df["Date"] >= "2022-06-01")].copy()
        t["ema_50"] = t["Close"].ewm(span=50).mean()
        lw = 2.5 if ticker == "NVDA" else 0.8
        ax.plot(t["Date"], t["Close"], alpha=0.3, linewidth=0.5, color="gray")
        ax.plot(t["Date"], t["ema_50"], label=ticker, linewidth=lw)
    ax.set_title("50-Day EMA — Key AI Stocks", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("viz_06_ai_explosion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_06_ai_explosion.png")


# ─── 7. Volume & volatility signatures during crashes ───────────────────
def plot_volatility_signatures(df):
    nvda = df[df["Ticker"] == "NVDA"].copy()
    nvda["vol_21d"] = nvda["ret"].rolling(21).std() * np.sqrt(252) * 100
    nvda["volume_ma"] = nvda["Volume"].rolling(21).mean()
    nvda["rel_vol"] = nvda["Volume"] / nvda["volume_ma"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    ax.plot(nvda["Date"], nvda["Close"], color="#42a5f5", linewidth=1)
    ax.fill_between(nvda["Date"], nvda["Close"], alpha=0.1, color="#42a5f5")
    ax.set_ylabel("Price ($)")
    ax.set_title("NVDA — Volatility & Volume Signatures During Market Events", fontsize=14, fontweight="bold")
    add_event_spans(ax, alpha=0.1)

    ax = axes[1]
    ax.plot(nvda["Date"], nvda["vol_21d"], color="#ffa726", linewidth=0.8)
    ax.fill_between(nvda["Date"], nvda["vol_21d"], alpha=0.2, color="#ffa726")
    ax.set_ylabel("Volatility (%)")
    threshold = nvda["vol_21d"].quantile(0.9)
    ax.axhline(y=threshold, color="#ef5350", linestyle="--", linewidth=0.5,
               label=f"90th pctile: {threshold:.0f}%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(nvda["Date"], nvda["rel_vol"], color="#ab47bc", linewidth=0.8)
    ax.fill_between(nvda["Date"], nvda["rel_vol"], alpha=0.2, color="#ab47bc")
    ax.set_ylabel("Relative Volume")
    ax.axhline(y=1.0, color="#555", linestyle="--", linewidth=0.5)
    ax.axhline(y=2.0, color="#ef5350", linestyle="--", linewidth=0.5,
               label="2x normal volume")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig("viz_07_volatility_signatures.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_07_volatility_signatures.png")


# ─── 8. Summary statistics table ────────────────────────────────────────
def plot_summary_table(df):
    top_tickers = ["NVDA", "AAPL", "AMZN", "MSFT", "GOOGL", "AMD", "META",
                   "NFLX", "DIS", "COST", "HD", "CSCO", "CRM", "SPOT", "ZM", "UBER"]
    top_tickers = [t for t in top_tickers if t in df["Ticker"].unique()]

    rows = []
    for ticker in top_tickers:
        t = df[df["Ticker"] == ticker].copy()
        t["ret"] = t["Close"].pct_change()
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
            cell.set_text_props(weight="bold", color="#42a5f5" if cell.get_text().get_text() == "NVDA" else "#ccc")
            cell.set_facecolor("#1f1f1f")
        else:
            cell.set_facecolor("#1a1a1a")
            cell.set_text_props(color="#ccc")

    ax.set_title("Key Ticker Summary Statistics", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("viz_08_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ viz_08_summary_table.png")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows, {df['Ticker'].nunique()} tickers, {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")

    print("\nGenerating visualizations...")
    plot_nvda_history(df)
    plot_sector_performance(df)
    plot_event_closeup(df)
    plot_correlation_heatmaps(df)
    plot_covid_crash(df)
    plot_ai_explosion(df)
    plot_volatility_signatures(df)
    plot_summary_table(df)
    print("\n✅ All 8 visualizations saved to current directory!")
