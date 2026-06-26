import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, get_ticker_data, train_test_split_ticker, compute_metrics
from utils.plots import plot_actual_vs_predicted

st.title("📐 Statistical Models — Classical Baselines")

st.markdown(
    """
    Classical time series models serve as essential **baselines** for evaluating more complex algorithms.
    These models are simple, fast, and interpretable.

    | Model | Description | Window |
    |-------|-------------|--------|
    | **SMA** | Simple Moving Average | 10-day |
    | **WMA** | Linearly Weighted Moving Average (more weight to recent) | 10-day |
    | **Median** | Rolling Median (robust to outliers) | 20-day |
    | **ETS** | Exponentially Weighted Moving Average (EMA) | span=10 |
    | **Drift** | Forecast extrapolates the average slope from start to current | full history |
    """
)

df = load_raw_data()
tickers = get_ticker_list(df)

col1, col2 = st.columns(2)
ticker = col1.selectbox("Ticker", tickers, key="stats_ticker")
train_pct = col2.slider("Train %", 0.5, 0.95, 0.8, 0.05)

model_choices = st.multiselect(
    "Select Models",
    ["SMA", "WMA", "Median", "ETS", "Drift"],
    default=["SMA", "WMA", "ETS"],
)

def calc_wma(prices):
    w = np.arange(1, len(prices) + 1)
    return np.dot(prices, w) / w.sum()

if st.button("▶ Run Statistical Models", type="primary"):
    td = get_ticker_data(df, ticker)
    train, test = train_test_split_ticker(td, train_pct)
    all_data = pd.concat([train, test])

    results = {}
    for model_name in model_choices:
        if model_name == "SMA":
            preds = all_data["Close"].rolling(10).mean()
        elif model_name == "WMA":
            preds = all_data["Close"].rolling(10).apply(calc_wma, raw=True)
        elif model_name == "Median":
            preds = all_data["Close"].rolling(20).median()
        elif model_name == "ETS":
            preds = all_data["Close"].ewm(span=10, adjust=False).mean()
        elif model_name == "Drift":
            cumcount = all_data.groupby("Ticker").cumcount() + 1 if "Ticker" in all_data.columns else range(1, len(all_data) + 1)
            first_close = all_data["Close"].iloc[0]
            seq_num = np.arange(1, len(all_data) + 1)
            preds = all_data["Close"] + (all_data["Close"] - first_close) / (seq_num - 1)
            preds.iloc[0] = all_data["Close"].iloc[0]

        test_preds = preds.iloc[len(train):].values
        test_actual = test["Close"].values

        mask = ~(np.isnan(test_preds) | np.isnan(test_actual))
        test_preds = test_preds[mask]
        test_actual = test_actual[mask]
        test_dates = test["Date"].values[mask]

        if len(test_preds) > 0:
            metrics = compute_metrics(test_actual, test_preds)
            results[model_name] = {
                "metrics": metrics,
                "preds": test_preds,
                "actual": test_actual,
                "dates": test_dates,
            }

    if results:
        metrics_df = pd.DataFrame({
            name: res["metrics"] for name, res in results.items()
        }).T
        st.subheader("Performance Comparison")
        st.dataframe(metrics_df.style.highlight_min(axis=0, color="#FFCDD2")
                     .highlight_max(axis=0, color="#C8E6C9"), use_container_width=True)

        st.subheader("Actual vs Predicted (Close Price)")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(test["Date"].values, test["Close"].values, color="#333", linewidth=2, label="Actual", alpha=0.8)
        colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
        for i, (name, res) in enumerate(results.items()):
            ax.plot(res["dates"], res["preds"], color=colors[i % len(colors)],
                    linewidth=1.2, label=name, linestyle="--", alpha=0.8)
        ax.set_title(f"Statistical Models — {ticker}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
