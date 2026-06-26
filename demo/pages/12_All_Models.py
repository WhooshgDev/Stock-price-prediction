import os, sys, warnings
import streamlit as st
import pandas as pd
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, get_ticker_data, train_test_split_ticker, compute_metrics
from utils.forecast import (
    arima_forecast, prophet_forecast, xgboost_forecast, hmm_forecast,
    hmm_xgboost_forecast, tabnet_forecast, tft_forecast,
    stat_forecast, build_forecast_table, return_to_price,
)

st.title("📈 Real Return Forecast")

st.markdown(
    """
    Select a ticker and forecast horizon. All models train on **all available historical data**
    and predict forward returns. Results show predicted direction and magnitude.
    """
)

df = load_raw_data()
tickers = get_ticker_list(df)

col1, col2 = st.columns([2, 1])
ticker = col1.selectbox("Ticker", tickers, index=list(tickers).index("AAPL") if "AAPL" in tickers else 0)
horizon = col2.slider("Forecast Horizon (days)", 1, 30, 5)

st.sidebar.header("Models")
all_models = {
    "ARIMA(1,0,0)": True,
    "XGBoost": True,
    "HMM-XGBoost": True,
    "TabNet": False,
    "TFT (Transformer)": False,
    "Prophet": True,
    "HMM (3-state)": False,
    "SMA(10)": False,
    "WMA(10)": False,
    "Median(10)": False,
    "ETS(10)": False,
    "Drift": False,
}
selected = {}
for name, default in all_models.items():
    selected[name] = st.sidebar.checkbox(name, value=default)

if st.button("▶ Run Forecast", type="primary"):
    td = get_ticker_data(df, ticker).dropna()
    last_close = td["Close"].values[-1]
    last_date = td["Date"].values[-1]
    ret_series = td["ret"].dropna().values
    close_series = td["Close"].dropna().values

    status = st.status(f"Forecasting {ticker} — {horizon} days ahead...", expanded=True)
    results = []

    if "ARIMA(1,0,0)" in selected and selected["ARIMA(1,0,0)"]:
        status.write("ARIMA...")
        try:
            pred = arima_forecast(ret_series, horizon)
            results.append(build_forecast_table("ARIMA(1,0,0)", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "XGBoost" in selected and selected["XGBoost"]:
        status.write("XGBoost...")
        try:
            pred = xgboost_forecast(ret_series, [1, 2, 3, 5, 10], horizon)
            results.append(build_forecast_table("XGBoost", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "HMM-XGBoost" in selected and selected["HMM-XGBoost"]:
        status.write("HMM-XGBoost...")
        try:
            pred = hmm_xgboost_forecast(ret_series, [1, 2, 3, 5, 10], horizon)
            results.append(build_forecast_table("HMM-XGBoost", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "TabNet" in selected and selected["TabNet"]:
        status.write("TabNet...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = tabnet_forecast(td, horizon)
            results.append(build_forecast_table("TabNet", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "TFT (Transformer)" in selected and selected["TFT (Transformer)"]:
        status.write("TFT...")
        try:
            pred = tft_forecast(ticker, horizon)
            results.append(build_forecast_table("TFT", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "Prophet" in selected and selected["Prophet"]:
        status.write("Prophet...")
        try:
            pred = prophet_forecast(td, horizon)
            results.append(build_forecast_table("Prophet", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "HMM (3-state)" in selected and selected["HMM (3-state)"]:
        status.write("HMM...")
        try:
            pred = hmm_forecast(ret_series, horizon)
            results.append(build_forecast_table("HMM", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "SMA(10)" in selected and selected["SMA(10)"]:
        status.write("SMA...")
        try:
            pred = stat_forecast(close_series, horizon, "SMA", 10)
            results.append(build_forecast_table("SMA(10)", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "WMA(10)" in selected and selected["WMA(10)"]:
        status.write("WMA...")
        try:
            pred = stat_forecast(close_series, horizon, "WMA", 10)
            results.append(build_forecast_table("WMA(10)", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "Median(10)" in selected and selected["Median(10)"]:
        status.write("Median...")
        try:
            pred = stat_forecast(close_series, horizon, "Median", 10)
            results.append(build_forecast_table("Median(10)", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "ETS(10)" in selected and selected["ETS(10)"]:
        status.write("ETS...")
        try:
            pred = stat_forecast(close_series, horizon, "ETS", 10)
            results.append(build_forecast_table("ETS(10)", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    if "Drift" in selected and selected["Drift"]:
        status.write("Drift...")
        try:
            pred = stat_forecast(close_series, horizon, "Drift")
            results.append(build_forecast_table("Drift", pred, last_close, horizon))
            status.write("  OK")
        except Exception as e:
            status.write(f"  FAIL: {e}")

    status.update(label=f"Done — {len(results)} models", state="complete")

    if not results:
        st.warning("No models selected.")
        st.stop()

    combined = pd.concat(results, ignore_index=True)

    st.subheader(f"{ticker} — {horizon}-Day Return Forecast")
    st.caption(f"Latest data: {pd.Timestamp(last_date).strftime('%Y-%m-%d')}  |  Last Close: ${last_close:.2f}")

    pivot_dir = combined.pivot_table(index="Step Ahead", columns="Model", values="Direction", aggfunc="first")
    pivot_ret = combined.pivot_table(index="Step Ahead", columns="Model", values="Predicted Return (%)", aggfunc="first")
    pivot_price = combined.pivot_table(index="Step Ahead", columns="Model", values="Predicted Price", aggfunc="first")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Predicted Return (%)**")
        st.dataframe(pivot_ret.style.format("{:.2f}").map(
            lambda v: "color: green" if isinstance(v, (int, float)) and v > 0 else ("color: red" if isinstance(v, (int, float)) and v < 0 else ""),
        ), use_container_width=True)
    with col_b:
        st.markdown("**Direction**")
        st.dataframe(pivot_dir, use_container_width=True)

    st.markdown("**Predicted Price**")
    st.dataframe(pivot_price.style.format("${:.2f}"), use_container_width=True)

    combined["Predicted Return (%)"] = combined["Predicted Return (%)"].astype(float)
    combined["Predicted Price"] = combined["Predicted Price"].astype(float)

    st.subheader("Return Forecast Plot")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5))
    for m in combined["Model"].unique():
        sub = combined[combined["Model"] == m]
        ax.plot(sub["Step Ahead"], sub["Predicted Return (%)"], marker="o", label=m, linewidth=1.5)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Predicted Return (%)")
    ax.set_title(f"{ticker} — Forward Return Forecast ({horizon} days)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

    csv = combined.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", csv, f"{ticker}_forecast_{horizon}d.csv", "text/csv")

    up_count = (combined["Predicted Return (%)"] > 0).sum()
    down_count = (combined["Predicted Return (%)"] < 0).sum()
    total_predictions = len(combined)
    if total_predictions > 0:
        st.info(f"**Consensus:** {up_count}/{total_predictions} predictions bullish ({up_count/total_predictions*100:.0f}%), "
                f"{down_count}/{total_predictions} bearish ({down_count/total_predictions*100:.0f}%)")