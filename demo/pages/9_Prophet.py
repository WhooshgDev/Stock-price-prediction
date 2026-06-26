import os, sys
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, get_ticker_data, train_test_split_ticker, compute_metrics
from utils.plots import plot_forecast_with_uncertainty

st.title("🔮 Prophet — Additive Forecasting")

st.markdown(
    """
    **Prophet** (by Meta/Facebook) decomposes time series into:
    - **Trend** (piecewise linear or logistic growth)
    - **Seasonality** (weekly, yearly, daily)
    - **Holiday effects**

    It is robust to missing data and handles outliers well. On financial data,
    it captures trend shifts via automatic changepoint detection.

    *Note: Prophet is relatively fast; predictions include uncertainty intervals.*
    """
)

df = load_raw_data()
tickers = get_ticker_list(df)

col1, col2 = st.columns(2)
ticker = col1.selectbox("Ticker", tickers, key="prophet_ticker")
train_pct = col2.slider("Train %", 0.5, 0.95, 0.8, 0.05)

col3, col4 = st.columns(2)
weekly_seas = col3.checkbox("Weekly Seasonality", True)
changepoint_prior = col4.select_slider(
    "Changepoint Prior Scale",
    options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    value=0.05,
)

if st.button("▶ Run Prophet", type="primary"):
    with st.spinner("Fitting Prophet model..."):
        td = get_ticker_data(df, ticker).dropna()
        train, test = train_test_split_ticker(td, train_pct)

        prophet_train = train[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet(
            daily_seasonality=False,
            yearly_seasonality=False,
            weekly_seasonality=weekly_seas,
            changepoint_prior_scale=changepoint_prior,
        )
        model.fit(prophet_train)

        future = test[["Date"]].rename(columns={"Date": "ds"})
        forecast = model.predict(future)

        y_true = test["Close"].values
        y_pred = forecast["yhat"].values
        y_lower = forecast["yhat_lower"].values
        y_upper = forecast["yhat_upper"].values

        metrics = compute_metrics(y_true, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", metrics["MAE"])
    col2.metric("RMSE", metrics["RMSE"])
    col3.metric("Directional Accuracy", f"{metrics['Directional Accuracy (%)']}%")
    col4.metric("Correlation", metrics["Correlation"])

    st.subheader("Forecast vs Actual (Close Price)")
    all_dates = np.concatenate([train["Date"].values, test["Date"].values])
    all_history = train["Close"].values
    fig = plot_forecast_with_uncertainty(
        all_dates, all_history, y_pred, y_lower, y_upper,
        title="Prophet — Close Price Forecast", 
    )
    st.pyplot(fig)

    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
