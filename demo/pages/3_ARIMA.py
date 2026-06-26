import os, sys
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, get_ticker_data, train_test_split_ticker, compute_metrics
from utils.plots import plot_actual_vs_predicted, plot_residuals

st.title("📈 ARIMA — AutoRegressive Integrated Moving Average")

st.markdown(
    """
    **ARIMA(p,d,q)** models the autocorrelation in the time series after differencing to achieve stationarity.
    - `p` = autoregressive order (lags of the dependent variable)
    - `d` = differencing order (to remove trend)
    - `q` = moving average order (lags of forecast errors)

    The best model found via Auto ARIMA on this data: **ARIMA(1,0,0)** with AIC = −22,871.84.
    """
)

df = load_raw_data()
tickers = get_ticker_list(df)

col1, col2, col3 = st.columns(3)
ticker = col1.selectbox("Ticker", tickers, key="arima_ticker")
p = col2.number_input("AR order (p)", min_value=0, max_value=5, value=1)
d = col3.number_input("Difference (d)", min_value=0, max_value=2, value=0)

col4, col5 = st.columns(2)
q = col4.number_input("MA order (q)", min_value=0, max_value=5, value=0)
train_pct = col5.slider("Train %", 0.5, 0.95, 0.8, 0.05)

if st.button("▶ Run ARIMA", type="primary"):
    with st.spinner("Fitting ARIMA model..."):
        td = get_ticker_data(df, ticker).dropna()
        train, test = train_test_split_ticker(td, train_pct)
        y_train, y_test = train["Close"].values, test["Close"].values

        model = ARIMA(y_train, order=(p, d, q))
        fitted = model.fit()

        preds = fitted.forecast(steps=len(y_test))

        metrics = compute_metrics(y_test, preds)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", metrics["MAE"])
    col2.metric("RMSE", metrics["RMSE"])
    col3.metric("Directional Accuracy", f"{metrics['Directional Accuracy (%)']}%")
    col4.metric("Correlation", metrics["Correlation"])

    st.subheader("Model Summary")
    st.text(str(fitted.summary()))

    st.subheader("Actual vs Predicted (Close Price)")
    fig = plot_actual_vs_predicted(
        test["Date"].values, y_test, preds,
        title=f"ARIMA({p},{d},{q}) — Close Price", ticker=ticker
    )
    st.pyplot(fig)

    st.subheader("Residual Analysis")
    fig2 = plot_residuals(y_test, preds)
    st.pyplot(fig2)

    st.subheader("ARIMA Coefficients")
    coef_df = pd.DataFrame({
        "Coefficient": fitted.params,
        "Std Error": fitted.bse,
        "P-value": fitted.pvalues,
    })
    st.dataframe(coef_df, use_container_width=True)
