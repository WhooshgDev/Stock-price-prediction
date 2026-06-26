import os, sys
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, get_ticker_data, train_test_split_ticker, compute_metrics
from utils.plots import plot_actual_vs_predicted, plot_feature_importance

st.title("🌲 XGBoost — Gradient Boosted Trees")

st.markdown(
    """
    **XGBoost** builds an ensemble of decision trees sequentially, each correcting the errors of the previous.
    Best parameters found on this dataset: `max_depth=5`, `learning_rate=0.001`, `n_estimators=100`.

    Features used: lagged returns, rolling volatility, volume ratios, price range indicators.
    """
)

df = load_raw_data()
tickers = get_ticker_list(df)

col1, col2 = st.columns(2)
ticker = col1.selectbox("Ticker", tickers, key="xgb_ticker")
train_pct = col2.slider("Train %", 0.5, 0.95, 0.8, 0.05)

col3, col4, col5 = st.columns(3)
n_estimators = col3.number_input("Trees", min_value=10, max_value=500, value=100, step=10)
max_depth = col4.number_input("Max Depth", min_value=1, max_value=15, value=5)
lr = col5.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], value=0.001)

if st.button("▶ Run XGBoost", type="primary"):
    with st.spinner("Training XGBoost model..."):
        td = get_ticker_data(df, ticker).dropna()

        td["lag_1"] = td["ret"].shift(1)
        td["lag_2"] = td["ret"].shift(2)
        td["lag_3"] = td["ret"].shift(3)
        td["lag_5"] = td["ret"].shift(5)
        td["lag_10"] = td["ret"].shift(10)
        td["vol_5"] = td["log_ret"].rolling(5).std()
        td["vol_10"] = td["log_ret"].rolling(10).std()
        td["sma_ratio"] = td["Close"] / td["SMA_10"]
        td["vol_ratio"] = td["Volume"] / td["Volume_MA_10"]
        td["hl_ratio"] = td["high_low_pct"]
        td["co_ratio"] = td["close_open_pct"]

        feature_cols = [
            "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
            "vol_5", "vol_10", "vol_21",
            "sma_ratio", "vol_ratio", "hl_ratio", "co_ratio",
        ]
        td = td.dropna()
        train, test = train_test_split_ticker(td, train_pct)

        X_train = train[feature_cols].values
        y_train = train["ret"].values
        X_test = test[feature_cols].values
        y_test = test["ret"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=lr, random_state=42, n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        metrics = compute_metrics(y_test, preds)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", metrics["MAE"])
    col2.metric("RMSE", metrics["RMSE"])
    col3.metric("Directional Accuracy", f"{metrics['Directional Accuracy (%)']}%")
    col4.metric("Correlation", metrics["Correlation"])

    st.subheader("Actual vs Predicted Returns")
    fig = plot_actual_vs_predicted(
        test["Date"].values, y_test, preds,
        title="XGBoost — Return Prediction", ticker=ticker
    )
    st.pyplot(fig)

    st.subheader("Feature Importance")
    fig2 = plot_feature_importance(model.feature_importances_, feature_cols)
    st.pyplot(fig2)
