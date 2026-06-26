import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, get_ticker_data, train_test_split_ticker, compute_metrics

st.title("⚖️ Model Comparison")

st.markdown(
    """
    Run all major algorithms on the same ticker and compare their performance side-by-side.
    Each model uses default hyperparameters for a fair baseline comparison.

    *Note: Training all models may take several minutes.*
    """
)

df = load_raw_data()
tickers = get_ticker_list(df)

col1, col2 = st.columns(2)
ticker = col1.selectbox("Ticker", tickers, key="comp_ticker")
train_pct = col2.slider("Train %", 0.5, 0.95, 0.8, 0.05)

models_to_run = st.multiselect(
    "Models to Compare",
    ["ARIMA(1,0,0)", "XGBoost", "HMM-XGBoost Hybrid", "Prophet", "SMA", "ETS", "Drift"],
    default=["ARIMA(1,0,0)", "XGBoost", "SMA", "ETS"],
)

if st.button("▶ Run All Selected Models", type="primary"):
    with st.spinner("Preparing data..."):
        td = get_ticker_data(df, ticker).dropna()
        train, test = train_test_split_ticker(td, train_pct)
        y_test = test["ret"].values

    results = {}

    if "ARIMA(1,0,0)" in models_to_run:
        with st.spinner("ARIMA..."):
            try:
                model = ARIMA(train["Close"].values, order=(1, 0, 0))
                fitted = model.fit()
                forecast = fitted.forecast(steps=len(test))
                ret_pred = (forecast - test["Close"].values) / test["Close"].values
                results["ARIMA(1,0,0)"] = compute_metrics(y_test, ret_pred)
            except Exception as e:
                results["ARIMA(1,0,0)"] = {"MAE": None, "RMSE": None, "Directional Accuracy (%)": None, "Correlation": None, "error": str(e)}

    if "XGBoost" in models_to_run:
        with st.spinner("XGBoost..."):
            feat_cols = ["lag_1", "lag_2", "lag_3", "lag_5", "vol_5", "vol_10", "sma_ratio", "vol_ratio"]
            tdx = td.copy()
            tdx["lag_1"] = tdx["ret"].shift(1)
            tdx["lag_2"] = tdx["ret"].shift(2)
            tdx["lag_3"] = tdx["ret"].shift(3)
            tdx["lag_5"] = tdx["ret"].shift(5)
            tdx["vol_5"] = tdx["log_ret"].rolling(5).std()
            tdx["vol_10"] = tdx["log_ret"].rolling(10).std()
            tdx["sma_ratio"] = tdx["Close"] / tdx["SMA_10"]
            tdx["vol_ratio"] = tdx["Volume"] / tdx["Volume_MA_10"]
            tdx = tdx.dropna()
            tr, te = train_test_split_ticker(tdx, train_pct)
            scaler = StandardScaler()
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.01, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(scaler.fit_transform(tr[feat_cols].values), tr["ret"].values)
            preds = model.predict(scaler.transform(te[feat_cols].values))
            results["XGBoost"] = compute_metrics(te["ret"].values, preds)

    if "HMM-XGBoost Hybrid" in models_to_run:
        with st.spinner("HMM-XGBoost Hybrid..."):
            feat_cols = ["lag_1", "lag_2", "lag_3", "lag_5", "vol_5", "vol_10", "sma_ratio", "vol_ratio"]
            tdx = td.copy()
            tdx["lag_1"] = tdx["ret"].shift(1)
            tdx["lag_2"] = tdx["ret"].shift(2)
            tdx["lag_3"] = tdx["ret"].shift(3)
            tdx["lag_5"] = tdx["ret"].shift(5)
            tdx["vol_5"] = tdx["log_ret"].rolling(5).std()
            tdx["vol_10"] = tdx["log_ret"].rolling(10).std()
            tdx["sma_ratio"] = tdx["Close"] / tdx["SMA_10"]
            tdx["vol_ratio"] = tdx["Volume"] / tdx["Volume_MA_10"]
            tdx = tdx.dropna()
            tr, te = train_test_split_ticker(tdx, train_pct)
            hmm_m = hmm.GaussianHMM(n_components=3, covariance_type="diag", random_state=42, n_iter=1000)
            hmm_m.fit(tr[["ret"]].values)
            tr_states = hmm_m.predict(tr[["ret"]].values)
            te_states = hmm_m.predict(te[["ret"]].values)
            n_hmm_states = 3
            state_ohe_tr = np.zeros((len(tr_states), n_hmm_states))
            state_ohe_tr[np.arange(len(tr_states)), tr_states] = 1
            state_ohe_te = np.zeros((len(te_states), n_hmm_states))
            state_ohe_te[np.arange(len(te_states)), te_states] = 1
            X_tr = np.concatenate([tr[feat_cols].values, state_ohe_tr], axis=1)
            X_te = np.concatenate([te[feat_cols].values, state_ohe_te], axis=1)
            scaler = StandardScaler()
            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.01, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(scaler.fit_transform(X_tr), tr["ret"].values)
            preds = model.predict(scaler.transform(X_te))
            results["HMM-XGBoost Hybrid"] = compute_metrics(te["ret"].values, preds)

    if "Prophet" in models_to_run:
        with st.spinner("Prophet..."):
            try:
                p_train = train[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
                p_model = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True, changepoint_prior_scale=0.05)
                p_model.fit(p_train)
                future = test[["Date"]].rename(columns={"Date": "ds"})
                forecast = p_model.predict(future)
                ret_pred = (forecast["yhat"].values - test["Close"].values) / test["Close"].values
                results["Prophet"] = compute_metrics(y_test, ret_pred)
            except Exception as e:
                results["Prophet"] = {"MAE": None, "RMSE": None, "Directional Accuracy (%)": None, "Correlation": None, "error": str(e)}

    if "SMA" in models_to_run:
        with st.spinner("SMA..."):
            all_close = np.concatenate([train["Close"].values, test["Close"].values])
            sma = pd.Series(all_close).rolling(10).mean().values[len(train):]
            ret_pred = (sma - test["Close"].values) / test["Close"].values
            mask = ~(np.isnan(ret_pred) | np.isnan(y_test))
            results["SMA (10)"] = compute_metrics(y_test[mask], ret_pred[mask]) if mask.sum() > 0 else {"error": "No valid predictions"}

    if "ETS" in models_to_run:
        with st.spinner("ETS..."):
            all_close = np.concatenate([train["Close"].values, test["Close"].values])
            ets = pd.Series(all_close).ewm(span=10, adjust=False).mean().values[len(train):]
            ret_pred_ets = (ets - test["Close"].values) / test["Close"].values
            mask = ~np.isnan(ret_pred_ets)
            results["ETS (EMA)"] = compute_metrics(y_test[mask], ret_pred_ets[mask]) if mask.sum() > 0 else {"error": "No valid predictions"}

    if "Drift" in models_to_run:
        with st.spinner("Drift..."):
            n_all = len(train) + len(test)
            seq = np.arange(1, n_all + 1)
            first_close = train["Close"].values[0]
            all_close = np.concatenate([train["Close"].values, test["Close"].values])
            drift_pred = all_close + (all_close - first_close) / (seq - 1)
            drift_pred[0] = all_close[0]
            drift_test = drift_pred[len(train):]
            ret_pred_d = (drift_test - test["Close"].values) / test["Close"].values
            results["Drift"] = compute_metrics(y_test, ret_pred_d)

    if results:
        metrics_df = pd.DataFrame(results).T
        metrics_df = metrics_df.drop(columns=["error"], errors="ignore")
        metrics_df = metrics_df.dropna(how="all")

        st.subheader("Comparison Results")
        st.dataframe(
            metrics_df.style.highlight_min(axis=0, color="#FFCDD2", subset=["MAE", "RMSE"])
            .highlight_max(axis=0, color="#C8E6C9", subset=["Directional Accuracy (%)", "Correlation"]),
            use_container_width=True,
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        valid_df = metrics_df.dropna()
        x = np.arange(len(valid_df))
        w = 0.3
        axes[0].bar(x - w / 2, valid_df["MAE"].values, w, color="#2196F3", edgecolor="#333")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(valid_df.index, rotation=45, ha="right", fontsize=9)
        axes[0].set_ylabel("MAE")
        axes[0].set_title("Mean Absolute Error", fontsize=12, fontweight="bold")

        axes[1].bar(x - w / 2, valid_df["Directional Accuracy (%)"].values, w, color="#4CAF50", edgecolor="#333")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(valid_df.index, rotation=45, ha="right", fontsize=9)
        axes[1].set_ylabel("Directional Accuracy (%)")
        axes[1].set_title("Directional Accuracy", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
