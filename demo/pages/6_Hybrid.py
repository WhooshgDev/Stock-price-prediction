import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, get_ticker_data, train_test_split_ticker, compute_metrics
from utils.plots import plot_actual_vs_predicted, plot_feature_importance

st.title("🧬 HMM-XGBoost Hybrid")

st.markdown(
    """
    This **hybrid** model first fits a Gaussian HMM to detect market regimes,
    then uses the decoded hidden states as additional features for an XGBoost regressor.

    The idea: the HMM captures regime shifts (bull/bear/volatile), and the XGBoost
    learns non-linear relationships between technical features **and** the current regime.

    Pipeline:
    1. Fit HMM on training returns → decode states
    2. Create features: lagged returns + rolling stats + **one-hot encoded state**
    3. Train XGBoost on features + state indicators
    4. Predict on test set
    """
)

df = load_raw_data()
tickers = get_ticker_list(df)

col1, col2, col3 = st.columns(3)
ticker = col1.selectbox("Ticker", tickers, key="hybrid_ticker")
n_states = col2.number_input("HMM States", min_value=2, max_value=5, value=3)
train_pct = col3.slider("Train %", 0.5, 0.95, 0.8, 0.05)

col4, col5, col6 = st.columns(3)
n_estimators = col4.number_input("XGBoost Trees", min_value=10, max_value=300, value=100, step=10)
max_depth = col5.number_input("Max Depth", min_value=1, max_value=10, value=4)
lr = col6.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)

if st.button("▶ Run Hybrid", type="primary"):
    with st.spinner("Fitting HMM → XGBoost pipeline..."):
        td = get_ticker_data(df, ticker).dropna()

        td["lag_1"] = td["ret"].shift(1)
        td["lag_2"] = td["ret"].shift(2)
        td["lag_3"] = td["ret"].shift(3)
        td["lag_5"] = td["ret"].shift(5)
        td["vol_5"] = td["log_ret"].rolling(5).std()
        td["vol_10"] = td["log_ret"].rolling(10).std()
        td["sma_ratio"] = td["Close"] / td["SMA_10"]
        td["vol_ratio"] = td["Volume"] / td["Volume_MA_10"]

        base_features = ["lag_1", "lag_2", "lag_3", "lag_5", "vol_5", "vol_10", "sma_ratio", "vol_ratio"]
        td = td.dropna()
        train, test = train_test_split_ticker(td, train_pct)

        hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", random_state=42, n_iter=1000)
        hmm_model.fit(train[["ret"]].values)
        train_states = hmm_model.predict(train[["ret"]].values)
        test_states = hmm_model.predict(test[["ret"]].values)

        X_train_base = train[base_features].values
        X_test_base = test[base_features].values
        state_train_arr = np.zeros((len(train_states), n_states))
        state_train_arr[np.arange(len(train_states)), train_states] = 1
        state_test_arr = np.zeros((len(test_states), n_states))
        state_test_arr[np.arange(len(test_states)), test_states] = 1

        X_train = np.concatenate([X_train_base, state_train_arr], axis=1)
        X_test = np.concatenate([X_test_base, state_test_arr], axis=1)
        y_train, y_test = train["ret"].values, test["ret"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(X_train_s, y_train)
        preds = xgb.predict(X_test_s)

        metrics = compute_metrics(y_test, preds)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", metrics["MAE"])
    col2.metric("RMSE", metrics["RMSE"])
    col3.metric("Directional Accuracy", f"{metrics['Directional Accuracy (%)']}%")
    col4.metric("Correlation", metrics["Correlation"])

    st.subheader("HMM State Distribution (Test Set)")
    state_counts = pd.Series(test_states).value_counts().sort_index()
    state_df = pd.DataFrame({
        "State": [f"State {i}" for i in state_counts.index],
        "Count": state_counts.values,
        "Proportion": [f"{v / len(test_states):.1%}" for v in state_counts.values],
    })
    st.dataframe(state_df, use_container_width=True)

    st.subheader("Actual vs Predicted Returns")
    fig = plot_actual_vs_predicted(
        test["Date"].values, y_test, preds,
        title="HMM-XGBoost Hybrid — Return Prediction", ticker=ticker
    )
    st.pyplot(fig)

    feature_names = base_features + [f"state_{i}" for i in range(n_states)]
    st.subheader("Feature Importance (with HMM states)")
    fig2 = plot_feature_importance(xgb.feature_importances_, feature_names)
    st.pyplot(fig2)
