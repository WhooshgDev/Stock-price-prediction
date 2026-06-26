import os, sys
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, get_ticker_data, train_test_split_ticker, compute_metrics
from utils.plots import plot_actual_vs_predicted, plot_feature_importance

st.title("📋 TabNet — Transformer-Style Tabular Network")

st.markdown(
    """
    **TabNet** uses a sequential attention mechanism to select which features to attend to at each step,
    mimicking the decision-making process of tree-based models while using deep learning.

    Features: Open, High, Low, Close, Volume, SMA_10, SMA_50, Rolling_Median_20.
    Targets: Direction (classification), Return (regression), Close (regression).

    *Note: Training TabNet from scratch is heavy. Below we demo a simplified training run on a single ticker.*
    """
)

df = load_raw_data()
tickers = get_ticker_list(df)

col1, col2 = st.columns(2)
ticker = col1.selectbox("Ticker", tickers, key="tabnet_ticker")
train_pct = col2.slider("Train %", 0.5, 0.95, 0.7, 0.05)

max_epochs = st.number_input("Max Epochs", min_value=5, max_value=100, value=20, step=5)

if st.button("▶ Run TabNet", type="primary"):
    with st.spinner("Training TabNet regressor (this may take a while)..."):
        td = get_ticker_data(df, ticker).dropna()
        train, test = train_test_split_ticker(td, train_pct)

        cont_features = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "SMA_50"]
        cat_features = []

        X_train = train[cont_features].values
        X_test = test[cont_features].values
        y_train = train["ret"].values.reshape(-1, 1)
        y_test = test["ret"].values.reshape(-1, 1)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = TabNetRegressor(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params=dict(step_size=50, gamma=0.9),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
            verbose=0,
            device_name="auto",
        )
        model.fit(
            X_train=X_train_s, y_train=y_train,
            eval_set=[(X_train_s, y_train)],
            max_epochs=max_epochs,
            patience=10,
            batch_size=256,
            virtual_batch_size=64,
        )
        preds = model.predict(X_test_s).flatten()

        metrics = compute_metrics(y_test.flatten(), preds)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", metrics["MAE"])
    col2.metric("RMSE", metrics["RMSE"])
    col3.metric("Directional Accuracy", f"{metrics['Directional Accuracy (%)']}%")
    col4.metric("Correlation", metrics["Correlation"])

    st.subheader("Actual vs Predicted Returns")
    fig = plot_actual_vs_predicted(
        test["Date"].values, y_test.flatten(), preds,
        title="TabNet — Return Prediction", ticker=ticker
    )
    st.pyplot(fig)
