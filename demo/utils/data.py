import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "World-Stock-Prices-Dataset.csv")


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.drop(columns=["Capital Gains", "Brand_Name"], errors="ignore")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


@st.cache_data
def get_ticker_list(df: Optional[pd.DataFrame] = None) -> list:
    if df is None:
        df = load_raw_data()
    return sorted(df["Ticker"].unique())


@st.cache_data
def get_ticker_data(_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    sub = _df[_df["Ticker"] == ticker].copy()
    sub["ret"] = sub["Close"].pct_change()
    sub["log_ret"] = np.log(sub["Close"] / sub["Close"].shift(1))
    sub["SMA_10"] = sub["Close"].rolling(10).mean()
    sub["SMA_50"] = sub["Close"].rolling(50).mean()
    sub["Volume_MA_10"] = sub["Volume"].rolling(10).mean()
    sub["vol_21"] = sub["log_ret"].rolling(21).std() * np.sqrt(252) * 100
    sub["vol_63"] = sub["log_ret"].rolling(63).std() * np.sqrt(252) * 100
    sub["high_low_pct"] = (sub["High"] - sub["Low"]) / sub["Close"]
    sub["close_open_pct"] = (sub["Close"] - sub["Open"]) / sub["Open"]
    return sub


def train_test_split_ticker(
    df: pd.DataFrame, train_pct: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * train_pct)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    y_true_s = np.sign(y_true)
    y_pred_s = np.sign(y_pred)
    directional = float(np.mean(y_true_s == y_pred_s) * 100)
    if len(y_true) > 1:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = 0.0
    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "Directional Accuracy (%)": round(directional, 2),
        "Correlation": round(corr, 4),
    }
