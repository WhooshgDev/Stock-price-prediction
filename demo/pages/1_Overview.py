import os, sys
import streamlit as st
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, get_ticker_data

st.title("🏠 Overview")
st.markdown(
    """
    This interactive dashboard demonstrates **10 different algorithms** for stock price prediction,
    applied to the World Stock Prices Dataset (310k+ rows, 62 tickers, 2000–2025).

    ---
    ### Algorithms Available
    | # | Algorithm | Type | Page |
    |---|-----------|------|------|
    | 1 | **ARIMA** | Statistical / Time Series | 📈 ARIMA |
    | 2 | **XGBoost** | Gradient Boosting (Tree-based) | 🌲 XGBoost |
    | 3 | **HMM (Gaussian HMM)** | Unsupervised / Regime Detection | 🔮 HMM |
    | 4 | **HMM-XGBoost Hybrid** | Ensemble (States → Features) | 🧬 Hybrid |
    | 5 | **Temporal Fusion Transformer** | Deep Learning (Attention) | 🧠 TFT |
    | 6 | **TabNet** | Deep Learning (Transformer-style) | 📋 TabNet |
    | 7 | **Prophet** | Bayesian / Additive Forecasting | 🔮 Prophet |
    | 8–12 | **SMA / WMA / Median / ETS / Drift** | Classical Statistical | 📐 Statistical |

    ---
    ### Dataset
    - **Source:** `data/World-Stock-Prices-Dataset.csv`
    - **Rows:** 310,122
    - **Tickers:** 62 (AAPL, MSFT, GOOGL, AMZN, TSLA, ...)
    - **Date Range:** 2000-01-03 to 2025-07-03
    - **Target:** Daily **return** (price change %), or price level for classical models

    ---
    ### How to Use
    1. Navigate using the left sidebar
    2. Select a **ticker** and **date range** on each page
    3. Click **Run** to train/predict (or load cached results)
    4. Compare all models on the **Comparison** page
    """
)

with st.expander("Quick Dataset Preview"):
    df = load_raw_data()
    st.dataframe(df.head(10), use_container_width=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Tickers", df["Ticker"].nunique())
    col3.metric("Date Range", f"{df['Date'].min():%Y-%m-%d} – {df['Date'].max():%Y-%m-%d}")
    col4.metric("Industries", df["Industry_Tag"].nunique())
