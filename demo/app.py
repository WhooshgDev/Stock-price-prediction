import streamlit as st

st.set_page_config(
    page_title="Stock Price Prediction — Algorithm Demo Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview = st.Page("pages/1_Overview.py", title="Overview", icon="🏠")
all_models = st.Page("pages/12_All_Models.py", title="All Models Forecast", icon="📊")
arima = st.Page("pages/3_ARIMA.py", title="ARIMA", icon="📈")
prophet = st.Page("pages/9_Prophet.py", title="Prophet", icon="🔮")
xgboost = st.Page("pages/4_XGBoost.py", title="XGBoost", icon="🌲")
hmm = st.Page("pages/5_HMM.py", title="HMM", icon="🔮")
hybrid = st.Page("pages/6_Hybrid.py", title="HMM-XGBoost Hybrid", icon="🧬")
tabnet = st.Page("pages/8_TabNet.py", title="TabNet", icon="📋")
tft = st.Page("pages/7_TFT.py", title="TFT (Transformer)", icon="🧠")
stats = st.Page("pages/10_Statistics.py", title="Statistical Models", icon="📐")
comparison = st.Page("pages/11_Comparison.py", title="Comparison", icon="⚖️")

pg = st.navigation([
    overview,
    all_models,
    arima, prophet, xgboost, hmm, hybrid, tabnet, tft, stats,
    comparison,
])
pg.run()