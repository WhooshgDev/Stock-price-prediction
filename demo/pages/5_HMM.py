import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, get_ticker_data, train_test_split_ticker

st.title("🔮 HMM — Gaussian Hidden Markov Model")

st.markdown(
    """
    A **Gaussian HMM** models the market as transitioning between discrete hidden states (regimes).
    Each state has its own Gaussian distribution of returns.

    - **State 0:** Bull market (positive mean, moderate volatility)
    - **State 1:** Bear market (negative mean, low/moderate volatility)  
    - **State 2:** High-volatility market (near-zero mean, very high volatility)

    The Viterbi algorithm decodes the most likely state sequence.
    """
)

df = load_raw_data()
tickers = get_ticker_list(df)

col1, col2, col3 = st.columns(3)
ticker = col1.selectbox("Ticker", tickers, key="hmm_ticker")
n_states = col2.number_input("Number of States", min_value=2, max_value=6, value=3)
covar_type = col3.selectbox("Covariance Type", ["full", "diag", "spherical", "tied"])

train_pct = st.slider("Train %", 0.5, 0.95, 0.8, 0.05)

if st.button("▶ Run HMM", type="primary"):
    with st.spinner("Fitting HMM..."):
        td = get_ticker_data(df, ticker).dropna()
        train, test = train_test_split_ticker(td, train_pct)

        X_train = train[["ret"]].values
        X_test = test[["ret"]].values

        model = hmm.GaussianHMM(
            n_components=n_states, covariance_type=covar_type,
            random_state=42, n_iter=1000, tol=1e-4
        )
        model.fit(X_train)

        train_states = model.predict(X_train)
        test_states = model.predict(X_test)
        log_likelihood = model.score(X_test)

        all_dates = np.concatenate([train["Date"].values, test["Date"].values])
        all_states = np.concatenate([train_states, test_states])
        all_returns = np.concatenate([train["ret"].values, test["ret"].values])
        all_prices = np.concatenate([train["Close"].values, test["Close"].values])

        state_means = model.means_.flatten()
        state_covs = np.sqrt(model.covars_).flatten() if model.covariance_type == "spherical" else np.array([np.sqrt(np.diag(c))[0] for c in model.covars_])
        trans_mat = model.transmat_

    col1, col2, col3 = st.columns(3)
    col1.metric("States", n_states)
    col2.metric("Log-Likelihood (Test)", f"{log_likelihood:.2f}")
    col3.metric("Test Obs.", len(X_test))

    st.subheader("State Parameters")
    param_df = pd.DataFrame({
        "State": [f"State {i}" for i in range(n_states)],
        "Mean Return (%)": [f"{m * 100:.4f}" for m in state_means],
        "Std Dev (%)": [f"{s * 100:.4f}" for s in state_covs],
        "Interpretation": [
            "Bull" if m > 0.0005 else ("Bear" if m < -0.0005 else "Volatile / Neutral")
            for m in state_means
        ],
    })
    st.dataframe(param_df, use_container_width=True)

    st.subheader("Transition Matrix")
    trans_df = pd.DataFrame(
        trans_mat,
        columns=[f"to State {i}" for i in range(n_states)],
        index=[f"from State {i}" for i in range(n_states)],
    )
    st.dataframe(trans_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Hidden State Assignments Over Time")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    colors = ["#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#2196F3", "#607D8B"]
    state_colors = [colors[s % len(colors)] for s in all_states]

    ax1.plot(all_dates, all_prices, color="#333", linewidth=1, alpha=0.7)
    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.set_title(f"HMM State Assignments — {ticker}", fontsize=13, fontweight="bold")
    for i in range(n_states):
        mask = all_states == i
        ax1.scatter(all_dates[mask], all_prices[mask], color=colors[i % len(colors)],
                    s=3, label=f"State {i}", alpha=0.5)
    ax1.legend(fontsize=8, loc="upper left")

    ax2.scatter(all_dates, all_returns, c=state_colors, s=2, alpha=0.6)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Return", fontsize=11)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
