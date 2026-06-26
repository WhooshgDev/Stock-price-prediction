import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.data import load_raw_data, get_ticker_list, compute_metrics

st.title("🧠 TFT — Temporal Fusion Transformer")

st.markdown(
    """
    The **Temporal Fusion Transformer** is a deep learning architecture for multi-horizon time series forecasting.
    It uses self-attention to identify long-range dependencies and handles multiple time series jointly.

    - **Architecture:** LSTM encoder-decoder + multi-head self-attention + gated skip connections
    - **Target:** Daily return (pct change)
    - **Context:** 60-day encoder window → **5-step ahead return predictions**
    - **Trained on:** All available tickers jointly
    """
)

df = load_raw_data()
tickers = get_ticker_list(df)
ticker = st.selectbox("Select Ticker", tickers, key="tft_ticker")

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tft", "checkpoints")

if st.button("▶ Run TFT Prediction", type="primary"):
    status = st.status("Running TFT...", expanded=True)

    try:
        status.write("Loading checkpoint...")
        import glob
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.ckpt")))
        if not ckpts:
            st.error(f"No checkpoints found in {CKPT_DIR}")
            st.stop()
        best_ckpt = [c for c in ckpts if "epoch=05" in c] or ckpts
        CKPT_PATH = best_ckpt[-1]

        status.write("Loading data and model...")
        from demo.tft.model import DATA_PATH, ENCODER_LENGTH, PREDICTION_LENGTH, BATCH_SIZE, load_and_preprocess, create_datasets, _worker_init_fn
        from pytorch_forecasting import TemporalFusionTransformer

        data_df = load_and_preprocess(DATA_PATH)
        training, validation = create_datasets(data_df)
        model = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH, map_location="cpu")

        status.write(f"Running predictions for {ticker}...")
        val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
        raw_preds = model.predict(val_dl, mode="raw", return_x=True)

        output = raw_preds[0]
        x = raw_preds[1]

        y_actual = x["decoder_target"].detach().cpu().numpy()
        y_pred_all = output.prediction.detach().cpu().numpy()
        y_pred = y_pred_all[:, :, 3]
        groups = x["groups"].detach().cpu().numpy()

        ticker_encoder = training._categorical_encoders["__group_id__Ticker"]
        ticker_idx = None
        for name, i in ticker_encoder.classes_.items():
            if name == ticker:
                ticker_idx = i
                break

        if ticker_idx is None:
            st.error(f"Ticker {ticker} not found in training data")
            st.stop()

        mask = groups[:, 0] == ticker_idx
        if mask.sum() == 0:
            st.warning(f"No validation samples for {ticker}.")
            st.stop()

        ticker_actual = y_actual[mask]
        ticker_pred = y_pred[mask]

        actual_vals = ticker_actual.flatten()
        pred_vals = ticker_pred.flatten()

        mets = compute_metrics(actual_vals, pred_vals)

        status.update(label="Done", state="complete")

        st.success(f"Return predictions for {ticker}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mets['MAE']:.6f}")
        col2.metric("RMSE", f"{mets['RMSE']:.6f}")
        col3.metric("Directional Accuracy", f"{mets['Directional Accuracy (%)']:.2f}%")
        col4.metric("Correlation", f"{mets['Correlation']:.4f}")

        n_samples = ticker_actual.shape[0]
        pred_table = pd.DataFrame({
            "Sample": np.repeat(np.arange(n_samples), PREDICTION_LENGTH),
            "Step Ahead": np.tile(np.arange(1, PREDICTION_LENGTH + 1), n_samples),
            "Predicted Return": pred_vals,
            "Actual Return": actual_vals,
        })
        pred_table["Direction"] = pred_table.apply(lambda r: "✓" if np.sign(r["Predicted Return"]) == np.sign(r["Actual Return"]) else "✗", axis=1)

        st.subheader("Predicted Returns (5-Step Horizon)")
        st.dataframe(pred_table, use_container_width=True)

        st.subheader("Prediction Plot")
        fig, ax = plt.subplots(figsize=(12, 5))
        x_idx = np.arange(len(actual_vals))
        ax.plot(x_idx, actual_vals, color="#2196F3", linewidth=1.5, label="Actual", alpha=0.8)
        ax.plot(x_idx, pred_vals, color="#FF5722", linewidth=1.5, label="Predicted", alpha=0.8, linestyle="--")
        ax.set_title(f"TFT — 5-Step Return Forecast — {ticker}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Return")
        ax.legend()
        ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        status.update(label="Failed", state="error")
        st.error(f"TFT prediction failed: {e}")
        import traceback
        st.code(traceback.format_exc())