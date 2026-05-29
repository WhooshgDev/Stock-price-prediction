import pandas as pd
import numpy as np
import torch
import warnings
import os
import json

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTORCH_WARN_ONCE"] = "0"

from tft import (
    DATA_PATH, ENCODER_LENGTH, PREDICTION_LENGTH, BATCH_SIZE,
    load_and_preprocess, create_datasets,
)

CKPT_PATH = "lightning_logs/version_2/checkpoints/tft-epoch=01-val_loss=3.9947.ckpt"


def main():
    print("[1/3] Loading and preprocessing data...")
    df = load_and_preprocess(DATA_PATH)
    print(f"     {len(df)} rows, {df['Ticker'].nunique()} tickers")

    print("[2/3] Recreating datasets and loading model...")
    training, validation = create_datasets(df)
    model = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH)
    print(f"     Loaded checkpoint: {CKPT_PATH}")
    print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("[3/3] Running predictions on validation set...")
    val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    raw_preds = model.predict(val_dl, mode="raw", return_x=True)

    output = raw_preds[0]
    x = raw_preds[1]

    y_actual = x["decoder_target"].detach().cpu().numpy()
    y_pred_all = output.prediction.detach().cpu().numpy()
    y_pred = y_pred_all[:, :, 3]
    groups = x["groups"].detach().cpu().numpy()
    target_scale = x["target_scale"].detach().cpu().numpy()

    mae = float(np.mean(np.abs(y_actual - y_pred)))
    rmse = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100)
    smape = float(np.mean(2 * np.abs(y_actual - y_pred) / (np.abs(y_actual) + np.abs(y_pred) + 1e-8)) * 100)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}
    print(f"\n{'='*50}")
    print("VALIDATION METRICS")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"{'='*50}\n")

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  metrics.json saved")

    ticker_encoder = training._categorical_encoders["__group_id__Ticker"]
    ticker_map = {i: name for name, i in ticker_encoder.classes_.items()}

    rows = []
    for i in range(len(y_pred)):
        ticker = ticker_map.get(int(groups[i, 0]), "UNKNOWN")
        mean = target_scale[i, 0]
        std = target_scale[i, 1]
        for t in range(y_pred.shape[1]):
            rows.append({
                "Ticker": ticker,
                "sample_idx": i,
                "step": t,
                "Actual": float(y_actual[i, t]),
                "Predicted": float(y_pred[i, t]),
                "Actual_denorm": float(y_actual[i, t] * std + mean),
                "Predicted_denorm": float(y_pred[i, t] * std + mean),
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv("predictions.csv", index=False)
    print(f"  predictions.csv saved ({len(pred_df)} rows)")

    print("\nPer-ticker metrics:")
    ticker_metrics = []
    for ticker in pred_df["Ticker"].unique():
        sub = pred_df[pred_df["Ticker"] == ticker]
        a, p = sub["Actual_denorm"].values, sub["Predicted_denorm"].values
        tm = {
            "Ticker": ticker,
            "MAE": float(np.mean(np.abs(a - p))),
            "RMSE": float(np.sqrt(np.mean((a - p) ** 2))),
            "MAPE": float(np.mean(np.abs((a - p) / (a + 1e-8))) * 100),
        }
        ticker_metrics.append(tm)

    tm_df = pd.DataFrame(ticker_metrics)
    tm_df = tm_df.sort_values("MAPE", ascending=True)
    tm_df.to_csv("ticker_metrics.csv", index=False)
    print(f"  ticker_metrics.csv saved")

    print("\nTop 10 best (by MAPE):")
    print(tm_df.head(10).to_string(index=False))
    print("\nBottom 10 worst (by MAPE):")
    print(tm_df.tail(10).to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
