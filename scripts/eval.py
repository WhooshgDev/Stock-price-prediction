import logging
import os

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
os.environ["PYTORCH_WARN_ONCE"] = "0"

import warnings
import json
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")

OUTPUT_DIR = "outputs"

from tft import (
    DATA_PATH, ENCODER_LENGTH, PREDICTION_LENGTH, BATCH_SIZE,
    load_and_preprocess, create_datasets, _worker_init_fn,
)


def load_best_checkpoint() -> str:
    import glob
    ckpts = sorted(glob.glob("lightning_logs/version_19/checkpoints/*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in lightning_logs/version_19/checkpoints/")
    best = [c for c in ckpts if "epoch=05" in c]
    if not best:
        best = ckpts
    print(f"     Using checkpoint: {best[-1]}")
    return best[-1]


def main():
    CKPT_PATH = load_best_checkpoint()

    print("[1/3] Loading and preprocessing data...")
    df = load_and_preprocess(DATA_PATH)
    print(f"     {len(df)} rows, {df['Ticker'].nunique()} tickers")

    print("[2/3] Recreating datasets and loading model...")
    training, validation = create_datasets(df)
    model = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH)
    print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("[3/3] Running predictions on validation set...")
    val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=11, persistent_workers=True, worker_init_fn=_worker_init_fn)
    raw_preds = model.predict(val_dl, mode="raw", return_x=True)

    output = raw_preds[0]
    x = raw_preds[1]

    y_actual = x["decoder_target"].detach().cpu().numpy()
    y_pred_all = output.prediction.detach().cpu().numpy()
    y_pred = y_pred_all[:, :, 3]
    groups = x["groups"].detach().cpu().numpy()

    mae = float(np.mean(np.abs(y_actual - y_pred)))
    rmse = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))
    actual_sign = np.sign(y_actual)
    pred_sign = np.sign(y_pred)
    directional_acc = float(np.mean(actual_sign == pred_sign) * 100)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": directional_acc,
    }
    print(f"\n{'='*50}")
    print("VALIDATION METRICS (returns)")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")
    print(f"{'='*50}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  {OUTPUT_DIR}/metrics.json saved")

    ticker_encoder = training._categorical_encoders["__group_id__Ticker"]
    ticker_map = {i: name for name, i in ticker_encoder.classes_.items()}

    rows = []
    for i in range(len(y_pred)):
        ticker = ticker_map.get(int(groups[i, 0]), "UNKNOWN")
        for t in range(y_pred.shape[1]):
            rows.append({
                "Ticker": ticker,
                "sample_idx": i,
                "step": t,
                "Actual": float(y_actual[i, t]),
                "Predicted": float(y_pred[i, t]),
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
    print(f"  {OUTPUT_DIR}/predictions.csv saved ({len(pred_df)} rows)")

    print("\nPer-ticker directional accuracy:")
    ticker_metrics = []
    for ticker in pred_df["Ticker"].unique():
        sub = pred_df[pred_df["Ticker"] == ticker]
        a, p = sub["Actual"].values, sub["Predicted"].values
        acc = float(np.mean(np.sign(a) == np.sign(p)) * 100)
        ticker_metrics.append({"Ticker": ticker, "Directional_Accuracy": round(acc, 2)})

    tm_df = pd.DataFrame(ticker_metrics).sort_values("Directional_Accuracy", ascending=False)
    tm_df.to_csv(f"{OUTPUT_DIR}/ticker_metrics.csv", index=False)
    print(f"  {OUTPUT_DIR}/ticker_metrics.csv saved")

    print("\nTop 5:")
    print(tm_df.head(5).to_string(index=False))
    print("\nBottom 5:")
    print(tm_df.tail(5).to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
