import logging
import os

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
os.environ["PYTORCH_WARN_ONCE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
import json
import pandas as pd
import numpy as np
import torch

from typing import Tuple
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import NaNLabelEncoder
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
warnings.filterwarnings("ignore", category=FutureWarning, module="lightning")

torch.set_float32_matmul_precision("high")


def _worker_init_fn(worker_id: int) -> None:
    import logging
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)


OUTPUT_DIR = "outputs"
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "World-Stock-Prices-Dataset.csv")
MAX_EPOCHS = 30
BATCH_SIZE = 256
HIDDEN_SIZE = 128
LSTM_LAYERS = 2
DROPOUT = 0.30
ATTENTION_HEADS = 4
ENCODER_LENGTH = 60
PREDICTION_LENGTH = 5

REAL_FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
    "log_ret",
    "ret_5d", "ret_21d", "ret_63d",
    "vol_5d", "vol_21d", "vol_63d",
    "volume_change", "volume_ma_ratio",
    "excess_ret", "relative_volume", "excess_sector_ret",
]

TARGET = "ret"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["ret"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change())
    df["log_ret"] = df.groupby("Ticker")["Close"].transform(lambda x: np.log(x / x.shift(1)))
    for w in [5, 21, 63]:
        df[f"ret_{w}d"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change(w))
        df[f"vol_{w}d"] = df.groupby("Ticker")["log_ret"].transform(
            lambda x, ww=w: x.rolling(ww, min_periods=1).std()
        )
    df["volume_change"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.pct_change())
    df["volume_ma_ratio"] = (
        df.groupby("Ticker")["Volume"].transform(lambda x: x / x.rolling(21, min_periods=1).mean())
    )
    return df


def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    daily_med_ret = df.groupby("Date")["ret"].transform("median")
    df["excess_ret"] = df["ret"] - daily_med_ret
    daily_med_vol = df.groupby("Date")["Volume"].transform("median")
    df["relative_volume"] = df["Volume"] / daily_med_vol.replace(0, np.nan)
    sector_daily = df.groupby(["Date", "Industry_Tag"])["ret"].transform("median")
    df["excess_sector_ret"] = df["ret"] - sector_daily
    return df


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.drop(columns=["Capital Gains", "Brand_Name"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Ticker", "Date"])

    ticker_counts = df.groupby("Ticker").size()
    valid_tickers = ticker_counts[ticker_counts >= ENCODER_LENGTH + PREDICTION_LENGTH].index
    df = df[df["Ticker"].isin(valid_tickers)].copy()

    df = add_returns(df)
    df = add_market_context(df)

    df["time_idx"] = df.groupby("Ticker").cumcount().astype(int)
    df["month"] = df["Date"].dt.month.astype(str)
    df["day_of_week"] = df["Date"].dt.dayofweek.astype(str)
    df["year"] = df["Date"].dt.year.astype(str)
    df["day_of_month"] = df["Date"].dt.day.astype(str)

    for col in ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]:
        df[col] = df[col].astype(float)

    df["Industry_Tag"] = df["Industry_Tag"].astype(str)
    df["Country"] = df["Country"].astype(str)

    df = df.replace([np.inf, -np.inf], np.nan)
    for c in REAL_FEATURES + [TARGET]:
        df[c] = df[c].fillna(0.0)

    return df.reset_index(drop=True)


def create_datasets(df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    training_cutoff = df["time_idx"].max() - PREDICTION_LENGTH

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target=TARGET,
        group_ids=["Ticker"],
        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=PREDICTION_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=["Industry_Tag", "Country"],
        static_reals=[],
        time_varying_known_categoricals=["month", "day_of_week", "year", "day_of_month"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=REAL_FEATURES,
        target_normalizer=GroupNormalizer(groups=["Ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        categorical_encoders={
            "Industry_Tag": NaNLabelEncoder(add_nan=True),
            "Country": NaNLabelEncoder(add_nan=True),
            "month": NaNLabelEncoder(add_nan=True),
            "day_of_week": NaNLabelEncoder(add_nan=True),
            "year": NaNLabelEncoder(add_nan=True),
            "day_of_month": NaNLabelEncoder(add_nan=True),
        },
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff + 1)
    return training, validation


def create_model(training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        attention_head_size=ATTENTION_HEADS,
        hidden_continuous_size=HIDDEN_SIZE // 2,
        output_size=7,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
        weight_decay=1e-5,
        mask_bias=-1e4,
    )


def evaluate_model(
    model: TemporalFusionTransformer,
    validation: TimeSeriesDataSet,
) -> pd.DataFrame:
    val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=11, persistent_workers=True, worker_init_fn=_worker_init_fn)

    raw_preds = model.predict(val_dl, mode="raw", return_x=True, n_jobs=1)

    out = raw_preds[0]
    x = raw_preds[1]

    y_true = x["decoder_target"].detach().cpu().numpy()
    y_pred_all = out.prediction.detach().cpu().numpy()
    y_pred = y_pred_all[:, :, 3]
    groups = x["groups"].detach().cpu().numpy()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    actual_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    directional_acc = float(np.mean(actual_sign == pred_sign) * 100)

    hit_rate_at_mean = float(np.mean(
        (y_true * y_pred) > 0
    ) * 100)

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

    ticker_encoder = validation._categorical_encoders["__group_id__Ticker"]
    ticker_map = {i: name for name, i in ticker_encoder.classes_.items()}

    rows = []
    for i in range(len(y_pred)):
        ticker = ticker_map.get(int(groups[i, 0]), "UNKNOWN")
        for t in range(y_pred.shape[1]):
            rows.append({
                "Ticker": ticker,
                "sample_idx": i,
                "step": t,
                "Actual": float(y_true[i, t]),
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
        acc = np.mean(np.sign(a) == np.sign(p)) * 100
        ticker_metrics.append({
            "Ticker": ticker,
            "Directional_Accuracy": round(acc, 2),
        })

    tm_df = pd.DataFrame(ticker_metrics).sort_values("Directional_Accuracy", ascending=False)
    tm_df.to_csv(f"{OUTPUT_DIR}/ticker_metrics.csv", index=False)
    print(f"  {OUTPUT_DIR}/ticker_metrics.csv saved")

    print("\nTop 5 (by directional accuracy):")
    print(tm_df.head(5).to_string(index=False))
    print("\nBottom 5:")
    print(tm_df.tail(5).to_string(index=False))

    return pred_df


def main():
    set_seed()

    print("[1/4] Loading and preprocessing data...")
    df = load_and_preprocess(DATA_PATH)
    print(f"     Total rows: {len(df)}, Tickers: {df['Ticker'].nunique()}")
    print(f"     Target: {TARGET} (daily return)")
    print(f"     Features: {REAL_FEATURES}")

    print("[2/4] Creating TimeSeriesDataSet...")
    training, validation = create_datasets(df)

    train_dl = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=11, persistent_workers=True, worker_init_fn=_worker_init_fn)
    val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=11, persistent_workers=True, worker_init_fn=_worker_init_fn)

    print("[3/4] Creating TFT model...")
    model = create_model(training)
    print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")

    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    lr_logger = LearningRateMonitor()
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="tft-{epoch:02d}-{val_loss:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    last_ckpt = f"{OUTPUT_DIR}/tft_last.ckpt" if os.path.exists(f"{OUTPUT_DIR}/tft_last.ckpt") else None

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        precision="16-mixed",
        callbacks=[early_stop, lr_logger, checkpoint],
        enable_progress_bar=True,
    )

    if last_ckpt:
        print(f"     Resuming from {last_ckpt}")

    print("[4/4] Training...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=last_ckpt)

    print(f"     Best model: {checkpoint.best_model_path}")
    trainer.save_checkpoint(f"{OUTPUT_DIR}/tft_last.ckpt")
    print(f"     {OUTPUT_DIR}/tft_last.ckpt saved (for resume)")

    print("\n[5/5] Evaluating on validation set...")
    pred_df = evaluate_model(model, validation)

    print("     Done!")
    return model, training, validation, pred_df


if __name__ == "__main__":
    model, training, validation, pred_df = main()
