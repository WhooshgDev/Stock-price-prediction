import pandas as pd
import numpy as np
import torch
import warnings
import os
import json

from typing import Tuple
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE, MAPE
from pytorch_forecasting.data import NaNLabelEncoder
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTORCH_WARN_ONCE"] = "0"

DATA_PATH = "World-Stock-Prices-Dataset.csv"
MAX_EPOCHS = 30
BATCH_SIZE = 128
HIDDEN_SIZE = 128
LSTM_LAYERS = 2
DROPOUT = 0.15
ATTENTION_HEADS = 4
ENCODER_LENGTH = 120
PREDICTION_LENGTH = 30

REAL_FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
    "ret", "log_ret",
    "ret_5d", "ret_21d", "ret_63d",
    "vol_5d", "vol_21d", "vol_63d",
    "volume_change", "volume_ma_ratio",
    "excess_ret", "relative_volume", "excess_sector_ret",
]


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
    for c in REAL_FEATURES:
        df[c] = df[c].fillna(0.0)

    return df.reset_index(drop=True)


def create_datasets(df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    training_cutoff = df["time_idx"].max() - PREDICTION_LENGTH

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="Close",
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
    )


def evaluate_model(
    model: TemporalFusionTransformer,
    validation: TimeSeriesDataSet,
    df: pd.DataFrame,
) -> pd.DataFrame:
    val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    raw_preds = model.predict(val_dl, mode="raw", return_x=True, n_jobs=1)

    y_true = raw_preds[1]["decoder_target"].detach().cpu().numpy()
    y_pred = raw_preds[0]["prediction"].detach().cpu().numpy()
    y_pred_median = y_pred[:, :, 3]

    mae = float(np.mean(np.abs(y_true - y_pred_median)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred_median) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred_median) / (y_true + 1e-8))) * 100)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    print(f"\n{'='*50}")
    print("VALIDATION METRICS")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"{'='*50}\n")

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  → metrics.json saved")

    index = raw_preds[1]["decoder_index"].detach().cpu().numpy().ravel()
    groups = raw_preds[1]["decoder_groups"].detach().cpu().numpy()

    ticker_map = {v: k for k, v in validation.categorical_encoders["Ticker"].classes_.items()}

    rows = []
    for i in range(len(y_pred_median)):
        ticker = ticker_map.get(int(groups[i, 0]), "UNKNOWN")
        for t in range(y_pred_median.shape[1]):
            rows.append({
                "Ticker": ticker,
                "time_idx": int(index[i * y_pred_median.shape[1] + t]),
                "Actual": float(y_true[i, t]),
                "Predicted": float(y_pred_median[i, t]),
            })

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df.merge(df[["Ticker", "time_idx", "Date", "Industry_Tag", "Country"]],
                            on=["Ticker", "time_idx"], how="left")
    pred_df.to_csv("predictions.csv", index=False)
    print("  → predictions.csv saved")

    return pred_df


def main() -> Tuple[TemporalFusionTransformer, TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame]:
    set_seed()

    print("[1/4] Loading and preprocessing data...")
    df = load_and_preprocess(DATA_PATH)
    nvda = df[df["Ticker"] == "NVDA"]
    print(f"     Total rows: {len(df)}, Tickers: {df['Ticker'].nunique()}")
    print(f"     NVDA rows: {len(nvda)}, NVDA date range: {nvda['Date'].min():%Y-%m-%d} to {nvda['Date'].max():%Y-%m-%d}")
    print(f"     Features: {REAL_FEATURES}")

    print("[2/4] Creating TimeSeriesDataSet...")
    training, validation = create_datasets(df)

    train_dl = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    print("[3/4] Creating TFT model...")
    model = create_model(training)
    print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")

    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    lr_logger = LearningRateMonitor()
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="tft-{epoch:02d}-{val_loss:.4f}")

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, lr_logger, checkpoint],
        enable_progress_bar=True,
    )

    print("[4/4] Training...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print(f"     Best model: {checkpoint.best_model_path}")
    trainer.save_checkpoint("tft_final.ckpt")
    print("     → tft_final.ckpt saved")

    print("\n[5/5] Evaluating on validation set...")
    pred_df = evaluate_model(model, validation, df)

    print("     Done!")

    return model, training, validation, pred_df


if __name__ == "__main__":
    model, training, validation, pred_df = main()
