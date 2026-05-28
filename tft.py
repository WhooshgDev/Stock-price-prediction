import pandas as pd
import numpy as np
import torch
import warnings

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import NaNLabelEncoder
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
warnings.filterwarnings("ignore")
import os
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


def add_returns(df):
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["ret"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change())
    df["log_ret"] = df.groupby("Ticker")["Close"].transform(lambda x: np.log(x / x.shift(1)))
    for w in [5, 21, 63]:
        df[f"ret_{w}d"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change(w))
        df[f"vol_{w}d"] = df.groupby("Ticker")["log_ret"].transform(lambda x: x.rolling(w, min_periods=w).std())
    df["volume_change"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.pct_change())
    df["volume_ma_ratio"] = (
        df.groupby("Ticker")["Volume"].transform(lambda x: x / x.rolling(21, min_periods=21).mean())
    )
    return df


def add_market_context(df):
    daily_med_ret = df.groupby("Date")["ret"].transform("median")
    df["excess_ret"] = df["ret"] - daily_med_ret
    daily_med_vol = df.groupby("Date")["Volume"].transform("median")
    df["relative_volume"] = df["Volume"] / daily_med_vol.replace(0, np.nan)
    sector_daily = df.groupby(["Date", "Industry_Tag"])["ret"].transform("median")
    df["excess_sector_ret"] = df["ret"] - sector_daily
    return df


def load_and_preprocess(path):
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
    real_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
                 "ret_5d", "ret_21d", "ret_63d", "vol_21d", "vol_63d",
                 "volume_change", "volume_ma_ratio", "excess_ret", "relative_volume", "excess_sector_ret"]
    for c in real_cols:
        df[c] = df[c].fillna(0.0)
    df = df.reset_index(drop=True)
    df["time_idx"] = df.groupby("Ticker").cumcount().astype(int)

    return df


def create_datasets(df):
    max_encoder = ENCODER_LENGTH
    training_cutoff = df["time_idx"].max() - PREDICTION_LENGTH

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="Close",
        group_ids=["Ticker"],
        min_encoder_length=max_encoder,
        max_encoder_length=max_encoder,
        min_prediction_length=PREDICTION_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=["Industry_Tag", "Country"],
        static_reals=[],
        time_varying_known_categoricals=["month", "day_of_week", "year", "day_of_month"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "Open", "High", "Low", "Close", "Volume",
            "Dividends", "Stock Splits",
            "ret_5d", "ret_21d", "ret_63d",
            "vol_21d", "vol_63d",
            "volume_change", "volume_ma_ratio",
            "excess_ret", "relative_volume", "excess_sector_ret",
        ],
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


def create_model(training):
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


def main():
    print("[1/4] Loading and preprocessing data...")
    df = load_and_preprocess(DATA_PATH)
    nvda = df[df["Ticker"] == "NVDA"]
    print(f"     Total rows: {len(df)}, Tickers: {df['Ticker'].nunique()}")
    print(f"     NVDA rows: {len(nvda)}, NVDA date range: {nvda['Date'].min():%Y-%m-%d} to {nvda['Date'].max():%Y-%m-%d}")
    print(f"     Features: {[c for c in df.columns if c not in ['Date','Ticker','Brand_Name']]}")

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
    print("     Done!")

    return model, training, validation


if __name__ == "__main__":
    model, training, validation = main()
