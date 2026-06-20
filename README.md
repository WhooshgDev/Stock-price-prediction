# Stock Return Prediction with Temporal Fusion Transformer

Multi-ticker daily **return** forecasting using a **Temporal Fusion Transformer (TFT)** across 61 tickers spanning 7 countries and 23 industry sectors (2000–2025).

## Dataset

**World-Stock-Prices-Dataset.csv** — 310K rows of daily OHLCV data for 61 tickers.

**Target:** daily return `ret = (Close(t) - Close(t-1)) / Close(t-1)`

**Features (19):** Open, High, Low, Close, Volume, log returns (1d/5d/21d/63d), rolling volatility, volume MA ratio, excess returns, relative volume.

**Encoder:** 60 days → **Prediction:** 5 days

## Model

**Temporal Fusion Transformer — 1.8M parameters**

| Component | Value |
|-----------|-------|
| Hidden size | 128 |
| LSTM layers | 2 |
| Attention heads | 4 |
| Dropout | 0.30 |
| Weight decay | 1e-5 |
| Loss | QuantileLoss (p10–p70) |

## Results

### Validation Metrics

| Metric | Value |
|--------|-------|
| Directional Accuracy | 64.85% |
| MAE | 1.22% |
| RMSE | 1.96% |
| Precision | 65.64% |
| Recall | 100% |
| F1 Score | 79.26% |

### Per-Step Directional Accuracy

| Step | Accuracy |
|------|----------|
| 1 | 87.88% |
| 2 | 51.52% |
| 3 | 75.76% |
| 4 | 51.52% |
| 5 | 57.58% |

### Diagnostic Analysis

The model predicted **all 165 samples as positive** (positive bias). In the validation window, actual returns were positive 107/165 times (64.85%), so the 64.85% accuracy reflects the market's overall upward drift during this period rather than genuine directional prediction.

| Check | Finding |
|-------|---------|
| Predicted positives | 165/165 (100%) |
| True negatives | 0/56 (0%) |
| Pearson IC | −0.16 (p=0.03, anti-correlated) |
| Spearman IC | −0.12 (p=0.14, no correlation) |

The negative Information Coefficient confirms the model outputs a nearly-constant slightly-positive value rather than learning directional patterns. This happens because QuantileLoss on raw returns learns the average drift — the model minimizes loss by predicting the mean return, which is positive.

### Comparison with Previous Model

| Metric | Old (Price) | New (Return) |
|--------|-------------|--------------|
| Target | Close price | Daily return |
| Parameters | 1.8M | 1.8M |
| Directional accuracy | 50.37% | 64.85% |
| Signal | None (random) | None (long bias) |
| Training stability | Overfit by epoch 2 | Stable through epoch 5 |

## Lesson

Predicting raw returns with regression learns the **mean drift**, not direction. For directional prediction, the target should be de-meaned or framed as binary classification. The model architecture and training pipeline work correctly — the target formulation needs refinement.

## Next Steps

1. De-meaned target: predict `ret - rolling_mean_ret` to remove drift
2. Binary classification: predict up/down directly
3. Larger validation window: 5 days per ticker is insufficient
