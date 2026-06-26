"""Utilities for forward (real) forecasting — train on all data, predict ahead."""
import numpy as np
import pandas as pd
import warnings


def make_features(series: np.ndarray, lags: list[int] = None, n_steps: int = 1) -> np.ndarray:
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    hist = list(series)
    out = []
    for step in range(n_steps):
        if step > 0:
            hist.append(out[-1])
        row = []
        for lag in lags:
            val = hist[-lag] if len(hist) >= lag else 0.0
            row.append(val)
        vol5 = np.std(hist[-5:]) if len(hist) >= 5 else 0.0
        vol10 = np.std(hist[-10:]) if len(hist) >= 10 else 0.0
        row.extend([vol5, vol10])
        out.append(np.array(row))
    return np.array(out)


def make_features_with_hmm(series: np.ndarray, hmm_model, lags: list[int] = None, n_steps: int = 1) -> np.ndarray:
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    base_feat = make_features(series, lags, n_steps)
    state_onehot = np.zeros((n_steps, 3))
    state_onehot[np.arange(n_steps), 1] = 1
    return np.concatenate([base_feat, state_onehot], axis=1)


def return_to_price(ret_pred: np.ndarray, last_close: float) -> np.ndarray:
    prices = [last_close]
    for r in ret_pred:
        prices.append(prices[-1] * (1 + r))
    return np.array(prices[1:])


def arima_forecast(train_ret: np.ndarray, steps: int, order=(1, 0, 0)):
    """ARIMA forward forecast."""
    from statsmodels.tsa.arima.model import ARIMA
    m = ARIMA(train_ret, order=order)
    fitted = m.fit()
    return np.asarray(fitted.forecast(steps=steps))


def prophet_forecast(df: pd.DataFrame, steps: int):
    """Prophet forward forecast on Close price, returns predicted returns."""
    from prophet import Prophet
    train_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True, changepoint_prior_scale=0.05)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=steps, include_history=False)
    forecast = m.predict(future)
    last_close = df["Close"].values[-1]
    pred_prices = forecast["yhat"].values
    ret = (pred_prices - last_close) / last_close
    return ret


def xgboost_forecast(series: np.ndarray, lags: list[int], steps: int, xgb_params: dict = None):
    """XGBoost recursive forward forecast on returns."""
    from xgboost import XGBRegressor
    from sklearn.preprocessing import StandardScaler
    if xgb_params is None:
        xgb_params = dict(n_estimators=100, max_depth=5, learning_rate=0.01, random_state=42, n_jobs=-1, verbosity=0)
    
    n = len(series)
    feat_cols = [f"lag_{l}" for l in lags] + ["vol_5", "vol_10"]
    feat_rows = []
    targets = []
    for i in range(max(lags), n):
        row = [series[i - lag] for lag in lags]
        row.append(np.std(series[i-5:i]) if i >= 5 else 0.0)
        row.append(np.std(series[i-10:i]) if i >= 10 else 0.0)
        feat_rows.append(row)
        targets.append(series[i])
    
    X = np.array(feat_rows)
    y = np.array(targets)
    scaler = StandardScaler()
    model = XGBRegressor(**xgb_params)
    model.fit(scaler.fit_transform(X), y)
    
    preds = []
    hist = list(series)
    for _ in range(steps):
        row = [hist[-lag] for lag in lags]
        row.append(np.std(hist[-5:]) if len(hist) >= 5 else 0.0)
        row.append(np.std(hist[-10:]) if len(hist) >= 10 else 0.0)
        p = model.predict(scaler.transform([np.array(row)]))[0]
        preds.append(p)
        hist.append(p)
    return np.array(preds)


def hmm_xgboost_forecast(series: np.ndarray, lags: list[int], steps: int, hmm_params: dict = None, xgb_params: dict = None):
    """HMM-XGBoost hybrid recursive forward forecast."""
    from hmmlearn import hmm
    from xgboost import XGBRegressor
    from sklearn.preprocessing import StandardScaler
    if hmm_params is None:
        hmm_params = dict(n_components=3, covariance_type="diag", random_state=42, n_iter=1000)
    if xgb_params is None:
        xgb_params = dict(n_estimators=100, max_depth=4, learning_rate=0.01, random_state=42, n_jobs=-1, verbosity=0)
    
    n = len(series)
    feat_rows = []
    targets = []
    for i in range(max(lags), n):
        row = [series[i - lag] for lag in lags]
        row.append(np.std(series[i-5:i]) if i >= 5 else 0.0)
        row.append(np.std(series[i-10:i]) if i >= 10 else 0.0)
        feat_rows.append(row)
        targets.append(series[i])
    X_base = np.array(feat_rows)
    y = np.array(targets)
    
    hmm_model = hmm.GaussianHMM(**hmm_params)
    ret_2d = series[max(lags):].reshape(-1, 1)
    hmm_model.fit(ret_2d)
    states = hmm_model.predict(ret_2d)
    sohe = np.zeros((len(states), 3))
    sohe[np.arange(len(states)), states] = 1
    X_full = np.concatenate([X_base, sohe], axis=1)
    
    scaler = StandardScaler()
    model = XGBRegressor(**xgb_params)
    model.fit(scaler.fit_transform(X_full), y)
    
    preds = []
    hist = list(series)
    for _ in range(steps):
        row = [hist[-lag] for lag in lags]
        row.append(np.std(hist[-5:]) if len(hist) >= 5 else 0.0)
        row.append(np.std(hist[-10:]) if len(hist) >= 10 else 0.0)
        hmm_state = hmm_model.predict([[hist[-1]]])[0]
        sohe_row = np.zeros(3)
        sohe_row[hmm_state] = 1
        feat_row = np.concatenate([np.array(row), sohe_row])
        p = model.predict(scaler.transform([feat_row]))[0]
        preds.append(p)
        hist.append(p)
    return np.array(preds)


def hmm_forecast(series: np.ndarray, steps: int, n_components: int = 3):
    """HMM forward forecast — predict next state's mean return."""
    from hmmlearn import hmm as hmm_learn
    model = hmm_learn.GaussianHMM(n_components=n_components, covariance_type="diag", random_state=42, n_iter=1000)
    model.fit(series.reshape(-1, 1))
    state_means = model.means_.flatten()
    last_state = model.predict(series.reshape(-1, 1))[-1]
    pred = state_means[last_state]
    return np.full(steps, pred)


def tabnet_forecast(td: pd.DataFrame, steps: int, max_epochs: int = 20):
    """TabNet forward forecast — train on all data, recursive predict."""
    from pytorch_tabnet.tab_model import TabNetRegressor
    from sklearn.preprocessing import StandardScaler
    import torch
    
    cont_features = ["Open", "High", "Low", "Close", "Volume", "SMA_10", "SMA_50"]
    available = [c for c in cont_features if c in td.columns]
    if "SMA_10" not in td.columns:
        td = td.copy()
        td["SMA_10"] = td["Close"].rolling(10, min_periods=1).mean()
        td["SMA_50"] = td["Close"].rolling(50, min_periods=1).mean()
        available = cont_features
    
    n = len(td)
    X_all = td[available].values
    ret = td["ret"].values
    
    mask = ~np.isnan(ret) & ~np.isnan(X_all).any(axis=1)
    X_clean = X_all[mask]
    y_clean = ret[mask]
    
    X_train = X_clean[:-1]
    y_train = y_clean[1:]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = TabNetRegressor(
        optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(step_size=50, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax", verbose=0, device_name="auto",
    )
    model.fit(
        X_train=X_scaled, y_train=y_train.reshape(-1, 1),
        eval_set=[(X_scaled, y_train.reshape(-1, 1))],
        max_epochs=max_epochs, patience=5, batch_size=256, virtual_batch_size=64,
    )
    
    preds = []
    last_feat = X_clean[-1:]
    for _ in range(steps):
        p = model.predict(scaler.transform(last_feat))[0, 0]
        preds.append(p)
    return np.array(preds)


def tft_forecast(ticker: str, steps: int = 5):
    """TFT forward forecast — load pre-trained checkpoint, predict last encoder window."""
    import os, sys, glob
    from pytorch_forecasting import TemporalFusionTransformer
    import numpy as np

    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    _demo = os.path.join(_root, "demo")
    ckpt_dir = os.path.join(_demo, "tft", "checkpoints")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpts:
        raise FileNotFoundError(f"No TFT checkpoints found in {ckpt_dir}")

    best_ckpt = [c for c in ckpts if "epoch=05" in c] or ckpts
    ckpt_path = best_ckpt[-1]

    from demo.tft.model import DATA_PATH, ENCODER_LENGTH, PREDICTION_LENGTH, BATCH_SIZE, load_and_preprocess, create_datasets

    data_df = load_and_preprocess(DATA_PATH)
    training, validation = create_datasets(data_df)
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu")

    val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    raw_preds = model.predict(val_dl, mode="raw", return_x=True)

    output = raw_preds[0]
    x = raw_preds[1]

    y_pred_all = output.prediction.detach().cpu().numpy()
    y_pred = y_pred_all[:, :, 3]
    groups = x["groups"].detach().cpu().numpy()

    ticker_encoder = validation._categorical_encoders["__group_id__Ticker"]
    ticker_idx = None
    for name, i in ticker_encoder.classes_.items():
        if name == ticker:
            ticker_idx = i
            break
    if ticker_idx is None:
        raise ValueError(f"Ticker {ticker} not found in TFT training data")

    mask = groups[:, 0] == ticker_idx
    if mask.sum() == 0:
        raise ValueError(f"No validation predictions for {ticker}")

    preds = y_pred[mask].flatten()
    n_out = min(steps, len(preds))
    return preds[:n_out]


def stat_forecast(close: np.ndarray, steps: int, method: str = "SMA", window: int = 10):
    """Statistical model forward forecast on price, returns predicted returns."""
    if method == "SMA":
        last_val = np.mean(close[-window:])
    elif method == "WMA":
        w = np.arange(1, window + 1)
        last_val = np.dot(close[-window:], w) / w.sum()
    elif method == "Median":
        last_val = np.median(close[-window:])
    elif method == "ETS":
        s = pd.Series(close)
        last_val = s.ewm(span=window, adjust=False).mean().values[-1]
    elif method == "Drift":
        n = len(close)
        slope = (close[-1] - close[0]) / (n - 1)
        last_val = close[-1]
    
    pred_prices = np.full(steps, last_val) if method != "Drift" else close[-1] + slope * np.arange(1, steps + 1)
    ret = (pred_prices - close[-1]) / close[-1]
    return ret


def build_forecast_table(model_name: str, pred_returns: np.ndarray, last_close: float, steps: int) -> pd.DataFrame:
    """Build a clean forecast results table."""
    pred_prices = return_to_price(pred_returns, last_close)
    rows = []
    for i in range(min(steps, len(pred_returns))):
        direction = "🟢 Up" if pred_returns[i] > 0 else ("🔴 Down" if pred_returns[i] < 0 else "⚪ Flat")
        rows.append({
            "Model": model_name,
            "Step Ahead": i + 1,
            "Predicted Return (%)": round(pred_returns[i] * 100, 4),
            "Predicted Price": round(pred_prices[i], 2),
            "Direction": direction,
        })
    return pd.DataFrame(rows)