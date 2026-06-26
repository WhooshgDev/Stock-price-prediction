import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#000000",
    "axes.titlecolor": "#000000",
    "text.color": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#cccccc",
    "legend.labelcolor": "#000000",
    "grid.color": "#dddddd",
    "grid.alpha": 0.5,
})


def plot_actual_vs_predicted(
    dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray,
    title: str = "Actual vs Predicted", ticker: str = ""
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, y_true, color="#2196F3", linewidth=1.5, label="Actual", alpha=0.8)
    ax.plot(dates, y_pred, color="#FF5722", linewidth=1.5, label="Predicted", alpha=0.8, linestyle="--")
    ax.set_title(f"{title} — {ticker}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price / Return")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residuals") -> plt.Figure:
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.scatter(y_pred, residuals, alpha=0.5, color="#2196F3", s=20)
    ax1.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted")
    ax2.hist(residuals, bins=30, color="#2196F3", edgecolor="white", alpha=0.7)
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance: np.ndarray, feature_names: list, title: str = "Feature Importance"
) -> plt.Figure:
    idx = np.argsort(importance)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(idx)))
    ax.barh(range(len(idx)), importance[idx], color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_price_with_signals(
    dates: pd.Series, price: np.ndarray,
    buy_signals: np.ndarray = None, sell_signals: np.ndarray = None,
    title: str = "Price with Trading Signals"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, price, color="#333", linewidth=1.2, alpha=0.7)
    if buy_signals is not None:
        ax.scatter(dates[buy_signals], price[buy_signals],
                   color="green", marker="^", s=80, label="Buy", alpha=0.8)
    if sell_signals is not None:
        ax.scatter(dates[sell_signals], price[sell_signals],
                   color="red", marker="v", s=80, label="Sell", alpha=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_forecast_with_uncertainty(
    dates: pd.Series, historical: np.ndarray,
    forecast: np.ndarray, lower: np.ndarray = None, upper: np.ndarray = None,
    title: str = "Forecast with Uncertainty"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates[:len(historical)], historical, color="#2196F3", linewidth=1.5, label="Historical")
    f_start = len(historical)
    f_dates = dates[f_start:f_start + len(forecast)]
    ax.plot(f_dates, forecast, color="#FF5722", linewidth=1.5, label="Forecast", linestyle="--")
    if lower is not None and upper is not None:
        ax.fill_between(f_dates, lower, upper, color="#FF5722", alpha=0.15, label="95% CI")
    ax.axvline(x=dates[f_start - 1], color="gray", linestyle=":", alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
