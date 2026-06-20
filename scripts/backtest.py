import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import os

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = "outputs"

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "axes.titlecolor": "#fff",
    "text.color": "#ccc",
    "xtick.color": "#888",
    "ytick.color": "#888",
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": "#333",
    "legend.labelcolor": "#ccc",
    "grid.color": "#2a2a2a",
    "grid.alpha": 0.5,
})


def backtest_trades(pred_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for ticker in pred_df["Ticker"].unique():
        sub = pred_df[pred_df["Ticker"] == ticker].sort_values(["sample_idx", "step"])
        for sidx in sub["sample_idx"].unique():
            sample = sub[sub["sample_idx"] == sidx].sort_values("step")
            returns = sample["Actual"].values
            preds = sample["Predicted"].values
            for t in range(len(returns)):
                signal = 1 if preds[t] > 0 else -1
                trade_return = signal * returns[t]
                correct = int(np.sign(preds[t]) == np.sign(returns[t]))

                records.append({
                    "Ticker": ticker,
                    "sample_idx": sidx,
                    "step": t,
                    "signal": signal,
                    "predicted_return": float(preds[t]),
                    "actual_return": float(returns[t]),
                    "trade_return": float(trade_return),
                    "correct": correct,
                })

    trades = pd.DataFrame(records)
    trades["cumulative_return"] = trades.groupby("Ticker")["trade_return"].cumsum()
    return trades


def compute_metrics(trades: pd.DataFrame) -> dict:
    total_trades = len(trades)
    wins = trades["correct"].sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    avg_trade_return = trades["trade_return"].mean()
    total_return = trades["trade_return"].sum()
    std_return = trades["trade_return"].std()
    sharpe = (avg_trade_return / std_return * np.sqrt(252)) if std_return > 0 else 0

    cumulative = trades["trade_return"].values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = float(np.min(drawdown))

    metrics = {
        "Total_Trades": int(total_trades),
        "Win_Rate_Pct": round(win_rate, 2),
        "Avg_Trade_Return_Pct": round(avg_trade_return * 100, 4),
        "Total_Return_Pct": round(total_return * 100, 4),
        "Sharpe_Ratio": round(sharpe, 4),
        "Max_Drawdown_Pct": round(max_dd * 100, 4),
    }
    return metrics


def plot_equity_curve(trades: pd.DataFrame) -> None:
    equity = trades.groupby("Ticker")["trade_return"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(equity)))
    bars = ax.barh(equity.index, equity.values * 100, color=colors)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("Total Return (%)", fontsize=12)
    ax.set_title("Per-Ticker P&L (Return-Based Strategy)", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, equity.values):
        if val > 0:
            ax.text(val * 100 + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"+{val * 100:.2f}%", va="center", fontsize=8, color="#4caf50")
        else:
            ax.text(val * 100 - 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val * 100:.2f}%", va="center", ha="right", fontsize=8, color="#f44336")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/backtest_equity.png", dpi=150)
    plt.close()
    print(f"  {OUTPUT_DIR}/backtest_equity.png saved")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[1/2] Loading predictions...")
    pred_df = pd.read_csv(f"{OUTPUT_DIR}/predictions.csv")
    print(f"     {len(pred_df)} rows, {pred_df['Ticker'].nunique()} tickers")

    print("[2/2] Running backtest...")
    trades = backtest_trades(pred_df)
    metrics = compute_metrics(trades)

    print(f"\n{'='*50}")
    print("BACKTEST RESULTS")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
    print(f"{'='*50}\n")

    with open(f"{OUTPUT_DIR}/backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  {OUTPUT_DIR}/backtest_metrics.json saved")

    trades.to_csv(f"{OUTPUT_DIR}/backtest_trades.csv", index=False)
    print(f"  {OUTPUT_DIR}/backtest_trades.csv saved ({len(trades)} rows)")

    plot_equity_curve(trades)

    print("\nPer-ticker summary:")
    summary = trades.groupby("Ticker").agg(
        Trades=("trade_return", "count"),
        Win_Rate=("correct", lambda x: x.mean() * 100),
        Total_Return=("trade_return", "sum"),
    ).sort_values("Total_Return", ascending=False)
    summary["Total_Return"] = summary["Total_Return"] * 100
    summary = summary.round(2)
    summary.to_csv(f"{OUTPUT_DIR}/backtest_summary.csv")
    print(summary.to_string())
    print("\nDone!")


if __name__ == "__main__":
    main()
