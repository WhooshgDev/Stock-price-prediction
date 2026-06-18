import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import json

from typing import List, Tuple
from matplotlib.axes import Axes

warnings.filterwarnings("ignore", category=FutureWarning)

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
            vals = sample["Actual_denorm"].values
            preds = sample["Predicted_denorm"].values
            for t in range(len(vals) - 1):
                current_price = vals[t]
                predicted_next = preds[t + 1]
                actual_next = vals[t + 1]

                pred_move = (predicted_next - current_price) / current_price
                actual_move = (actual_next - current_price) / current_price

                signal = 1 if pred_move > 0 else -1
                trade_return = signal * actual_move
                correct = int(np.sign(pred_move) == np.sign(actual_move))

                records.append({
                    "Ticker": ticker,
                    "sample_idx": sidx,
                    "entry_step": t,
                    "current_price": current_price,
                    "predicted_next": predicted_next,
                    "actual_next": actual_next,
                    "pred_move": float(pred_move),
                    "actual_move": float(actual_move),
                    "signal": signal,
                    "trade_return": float(trade_return),
                    "correct": correct,
                })
    return pd.DataFrame(records)


def compute_strategy_metrics(trades: pd.DataFrame) -> dict:
    total_trades = len(trades)
    wins = trades["correct"].sum()
    win_rate = wins / total_trades * 100

    cumulative = (1 + trades["trade_return"]).cumprod()
    total_return = float(cumulative.iloc[-1] - 1) if len(cumulative) > 0 else 0

    mean_return = trades["trade_return"].mean()
    std_return = trades["trade_return"].std()
    sharpe = float(np.sqrt(252) * mean_return / std_return) if std_return > 0 else 0

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min())

    avg_win = float(trades[trades["correct"] == 1]["trade_return"].mean()) if wins > 0 else 0
    avg_loss = float(trades[trades["correct"] == 0]["trade_return"].mean()) if wins < total_trades else 0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_return_pct": total_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "avg_win_pct": avg_win * 100,
        "avg_loss_pct": avg_loss * 100,
        "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
    }


def plot_equity_curve(trades: pd.DataFrame, metrics: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    cumulative = (1 + trades["trade_return"]).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    ax = axes[0]
    ax.plot(cumulative.index, cumulative.values, color="#42a5f5", linewidth=1)
    ax.fill_between(cumulative.index, 1, cumulative.values, alpha=0.1, color="#42a5f5")
    ax.axhline(y=1, color="#555", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Equity ($)")
    ax.set_title("Backtest Equity Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    text = (
        f"Total Return: {metrics['total_return_pct']:.2f}%  |  "
        f"Win Rate: {metrics['win_rate']:.1f}%  |  "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
        f"Max DD: {metrics['max_drawdown_pct']:.1f}%"
    )
    ax.text(0.5, 0.05, text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#ccc",
            bbox=dict(boxstyle="round", facecolor="#2a2a2a", edgecolor="#333", alpha=0.9))

    ax = axes[1]
    ax.fill_between(drawdown.index, 0, drawdown.values * 100, color="#ef5350", alpha=0.5)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("viz_backtest_equity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] viz_backtest_equity.png")


def plot_ticker_performance(trades: pd.DataFrame) -> None:
    ticker_stats = []
    for ticker in trades["Ticker"].unique():
        sub = trades[trades["Ticker"] == ticker]
        t_wins = sub["correct"].sum()
        t_total = len(sub)
        t_ret = (1 + sub["trade_return"]).prod() - 1
        ticker_stats.append({
            "Ticker": ticker,
            "Trades": t_total,
            "Win Rate": t_wins / t_total * 100,
            "Return": float(t_ret * 100),
        })

    ts = pd.DataFrame(ticker_stats).sort_values("Return", ascending=False)
    colors = ["#66bb6a" if r > 0 else "#ef5350" for r in ts["Return"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(ts)), ts["Return"].values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(ts)))
    ax.set_yticklabels(ts["Ticker"].values, fontsize=8)
    ax.set_xlabel("Strategy Return (%)")
    ax.set_title("Per-Ticker Backtest Return", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="#555", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    for bar, val, wr in zip(bars, ts["Return"].values, ts["Win Rate"].values):
        label = f"{val:+.1f}% (WR: {wr:.0f}%)"
        ax.text(bar.get_width() + 0.3 if val > 0 else bar.get_width() - 0.3,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=7, color="#ccc",
                ha="left" if val > 0 else "right")

    plt.tight_layout()
    plt.savefig("viz_backtest_ticker_pnl.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] viz_backtest_ticker_pnl.png")


def plot_directional_accuracy(trades: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    step_acc = trades.groupby("entry_step")["correct"].agg(["count", "sum"])
    step_acc["accuracy"] = step_acc["sum"] / step_acc["count"] * 100
    ax.plot(step_acc.index, step_acc["accuracy"].values, color="#42a5f5", linewidth=1.5, marker="o", markersize=4)
    ax.axhline(y=50, color="#555", linestyle="--", linewidth=0.5, label="50% (random)")
    ax.set_xlabel("Prediction Step (t+n days ahead)")
    ax.set_ylabel("Directional Accuracy (%)")
    ax.set_title("Directional Accuracy by Step Ahead", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 30, 5))

    ax = axes[1]
    overall = trades.groupby("Ticker")["correct"].mean() * 100
    colors = ["#66bb6a" if v > 50 else "#ef5350" for v in overall.values]
    ax.barh(range(len(overall)), overall.values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(overall)))
    ax.set_yticklabels(overall.index, fontsize=7)
    ax.set_xlabel("Directional Accuracy (%)")
    ax.set_title("Per-Ticker Directional Accuracy", fontsize=13, fontweight="bold")
    ax.axvline(x=50, color="#555", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("viz_backtest_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] viz_backtest_accuracy.png")


def main():
    print("Loading predictions...")
    pred_df = pd.read_csv("predictions.csv")
    print(f"  {len(pred_df)} rows, {pred_df['Ticker'].nunique()} tickers")

    print("\nRunning backtest simulation...")
    trades = backtest_trades(pred_df)
    print(f"  Total trades: {len(trades)}")

    metrics = compute_strategy_metrics(trades)
    print(f"\n{'='*55}")
    print("BACKTEST RESULTS")
    print(f"{'='*55}")
    print(f"  Total Trades:       {metrics['total_trades']}")
    print(f"  Win Rate:           {metrics['win_rate']:.2f}%")
    print(f"  Total Return:       {metrics['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Avg Win:            {metrics['avg_win_pct']:.2f}%")
    print(f"  Avg Loss:           {metrics['avg_loss_pct']:.2f}%")
    print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"{'='*55}")

    with open("backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n  backtest_metrics.json saved")

    trades.to_csv("backtest_trades.csv", index=False)
    print("  backtest_trades.csv saved")

    print("\nGenerating backtest visualizations...")
    plot_equity_curve(trades, metrics)
    plot_ticker_performance(trades)
    plot_directional_accuracy(trades)
    print("\nDone! 3 backtest visualizations saved.")


if __name__ == "__main__":
    main()
