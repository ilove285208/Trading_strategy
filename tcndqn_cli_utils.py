import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from LoadStockData import LoadStockData


PRIMARY_MARKET = "TWII"
PRIMARY_FILENAME = "StockData/TWII.csv"
AUX_MARKET_NAMES = [
    "S&P_500",
    "Dow_Jones",
    "NASDAQ",
    "PHLX",
    "N225",
    "FTSE",
    "FCHI",
    "DAX",
    "000001.SS",
]
AUX_MARKET_FILES = [
    "StockData/S&P_500.csv",
    "StockData/Dow_Jones.csv",
    "StockData/NASDAQ.csv",
    "StockData/PHLX.csv",
    "StockData/N225.csv",
    "StockData/FTSE.csv",
    "StockData/FCHI.csv",
    "StockData/DAX.csv",
    "StockData/000001.SS.csv",
]
ACTION_LABELS = ["sell", "hold", "buy"]


def parse_layer_channels(value):
    if isinstance(value, (list, tuple)):
        return [int(channel) for channel in value]
    return [int(channel.strip()) for channel in str(value).split(",") if channel.strip()]


def shift_date_int(date_int, days):
    date_obj = datetime.strptime(str(date_int), "%Y%m%d")
    shifted = date_obj + timedelta(days=days)
    return int(shifted.strftime("%Y%m%d"))


def resolve_device(device_arg):
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path_like):
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir(root, label, start_date, end_date):
    return Path(root) / f"{label}_{start_date}_{end_date}"


def load_market_data(data_start_date, data_end_date, field_names):
    stock_data = LoadStockData(
        PRIMARY_MARKET,
        list(field_names),
        PRIMARY_FILENAME,
        (data_start_date, data_end_date),
    )
    stock_data.AddNewData(AUX_MARKET_NAMES, AUX_MARKET_FILES)

    selected_market_names = [PRIMARY_MARKET] + AUX_MARKET_NAMES
    dates, prices = stock_data.get_prices(
        (data_start_date, data_end_date),
        list(field_names),
        selected_market_names,
    )
    return dates, torch.FloatTensor(prices), selected_market_names


def plot_action_distribution(action_distribution_history, output_dir, action_labels=None):
    if not action_distribution_history:
        return None

    labels = action_labels or ACTION_LABELS
    history = np.array(action_distribution_history)
    x = np.arange(1, history.shape[0] + 1)

    plt.figure(figsize=(12, 6))
    for idx, label in enumerate(labels):
        plt.plot(x, history[:, idx], label=label)

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Action Distribution", fontsize=14)
    plt.title("Action Distribution Over Episodes", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()

    output_path = Path(output_dir) / "action_distribution.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_training_curves(loss_history, reward_history, output_dir):
    if not loss_history or not reward_history:
        return None

    episodes = range(1, len(loss_history) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(episodes, loss_history, "-b", label="Loss")
    ax1.set_xlabel("Episode", fontsize=16)
    ax1.set_ylabel("Loss", color="blue", fontsize=16)

    ax2 = ax1.twinx()
    ax2.plot(episodes, reward_history, "-r", label="Reward")
    ax2.set_ylabel("Reward", color="red", fontsize=16)

    plt.title("Loss and Reward per Episode", fontsize=16)
    output_path = Path(output_dir) / "training_curves.png"
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def save_json(path_like, payload):
    path = Path(path_like)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json(path_like):
    path = Path(path_like)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_checkpoints(model_dir):
    model_dir = Path(model_dir)
    checkpoints = []
    for path in model_dir.glob("model*.pt"):
        stem = path.stem.replace("model", "")
        if stem.isdigit():
            checkpoints.append((int(stem), path))
    return [path for _, path in sorted(checkpoints, key=lambda item: item[0])]


def compute_nav_series(daily_rewards, reward_scale):
    if len(daily_rewards) == 0:
        return np.array([], dtype=np.float64)

    daily_returns = np.array(daily_rewards, dtype=np.float64) / reward_scale
    return np.cumprod(1.0 + daily_returns)


def compute_max_drawdown(nav_series):
    if len(nav_series) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(nav_series)
    drawdowns = nav_series / running_peak - 1.0
    return float(drawdowns.min())


def summarize_daily_rewards(daily_rewards, reward_scale):
    if len(daily_rewards) == 0:
        return {
            "total_reward": 0.0,
            "mean_daily_reward": 0.0,
            "std_daily_reward": 0.0,
            "sharpe": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
        }

    rewards = np.array(daily_rewards, dtype=np.float64)
    daily_returns = rewards / reward_scale
    nav_series = compute_nav_series(daily_rewards, reward_scale)
    ending_nav = float(nav_series[-1]) if len(nav_series) > 0 else 1.0
    years = max(len(daily_returns) / 252.0, 1 / 252.0)

    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * math.sqrt(252))

    cagr = ending_nav ** (1.0 / years) - 1.0 if ending_nav > 0 else -1.0

    return {
        "total_reward": float(rewards.sum()),
        "mean_daily_reward": float(rewards.mean()),
        "std_daily_reward": float(rewards.std()),
        "sharpe": sharpe,
        "cagr": float(cagr),
        "max_drawdown": compute_max_drawdown(nav_series),
    }


def compute_turnover(positions):
    if len(positions) == 0:
        return 0.0
    previous = 0
    turnover = 0.0
    for position in positions:
        turnover += abs(position - previous)
        previous = position
    return float(turnover)


def write_summary_csv(path_like, rows):
    path = Path(path_like)
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_checkpoint_rewards(summary_rows, output_dir):
    if not summary_rows:
        return None

    checkpoints = [row["checkpoint"] for row in summary_rows]
    rewards = [row["total_reward"] for row in summary_rows]

    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, rewards, "-r")
    plt.xlabel("Checkpoint", fontsize=16)
    plt.ylabel("Total Reward", fontsize=16)
    plt.title("Backtest Reward per Checkpoint", fontsize=16)
    plt.grid(True)

    output_path = Path(output_dir) / "checkpoint_rewards.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_equity_curve(daily_rewards, reward_scale, output_dir, filename="equity_curve.png"):
    nav_series = compute_nav_series(daily_rewards, reward_scale)
    if len(nav_series) == 0:
        return None

    plt.figure(figsize=(10, 6))
    plt.plot(nav_series, color="black")
    plt.xlabel("Step", fontsize=16)
    plt.ylabel("NAV", fontsize=16)
    plt.title("Equity Curve", fontsize=16)
    plt.grid(True)

    output_path = Path(output_dir) / filename
    plt.savefig(output_path)
    plt.close()
    return output_path
