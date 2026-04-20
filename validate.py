import argparse
from pathlib import Path

import torch

from TCNDQN_DQNAgent import DQNAgent
from TCNDQN_Environment import Environment
from tcndqn_cli_utils import (
    ACTION_LABELS,
    compute_turnover,
    ensure_dir,
    list_checkpoints,
    load_json,
    load_market_data,
    parse_layer_channels,
    plot_checkpoint_rewards,
    plot_equity_curve,
    resolve_device,
    save_json,
    summarize_daily_rewards,
    write_summary_csv,
)


SELECTION_METRICS = [
    "sharpe",
    "total_reward",
    "cagr",
    "mean_daily_reward",
    "max_drawdown",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate checkpoints and auto-select the best model."
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--validation-dir", type=str, default=None)
    parser.add_argument("--validation-start-date", type=int, default=None)
    parser.add_argument("--validation-end-date", type=int, default=None)
    parser.add_argument("--data-start-date", type=int, default=None)
    parser.add_argument("--data-end-date", type=int, default=None)
    parser.add_argument("--field-names", type=str, default=None)
    parser.add_argument("--tcn-window", type=int, default=None)
    parser.add_argument("--layer-channels", type=str, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--const", type=float, default=None)
    parser.add_argument("--fee-rate", type=float, default=None)
    parser.add_argument("--slippage-rate", type=float, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=SELECTION_METRICS,
        default="sharpe",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    return parser


def resolve_arg(value, config, key, default=None):
    if value is not None:
        return value
    if config is not None and key in config:
        return config[key]
    return default


def run_checkpoint_validation(agent, env, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()

    env.reset()
    start_step = env.start_step
    end_step = env.end_step

    daily_rewards = []
    actions = []
    positions = []

    for step in range(start_step, end_step):
        done = step == end_step - 1
        price = env.get_input_data(step)
        costs = env.get_assets_state(step)
        action = agent.greedy_action(price, costs)
        env.action_execution(action, step)
        reward = float(env.get_reward(step, done, type="test").item())

        daily_rewards.append(reward)
        actions.append(action)
        positions.append(env.position)

    return daily_rewards, actions, positions


def build_sort_key(metric):
    def sort_key(row):
        return (
            row[metric],
            row["sharpe"],
            row["total_reward"],
            row["cagr"],
            row["max_drawdown"],
        )

    return sort_key


def main():
    args = build_parser().parse_args()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    config = load_json(config_path) if config_path.exists() else None

    validation_start_date = resolve_arg(
        args.validation_start_date, config, "train_end_date"
    )
    validation_end_date = resolve_arg(args.validation_end_date, config, "data_end_date")
    if validation_start_date is None or validation_end_date is None:
        raise ValueError(
            "Validation dates are required when config.json is unavailable."
        )

    data_start_date = resolve_arg(
        args.data_start_date, config, "data_start_date", validation_start_date
    )
    data_end_date = resolve_arg(
        args.data_end_date, config, "data_end_date", validation_end_date
    )
    field_names_value = resolve_arg(
        args.field_names, config, "field_names", ["close", "change"]
    )
    field_names = (
        [field.strip() for field in field_names_value.split(",") if field.strip()]
        if isinstance(field_names_value, str)
        else list(field_names_value)
    )
    layer_channels_value = resolve_arg(args.layer_channels, config, "layer_channels")
    if layer_channels_value is None:
        raise ValueError("layer_channels are required when config.json is unavailable.")
    layer_channels = parse_layer_channels(layer_channels_value)

    tcn_window = resolve_arg(args.tcn_window, config, "tcn_window")
    kernel_size = resolve_arg(args.kernel_size, config, "kernel_size")
    sliding_window = resolve_arg(args.sliding_window, config, "sliding_window", 60)
    const = resolve_arg(args.const, config, "const", 0.0)
    fee_rate = resolve_arg(args.fee_rate, config, "fee_rate", 0.0005)
    slippage_rate = resolve_arg(args.slippage_rate, config, "slippage_rate", 0.0005)
    reward_scale = resolve_arg(args.reward_scale, config, "reward_scale", 100.0)
    device = resolve_device(args.device or (config.get("device") if config else None))

    if tcn_window is None or kernel_size is None:
        raise ValueError("tcn_window and kernel_size are required.")

    validation_dir = (
        Path(args.validation_dir)
        if args.validation_dir is not None
        else ensure_dir(model_dir / "validation")
    )
    validation_dir = ensure_dir(validation_dir)

    dates, prices, selected_markets = load_market_data(
        data_start_date, data_end_date, field_names
    )
    env = Environment(
        prices,
        dates,
        start_date=validation_start_date,
        end_date=validation_end_date,
        tcn_window=tcn_window,
        sliding_window=sliding_window,
        const=const,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        reward_scale=reward_scale,
        device=device,
    )
    agent = DQNAgent(
        price_size=len(selected_markets),
        cost_size=2,
        layer_channels=layer_channels,
        kernel_size=kernel_size,
        action_size=len(ACTION_LABELS),
        device=device,
        epsilon=0.0,
    )

    checkpoints = list_checkpoints(model_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No model*.pt checkpoints found in {model_dir}")

    summary_rows = []
    checkpoint_results = {}

    for checkpoint_path in checkpoints:
        checkpoint_id = int(checkpoint_path.stem.replace("model", ""))
        daily_rewards, actions, positions = run_checkpoint_validation(
            agent, env, checkpoint_path, device
        )
        summary = summarize_daily_rewards(daily_rewards, reward_scale)
        summary_row = {
            "checkpoint": checkpoint_id,
            "total_reward": summary["total_reward"],
            "mean_daily_reward": summary["mean_daily_reward"],
            "std_daily_reward": summary["std_daily_reward"],
            "sharpe": summary["sharpe"],
            "cagr": summary["cagr"],
            "max_drawdown": summary["max_drawdown"],
            "turnover": compute_turnover(positions),
        }
        summary_rows.append(summary_row)
        checkpoint_results[checkpoint_id] = {
            "checkpoint": checkpoint_id,
            "checkpoint_path": str(checkpoint_path),
            "daily_rewards": daily_rewards,
            "actions": actions,
            "positions": positions,
            "summary": summary_row,
        }

    summary_rows.sort(key=lambda row: row["checkpoint"])
    write_summary_csv(validation_dir / "summary.csv", summary_rows)
    plot_checkpoint_rewards(summary_rows, validation_dir)

    ranked_rows = sorted(
        summary_rows,
        key=build_sort_key(args.selection_metric),
        reverse=True,
    )
    write_summary_csv(
        validation_dir / f"ranking_by_{args.selection_metric}.csv",
        ranked_rows,
    )

    best_checkpoint = ranked_rows[0]["checkpoint"]
    best_result = checkpoint_results[best_checkpoint]
    plot_equity_curve(
        best_result["daily_rewards"],
        reward_scale,
        validation_dir,
        filename=f"selected_model_{best_checkpoint}_equity_curve.png",
    )

    selection_payload = {
        "model_dir": str(model_dir),
        "validation_dir": str(validation_dir),
        "validation_start_date": validation_start_date,
        "validation_end_date": validation_end_date,
        "data_start_date": data_start_date,
        "data_end_date": data_end_date,
        "field_names": field_names,
        "selected_markets": selected_markets,
        "selection_metric": args.selection_metric,
        "selected_checkpoint": best_checkpoint,
        "selected_checkpoint_path": best_result["checkpoint_path"],
        "selected_summary": best_result["summary"],
        "top_ranked_checkpoints": ranked_rows[: max(args.top_k, 1)],
    }
    save_json(validation_dir / "selected_checkpoint.json", selection_payload)
    save_json(model_dir / "selected_checkpoint.json", selection_payload)
    save_json(
        validation_dir / "validation_config.json",
        {
            "model_dir": str(model_dir),
            "validation_dir": str(validation_dir),
            "validation_start_date": validation_start_date,
            "validation_end_date": validation_end_date,
            "data_start_date": data_start_date,
            "data_end_date": data_end_date,
            "field_names": field_names,
            "selected_markets": selected_markets,
            "tcn_window": tcn_window,
            "layer_channels": layer_channels,
            "kernel_size": kernel_size,
            "sliding_window": sliding_window,
            "const": const,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "reward_scale": reward_scale,
            "selection_metric": args.selection_metric,
            "device": device,
        },
    )

    print(
        "Validation finished. "
        f"Selected checkpoint: model{best_checkpoint}.pt "
        f"by {args.selection_metric}={best_result['summary'][args.selection_metric]:.6f}"
    )
    print(f"Outputs saved to: {validation_dir}")


if __name__ == "__main__":
    main()
