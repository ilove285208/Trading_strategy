import argparse
from pathlib import Path

from TCNDQN_DQNAgent import DQNAgent
from TCNDQN_Environment import Environment
from tcndqn_cli_utils import (
    ACTION_LABELS,
    default_output_dir,
    ensure_dir,
    load_market_data,
    parse_layer_channels,
    plot_action_distribution,
    plot_training_curves,
    resolve_device,
    save_json,
    shift_date_int,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train a TCN-Double-Dueling-DQN agent.")
    parser.add_argument("--train-start-date", type=int, required=True)
    parser.add_argument("--train-end-date", type=int, required=True)
    parser.add_argument("--data-start-date", type=int, default=None)
    parser.add_argument("--data-end-date", type=int, default=None)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label", type=str, default="tcndqn")

    parser.add_argument("--field-names", type=str, default="close,change")
    parser.add_argument("--tcn-window", type=int, default=256)
    parser.add_argument("--layer-channels", type=str, default="12,12,12,12,12,12,12")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--sliding-window", type=int, default=60)
    parser.add_argument("--const", type=float, default=0.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--reward-scale", type=float, default=100.0)

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99997)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()

    field_names = [field.strip() for field in args.field_names.split(",") if field.strip()]
    layer_channels = parse_layer_channels(args.layer_channels)
    device = resolve_device(args.device)

    data_start_date = (
        args.data_start_date
        if args.data_start_date is not None
        else shift_date_int(args.train_start_date, -args.lookback_days)
    )
    data_end_date = (
        args.data_end_date if args.data_end_date is not None else args.train_end_date
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_output_dir("model", args.label, args.train_start_date, args.train_end_date)
    )
    output_dir = ensure_dir(output_dir)

    dates, prices, selected_markets = load_market_data(
        data_start_date, data_end_date, field_names
    )

    env = Environment(
        prices,
        dates,
        start_date=args.train_start_date,
        end_date=args.train_end_date,
        tcn_window=args.tcn_window,
        sliding_window=args.sliding_window,
        const=args.const,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        reward_scale=args.reward_scale,
        device=device,
    )

    agent = DQNAgent(
        price_size=len(selected_markets),
        cost_size=2,
        layer_channels=layer_channels,
        kernel_size=args.kernel_size,
        action_size=len(ACTION_LABELS),
        device=device,
        batch_size=args.batch_size,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        memory_size=args.memory_size,
        output_dir=output_dir,
    )

    loss_history, reward_history, action_distribution_history = agent.train(
        env, n_episodes=args.episodes
    )

    plot_training_curves(loss_history, reward_history, output_dir)
    plot_action_distribution(action_distribution_history, output_dir, ACTION_LABELS)

    config = {
        "train_start_date": args.train_start_date,
        "train_end_date": args.train_end_date,
        "data_start_date": data_start_date,
        "data_end_date": data_end_date,
        "lookback_days": args.lookback_days,
        "field_names": field_names,
        "selected_markets": selected_markets,
        "tcn_window": args.tcn_window,
        "layer_channels": layer_channels,
        "kernel_size": args.kernel_size,
        "sliding_window": args.sliding_window,
        "const": args.const,
        "fee_rate": args.fee_rate,
        "slippage_rate": args.slippage_rate,
        "reward_scale": args.reward_scale,
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "memory_size": args.memory_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "device": device,
        "output_dir": str(output_dir),
        "label": args.label,
    }
    save_json(output_dir / "config.json", config)
    save_json(
        output_dir / "training_summary.json",
        {
            "final_epsilon": agent.epsilon,
            "loss_history": loss_history,
            "reward_history": reward_history,
            "action_distribution_history": [
                history.tolist() for history in action_distribution_history
            ],
        },
    )

    print(f"Training finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
