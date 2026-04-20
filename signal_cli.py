import argparse

from signal_runtime import generate_signal_payload, save_signal_payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate the latest flat/long/short signal from a selected checkpoint."
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--selection-file", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--signal-date", type=int, default=None)
    parser.add_argument("--replay-start-date", type=int, default=None)
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
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    signal_payload = generate_signal_payload(
        model_dir=args.model_dir,
        selection_file=args.selection_file,
        checkpoint=args.checkpoint,
        signal_date=args.signal_date,
        replay_start_date=args.replay_start_date,
        data_start_date=args.data_start_date,
        data_end_date=args.data_end_date,
        field_names=args.field_names,
        tcn_window=args.tcn_window,
        layer_channels=args.layer_channels,
        kernel_size=args.kernel_size,
        sliding_window=args.sliding_window,
        const=args.const,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        reward_scale=args.reward_scale,
        history_length=args.history_length,
        device=args.device,
    )
    output_path = save_signal_payload(signal_payload, args.output_path)

    q_values = signal_payload["q_values"]
    print(f"Signal date: {signal_payload['signal_date']}")
    print(f"Checkpoint: model{signal_payload['checkpoint']}.pt")
    print(
        f"Current position: {signal_payload['current_position_label']} "
        f"-> Target position: {signal_payload['target_position_label']}"
    )
    print(
        "Action / Q-values: "
        f"sell={q_values['sell']:.6f}, "
        f"hold={q_values['hold']:.6f}, "
        f"buy={q_values['buy']:.6f}"
    )
    print(f"Signal saved to: {output_path}")


if __name__ == "__main__":
    main()
