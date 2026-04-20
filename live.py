import argparse
from pathlib import Path

from live_broker import OrderRequest, create_broker_adapter
from signal_runtime import generate_signal_payload, save_signal_payload
from tcndqn_cli_utils import ensure_dir, save_json


def build_parser():
    parser = argparse.ArgumentParser(
        description="Live trading skeleton that turns model signals into broker orders."
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
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--symbol", type=str, default="TWII")
    parser.add_argument("--units-per-position", type=float, default=1.0)
    parser.add_argument("--order-type", type=str, default="market")
    parser.add_argument("--time-in-force", type=str, default="day")
    parser.add_argument(
        "--broker", choices=["dry-run", "file", "custom"], default="dry-run"
    )
    parser.add_argument("--custom-adapter", type=str, default=None)
    parser.add_argument("--current-quantity", type=float, default=0.0)
    parser.add_argument("--broker-state-path", type=str, default=None)
    parser.add_argument("--order-outbox", type=str, default="broker_outbox")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--live-dir", type=str, default=None)
    return parser


def quantity_to_side(delta_quantity):
    if delta_quantity > 0:
        return "buy"
    if delta_quantity < 0:
        return "sell"
    return "hold"


def main():
    args = build_parser().parse_args()

    model_dir = Path(args.model_dir)
    live_dir = (
        Path(args.live_dir) if args.live_dir is not None else ensure_dir(model_dir / "live")
    )
    live_dir = ensure_dir(live_dir)

    signal_payload = generate_signal_payload(
        model_dir=model_dir,
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
    save_signal_payload(signal_payload)

    adapter = create_broker_adapter(args)
    adapter.connect()
    try:
        broker_position = adapter.get_position(args.symbol)
        current_quantity = float(broker_position.quantity)
        target_quantity = (
            float(signal_payload["target_position"]) * float(args.units_per_position)
        )
        delta_quantity = target_quantity - current_quantity
        side = quantity_to_side(delta_quantity)

        order_request = OrderRequest(
            symbol=args.symbol,
            target_quantity=target_quantity,
            current_quantity=current_quantity,
            delta_quantity=delta_quantity,
            side=side,
            order_type=args.order_type,
            time_in_force=args.time_in_force,
            signal_payload=signal_payload,
        )

        if delta_quantity == 0:
            result_payload = {
                "status": "no_action",
                "message": "Broker position already matches the model target.",
            }
        elif not args.execute:
            result_payload = {
                "status": "preview_only",
                "message": "Preview mode. Re-run with --execute to submit the order.",
            }
        else:
            order_result = adapter.submit_target_order(order_request)
            result_payload = {
                "status": order_result.status,
                "order_id": order_result.order_id,
                "message": order_result.message,
                "payload": order_result.payload,
            }

        log_payload = {
            "model_dir": str(model_dir),
            "live_dir": str(live_dir),
            "broker": args.broker,
            "symbol": args.symbol,
            "execute": bool(args.execute),
            "current_quantity": current_quantity,
            "target_quantity": target_quantity,
            "delta_quantity": delta_quantity,
            "order_type": args.order_type,
            "time_in_force": args.time_in_force,
            "signal": signal_payload,
            "broker_position_metadata": broker_position.metadata,
            "result": result_payload,
        }
        log_path = live_dir / f"{signal_payload['signal_date']}_live.json"
        save_json(log_path, log_payload)
    finally:
        adapter.disconnect()

    print(f"Signal date: {signal_payload['signal_date']}")
    print(
        f"Broker quantity: {current_quantity:.4f} -> target {target_quantity:.4f} "
        f"(delta {delta_quantity:.4f})"
    )
    print(f"Status: {result_payload['status']}")
    if result_payload.get("message"):
        print(result_payload["message"])
    print(f"Live log saved to: {log_path}")


if __name__ == "__main__":
    main()
