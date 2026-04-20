import argparse
import csv
from pathlib import Path

from signal_runtime import generate_signal_payload, save_signal_payload
from tcndqn_cli_utils import ensure_dir, load_json, save_json


def build_parser():
    parser = argparse.ArgumentParser(
        description="Daily paper-trading runner with simulated positions and ledger."
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
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--units-per-position", type=float, default=1.0)
    parser.add_argument("--contract-multiplier", type=float, default=1.0)
    parser.add_argument("--paper-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


def default_state(initial_cash):
    return {
        "last_signal_date": None,
        "position": 0,
        "cash": float(initial_cash),
        "market_value": 0.0,
        "equity": float(initial_cash),
        "cumulative_transaction_cost": 0.0,
    }


def read_state(state_path, initial_cash):
    if state_path.exists():
        return load_json(state_path)
    return default_state(initial_cash)


def resolve_state_before_trade(state_path, runs_dir, signal_date, initial_cash, overwrite):
    current_state = read_state(state_path, initial_cash)
    if (
        overwrite
        and current_state["last_signal_date"] is not None
        and int(current_state["last_signal_date"]) == int(signal_date)
    ):
        previous_run_path = runs_dir / f"{int(signal_date)}_paper_trade.json"
        if previous_run_path.exists():
            previous_run = load_json(previous_run_path)
            return previous_run["state_before"]
        raise FileNotFoundError(
            f"Cannot overwrite signal date {signal_date} because {previous_run_path} is missing."
        )
    return current_state


def read_ledger_rows(ledger_path):
    if not ledger_path.exists():
        return []
    with ledger_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_ledger_rows(ledger_path, rows):
    if not rows:
        return
    with ledger_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_ledger_row(signal_payload, state_before, state_after, trade_summary):
    q_values = signal_payload["q_values"]
    return {
        "requested_signal_date": signal_payload["requested_signal_date"],
        "signal_date": signal_payload["signal_date"],
        "checkpoint": signal_payload["checkpoint"],
        "action": signal_payload["action"],
        "action_label": signal_payload["action_label"],
        "previous_position": state_before["position"],
        "target_position": signal_payload["target_position"],
        "delta_position": trade_summary["delta_position"],
        "close": trade_summary["close"],
        "trade_notional": trade_summary["trade_notional"],
        "transaction_cost": trade_summary["transaction_cost"],
        "cash_before": state_before["cash"],
        "cash_after": state_after["cash"],
        "market_value_after": state_after["market_value"],
        "equity_before": state_before["equity"],
        "equity_after": state_after["equity"],
        "daily_pnl": trade_summary["daily_pnl"],
        "cumulative_transaction_cost": state_after["cumulative_transaction_cost"],
        "q_sell": q_values["sell"],
        "q_hold": q_values["hold"],
        "q_buy": q_values["buy"],
    }


def apply_trade(
    state,
    signal_payload,
    units_per_position,
    contract_multiplier,
    fee_rate,
    slippage_rate,
):
    previous_position = int(state["position"])
    target_position = int(signal_payload["target_position"])
    close_price = float(signal_payload["current_close"])
    delta_position = target_position - previous_position

    trade_quantity = delta_position * float(units_per_position)
    notional_multiplier = float(contract_multiplier) * close_price
    trade_notional = abs(trade_quantity) * notional_multiplier
    transaction_cost = trade_notional * (float(fee_rate) + float(slippage_rate))
    execution_cashflow = -trade_quantity * notional_multiplier

    cash_after = float(state["cash"]) + execution_cashflow - transaction_cost
    market_value_after = (
        target_position * float(units_per_position) * notional_multiplier
    )
    equity_after = cash_after + market_value_after

    updated_state = {
        "last_signal_date": int(signal_payload["signal_date"]),
        "position": target_position,
        "cash": float(cash_after),
        "market_value": float(market_value_after),
        "equity": float(equity_after),
        "cumulative_transaction_cost": float(state["cumulative_transaction_cost"])
        + float(transaction_cost),
    }
    trade_summary = {
        "close": float(close_price),
        "delta_position": int(delta_position),
        "trade_notional": float(trade_notional),
        "transaction_cost": float(transaction_cost),
        "daily_pnl": float(equity_after - float(state["equity"])),
    }
    return updated_state, trade_summary


def main():
    args = build_parser().parse_args()

    model_dir = Path(args.model_dir)
    paper_dir = (
        Path(args.paper_dir)
        if args.paper_dir is not None
        else ensure_dir(model_dir / "paper_trading")
    )
    paper_dir = ensure_dir(paper_dir)
    runs_dir = ensure_dir(paper_dir / "runs")
    state_path = paper_dir / "state.json"
    ledger_path = paper_dir / "ledger.csv"

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

    signal_params = signal_payload["signal_params"]
    fee_rate = float(signal_params["fee_rate"])
    slippage_rate = float(signal_params["slippage_rate"])
    signal_date = int(signal_payload["signal_date"])

    state_before = resolve_state_before_trade(
        state_path, runs_dir, signal_date, args.initial_cash, args.overwrite
    )
    if (
        state_before["last_signal_date"] is not None
        and int(state_before["last_signal_date"]) == signal_date
        and not args.overwrite
    ):
        raise ValueError(
            f"paper_trading already processed signal date {signal_date}. "
            "Use --overwrite to replace the latest entry."
        )

    state_after, trade_summary = apply_trade(
        state_before,
        signal_payload,
        args.units_per_position,
        args.contract_multiplier,
        fee_rate,
        slippage_rate,
    )
    ledger_row = build_ledger_row(signal_payload, state_before, state_after, trade_summary)

    ledger_rows = read_ledger_rows(ledger_path)
    if (
        ledger_rows
        and int(ledger_rows[-1]["signal_date"]) == signal_date
        and args.overwrite
    ):
        ledger_rows[-1] = {key: str(value) for key, value in ledger_row.items()}
    else:
        ledger_rows.append({key: str(value) for key, value in ledger_row.items()})
    write_ledger_rows(ledger_path, ledger_rows)
    save_json(state_path, state_after)

    run_payload = {
        "paper_dir": str(paper_dir),
        "state_before": state_before,
        "state_after": state_after,
        "trade_summary": trade_summary,
        "signal": signal_payload,
        "units_per_position": float(args.units_per_position),
        "contract_multiplier": float(args.contract_multiplier),
    }
    run_path = runs_dir / f"{signal_date}_paper_trade.json"
    save_json(run_path, run_payload)

    print(f"Signal date: {signal_date}")
    print(
        f"Paper position: {state_before['position']} -> {state_after['position']}, "
        f"daily_pnl={trade_summary['daily_pnl']:.2f}, equity={state_after['equity']:.2f}"
    )
    print(f"Ledger updated: {ledger_path}")
    print(f"Run log saved to: {run_path}")


if __name__ == "__main__":
    main()
