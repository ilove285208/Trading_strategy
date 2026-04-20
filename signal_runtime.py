from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path

import torch

from TCNDQN_DQNAgent import DQNAgent
from TCNDQN_Environment import Environment
from tcndqn_cli_utils import (
    ACTION_LABELS,
    ensure_dir,
    load_json,
    load_market_data,
    parse_layer_channels,
    resolve_device,
    save_json,
)


POSITION_LABELS = {
    -1: "short",
    0: "flat",
    1: "long",
}


def resolve_arg(value, config, key, default=None):
    if value is not None:
        return value
    if config is not None and key in config:
        return config[key]
    return default


def default_signal_date():
    return int(datetime.today().strftime("%Y%m%d"))


def resolve_selection_file(model_dir, explicit_path=None):
    model_dir = Path(model_dir)
    if explicit_path is not None:
        return Path(explicit_path)

    root_selection = model_dir / "selected_checkpoint.json"
    if root_selection.exists():
        return root_selection

    validation_selection = model_dir / "validation" / "selected_checkpoint.json"
    if validation_selection.exists():
        return validation_selection

    return None


def first_index_on_or_after(dates, target_date):
    date_ints = [int(date) for date in dates]
    idx = bisect_left(date_ints, int(target_date))
    if idx >= len(date_ints):
        raise ValueError(f"No trading date on or after {target_date}.")
    return idx


def last_index_on_or_before(dates, target_date):
    date_ints = [int(date) for date in dates]
    idx = bisect_right(date_ints, int(target_date)) - 1
    if idx < 0:
        raise ValueError(f"No trading date on or before {target_date}.")
    return idx


def load_checkpoint(agent, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()


def greedy_q_values(agent, price, costs):
    with torch.no_grad():
        q_values = agent.policy_net(price.to(agent.device), costs.to(agent.device))
    return q_values.squeeze(0).detach().cpu()


def replay_until_signal(agent, env, dates, replay_start_step, signal_step, history_length):
    replay_history = []

    for step in range(replay_start_step, signal_step):
        price = env.get_input_data(step)
        costs = env.get_assets_state(step)
        q_values = greedy_q_values(agent, price, costs)
        action = int(torch.argmax(q_values).item())
        env.action_execution(action, step)

        replay_history.append(
            {
                "date": int(dates[step]),
                "action": action,
                "action_label": ACTION_LABELS[action],
                "target_position": int(env.position),
                "target_position_label": POSITION_LABELS[int(env.position)],
                "q_values": {
                    ACTION_LABELS[idx]: float(q_values[idx].item())
                    for idx in range(len(ACTION_LABELS))
                },
            }
        )

    if history_length <= 0:
        return []
    return replay_history[-history_length:]


def load_signal_context(model_dir, selection_file=None):
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    config = load_json(config_path) if config_path.exists() else None

    resolved_selection_file = resolve_selection_file(model_dir, selection_file)
    selection = (
        load_json(resolved_selection_file)
        if resolved_selection_file is not None and resolved_selection_file.exists()
        else None
    )
    return model_dir, config, resolved_selection_file, selection


def generate_signal_payload(
    model_dir,
    selection_file=None,
    checkpoint=None,
    signal_date=None,
    replay_start_date=None,
    data_start_date=None,
    data_end_date=None,
    field_names=None,
    tcn_window=None,
    layer_channels=None,
    kernel_size=None,
    sliding_window=None,
    const=None,
    fee_rate=None,
    slippage_rate=None,
    reward_scale=None,
    history_length=5,
    device=None,
):
    model_dir, config, resolved_selection_file, selection = load_signal_context(
        model_dir, selection_file
    )

    checkpoint_id = checkpoint
    if checkpoint_id is None and selection is not None:
        checkpoint_id = selection.get("selected_checkpoint")
    if checkpoint_id is None:
        raise ValueError(
            "Checkpoint is required. Run validate.py first or pass --checkpoint explicitly."
        )

    checkpoint_path = model_dir / f"model{checkpoint_id}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    requested_signal_date = signal_date or default_signal_date()
    replay_start_date = (
        replay_start_date
        or (selection.get("validation_start_date") if selection else None)
        or (config.get("train_end_date") if config else None)
        or requested_signal_date
    )

    data_start_date = data_start_date or (
        config.get("data_start_date") if config else replay_start_date
    )
    data_start_date = min(int(data_start_date), int(replay_start_date))
    data_end_date = data_end_date or requested_signal_date

    field_names_value = resolve_arg(
        field_names, config, "field_names", ["close", "change"]
    )
    resolved_field_names = (
        [field.strip() for field in field_names_value.split(",") if field.strip()]
        if isinstance(field_names_value, str)
        else list(field_names_value)
    )

    layer_channels_value = resolve_arg(layer_channels, config, "layer_channels")
    if layer_channels_value is None:
        raise ValueError("layer_channels are required when config.json is unavailable.")
    resolved_layer_channels = parse_layer_channels(layer_channels_value)

    resolved_tcn_window = resolve_arg(tcn_window, config, "tcn_window")
    resolved_kernel_size = resolve_arg(kernel_size, config, "kernel_size")
    resolved_sliding_window = resolve_arg(sliding_window, config, "sliding_window", 60)
    resolved_const = resolve_arg(const, config, "const", 0.0)
    resolved_fee_rate = resolve_arg(fee_rate, config, "fee_rate", 0.0005)
    resolved_slippage_rate = resolve_arg(
        slippage_rate, config, "slippage_rate", 0.0005
    )
    resolved_reward_scale = resolve_arg(reward_scale, config, "reward_scale", 100.0)
    resolved_device = resolve_device(device or (config.get("device") if config else None))

    if resolved_tcn_window is None or resolved_kernel_size is None:
        raise ValueError("tcn_window and kernel_size are required.")

    dates, prices, selected_markets = load_market_data(
        data_start_date, data_end_date, resolved_field_names
    )
    signal_step = last_index_on_or_before(dates, requested_signal_date)
    replay_start_step = first_index_on_or_after(dates, replay_start_date)
    if replay_start_step > signal_step:
        raise ValueError(
            "replay-start-date is after the resolved signal date. "
            "Please choose an earlier replay start."
        )

    actual_signal_date = int(dates[signal_step])
    actual_replay_start_date = int(dates[replay_start_step])

    env = Environment(
        prices,
        dates,
        start_date=actual_replay_start_date,
        end_date=actual_signal_date,
        tcn_window=resolved_tcn_window,
        sliding_window=resolved_sliding_window,
        const=resolved_const,
        fee_rate=resolved_fee_rate,
        slippage_rate=resolved_slippage_rate,
        reward_scale=resolved_reward_scale,
        close_position_on_done=False,
        device=resolved_device,
    )
    agent = DQNAgent(
        price_size=len(selected_markets),
        cost_size=2,
        layer_channels=resolved_layer_channels,
        kernel_size=resolved_kernel_size,
        action_size=len(ACTION_LABELS),
        device=resolved_device,
        epsilon=0.0,
    )
    load_checkpoint(agent, checkpoint_path, resolved_device)

    env.reset()
    recent_history = replay_until_signal(
        agent,
        env,
        dates,
        replay_start_step,
        signal_step,
        history_length,
    )

    price = env.get_input_data(signal_step)
    costs = env.get_assets_state(signal_step)
    q_values = greedy_q_values(agent, price, costs)
    action = int(torch.argmax(q_values).item())
    target_position = action - 1
    current_price = float(env.prices[0, 0, signal_step].item())

    return {
        "model_dir": str(model_dir),
        "checkpoint": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "selection_file": (
            str(resolved_selection_file) if resolved_selection_file is not None else None
        ),
        "selection_metric": selection.get("selection_metric") if selection else None,
        "requested_signal_date": requested_signal_date,
        "signal_date": actual_signal_date,
        "replay_start_date": actual_replay_start_date,
        "current_position": int(env.count),
        "current_position_label": POSITION_LABELS[int(env.count)],
        "current_avg_cost": float(env.avgcost.item()),
        "current_unrealized_pnl_ratio": float(env.unreal_PNL.item()),
        "current_close": current_price,
        "action": action,
        "action_label": ACTION_LABELS[action],
        "target_position": target_position,
        "target_position_label": POSITION_LABELS[target_position],
        "next_trading_date": (
            int(dates[signal_step + 1]) if signal_step < len(dates) - 1 else None
        ),
        "q_values": {
            ACTION_LABELS[idx]: float(q_values[idx].item())
            for idx in range(len(ACTION_LABELS))
        },
        "recent_replay_actions": recent_history,
        "field_names": resolved_field_names,
        "selected_markets": selected_markets,
        "signal_params": {
            "data_start_date": int(data_start_date),
            "data_end_date": int(data_end_date),
            "tcn_window": int(resolved_tcn_window),
            "layer_channels": list(resolved_layer_channels),
            "kernel_size": int(resolved_kernel_size),
            "sliding_window": int(resolved_sliding_window),
            "const": float(resolved_const),
            "fee_rate": float(resolved_fee_rate),
            "slippage_rate": float(resolved_slippage_rate),
            "reward_scale": float(resolved_reward_scale),
            "device": resolved_device,
            "history_length": int(history_length),
        },
    }


def default_signal_output_path(model_dir, signal_date):
    model_dir = Path(model_dir)
    return ensure_dir(model_dir / "signals") / f"{int(signal_date)}_signal.json"


def save_signal_payload(payload, output_path=None):
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else default_signal_output_path(payload["model_dir"], payload["signal_date"])
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(resolved_output_path, payload)
    return resolved_output_path
