from datetime import date, timedelta

import pytest
import torch

from TCNDQN_Environment import Environment


def build_environment(days=20, markets=2, tcn_window=10):
    closes = (
        torch.arange(100, 100 + days, dtype=torch.float32)
        .unsqueeze(1)
        .repeat(1, markets)
    )
    changes = torch.arange(days, dtype=torch.float32).unsqueeze(1).repeat(1, markets)
    data = torch.stack([closes, changes], dim=1)
    dates = [
        int((date(2020, 1, 1) + timedelta(days=i)).strftime("%Y%m%d"))
        for i in range(days)
    ]
    return Environment(
        data,
        dates,
        start_date=dates[0],
        end_date=dates[-1],
        tcn_window=tcn_window,
        device="cpu",
    )


@pytest.mark.parametrize("step", [0, 9, 15])
def test_price_and_cost_states_share_the_same_window_length(step):
    env = build_environment()

    price_state = env.get_input_data(step)
    cost_state = env.get_assets_state(step)

    assert price_state.shape == (1, 2, env.tcn_window)
    assert cost_state.shape == (1, 2, env.tcn_window)
    assert price_state.shape[-1] == cost_state.shape[-1]


def test_price_state_includes_the_current_timestep():
    env = build_environment()
    step = 15

    price_state = env.get_input_data(step)
    start = step - env.tcn_window + 1

    torch.testing.assert_close(price_state[0, :, 0], env.changes[0, :, start])
    torch.testing.assert_close(price_state[0, :, -1], env.changes[0, :, step])


def test_cost_state_is_aligned_to_the_same_step_after_trade_execution():
    env = build_environment()
    step = 9

    env.action_execution(2, step)
    cost_state = env.get_assets_state(step)

    start = max(0, step - env.tcn_window + 1)
    mean_price = torch.mean(env.prices[0, 0, start : step + 1])
    expected_avg_cost = env.avgcost / mean_price

    torch.testing.assert_close(cost_state[0, 0, -1], expected_avg_cost)
    torch.testing.assert_close(cost_state[0, 1, -1], env.unreal_PNL)
