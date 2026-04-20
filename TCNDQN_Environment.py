from datetime import datetime

import torch


class Environment:
    def __init__(
        self,
        data,
        dates,
        start_date,
        end_date,
        tcn_window=70,
        sliding_window=60,
        const=10,
        fee_rate=0.0005,
        slippage_rate=0.0005,
        reward_scale=100.0,
        close_position_on_done=True,
        device="cpu",
    ):
        self.device = device
        self.dates = dates
        self.tcn_window = tcn_window
        self.sliding_window = sliding_window
        self.const = const
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.reward_scale = reward_scale
        self.close_position_on_done = close_position_on_done

        self.changes = data[:, 1, :].to(device)
        self.changes = self.changes.unsqueeze(dim=0)
        self.changes = torch.transpose(self.changes, 1, 2)
        self.changes = self.zscore_normal(self.changes)

        self.prices = data[:, 0, :].to(device)
        self.prices = self.prices.unsqueeze(dim=0)
        self.prices = torch.transpose(self.prices, 1, 2)

        self.costs = torch.zeros(1, 2, self.tcn_window, device=device)
        self.start_step = self.get_step(start_date)
        self.end_step = self.get_step(end_date)
        self.reset()

    def reset(self):
        self.count = 0
        self.previous_count = 0
        self.position = 0
        self.avgcost = torch.zeros(1, device=self.device)
        self.unreal_PNL = torch.zeros(1, device=self.device)
        self.real_PNL = torch.zeros(1, device=self.device)
        self.last_trade_cost = 0.0
        self.costs.zero_()
        self._invalidate_cost_cache()

    def _invalidate_cost_cache(self):
        self._last_cost_signature = None
        self._last_cost_state = None

    def load_costs(self, path):
        self.costs = torch.load(
            f"{path}/final_cost.pt", map_location=self.device, weights_only=True
        ).to(
            self.device
        )
        self._invalidate_cost_cache()
        return self.costs

    def zscore_normal(self, data):
        mean = torch.mean(data, dim=-1, keepdim=True)
        std = torch.std(data, dim=-1, keepdim=True)
        std[std == 0] = 1
        return (data - mean) / std

    def get_step(self, startdate):
        target_date = datetime.strptime(str(startdate), "%Y%m%d")

        for i, date in enumerate(self.dates):
            current_date = datetime.strptime(str(int(date)), "%Y%m%d")
            if current_date >= target_date:
                return i
        return len(self.dates) - 1

    def get_date(self, step):
        return self.dates[step]

    def _validate_step(self, step):
        if step < 0 or step >= self.changes.shape[2]:
            raise ValueError(
                f"step {step} is out of range [0, {self.changes.shape[2] - 1}]"
            )

    def get_input_data(self, step):
        self._validate_step(step)
        start = max(0, step - self.tcn_window + 1)
        window = self.changes[:, :, start : step + 1]

        if window.shape[-1] < self.tcn_window:
            pad_width = self.tcn_window - window.shape[-1]
            padding = torch.zeros(
                window.shape[0], window.shape[1], pad_width, device=self.device
            )
            window = torch.cat([padding, window], dim=-1)

        return window

    def _trade_cost_rate(self, position_change):
        return abs(position_change) * (self.fee_rate + self.slippage_rate)

    def _position_to_signed_cost(self, price, position):
        if position == 0:
            return torch.zeros(1, device=self.device)

        cost = torch.tensor([price], dtype=torch.float32, device=self.device)
        return cost if position > 0 else -cost

    def action_execution(self, action, step):
        self._validate_step(step)
        if step >= self.prices.shape[2] - 1:
            raise IndexError(f"step = {step} has no next-day price for reward")

        # Action indices map to target inventory: 0/1/2 -> short/flat/long -> -1/0/+1.
        desired_position = int(action) - 1
        execution_price = float(self.prices[0, 0, step].item())
        previous_position = self.count
        position_change = desired_position - previous_position

        self.previous_count = previous_position
        self.position = desired_position
        self.real_PNL = torch.zeros(1, device=self.device)
        self.last_trade_cost = self._trade_cost_rate(position_change)

        if desired_position != previous_position:
            self.avgcost = self._position_to_signed_cost(execution_price, desired_position)

        self.count = desired_position
        if self.count == 0:
            self.unreal_PNL = torch.zeros(1, device=self.device)

        self._invalidate_cost_cache()

    def _daily_return(self, step):
        current_price = float(self.prices[0, 0, step].item())
        next_price = float(self.prices[0, 0, step + 1].item())
        if current_price == 0:
            return 0.0
        return (next_price - current_price) / current_price

    def get_reward(self, step, done=0, type="train"):
        self._validate_step(step)
        if step >= self.prices.shape[2] - 1:
            raise IndexError(f"step = {step} has no next-day price for reward")

        daily_return = self._daily_return(step)
        reward = self.count * daily_return - self.last_trade_cost
        self.last_trade_cost = 0.0

        if done and self.close_position_on_done and self.count != 0:
            close_out_cost = self._trade_cost_rate(self.count)
            reward -= close_out_cost
            self.count = 0
            self.position = 0
            self.avgcost = torch.zeros(1, device=self.device)
            self.unreal_PNL = torch.zeros(1, device=self.device)
            self._invalidate_cost_cache()

        reward_tensor = torch.tensor(
            [reward * self.reward_scale], dtype=torch.float32, device=self.device
        )
        return reward_tensor

    def _compute_unrealized_ratio(self, current_price):
        if self.count == 0:
            return torch.zeros(1, device=self.device)

        entry_price = torch.abs(self.avgcost)
        if float(entry_price.item()) == 0.0:
            return torch.zeros(1, device=self.device)

        if self.count > 0:
            return (current_price - entry_price) / entry_price
        return (entry_price - current_price) / entry_price

    def get_assets_state(self, step):
        self._validate_step(step)

        state_signature = (
            step,
            int(self.count),
            float(self.avgcost.item()),
        )
        if self._last_cost_signature == state_signature and self._last_cost_state is not None:
            return self._last_cost_state.clone()

        current_price = self.prices[0, 0, step].reshape(1)
        start = max(0, step - self.tcn_window + 1)
        mean_window = self.prices[0, 0, start : step + 1]
        mean_price = torch.mean(mean_window) if mean_window.numel() > 0 else current_price
        if float(torch.abs(mean_price).item()) == 0.0:
            mean_price = torch.ones(1, device=self.device)

        self.unreal_PNL = self._compute_unrealized_ratio(current_price)
        avgcost = self.avgcost.clone() if self.count != 0 else torch.zeros(1, device=self.device)

        self.costs[:, :, :-1] = self.costs[:, :, 1:]
        self.costs[:, 0, -1] = avgcost
        self.costs[:, 1, -1] = self.unreal_PNL

        costs = self.costs.clone()
        costs[:, 0, :] = costs[:, 0, :] / mean_price

        self._last_cost_signature = state_signature
        self._last_cost_state = costs.clone()
        return costs
