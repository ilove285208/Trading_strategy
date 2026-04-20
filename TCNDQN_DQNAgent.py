from collections import deque
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from TCNDQN import TCN_DQN


plt.rcParams["font.family"] = "Times New Roman"


class DQNAgent:
    def __init__(
        self,
        price_size,
        cost_size,
        layer_channels,
        kernel_size,
        action_size,
        device="cpu",
        batch_size=64,
        epsilon=1.0,
        epsilon_decay=0.99997,
        epsilon_min=0.01,
        learning_rate=0.007,
        gamma=0.95,
        memory_size=100000,
        output_dir=None,
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.min_memory_size = batch_size
        self.device = device
        self.output_dir = None
        self.set_output_dir(output_dir)

        self.policy_net = TCN_DQN(
            price_size, cost_size, layer_channels, kernel_size, action_size
        ).to(device)
        self.target_net = TCN_DQN(
            price_size, cost_size, layer_channels, kernel_size, action_size
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss = nn.SmoothL1Loss()
        self.q_stats = {
            "predict_q_mean": [],
            "predict_q_std": [],
            "target_q_mean": [],
            "target_q_std": [],
        }

    def set_output_dir(self, output_dir):
        if output_dir is None:
            self.output_dir = None
            return

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def _model_q_values(self, model, price, costs):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            q_values = model(price.to(self.device), costs.to(self.device))
        if was_training:
            model.train()
        return q_values

    def greedy_action(self, price, costs):
        q_values = self._model_q_values(self.policy_net, price, costs)
        return int(torch.argmax(q_values, dim=1).item())

    def act(self, price, costs):
        if random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        return self.greedy_action(price, costs)

    def remember(self, price, costs, action, reward, next_price, next_costs, done):
        self.memory.append(
            (
                price.detach().cpu(),
                costs.detach().cpu(),
                int(action),
                float(reward),
                next_price.detach().cpu(),
                next_costs.detach().cpu(),
                float(done),
            )
        )

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        prices = torch.stack([x[0] for x in batch]).squeeze(1).to(self.device)
        costs = torch.stack([x[1] for x in batch]).squeeze(1).to(self.device)
        actions = torch.tensor([x[2] for x in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([x[3] for x in batch], dtype=torch.float32, device=self.device)
        next_prices = torch.stack([x[4] for x in batch]).squeeze(1).to(self.device)
        next_costs = torch.stack([x[5] for x in batch]).squeeze(1).to(self.device)
        dones = torch.tensor([x[6] for x in batch], dtype=torch.float32, device=self.device)

        predict_q_values = (
            self.policy_net(prices, costs).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        policy_was_training = self.policy_net.training
        target_was_training = self.target_net.training
        self.policy_net.eval()
        self.target_net.eval()
        with torch.no_grad():
            next_actions = self.policy_net(next_prices, next_costs).argmax(dim=1)
            next_q_values = (
                self.target_net(next_prices, next_costs)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        if policy_was_training:
            self.policy_net.train()
        if target_was_training:
            self.target_net.train()

        self.q_stats["predict_q_mean"].append(predict_q_values.mean().item())
        self.q_stats["predict_q_std"].append(predict_q_values.std().item())
        self.q_stats["target_q_mean"].append(target_q_values.mean().item())
        self.q_stats["target_q_std"].append(target_q_values.std().item())

        self.optimizer.zero_grad()
        loss = self.loss(predict_q_values, target_q_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        return float(loss.item())

    def _save_checkpoint(self, episode_index):
        if self.output_dir is None:
            return

        checkpoint_path = self.output_dir / f"model{episode_index + 1}.pt"
        torch.save(self.policy_net.state_dict(), checkpoint_path)

    def _save_q_statistics_plot(self):
        if self.output_dir is None or not self.q_stats["predict_q_mean"]:
            return

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.q_stats["predict_q_mean"], label="Predict Q Mean", alpha=0.7)
        plt.plot(self.q_stats["target_q_mean"], label="Target Q Mean", alpha=0.7)
        plt.title("Q-Value Mean Over Time", fontsize=16)
        plt.xlabel("Replay Iterations", fontsize=16)
        plt.ylabel("Mean Q-Value", fontsize=16)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.q_stats["predict_q_std"], label="Predict Q Std", alpha=0.7)
        plt.plot(self.q_stats["target_q_std"], label="Target Q Std", alpha=0.7)
        plt.title("Q-Value Standard Deviation Over Time", fontsize=16)
        plt.xlabel("Replay Iterations", fontsize=16)
        plt.ylabel("Std Q-Value", fontsize=16)
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "q_value_trends.png")
        plt.close()

    def train(self, env, n_episodes=100):
        self.policy_net.train()

        loss_history = []
        reward_history = []
        action_distribution_history = []
        step_count = 0

        for episode in range(n_episodes):
            env.reset()
            start_step = env.start_step
            end_step = env.end_step
            price = env.get_input_data(start_step)
            costs = env.get_assets_state(start_step)
            total_reward = 0.0
            episode_loss = 0.0
            day_count = 0
            action_counts = np.zeros(self.action_size, dtype=np.float32)

            for step in tqdm(
                range(start_step, end_step),
                desc=f"Episode {episode + 1}/{n_episodes}",
                leave=False,
            ):
                done = step == end_step - 1
                action = self.act(price, costs)
                action_counts[action] += 1

                env.action_execution(action, step)
                reward_tensor = env.get_reward(step, done, type="train")
                reward_value = float(reward_tensor.squeeze().item())
                total_reward += reward_value

                next_price = env.get_input_data(step + 1)
                next_costs = env.get_assets_state(step + 1)

                self.remember(
                    price, costs, action, reward_value, next_price, next_costs, done
                )
                price = next_price
                costs = next_costs

                day_count += 1
                step_count += 1

                if step_count % 1000 == 0:
                    self.update_target_network()

                if len(self.memory) >= self.min_memory_size:
                    loss = self.replay()
                    if loss is not None:
                        episode_loss += loss

            total_actions = max(float(action_counts.sum()), 1.0)
            action_distribution = action_counts / total_actions
            action_distribution_history.append(action_distribution)

            avg_loss = episode_loss / max(day_count, 1)
            print(f"Episode {episode + 1}, Action distribution: {action_distribution}")
            print(
                f"Episode {episode + 1}/{n_episodes}, Total Loss:{episode_loss:.4f}, "
                f"Avg. Loss:{avg_loss:.4f}, Epsilon: {self.epsilon:.4f}, "
                f"Total Reward: {total_reward:.4f}"
            )

            loss_history.append(avg_loss)
            reward_history.append(total_reward)
            self._save_checkpoint(episode)

        if self.q_stats["predict_q_mean"]:
            print("\nQ-Value Statistics Summary:")
            print(
                f"Predict Q Mean (Avg): {np.mean(self.q_stats['predict_q_mean']):.4f}, "
                f"Std (Avg): {np.mean(self.q_stats['predict_q_std']):.4f}"
            )
            print(
                f"Target Q Mean (Avg): {np.mean(self.q_stats['target_q_mean']):.4f}, "
                f"Std (Avg): {np.mean(self.q_stats['target_q_std']):.4f}"
            )

        self._save_q_statistics_plot()

        if self.output_dir is not None:
            torch.save(costs.detach().cpu(), self.output_dir / "final_cost.pt")

        return loss_history, reward_history, action_distribution_history
