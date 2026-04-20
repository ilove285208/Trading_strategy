from collections import deque
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from TCNDQN_noC import TCN_DQN


class DQNAgent:
    def __init__(
        self,
        price_size,
        layer_channels,
        kernel_size,
        action_size,
        device="cpu",
        batch_size=32,
        epsilon=1.0,
        epsilon_decay=0.99997,
        epsilon_min=0.01,
        learning_rate=0.002,
        gamma=0.95,
        memory_size=20000,
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.max_memory_size = memory_size
        self.device = device

        self.policy_net = TCN_DQN(price_size, layer_channels, kernel_size, action_size).to(
            device
        )
        self.target_net = TCN_DQN(price_size, layer_channels, kernel_size, action_size).to(
            device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss = nn.SmoothL1Loss()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def greedy_action(self, price):
        was_training = self.policy_net.training
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(price.to(self.device))
        if was_training:
            self.policy_net.train()
        return int(torch.argmax(q_values, dim=1).item())

    def act(self, price):
        if random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        return self.greedy_action(price)

    def remember(self, price, action, reward, next_state, done):
        self.memory.append(
            (
                price.detach().cpu(),
                int(action),
                float(reward),
                next_state.detach().cpu(),
                float(done),
            )
        )

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        prices = torch.stack([x[0] for x in batch]).squeeze(1).to(self.device)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device)
        next_prices = torch.stack([x[3] for x in batch]).squeeze(1).to(self.device)
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=self.device)

        predict_q_values = (
            self.policy_net(prices).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        policy_was_training = self.policy_net.training
        target_was_training = self.target_net.training
        self.policy_net.eval()
        self.target_net.eval()
        with torch.no_grad():
            next_actions = self.policy_net(next_prices).argmax(dim=1)
            next_q_values = (
                self.target_net(next_prices)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        if policy_was_training:
            self.policy_net.train()
        if target_was_training:
            self.target_net.train()

        self.optimizer.zero_grad()
        loss = self.loss(predict_q_values, target_q_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        return float(loss.item())

    def train(self, env, n_episodes=100):
        loss_history = []
        reward_history = []
        step_count = 0
        self.policy_net.train()

        for episode in range(n_episodes):
            env.reset()
            start_step = env.start_step
            total_reward = 0.0
            episode_loss = 0.0
            day_count = 0
            price = env.get_input_data(start_step)

            for step in tqdm(range(start_step, env.end_step), leave=False):
                action = self.act(price)
                env.stock_count(action)
                reward = float(env.get_reward(step).item())
                next_price = env.get_input_data(step + 1)

                done = step == env.end_step - 1
                self.remember(price, action, reward, next_price, done)
                price = next_price

                total_reward += reward
                day_count += 1
                step_count += 1

                if step_count % 1000 == 0:
                    self.update_target_network()
                if len(self.memory) >= self.batch_size:
                    loss = self.replay()
                    if loss is not None:
                        episode_loss += loss

            print(
                f"Episode {episode + 1}/{n_episodes}, Total Loss:{episode_loss:.4f},"
                f"Avg. Loss:{(episode_loss / max(day_count, 1)):.4f}, Epsilon: {self.epsilon:.4f}"
                f", Total Reward: {total_reward:.4f}"
            )

            loss_history.append(episode_loss / max(day_count, 1))
            reward_history.append(total_reward)

            path = Path("./model/No_Cost/")
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.policy_net.state_dict(), path / f"model{episode + 1}.pt")

        return loss_history, reward_history
