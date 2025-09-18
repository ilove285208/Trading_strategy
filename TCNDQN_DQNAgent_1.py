import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from collections import deque
from TCNDQN import TCN_DQN
from tqdm import tqdm
from date_utils import get_episode_range_rolling


class DQNAgent:
    def __init__(self, price_size, cost_size, layer_channels, kernel_size, action_size, device='cpu', batch_size=32, epsilon=1.0,
                  epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001, gamma=0.95, memory_size=10000):
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
        # DQN 模型
        self.policy_net = TCN_DQN(price_size, cost_size, layer_channels, kernel_size, action_size).to(device)
        self.target_net = TCN_DQN(price_size, cost_size, layer_channels, kernel_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        # self.loss = nn.SmoothL1Loss()
        
    def update_target_network(self):
        """ 使用 target_net 複製 policy_net 的參數 """
        # self.target_net.dqn.load_state_dict(self.policy_net.dqn.state_dict())
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Target network updated")

    def act(self, price, costs):
        """ 使用 epsilon-greedy 策略選擇動作 """
        if random.random() <= self.epsilon:
            return random.choice(range(self.action_size))  # 隨機選擇動作
        with torch.no_grad():
            q_values = self.policy_net(price, costs)
        return torch.argmax(q_values).item()  # 選擇預測的最佳動作

    def remember(self, price, costs, action, reward, next_state, next_costs, done):
        """ 儲存記憶 """
        self.memory.append((price, costs, action, reward, next_state, next_costs, done))

    def replay(self):
        """ 從記憶中隨機選擇 batch 進行訓練 """
        if len(self.memory) < self.batch_size:
            return

        # 隨機選擇批次
        batch = random.sample(self.memory, self.batch_size)

        prices = torch.stack([x[0] for x in batch]).squeeze(1).to(self.device)
        costs = torch.stack([x[1] for x in batch]).squeeze(1).to(self.device)
        actions = torch.LongTensor([x[2] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        next_prices = torch.stack([x[4] for x in batch]).squeeze(1).to(self.device) 
        next_costs = torch.stack([x[5] for x in batch]).squeeze(1).to(self.device)
        dones = torch.FloatTensor([x[6] for x in batch]).to(self.device)

        # 預測 Q 值
        predict_q_values = self.policy_net(prices, costs).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_prices, next_costs).max(1)[0]

        # 計算 target Q 值
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 清除前一輪梯度
        self.optimizer.zero_grad()

        # 計算損失
        loss = self.loss(predict_q_values, target_q_values)
        loss.backward()

        # 梯度修剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        # 更新 dqn 網絡
        self.optimizer.step()
        

        # 更新 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def train(self, env):
        """ 訓練 DQN  """
        loss_history = []
        reward_history = []
        step_count = 0

        base_start = env.start_date # 20000101
        episode = 0

        while True:
            start_date, stop_date = get_episode_range_rolling(base_start, episode)
            start_step = env.get_step(start_date) - 1
            stop_step = env.get_step(stop_date)
            env.reset()

            if stop_step > env.end_step:
                break

            price = env.get_input_data(start_step) # torch.Size([1, 10, tcn_window])
            costs = env.get_costs(start_step) # torch.Size([1, 2, tcn_window])
            total_reward = 0.0
            episode_loss = 0.0  # 每個 episode 的累積損失
            day_count = 0 # 計算平均Loss用的計數器
            done = False

            for step in tqdm(range(start_step, stop_step)):
                action = self.act(price,costs)
                env.stock_count(action) # 更新持有數量
                next_price = env.get_input_data(step + 1)
                next_costs = env.get_costs(step + 1)
                reward = env.get_reward(step)

                done = step == stop_step - 1 # 確認是不是最後一天
                self.remember(price, costs, action, reward, next_price, next_costs, done)

                # 更新狀態
                price = next_price
                costs = next_costs

                total_reward += reward
                day_count += 1
                step_count += 1
                # 每隔一定步數更新 target network
                # if step_count % 1000 == 0:
                #     self.update_target_network()
                if len(self.memory) >= self.batch_size:
                    loss = self.replay()  # 更新 DQN
                    episode_loss += loss
            # print(f"start_step:{start_step} to stop_step:{stop_step}")
            print(f"Episode {episode + 1}, Total Loss:{episode_loss}, Avg. Loss:{episode_loss / day_count}, Total Reward: {total_reward}")

            loss_history.append(episode_loss / day_count)
            reward_history.append(total_reward.detach().cpu().numpy())

            # 每隔一定步數更新 target network
            if (episode + 1) % 2 == 0:
                self.update_target_network()

            # 保存模型參數
            torch.save(self.policy_net.state_dict(), f'./model/model{episode + 1}.pt')

            # 計算episode
            episode += 1
        print(f"costs:{costs}")

        return loss_history, reward_history
    
