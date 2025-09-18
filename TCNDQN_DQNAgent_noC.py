import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

from collections import deque
from TCNDQN_noC import TCN_DQN
from tqdm import tqdm


class DQNAgent:
    def __init__(self, price_size, layer_channels, kernel_size, action_size, device="cpu", batch_size=32, epsilon=1.0,
                  epsilon_decay=0.99997, epsilon_min=0.01, learning_rate=0.002, gamma=0.95, memory_size=20000):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay # 0.99997大約30個epsiode後，epsilon會降到0.01
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.max_memory_size = memory_size
        self.device = device
        # DQN 模型
        self.policy_net = TCN_DQN(price_size, layer_channels, kernel_size, action_size).to(device)
        self.target_net = TCN_DQN(price_size, layer_channels, kernel_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
        
    def update_target_network(self):
        """ 使用 target_net 複製 policy_net 的參數 """
        # self.target_net.dqn.load_state_dict(self.policy_net.dqn.state_dict())
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # print("Target network updated")

    def act(self, price):
        """ 使用 epsilon-greedy 策略選擇動作 """
        if random.random() <= self.epsilon:
            return random.choice(range(self.action_size))  # 隨機選擇動作
        with torch.no_grad():
            q_values = self.policy_net(price)
        return torch.argmax(q_values).item()  # 選擇預測的最佳動作

    def remember(self, price, action, reward, next_state, done):
        """ 儲存記憶 """
        self.memory.append((price, action, reward, next_state, done))

    def replay(self):
        """ 從記憶中隨機選擇 batch 進行訓練 """
        if len(self.memory) < self.batch_size:
            return

        # 隨機選擇批次
        batch = random.sample(self.memory, self.batch_size)

        prices = torch.stack([x[0] for x in batch]).squeeze(1).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_prices = torch.stack([x[3] for x in batch]).squeeze(1).to(self.device) 
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)

        # 預測 Q 值
        predict_q_values = self.policy_net(prices).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN 的 target Q 值計算
        # with torch.no_grad():
        #     next_actions = self.policy_net(next_prices, next_costs).argmax(1)
        #     next_q_values = self.target_net(next_prices, next_costs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_prices).max(1)[0]

        # 計算 target Q 值
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 檢查是否有過度估計問題
        # print(f"Current Q Mean: {predict_q_values.mean().item():.2f}, Target Q Mean: {target_q_values.mean().item():.2f}")

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

    def train(self, env, n_episodes=100):
        """ 訓練 DQN  """
        loss_history = []
        reward_history = []
        step_count = 0
        for episode in range(n_episodes):
            env.reset()
            start_step = env.start_step - 1
            price = env.get_input_data(start_step) # torch.Size([1, 10, tcn_window])
            total_reward = 0.0
            done = False
            episode_loss = 0.0  # 每個 episode 的累積損失
            day_count = 0 # 計算平均Loss用的計數器

            for step in tqdm(range(start_step, env.end_step)):
                action = self.act(price)
                env.stock_count(action) # 更新持有數量
                next_price = env.get_input_data(step + 1)
                reward = env.get_reward(step)

                done = step == env.end_step - 1  # 確認是不是最後一天
                self.remember(price, action, reward, next_price, done)

                # 更新狀態
                price = next_price

                total_reward += reward
                day_count += 1
                step_count += 1
                
                # 每隔一定步數更新 target network
                if step_count % 1000 == 0:
                    self.update_target_network()
                # Experience Replay
                if len(self.memory) >= self.batch_size:
                    loss = self.replay()  # 更新 DQN
                    episode_loss += loss
            
            print(f"Episode {episode+1}/{n_episodes}, Total Loss:{episode_loss:.4f},"+
                  f"Avg. Loss:{(episode_loss / day_count):.4f}, Epsilon: {self.epsilon:.4f}"+
                  f", Total Reward: {total_reward:.4f}")

            loss_history.append(episode_loss / day_count)
            reward_history.append(total_reward.detach().cpu().numpy())

            # 保存模型參數
            path = f"./model/No_Cost/"
            Path(path).mkdir(parents=True, exist_ok=True)  # 如果資料夾不存在則創建
            torch.save(self.policy_net.state_dict(), f"{path}/model{episode + 1}.pt")


        return loss_history, reward_history
    
