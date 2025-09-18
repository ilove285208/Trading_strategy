import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

from collections import deque
from TCNDQN import TCN_DQN
from tqdm import tqdm

season = "2021-1"
trend = "" # 設定趨勢類型，分別為下跌、上漲和盤整
num = 2
if season == "":
    path = f"./model/{trend}_trend/{num}/"
else:
    path = f"./model/{season}/"
Path(path).mkdir(parents=True, exist_ok=True)  # 如果資料夾不存在則創建
print(path)

# 設置字體為 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"


class DQNAgent:
    def __init__(self, price_size, cost_size, layer_channels, kernel_size, action_size, device="cpu", batch_size=64, epsilon=1.0,
                  epsilon_decay=0.99997, epsilon_min=0.01, learning_rate=0.007, gamma=0.95, memory_size=100000):
        self.action_size = action_size
        self.epsilon = epsilon
        # 0.99997大約30個epsiode後，epsilon會降到0.01
        # 0.999982大約50個epsiode後，epsilon會降到0.01
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        # 設置reply的最小memory門檻
        self.min_memory_size = batch_size
        self.device = device
        # DQN 模型
        self.policy_net = TCN_DQN(price_size, cost_size, layer_channels, kernel_size, action_size).to(device)
        self.target_net = TCN_DQN(price_size, cost_size, layer_channels, kernel_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
        # 紀錄繪製圖表所需的數值
        self.q_stats = {
            'predict_q_mean': [],  # 記錄predict_q_values的均值
            'predict_q_std': [],   # 記錄predict_q_values的標準差
            'target_q_mean': [],   # 記錄target_q_values的均值
            'target_q_std': []     # 記錄target_q_values的標準差
        }

        
    def update_target_network(self):
        """ 使用 target_net 複製 policy_net 的參數 """
        # self.target_net.dqn.load_state_dict(self.policy_net.dqn.state_dict())
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # print("Target network updated")

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

        # 檢查是否有過度估計問題
        # 記錄 Q 值統計
        self.q_stats['predict_q_mean'].append(predict_q_values.mean().item())
        self.q_stats['predict_q_std'].append(predict_q_values.std().item())
        self.q_stats['target_q_mean'].append(target_q_values.mean().item())
        self.q_stats['target_q_std'].append(target_q_values.std().item())


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
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        return loss.item()

    def train(self, env, n_episodes=100):
        """ 訓練 DQN  """
        loss_history = []
        reward_history = []
        action_distribution_history = []  # 用於記錄每個 episode 的動作分佈
        step_count = 0

        for episode in range(n_episodes):
            env.reset()
            start_step = env.start_step - 1
            end_step = env.end_step
            price = env.get_input_data(start_step) # torch.Size([1, 10, tcn_window])
            costs = env.get_assets_state(start_step) # torch.Size([1, 2, tcn_window])
            total_reward = 0.0
            done = False
            episode_loss = 0.0  # 每個 episode 的累積損失
            day_count = 0 # 計算平均Loss用的計數器
            action_counts = np.zeros(self.action_size) # 記錄每個動作的次數

            # 每個episode會跑完所有天數
            for step in tqdm(range(start_step, end_step)):
                # 確認是不是最後一天
                done = step == end_step - 1  
                # 獲取當前動作
                action = self.act(price, costs)
                action_counts[action] += 1
                # 動作執行，更新持有數量以及資產狀態
                env.action_execution(action, step) 
                # 計算獎勵
                reward = env.get_reward(step, done, type="train") 
                total_reward += reward
                # 獲取下一步的狀態
                next_price = env.get_input_data(step + 1) 
                next_costs = env.get_assets_state(step + 1) 

                # 儲存經驗
                self.remember(price, costs, action, reward, next_price, next_costs, done)
                # 更新狀態
                price = next_price
                costs = next_costs



                day_count += 1
                step_count += 1
                # 每隔一定步數更新 target network
                if step_count % 1000 == 0:
                    self.update_target_network()
                # Experience Replay
                if len(self.memory) >= self.min_memory_size:
                    loss = self.replay()  # 更新 DQN
                    episode_loss += loss

            # 記錄每個 episode 的動作分佈
            action_distribution = action_counts / action_counts.sum()
            action_distribution_history.append(action_distribution)

            # 輸出每個episode的每個動作所佔的比例
            print(f"Episode {episode+1}, Action distribution: {action_counts / action_counts.sum()}")
            # 輸出每個episode的損失、獎勵和 epsilon 值
            print(f"Episode {episode+1}/{n_episodes}, Total Loss:{episode_loss:.4f}, "+
                  f"Avg. Loss:{(episode_loss / day_count):.4f}, Epsilon: {self.epsilon:.4f}"+
                  f", Total Reward: {float(total_reward):.4f}")

            loss_history.append(episode_loss / day_count)
            reward_history.append(total_reward.detach().cpu().numpy())

            # 保存模型參數
            torch.save(self.policy_net.state_dict(), f"{path}/model{episode + 1}.pt")

        # 輸出 Q 值統計總結
        print("\nQ-Value Statistics Summary:")
        print(f"Predict Q Mean (Avg): {np.mean(self.q_stats['predict_q_mean']):.4f}, "
            f"Std (Avg): {np.mean(self.q_stats['predict_q_std']):.4f}")
        print(f"Target Q Mean (Avg): {np.mean(self.q_stats['target_q_mean']):.4f}, "
            f"Std (Avg): {np.mean(self.q_stats['target_q_std']):.4f}")

        # 可視化 Q 值趨勢
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.q_stats['predict_q_mean'], label="Predict Q Mean", alpha=0.7)
        plt.plot(self.q_stats['target_q_mean'], label="Target Q Mean", alpha=0.7)
        plt.title("Q-Value Mean Over Time", fontsize=16)
        plt.xlabel("Replay Iterations", fontsize=16)
        plt.ylabel("Mean Q-Value", fontsize=16)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.q_stats['predict_q_std'], label="Predict Q Std", alpha=0.7)
        plt.plot(self.q_stats['target_q_std'], label="Target Q Std", alpha=0.7)
        plt.title("Q-Value Standard Deviation Over Time", fontsize=16)
        plt.xlabel("Replay Iterations", fontsize=16)
        plt.ylabel("Std Q-Value", fontsize=16)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{path}/q_value_trends.png")
        plt.show()

        # 保存costs參數
        torch.save(costs, f"{path}/final_cost.pt")

        return loss_history, reward_history, action_distribution_history
    
