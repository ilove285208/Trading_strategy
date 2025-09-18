import numpy as np
import torch
import torch.nn as nn

from datetime import datetime
from LoadStockData import LoadStockData



class Environment:
    def __init__(self, data, dates, start_date, end_date, tcn_window=70, sliding_window = 60, const = 10, device="cpu"):
        # 資料和日期
        self.changes = data[:,1,:]
        self.changes = self.changes.unsqueeze(dim=0)
        self.changes = torch.transpose(self.changes, 1, 2) 
        self.changes = self.zscore_normal(self.changes)

        self.prices = data[:,0,:]
        self.prices = self.prices.unsqueeze(dim=0)
        self.prices = torch.transpose(self.prices, 1, 2)
        
        self.dates = dates  # 這是日期列表，與股價資料對應
        self.tcn_window = tcn_window
        self.sliding_window = sliding_window
        self.const = const # 用來計算執行"HOLD"時的reward
        self.device = device

        # 初始化狀態變數
        self.reset()
        self.count = 0 # 計算持有股票張數
        self.position = 0 # 儲存當前操作方向(預設為hold)

        self.start_step = self.get_step(start_date) # 計算 start_date 對應的 step
        self.end_step = self.get_step(end_date) # 計算 end_date 對應的 step 
    
    def reset(self):
        """ 重置環境 """
        self.count = 0
        self.position = 0


    def zscore_normal(self, data):
        """ 正規化資料 """
        # print(f"original data:\n{data}")

        # 使用 PyTorch 的 torch.mean 和 torch.std 進行正規化
        mean = torch.mean(data, dim=-1, keepdim=True)
        std = torch.std(data, dim=-1, keepdim=True)   
        std[std == 0] = 1 # 避免標準差為0導致除以0的情況發生
        normalized_data = (data - mean) / std
        # print(f"normalized data:\n{normalized_data}")

        return normalized_data

    def get_step(self, startdate):
        """ 計算 step 開始的索引位置 """
        # 將 startdate 字符串轉換為日期格式
        startdate = datetime.strptime(str(startdate), "%Y%m%d")
        
        # 找到與 startdate 對應的日期在資料中的索引
        for i, date in enumerate(self.dates):
            date_str = str(int(date))
            current_date = datetime.strptime(date_str, "%Y%m%d")
            if current_date >= startdate:
                return i  # 返回第一個符合的索引位置
        return len(self.dates) - 1  # 如果找不到，返回資料最後一個位置
    
    def get_input_data(self, step):
        """ 提取指定步驟的前 tcn_window 天資料 """
        if step < 0 or step >= self.changes.shape[2]:
            raise ValueError(f"step {step} 超出資料範圍 [0, {self.changes.shape[2]-1}]")
        start = max(0, step - self.tcn_window + 1)
        return self.changes[:, :, start:step].to(self.device)
    
    def stock_count(self, action):
        self.position = action - 1 # action = 0,1,2 -> position = -1,0,1

        self.count += self.position

        self.entry_price = self.prices[:,0,self.start_step+1].item() # 記錄進場價格
        # print(f"action : {action}")
        # print(f"持有張數:{self.count}")
    
    def cal_ratio(self, prices):
        """ 計算未來滑動窗口內的最大和最小漲幅 """
        current_price = prices[:, 0]
        if current_price == 0:
            return torch.tensor(0.0), torch.tensor(0.0)  # 避免除零
        future_prices = prices[:, 1:]
        r_values = 100 * (future_prices - current_price) / current_price
        max_ratio = torch.max(r_values[r_values > 0]) if (r_values > 0).any() else 0.0
        min_ratio = torch.min(r_values[r_values < 0]) if (r_values < 0).any() else 0.0

        return max_ratio, min_ratio
    
    def get_reward(self, step, end = 0, type = "train"):
        """ 獎勵函數 """
        if type == "train":
            prices = self.prices[:, 0, (step + 1):(step + 1) + self.sliding_window]
            
            """ 使用專家資訊來當作獎勵函數-論文查閱 """
            max_ratio, min_ratio = self.cal_ratio(prices)

            """ 方案1 """
            # 計算 reward   
            if max_ratio > 0 or (max_ratio + min_ratio) > 0:
                reward = self.position * max_ratio
            elif min_ratio < 0 or (max_ratio + min_ratio) < 0:
                reward = self.position * min_ratio
            elif self.position == 0:
                reward = self.const - (max_ratio - min_ratio)

        elif type == "test":
            # 當前步驟的價格
            current_price = self.prices[:, 0, step + 1].item()
            reward = 0.0
            if end:  # 最後一天，平倉
                reward += current_price * self.count
                self.count = 0
            else:
                # 買賣價格
                reward = self.position * -current_price

        return torch.tensor(reward, dtype=torch.float32).to(self.device)

    def get_date(self, step):
        """ 返回當前步驟對應的日期 """
        return self.dates[step]


