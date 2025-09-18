import numpy as np
import torch
import torch.nn as nn

from datetime import datetime
from LoadStockData import LoadStockData
from dateutil.relativedelta import relativedelta



class Environment:
    def __init__(self, data, dates, start_date, end_date, tcn_window=70, sliding_window = 60, const = 10, device="cpu"):
        # 資料和日期
        self.changes = data[:,1,:].to(device)
        self.changes = self.changes.unsqueeze(dim=0)
        self.changes = torch.transpose(self.changes, 1, 2) 
        self.changes = self.zscore_normal(self.changes)

        self.prices = data[:,0,:].to(device)
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
        self.previous_count = self.count # 計算前一張數
        self.position = 0 # 儲存當前操作方向(預設為hold)
        self.avgcost = torch.zeros(1).to(device) # 計算持有股票平均成本
        self.unreal_PNL = torch.zeros(1).to(device) # 計算未實現損益
        self.real_PNL = torch.zeros(1).to(device) # 計算已實現損益

        self.costs = torch.zeros(1, 2, self.tcn_window).to(device) # 初始化成本特徵
        self.start_step = self.get_step(start_date) # 計算 start_date 對應的 step
        self.end_step = self.get_step(end_date) # 計算 end_date 對應的 step 
        
    
    def reset(self):
        """ 重置環境 """
        self.count = 0
        self.previous_count = self.count
        self.position = 0
        self.avgcost = torch.zeros(1).to(self.device)
        self.unreal_PNL = torch.zeros(1).to(self.device)
        self.real_PNL = torch.zeros(1).to(self.device)

    def load_costs(self, path):
        """ 載入成本特徵 """
        self.costs = torch.load(f"{path}/final_cost.pt").to(self.device)
        return self.costs

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
    
    def get_date(self, step):
        """ 返回當前步驟對應的日期 """
        return self.dates[step]
    
    def get_input_data(self, step):
        """ 提取指定步驟的前 tcn_window 天資料 """
        if step < 0 or step >= self.changes.shape[2]:
            raise ValueError(f"step {step} 超出資料範圍 [0, {self.changes.shape[2]-1}]")
        start = max(0, step - self.tcn_window + 1)
        return self.changes[:, :, start:step].to(self.device)
    
    def action_execution(self, action, step):
        """ 執行預測的動作 """
        self.position = action - 1 # action = 0,1,2 -> position = -1,0,1

        # 當前步驟的價格
        """ 預測出來的action是為了在明天執行，所以step要+1 """
        current_price = self.prices[:, 0, step + 1].item()
        # 檢查index是否越界
        if step + 1 >= self.prices.shape[2]:
            raise IndexError(f"step + 1 = {step+1} out of bounds")


        # 保存當前持倉數量，用於判斷是否平倉
        # self.count: 正數表示多頭持倉，負數表示空頭持倉，0 表示無持倉
        self.previous_count = self.count

        # 有平倉則計算已實現損益
        if self.previous_count > 0 and self.position == -1:
            # (多頭)遇到反向操作，平倉所有現有倉位 正 * (正) = 正
            self.real_PNL = self.previous_count * (current_price - self.avgcost)
            self.count = 0  # 清零持倉
            self.avgcost = torch.zeros(1).to(self.device)  # 重置平均成本
            self.unreal_PNL = torch.zeros(1).to(self.device)  # 重置未實現損益
        elif self.previous_count < 0 and self.position == 1:
            # (空頭)遇到反向操作，平倉所有現有倉位 負 * (負) = 正
            self.real_PNL = self.previous_count * (current_price - abs(self.avgcost))
            self.count = 0  # 清零持倉
            self.avgcost = torch.zeros(1).to(self.device)  # 重置平均成本
            self.unreal_PNL = torch.zeros(1).to(self.device)  # 重置未實現損益
        else:
            # 沒有平倉則已實現損益為零
            self.real_PNL = torch.zeros(1).to(self.device)
            # 更新持股數量
            self.count += self.position     


        
    def count_price_changes(self, prices):
        """
        計算未來 sliding window 天內每天的漲跌幅
        使用今天與昨天的價格進行比較計算漲跌幅

        Args:
            prices (torch.Tensor): 股價資料，形狀為 [1, sliding_window] 或 [sliding_window]
            sliding_window (int): 滑動窗口大小

        Returns:
            torch.Tensor: 每天的漲跌幅，形狀為 [sliding_window]
        """
        # 確保 prices 是 1 維張量
        if prices.dim() > 1:
            prices = prices.squeeze(0)

        # 計算每天的漲跌幅
        daily_changes = (prices[1:] - prices[:-1]) / prices[:-1] * 100  # 百分比漲跌幅
        
        # 初始化計數
        ratio = {"up": 0, "down": 0}
        # 上漲：漲跌幅 > 0
        ratio["up"] = daily_changes[daily_changes > 0].sum().item()

        # 下跌：漲跌幅 < 0
        ratio["down"] = daily_changes[daily_changes < 0].sum().item()

        return ratio
    
        
    def get_reward(self, step, done = 0, type = "train"):
        """ 獎勵函數 """
        if type == "train":
            """ 預測出來的action是為了在明天執行，所以step要+1 """
            start_date = datetime.strptime(str(int(self.get_date(step + 1))), "%Y%m%d")
            end_date = start_date + relativedelta(months=3)
            start_step = self.get_step(start_date.strftime("%Y%m%d"))
            end_step = self.get_step(end_date.strftime("%Y%m%d"))

            prices = self.prices[:, 0, start_step:end_step] # 未來3個月(含今天)的資料
            
            # 檢查index是否越界
            if end_step >= self.prices.shape[2]:
                raise IndexError(f"end_step = {end_step} out of bounds")
            
            """ 使用來當作判斷依據獎勵函數 """
            ratio = self.count_price_changes(prices)
            threshold = 0.5 * np.sqrt(self.sliding_window) # 設定平盤的漲跌幅門檻值

            if ratio["up"] + ratio["down"] > threshold:
                reward = ratio["up"] if self.position == 1 else -ratio["up"]
            elif ratio["up"] + ratio["down"] < -threshold:
                reward = abs(ratio["down"]) if self.position == -1 else -abs(ratio["down"])
            else:
                reward = (ratio["up"] - ratio["down"]) / 2 if self.position == 0 else -(ratio["up"] - ratio["down"]) / 2 

        elif type == "test":
            """ 預測出來的action是為了在明天執行，所以step要+1 """
            current_price = self.prices[:, 0, step + 1].item() # 當前步驟的價格(明天價格)
            if done:  # 最後一天，平倉
                if self.count > 0:
                    # self.count != 0 代表動作跟當前持倉方向相同
                    self.real_PNL = self.count * (current_price - self.avgcost)
                elif self.count < 0:
                    # self.count != 0 代表動作跟當前持倉方向相同
                    self.real_PNL = self.count * (current_price - abs(self.avgcost))
                else:
                    # self.count = 0 代表無持倉(self.count跟self.previous_count皆為0)或是action_execution已經執行平倉
                    pass
                self.count = 0  # 清空持倉 
                self.avgcost = torch.zeros(1).to(self.device) # 清空平均成本
                self.unreal_PNL = torch.zeros(1).to(self.device) # 清空未實現損益
            
            # 計算獎勵
            reward = self.real_PNL.item()

        # 動態調整 reward 的形狀
        if torch.is_tensor(reward):
            reward_tensor = reward.to(self.device)
        else:
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
        if reward_tensor.dim() == 0:  # 如果是標量，新增一個維度
            reward_tensor = reward_tensor.unsqueeze(0)
        return reward_tensor

    def get_assets_state(self, step):
        """ 計算資產特徵 """
        current_price = self.prices[0, 0, step].to(self.device) # 當前步驟的價格
        start = max(0, step - self.tcn_window + 1)
        mean_price = torch.mean(self.prices[0, 0, start:step]).to(self.device) # 平均價格

        # 計算未實現損益與平均成本(持倉價)
        # 假設：self.count 已經在動作後更新完畢（即已包含最新持倉）
        # 多頭 avgcost 為正，空頭 avgcost current_price 為正
        if self.count > 0:
            if self.position == 1:   # 多頭建倉或加碼時，需更新平均成本
                # 新平均成本 = (原持倉總成本 + 新成交價格) / 新總張數
                # 原總成本 = avgcost * (count - 1)，因為 count 已是更新後
                self.avgcost = (self.avgcost * (self.count - 1) + current_price) / self.count
            # 未實現損益 = 多頭報酬 = (現價 - |正成本|) 
            # 如果current_price比avgcost大，則報酬為正；如果current_price比avgcost小，則報酬為負
            self.unreal_PNL = current_price - abs(self.avgcost)

        elif self.count < 0:
            if self.position == -1:  # 空頭建倉或加碼時，需更新平均成本
                # 注意：avgcost 為負值，但 avgcost * (count + 1) 是正的
                # 新 avgcost 維持負值方向邏輯不變
                self.avgcost = (self.avgcost * (self.count + 1) + current_price) / self.count
            # 未實現損益 = 空頭報酬 = (|負成本| - 現價) 
            # 如果current_price比avgcost大，則報酬為負；如果current_price比avgcost小，則報酬為正
            self.unreal_PNL = abs(self.avgcost) - current_price

        else:
            # 無持倉則損益與成本歸零
            self.unreal_PNL = torch.zeros(1).to(self.device)
            self.avgcost = torch.zeros(1).to(self.device)
        
        avgcost = self.avgcost.clone() if self.avgcost != 0 else torch.tensor(0.0, device=self.device)
        unreal_PNL = self.unreal_PNL.clone() if self.unreal_PNL != 0 else torch.tensor(0.0, device=self.device)

        # print(f"avgcost:{avgcost} unreal_PNL:{unreal_PNL}")
        
        # print(f"before costs:{self.costs}")
        self.costs[:, :, :-1] = self.costs[:,:,1:] # 所有資料往前移
        self.costs[:, 0, -1] = avgcost  # 計算持有股票的成本
        self.costs[:, 1, -1] = unreal_PNL / abs(avgcost) if avgcost != 0 else 0.0 # 計算未實現損益比率
        # print(f"after costs:{self.costs}")

        costs = self.costs.clone() # 複製成本特徵
        
        # print(f"before costs:{costs[:, 0, -10:]}")
        costs[:, 0, :] = costs[:, 0, :] / mean_price # 計算持有股票的成本相較於平均價格的比率
        # print(f"after costs:{costs[:, 0, -10:]}")

        return costs.to(self.device)

    

    


