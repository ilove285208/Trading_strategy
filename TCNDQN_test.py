import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from LoadStockData import LoadStockData
from TCNDQN_Environment import Environment
from TCNDQN import TCN_DQN
from tqdm import tqdm

class DQNAgent:
    def __init__(self, price_size, cost_size, layer_channels, kernel_size, action_size, device='cpu'):
        self.device = device
        # DQN 模型
        self.policy_net = TCN_DQN(price_size, cost_size, layer_channels, kernel_size, action_size).to(device)
    
    def act(self, price, costs):
        """ 選擇最佳動作 (測試時不使用 epsilon-greedy) """
        with torch.no_grad():  # 測試時不需要計算梯度
            q_values = self.policy_net(price, costs)
            # print(q_values)
        return torch.argmax(q_values).item()  # 選擇預測的最佳動作
    
    
    def test(self, env, test_start_date, test_end_date, model_path):
        """ 測試 DQN 模型在給定的環境中運行表現 """
        # 加載訓練好的模型
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.policy_net.eval()
        env.reset()

        # 理論上是20210101~20210331，但因為DQN跑出來的action是要給明天執行的，所以要提前一天
        # 變成20201231~20210330
        start_step = env.get_step(test_start_date) - 1 # 20201231
        stop_step = env.get_step(test_end_date)        # 20210331，range函數取不到這個值    

        price = env.get_input_data(start_step).to(self.device)
        # print(f'price shape:{price.shape}')
        costs = env.load_costs() # torch.Size([1, 2, tcn_window])
        # print(f'costs:{costs}')

        total_reward = 0.0
        action_list = []

        for step in tqdm(range(start_step, stop_step)):
            # 使用 DQN 模型選擇動作
            action = self.act(price, costs)
            env.stock_count(action)
            # print(f"step:{step}, action:{action}")
            action_list.append(action) 
            next_price = env.get_input_data(step + 1)
            next_costs = env.get_costs(step + 1)
            # 判斷是否結束
            done = step == stop_step - 1
            # 根據動作執行並取得獎勵
            reward = env.get_reward(step, done ,type = "test")
            total_reward += reward.item()

            # print(costs)
            
            
            # 更新狀態
            price = next_price
            costs = next_costs

        return total_reward, action_list
    
    
    
    
# main program for testing
if __name__ == '__main__':

    # 在程式開始時清空文件內容
    with open('TCNDQN_strategy.txt', 'w') as file:
        file.write('')

    # 測試超參數
    model_path = './model/' 
    sliding_window = 60
    const = 10
    cost_names = ['avgcost','Total cost']
    start_date = 20000101  # 設定資料起始日期
    end_date = 20241231  # 設定資料結束日期
    
    test_start_date = 20220401  # 設定測試起始日期
    test_end_date = 20220630  # 設定測試結束日期
    
    # TCN hyper-parameters
    tcn_window = 70  # TCN 的時間窗口大小   
    # layer_channels = [8, 8, 8, 8, 8, 8, 8]  # 7 層
    # kernel_size = 3
    layer_channels = [8, 8, 8, 8, 8]  # 5 層
    kernel_size = 3
    # layer_channels = [8, 8, 8, 8]
    # kernel_size = 5

    # 設置設備
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # 載入數據
    aux_market_names = [
        "S&P_500",
        "Dow_Jones",
        "NASDAQ",
        "PHLX",
        "N225",
        "FTSE",
        "FCHI",
        "DAX",
        "000001.SS",
    ]
    filename = "StockData/TWII.csv"
    data_period = (start_date - 10000, end_date)
    field_names = ["close","change"]
    Stock_Data = LoadStockData("TWII", field_names, filename, data_period)
    Stock_Data.AddNewData(
        aux_market_names,
        [
            "StockData/S&P_500.csv",
            "StockData/Dow_Jones.csv",
            "StockData/NASDAQ.csv",
            "StockData/PHLX.csv",
            "StockData/N225.csv",
            "StockData/FTSE.csv",
            "StockData/FCHI.csv",
            "StockData/DAX.csv",
            "StockData/000001.SS.csv",
        ],
    )

    # 獲取所有股市的價格資料
    selected_market_names = ["TWII"] + aux_market_names
    dates, prices = Stock_Data.get_prices(data_period, field_names, selected_market_names)
    if len(field_names) < 2:
        prices = np.squeeze(prices, axis=1)

    # 將資料轉換為 torch tensor
    prices = torch.FloatTensor(prices)
    # print(f'price shape:{prices.shape}')
    # exit()

  
    # 初始化環境
    env = Environment(prices, dates, start_date=start_date, end_date = end_date,
    tcn_window=tcn_window, sliding_window=sliding_window, const = const, device=device)
    # 初始化 DQN 測試代理
    action_size = 3  # Buy, Sell, Hold
    price_size = len(selected_market_names)  # TCN 輸出的股價狀態維度
    cost_size = len(cost_names)  # TCN 輸出的成本狀態維度
    
    agent = DQNAgent(price_size, cost_size, layer_channels, kernel_size, action_size, device=device)


    folder_path = Path(model_path)
    # 統計 .pt 檔案數量
    txt_file_count = len(list(folder_path.glob('*.pt')))
    print(f"資料夾 '{folder_path}' 中的 .pt 檔案數量為：{txt_file_count}")
    
    model_reward = []
    action_list = []
    

    for idx in range(1,txt_file_count+1):
        total_reward, action_list = agent.test(env, test_start_date, test_end_date, model_path + f'model{idx}.pt')
        print(f"model{idx}, Total Reward: {total_reward}")
        model_reward.append(total_reward)
        # 使用 write() 方法寫入
        with open('TCNDQN_strategy.txt', 'a') as file:
            file.write(f'{idx - 1} : {action_list}\n')
        # print(f'action:{action_list}')

    

    x = [reward for reward in model_reward]
     # 設置字體為 Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(8, 5))
    plt.plot(x, '-r')
    plt.xlabel('model', fontsize=16)
    plt.ylabel('reward', fontsize=16)
    plt.title('reward per model', fontsize=16)
    plt.savefig('./picture/test/DQN+TCN_test.png')
    plt.show()
