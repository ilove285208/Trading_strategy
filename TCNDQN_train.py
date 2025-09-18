import torch
import numpy as np
import matplotlib.pyplot as plt

from LoadStockData import LoadStockData
# 包含資產特徵
from TCNDQN_Environment import Environment
from TCNDQN_DQNAgent import DQNAgent
# from TCNDQN_DQNAgent_1 import DQNAgent
import os

def delete_pt_files(folder_path):
    """刪除資料夾內的所有 .pt 檔案"""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) and file_path.endswith('.pt'):
                    os.unlink(file_path)  # 刪除 .pt 檔案
                    print(f"已刪除: {file_path}")
            except Exception as e:
                print(f"無法刪除 {file_path}。原因: {e}")
    else:
        print(f"資料夾 {folder_path} 不存在。")



# main program
if __name__ == '__main__':      
    # 程式開始時先清空.pt檔案
    # delete_pt_files('./model/')

    # training hyper-parameters
    batch_size = 32
    sliding_window = 60
    const = 10

    train_start_date = 20000101  # 設定訓練起始日期
    train_end_date = 20220331  # 設定訓練結束日期
    data_start_date = train_start_date - 20000  # 設定資料起始日期
    data_end_date = train_end_date + 9100 # 設定資料結束日期
    trend = "down" 

    action_type = ['sell','hold','buy'] # 設定動作類型
    field_names = ['close','change']
    cost_names = ['avgcost','Total cost']
    # TCN hyper-parameters
    tcn_window = 70  # TCN 的時間窗口大小   
    # layer_channels = [8, 8, 8, 8, 8, 8, 8]  # 7 層
    # kernel_size = 3
    layer_channels = [8, 8, 8, 8, 8]  # TCN 每層的通道數量
    kernel_size = 3
    # layer_channels = [8, 8, 8, 8]   # TCN 每層的通道數量
    # kernel_size = 5
    
    # check the device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # exit()

    # load data
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
    data_period = (data_start_date, data_end_date) 
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

    # get the prices of all stock markets
    selected_market_names = ["TWII"] + aux_market_names
    # selected_market_names = ["TWII"] 
    dates, prices = Stock_Data.get_prices(data_period, field_names, selected_market_names)
    if len(field_names) < 2:
        prices = np.squeeze(prices, axis=1)
    # print(dates)
    # exit()

    # 將資料轉換為 torch tensor
    prices = torch.FloatTensor(prices)
    # print(f'price shape:{prices.shape}')
    # exit()


    # 設定模型
    price_size = len(selected_market_names)  # TCN 輸出的股價狀態維度
    cost_size = len(cost_names)  # TCN 輸出的成本狀態維度

    # 包含資產特徵
    agent = DQNAgent(price_size, cost_size, layer_channels, kernel_size, action_size=len(action_type), device=device)
    # 初始化環境
    env = Environment(prices, dates, start_date=train_start_date, end_date=train_end_date, tcn_window=tcn_window, sliding_window=sliding_window, const = const, device=device)


    # 輸出從 startdate 開始的步驟
    # print(f"Start step for {start_date}, index:{env.start_step}")

    # 開始訓練
    loss_history, reward_history = agent.train(env)
    # 建立 x 軸資料
    x = range(len(loss_history))
    # 建立兩條線的 y 軸資料
    y1 = [loss for loss in loss_history]
    y2 = [reward for reward in reward_history]
     # 設置字體為 Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    # 建立主圖表
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 繪製第一條線並設置左側的 y 軸標籤
    ax1.plot(x, y1, '-b', label='Loss')
    ax1.set_xlabel('Episode', fontsize=16)
    ax1.set_ylabel('Loss', color='blue', fontsize=16)
    # 創建共享 x 軸的第二個 y 軸
    ax2 = ax1.twinx()

    # 繪製第二條線並設置右側的 y 軸標籤
    ax2.plot(x, y2, '-r', label='reward')
    ax2.set_ylabel('reward', color='red', fontsize=16)
    
    plt.title('Loss and reward per Episode', fontsize=16)
    if trend != "":
        plt.savefig(f'./picture/train/{trend}/TCNDQN_train.png')
    else:
        plt.savefig(f'./picture/train/TCNDQN_train.png')
    plt.show()