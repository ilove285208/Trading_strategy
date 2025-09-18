import torch
import torch.nn as nn
from DQN_Block import DQN
from TCN_Block import TCN


# TCN-DQN integration model
class TCN_DQN(nn.Module):
    def __init__(self, tcn_price_channels, tcn_cost_channels, tcn_layer_channels, tcn_kernel_size, dqn_output_channels):
        super(TCN_DQN, self).__init__()
        self.market_tcn = TCN(tcn_price_channels, tcn_layer_channels, tcn_kernel_size)
        self.asset_tcn = TCN(tcn_cost_channels, tcn_layer_channels, tcn_kernel_size)
        
        # 串接後的size = (2 * tcn_layer_channels[-1])
        self.dqn = DQN(input_channels=2 * tcn_layer_channels[-1], output_channels=dqn_output_channels)

    def forward(self, price, cost):
        # 分別經過 TCN 處理，並取最後一個時間步的特徵
        market_features = self.market_tcn(price)[:, :, -1]
        asset_features = self.asset_tcn(cost)[:, :, -1]
        
        # 融合股價和成本的特徵，這裡可以使用串接（也可以考慮加權或注意力）
        fused_features = torch.cat([market_features, asset_features], dim=-1)  # 串接兩個特徵

        return self.dqn(fused_features)

# main program
if __name__ == '__main__':
    # 定義數據
    oil_price = [84.83, 84.84, 85.14, 84.4, 85.53, 87.27, 86.63, 86.29, 84.96, 84.86, 82.82, 80.48, 81.51]
    oil_price2 = [54.83, 64.84, 75.14, 24.4, 55.53, 47.27, 66.63, 86.29, 54.96, 64.86, 92.82, 70.48, 41.51]
    x = torch.FloatTensor(oil_price).unsqueeze(dim=0).unsqueeze(dim=0)  # 增加 channel 和 batch_size
    x1 = torch.FloatTensor(oil_price2).unsqueeze(dim=0).unsqueeze(dim=0)  # 增加 channel 和 batch_size

    # 建立 TCN-DQN 模型並測試
    tcn_dqn_model = TCN_DQN(tcn_price_channels=1, tcn_cost_channels = 1,tcn_layer_channels=[8, 16, 24], tcn_kernel_size=3, dqn_output_channels=3)
    y = tcn_dqn_model(x, x1)
    print(f'y: {y}')