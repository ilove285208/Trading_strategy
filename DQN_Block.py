import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0.2):
        super(DQN, self).__init__()

        # 線性
        self.fc1 = nn.Linear(input_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_channels)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.fc3(x)
        
        return y

