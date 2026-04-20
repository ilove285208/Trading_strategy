import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0.2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_channels),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
