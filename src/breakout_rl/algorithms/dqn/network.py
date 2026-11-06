"""Nature CNN used by DQN."""
import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, stack_size=4, action_count=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(stack_size, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, action_count),
        )

    def forward(self, x):
        return self.network(x.float() / 255.0)
