"""CleanRL PQN CNN with LayerNorm and orthogonal initialization."""
import math
import torch
from torch import nn


def layer_init(layer):
    nn.init.orthogonal_(layer.weight, math.sqrt(2))
    nn.init.constant_(layer.bias, 0.0)
    return layer


class QNetwork(nn.Module):
    def __init__(self, stack_size=4, action_count=4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(stack_size, 32, 8, stride=4)),
            nn.LayerNorm([32,20,20]), nn.ReLU(),
            layer_init(nn.Conv2d(32,64,4,stride=2)),
            nn.LayerNorm([64,9,9]), nn.ReLU(),
            layer_init(nn.Conv2d(64,64,3,stride=1)),
            nn.LayerNorm([64,7,7]), nn.ReLU(), nn.Flatten(),
            layer_init(nn.Linear(3136,512)), nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512,action_count)),
        )

    def forward(self, obs):
        return self.network(obs.float() / 255.0)
