"""Atari CNN with value and advantage heads and mean aggregation."""
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, stack_size=4, action_count=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(stack_size, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten(),
        )
        self.value = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, action_count))

    def forward(self, obs):
        hidden = self.features(obs.float() / 255.0)
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
