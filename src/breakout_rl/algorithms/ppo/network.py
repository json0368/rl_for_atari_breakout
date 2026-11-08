"""CleanRL Atari actor-critic with orthogonal initialization."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, stack_size=4, action_count=4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(stack_size, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(), layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, action_count), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def distribution_and_value(self, obs):
        hidden = self.network(obs.float() / 255.0)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden).squeeze(-1)

    def get_value(self, obs):
        return self.critic(self.network(obs.float() / 255.0)).squeeze(-1)

    def get_action_and_value(self, obs, actions=None):
        distribution, value = self.distribution_and_value(obs)
        if actions is None:
            actions = distribution.sample()
        return actions, distribution.log_prob(actions), distribution.entropy(), value
