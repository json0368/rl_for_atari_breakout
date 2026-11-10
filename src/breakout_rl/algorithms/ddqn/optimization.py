"""Online action selection and target-network action evaluation."""
import torch
from ..dqn.optimization import update as dqn_update


@torch.no_grad()
def next_q_values(network, target, next_obs):
    actions = network(next_obs).argmax(dim=1, keepdim=True)
    return target(next_obs).gather(1, actions).squeeze(1)


def update(network, target, optimizer, batch, gamma, max_grad_norm=None):
    return dqn_update(network, target, optimizer, batch, gamma, max_grad_norm,
                      next_value_fn=next_q_values)
