"""Frozen Q(lambda) rollout targets and minibatch MSE updates."""
import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def q_lambda_returns(rewards, values, boundaries, next_value, gamma, q_lambda):
    # values[t] = max_a Q(s_t, a), recorded BEFORE any rollout updates.
    # Timeout rewards include gamma * max Q(final_obs); boundaries cut recursion.
    returns = torch.empty_like(rewards)
    following_return = next_value
    for t in reversed(range(len(rewards))):
        following_value = next_value if t == len(rewards) - 1 else values[t + 1]
        target = q_lambda * following_return + (1 - q_lambda) * following_value
        following_return = rewards[t] + gamma * target * (~boundaries[t]).float()
        returns[t] = following_return
    return returns


def update(network, optimizer, rollout, args):
    observations, actions, returns = rollout
    returns = returns.detach()
    indices = np.arange(len(actions))
    for _ in range(args.update_epochs):
        np.random.shuffle(indices)
        for ids in np.array_split(indices, min(args.num_minibatches, len(indices))):
            values = network(observations[ids]).gather(1, actions[ids, None]).squeeze(-1)
            loss = F.mse_loss(values, returns[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
            optimizer.step()
    return {"losses/td_loss": loss.item(), "losses/q_mean": values.mean().item(),
            "losses/grad_norm": float(grad_norm)}
