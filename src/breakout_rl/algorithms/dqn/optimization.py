"""DQN Bellman targets, MSE updates, and exploration schedule."""
import torch
from torch import nn
from torch.nn import functional as F

def td_target(rewards, terminated, next_q, gamma):
    # Time limits bootstrap; game-over and artificial life termination suppress bootstrap.
    return rewards + gamma * (~terminated).float() * next_q


def update(network, target, optimizer, batch, gamma, max_grad_norm=None, *, next_value_fn=None):
    obs, next_obs, actions, rewards, terminated = batch
    with torch.no_grad():
        next_q = (target(next_obs).max(1).values if next_value_fn is None
                  else next_value_fn(network, target, next_obs))
        expected = td_target(rewards, terminated, next_q, gamma)
    values = network(obs).gather(1, actions[:, None]).squeeze(1)
    loss = F.mse_loss(values, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in network.parameters() if p.grad is not None]))
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
    optimizer.step()
    return {"losses/td_loss": loss.item(), "losses/q_mean": values.mean().item(),
            "losses/grad_norm": float(grad_norm)}


def epsilon_at(step, start, end, duration):
    return start + min(step / max(duration, 1), 1.0) * (end - start)
