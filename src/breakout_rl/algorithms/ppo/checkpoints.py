"""Construct a PPO actor-critic from a policy snapshot."""
import torch
from .network import ActorCritic
from ...envs import EnvConfig


def build_policy(checkpoint, device):
    if checkpoint.get("format_version") != 1 or checkpoint.get("algorithm") != "ppo":
        raise ValueError("Unsupported PPO checkpoint")
    config = EnvConfig(**checkpoint["env_config"])
    agent = ActorCritic(config.stack_size, checkpoint["action_count"]).to(device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()
    return agent, config, checkpoint


def load_checkpoint(path, device):
    return build_policy(torch.load(path, map_location=device, weights_only=True), device)
