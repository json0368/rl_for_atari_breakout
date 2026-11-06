"""Construct DQN policies from project checkpoint files."""
import torch
from .network import QNetwork
from ...envs import EnvConfig

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("algorithm") != "dqn":
        raise ValueError("Unsupported checkpoint format or algorithm")
    saved_env = dict(checkpoint["env_config"])
    saved_env.setdefault("protocol", "legacy")
    config = EnvConfig(**saved_env)
    network = QNetwork(config.stack_size, checkpoint["action_count"]).to(device)
    network.load_state_dict(checkpoint["model"])
    network.eval()
    return network, config, checkpoint
