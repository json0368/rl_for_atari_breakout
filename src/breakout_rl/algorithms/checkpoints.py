"""Select the policy constructor declared in a checkpoint."""
import torch


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    algorithm = checkpoint.get("algorithm")
    if algorithm == "dqn":
        from .dqn.checkpoints import build_policy
    elif algorithm == "ppo":
        from .ppo.checkpoints import build_policy
    else:
        raise ValueError(f"Unsupported checkpoint algorithm: {algorithm}")
    return build_policy(checkpoint, device)
