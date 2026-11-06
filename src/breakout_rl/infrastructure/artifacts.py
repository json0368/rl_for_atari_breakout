"""JSON output and portable policy snapshots."""
import json
from dataclasses import asdict
from pathlib import Path
import torch

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(path, network, optimizer, env_config, train_config, step, frames, *, algorithm="dqn", action_count=4):
    """Policy snapshot, not an exact training-resume snapshot (no replay/RNG)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "format_version": 1, "algorithm": algorithm, "model": network.state_dict(),
        "optimizer": optimizer.state_dict(), "env_config": asdict(env_config),
        "train_config": train_config, "step": step, "frames": frames,
        "action_count": action_count,
    }, temporary)
    temporary.replace(path)
