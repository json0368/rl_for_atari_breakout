"""DQN arguments and validation; defaults align with pinned CleanRL."""
import argparse
from pathlib import Path

def parser():
    p = argparse.ArgumentParser(description="Train CleanRL-aligned DQN with SwanLab logging")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=80_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-frequency", type=int, default=4)
    p.add_argument("--target-network-frequency", type=int, default=1000)
    p.add_argument("--max-grad-norm", type=float, default=None, help="Optional deviation: gradient clipping (default disabled)")
    p.add_argument("--start-e", type=float, default=1.0)
    p.add_argument("--end-e", type=float, default=0.01)
    p.add_argument("--exploration-fraction", type=float, default=0.1)
    p.add_argument("--eval-frequency", type=int, default=100_000, help="0 disables evaluation")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--eval-seed", type=int, default=10_000)
    p.add_argument("--eval-epsilon", type=float, default=0.01)
    p.add_argument("--save-frequency", type=int, default=100_000)
    p.add_argument("--log-frequency", type=int, default=1000)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/runs"))
    p.add_argument("--run-name")
    p.add_argument("--swanlab-project", default="atari-breakout")
    p.add_argument("--swanlab-workspace")
    p.add_argument("--swanlab-mode", choices=["online", "local", "offline", "disabled"], default="online")
    p.add_argument("--swanlab-logdir", type=Path, default=Path("outputs/swanlab"))
    p.add_argument("--sticky-action-probability", type=float, default=0.0)
    p.add_argument("--max-episode-frames", type=int, default=108_000)
    return p


def validate(args):
    for key in ("total_timesteps", "buffer_size", "batch_size", "train_frequency",
                "target_network_frequency", "save_frequency", "log_frequency",
                "eval_episodes", "torch_threads"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.learning_starts < 0 or args.eval_frequency < 0:
        raise ValueError("learning-starts and eval-frequency must be nonnegative")
    if args.batch_size > args.buffer_size:
        raise ValueError("batch-size must not exceed buffer-size")
    if not 0 <= args.end_e <= args.start_e <= 1 or not 0 <= args.eval_epsilon <= 1:
        raise ValueError("Invalid epsilon range")
    if not 0 < args.exploration_fraction <= 1 or not 0 <= args.gamma <= 1:
        raise ValueError("Invalid exploration fraction or gamma")
    if args.learning_rate <= 0 or (args.max_grad_norm is not None and args.max_grad_norm <= 0):
        raise ValueError("learning-rate and max-grad-norm must be positive")
    if args.run_name and (args.run_name in (".", "..") or Path(args.run_name).name != args.run_name):
        raise ValueError("run-name must be a directory name, not a path")
