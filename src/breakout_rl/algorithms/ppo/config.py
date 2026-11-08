"""PPO Atari defaults and experiment settings."""
import argparse
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description="Train Atari PPO with SwanLab logging")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None, help="Optional limit on KL per update")
    p.add_argument("--eval-frequency", type=int, default=100_000, help="0 disables evaluation")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--eval-seed", type=int, default=10_000)
    p.add_argument("--eval-deterministic", action="store_true", help="Use the most probable action for evaluation")
    p.add_argument("--early-stopping-patience", type=int, default=3, help="Evaluations without a new best; 0 disables")
    p.add_argument("--save-frequency", type=int, default=100_000)
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
    for key in ("total_timesteps", "num_envs", "num_steps", "num_minibatches", "update_epochs",
                "eval_episodes", "save_frequency", "torch_threads"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    batch_size = args.num_envs * args.num_steps
    if batch_size % args.num_minibatches or batch_size // args.num_minibatches < 2:
        raise ValueError("rollout batch must divide into minibatches of at least 2 transitions")
    if args.total_timesteps % args.num_envs:
        raise ValueError("total-timesteps must be divisible by num-envs")
    if args.eval_frequency < 0 or args.early_stopping_patience < 0:
        raise ValueError("eval-frequency and early-stopping-patience must be nonnegative")
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError("gamma and gae-lambda must be in [0, 1]")
    if args.learning_rate <= 0 or args.clip_coef <= 0 or args.max_grad_norm <= 0:
        raise ValueError("learning-rate, clip-coef and max-grad-norm must be positive")
    if args.ent_coef < 0 or args.vf_coef < 0 or (args.target_kl is not None and args.target_kl <= 0):
        raise ValueError("Invalid entropy/value coefficient or target-kl")
    if args.run_name and (args.run_name in (".", "..") or Path(args.run_name).name != args.run_name):
        raise ValueError("run-name must be a directory name, not a path")
