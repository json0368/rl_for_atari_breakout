"""Evaluate a saved DQN policy."""
from pathlib import Path
import argparse
from ..algorithms.dqn.checkpoints import load_checkpoint
from ..algorithms.dqn.policy import act
from ..infrastructure.runtime import setup
from ..infrastructure.artifacts import write_json
from ..evaluation.runner import evaluate

def parser():
    p = argparse.ArgumentParser(description="Evaluate a saved DQN policy over complete games")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--device", default="auto")
    p.add_argument("--video-dir", type=Path)
    p.add_argument("--output", type=Path)
    return p


def run(args, *, human=False):
    if not 0 <= args.epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    device = setup(args.seed, args.device)
    network, config, checkpoint = load_checkpoint(args.checkpoint, device)
    result = evaluate(lambda obs, rng: act(network, obs, args.epsilon, rng, device),
                      config, episodes=args.episodes, seed=args.seed,
                      render_mode="human" if human else None,
                      video_dir=args.video_dir)
    result.update(epsilon=args.epsilon, checkpoint=str(args.checkpoint.resolve()), training_step=checkpoint["step"],
                  env_config=checkpoint["env_config"])
    output = args.output or args.checkpoint.with_suffix(".evaluation.json")
    write_json(output, result)
    print(f"mean={result['mean_return']:.2f} std={result['std_return']:.2f}; {output}")


def main():
    run(parser().parse_args())

if __name__ == "__main__":
    main()
