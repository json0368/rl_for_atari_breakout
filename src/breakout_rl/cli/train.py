"""Select and train a Breakout algorithm."""
import argparse


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--algorithm", choices=["dqn", "ppo", "pqn"], default="dqn")
    selected, remaining = selector.parse_known_args()
    if selected.algorithm == "ppo":
        from ..algorithms.ppo.config import parser
        from ..algorithms.ppo.trainer import train
    elif selected.algorithm == "pqn":
        from ..algorithms.pqn.config import parser
        from ..algorithms.pqn.trainer import train
    else:
        from ..algorithms.dqn.config import parser
        from ..algorithms.dqn.trainer import train
    p = parser()
    p.add_argument("--algorithm", choices=["dqn", "ppo", "pqn"], default=selected.algorithm,
                   help="Algorithm to train (default: dqn)")
    args = p.parse_args(remaining)
    train(args)


if __name__ == "__main__":
    main()
