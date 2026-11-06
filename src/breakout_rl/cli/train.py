"""Train a DQN agent."""
from ..algorithms.dqn.config import parser
from ..algorithms.dqn.trainer import train

def main():
    train(parser().parse_args())

if __name__ == "__main__":
    main()
