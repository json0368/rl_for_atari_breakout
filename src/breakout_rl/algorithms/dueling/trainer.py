"""Use Dueling networks in the common DQN training loop."""
from ..dqn.trainer import train as train_dqn
from .network import QNetwork


def train(args):
    return train_dqn(args, algorithm="dueling", network_factory=QNetwork)
