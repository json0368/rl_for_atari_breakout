"""Train Double DQN through the common DQN loop."""
from ..dqn.trainer import train as train_dqn
from .optimization import update


def train(args):
    return train_dqn(args, algorithm="ddqn", update_fn=update)
