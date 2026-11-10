"""Double DQN uses the same experimental settings as DQN."""
from ..dqn.config import parser as dqn_parser, validate


def parser():
    p = dqn_parser()
    p.description = "Train Double DQN with SwanLab logging"
    p.set_defaults(algorithm="ddqn")
    return p
