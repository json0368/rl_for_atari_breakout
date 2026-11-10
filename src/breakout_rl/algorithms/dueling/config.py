"""Dueling DQN shares the DQN experimental settings."""
from ..dqn.config import parser as dqn_parser, validate


def parser():
    p = dqn_parser()
    p.description = "Train Dueling DQN with SwanLab logging"
    p.set_defaults(algorithm="dueling")
    return p
