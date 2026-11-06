"""Shared Atari environment interface."""
from .atari import EnvConfig, make_env, frame_count

__all__ = ["EnvConfig", "make_env", "frame_count"]
