"""Shared Atari protocol with pinned CleanRL training wrappers."""
from dataclasses import dataclass
import ale_py
import gymnasium as gym
import numpy as np
from ..vendor.cleanrl.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv

from .legacy import AutoFire, ClipReward

gym.register_envs(ale_py)

@dataclass
class EnvConfig:
    env_id: str = "BreakoutNoFrameskip-v4"
    frame_skip: int = 4
    screen_size: int = 84
    stack_size: int = 4
    noop_max: int = 30
    repeat_action_probability: float = 0.0
    max_episode_frames: int = 108_000
    protocol: str = "cleanrl"

    def __post_init__(self):
        if self.env_id not in ("ALE/Breakout-v5", "BreakoutNoFrameskip-v4"):
            raise ValueError("Only Breakout environments are supported")
        if self.screen_size != 84 or self.stack_size < 1:
            raise ValueError("DQN requires 84x84 frames and a positive stack_size")
        if self.frame_skip < 1 or self.noop_max < 1 or self.max_episode_frames < 100:
            raise ValueError("Invalid frame skip, noop count, or episode limit")
        if not 0 <= self.repeat_action_probability <= 1:
            raise ValueError("repeat_action_probability must be in [0, 1]")
        if self.protocol not in ("cleanrl", "legacy"):
            raise ValueError("Unsupported environment protocol")

class FrameCounter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_frames = 0
    def step(self, action):
        before = self.unwrapped.ale.getFrameNumber()
        result = self.env.step(action)
        self.total_frames += self.unwrapped.ale.getFrameNumber() - before
        return result



def make_env(config: EnvConfig, *, training=False, render_mode=None, video_dir=None):
    if video_dir is not None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Video recording requires rgb_array")
        render_mode = "rgb_array"
    env = gym.make(config.env_id, frameskip=1, render_mode=render_mode,
                   repeat_action_probability=config.repeat_action_probability,
                   full_action_space=False, max_episode_steps=config.max_episode_frames)
    env = FrameCounter(env)
    if video_dir is not None:
        env = gym.wrappers.RecordVideo(env, str(video_dir), episode_trigger=lambda _: True, disable_logger=True)
    if config.protocol == "legacy":
        env = gym.wrappers.AtariPreprocessing(env, noop_max=config.noop_max,
            frame_skip=config.frame_skip, screen_size=config.screen_size,
            terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False)
        env = AutoFire(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if training:
            env = ClipReward(env)
    else:
        # Track real full-game scores BEFORE the artificial life-loss boundary.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=config.noop_max)
        env = MaxAndSkipEnv(env, skip=config.frame_skip)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        if training:
            env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (config.screen_size, config.screen_size))
        env = gym.wrappers.GrayscaleObservation(env)
    return gym.wrappers.FrameStackObservation(env, config.stack_size)

def frame_count(env):
    return int(env.get_wrapper_attr("total_frames"))
