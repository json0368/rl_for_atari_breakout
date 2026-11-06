"""Historical full-life protocol needed to evaluate existing saved models."""
import gymnasium as gym
import numpy as np

class AutoFire(gym.Wrapper):
    """Historical protocol retained only for loading existing checkpoints."""
    def __init__(self, env):
        super().__init__(env)
        self.fire = self.unwrapped.get_action_meanings().index("FIRE")
        self.lives = 0
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(10):
            obs, _, terminated, truncated, info = self.env.step(self.fire)
            if not (terminated or truncated):
                self.lives = self.unwrapped.ale.lives()
                return obs, info
            obs, info = self.env.reset()
        raise RuntimeError("Could not start Breakout")
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = self.unwrapped.ale.lives()
        if 0 < lives < self.lives and not (terminated or truncated):
            obs, extra, terminated, truncated, info = self.env.step(self.fire)
            reward += extra
        self.lives = self.unwrapped.ale.lives()
        return obs, reward, terminated, truncated, info


class ClipReward(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))
