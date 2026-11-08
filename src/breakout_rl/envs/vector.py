"""Synchronous Atari environments with same-step autoreset."""
import gymnasium as gym
import numpy as np
from .atari import make_env, frame_count


def make_vector_env(config, num_envs):
    return gym.vector.SyncVectorEnv(
        [lambda: make_env(config, training=True) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def total_frames(envs):
    return sum(frame_count(env) for env in envs.envs)


def episode_records(infos):
    # Gymnasium 1.x nests terminal episode stats inside a dict of arrays.
    for source in (infos, infos.get("final_info", {})):
        if "episode" not in source:
            continue
        mask = source["_episode"]
        episodes = source["episode"]
        for index in np.flatnonzero(mask):
            yield float(episodes["r"][index]), int(episodes["l"][index])


def timeout_observations(infos, terminated, truncated):
    indices = np.flatnonzero(truncated & ~terminated)
    if len(indices):
        return indices, np.stack([infos["final_obs"][i] for i in indices])
    return indices, None
