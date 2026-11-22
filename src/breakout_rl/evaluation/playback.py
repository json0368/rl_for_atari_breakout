"""Play a saved policy at Atari speed and publish frames for the desktop viewer."""

from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import perf_counter

import numpy as np

from ..algorithms.checkpoints import load_checkpoint
from ..envs import make_env
from .policy import action_selector


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray
    algorithm: str
    training_step: int
    episode: int
    score: float
    last_score: float | None


class Playback:
    def __init__(self, path, device, seed):
        self._playing = Event()
        self._stop = Event()
        self._lock = Lock()
        self._latest = None
        self._thread = Thread(target=self._run, args=(path, device, seed), name=f"breakout-{path.stem}")
        self._thread.start()

    def set_paused(self, paused):
        if paused:
            self._playing.clear()
        else:
            self._playing.set()

    def close(self):
        self._stop.set()
        self._playing.set()

    def take_latest(self):
        with self._lock:
            latest, self._latest = self._latest, None
            return latest

    def _publish(self, update):
        with self._lock:
            self._latest = update

    def _run(self, path, device, seed):
        try:
            network, config, checkpoint = load_checkpoint(path, device)
            select_action = action_selector(network, checkpoint, device)
            algorithm, training_step = checkpoint["algorithm"], int(checkpoint["step"])
            del checkpoint
            if self._stop.is_set():
                return

            env = make_env(config, render_mode="rgb_array")
            try:
                self._play(env, select_action, seed, algorithm, training_step, config.frame_skip / 60)
            finally:
                env.close()
        except Exception as exc:
            if not self._stop.is_set():
                self._publish(f"{type(exc).__name__}: {exc}")

    def _play(self, env, select_action, seed, algorithm, training_step, interval):
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        episode, score, last_score = 1, 0.0, None
        self._publish(Frame(env.render(), algorithm, training_step, episode, score, last_score))

        while not self._stop.is_set():
            self._playing.wait()
            if self._stop.is_set():
                break

            started = perf_counter()
            obs, reward, terminated, truncated, info = env.step(select_action(obs, rng))
            score += float(reward)
            if "episode" in info:
                last_score = float(info["episode"]["r"])
                obs, _ = env.reset(seed=seed + episode)
                episode += 1
                score = 0.0
            elif terminated or truncated:
                obs, _ = env.reset()

            self._publish(Frame(env.render(), algorithm, training_step, episode, score, last_score))
            self._stop.wait(max(0, interval - (perf_counter() - started)))
