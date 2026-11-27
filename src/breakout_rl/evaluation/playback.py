"""Play a saved policy at configurable speed, track episode/life stats, and record HUD MP4s."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Callable

import numpy as np

from ..algorithms.checkpoints import load_checkpoint
from ..envs import frame_count, make_env
from .policy import ACTION_NAMES, ActionDecision, action_selector, decision_selector
from .recording import EpisodeRecord, LifeLossEvent, Mp4Writer, render_card_frame

_DEFAULT_ACTION_SELECTOR = action_selector


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray | None
    algorithm: str
    training_step: int
    episode: int
    score: float
    last_score: float | None
    action: int | None = None
    action_values: tuple[float, ...] | None = None
    value_kind: str | None = None
    exploratory: bool = False
    step: int = 0
    frame: int = 0
    lives: int | None = None
    max_lives: int = 5
    target_episodes: int | None = None
    completed_episodes: int = 0
    finished: bool = False
    run_label: str = ""


def _safe_frame_count(env) -> int:
    try:
        return int(frame_count(env))
    except Exception:
        return 0


def _safe_lives(env) -> int | None:
    try:
        return int(env.unwrapped.ale.lives())
    except Exception:
        return None


class Playback:
    def __init__(
        self,
        path: Path,
        device,
        seed: int,
        *,
        max_episodes: int | None = None,
        speed_multiplier: float = 1.0,
        record_dir: Path | None = None,
        video_root: Path | None = None,
        epsilon: float = 0.01,
        deterministic: bool = False,
        on_frame: Callable[[Frame], None] | None = None,
        on_episode_end: Callable[[EpisodeRecord], None] | None = None,
        on_complete: Callable[[list[EpisodeRecord]], None] | None = None,
        episode_barrier=None,
    ):
        self.path = Path(path)
        self.seed = int(seed)
        self.max_episodes = max_episodes
        self.record_dir = Path(record_dir) if record_dir is not None else None
        self.video_root = Path(video_root) if video_root is not None else None
        self.epsilon = float(epsilon)
        self.deterministic = bool(deterministic)
        self._on_frame = on_frame
        self._on_episode_end = on_episode_end
        self._on_complete = on_complete
        self._episode_barrier = episode_barrier

        self._playing = Event()
        self._stop = Event()
        self._done = Event()
        self._lock = Lock()
        self._latest = None
        self._speed = float(speed_multiplier)
        self._records: list[EpisodeRecord] = []
        self._algorithm: str = ""
        self._training_step: int = 0
        self._run_label: str = f"{self.path.parent.name} / {self.path.name}" if self.path.parent.name else self.path.name
        self._error: str | None = None

        self._thread = Thread(
            target=self._run,
            args=(self.path, device, self.seed),
            name=f"breakout-{self.path.stem}",
            daemon=True,
        )
        self._thread.start()

    @property
    def records(self) -> list[EpisodeRecord]:
        with self._lock:
            return list(self._records)

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def training_step(self) -> int:
        return self._training_step

    @property
    def run_label(self) -> str:
        return self._run_label

    @property
    def error(self) -> str | None:
        return self._error

    def is_done(self) -> bool:
        return self._done.is_set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._playing.clear()
        else:
            self._playing.set()

    def set_speed(self, speed_multiplier: float) -> None:
        with self._lock:
            self._speed = max(0.0, float(speed_multiplier))

    def _current_speed(self) -> float:
        with self._lock:
            return self._speed

    def close(self) -> None:
        self._stop.set()
        self._playing.set()
        if self._episode_barrier is not None:
            try:
                self._episode_barrier.abort()
            except Exception:
                pass

    def take_latest(self):
        with self._lock:
            latest, self._latest = self._latest, None
            return latest

    def _publish(self, update) -> None:
        with self._lock:
            self._latest = update

    def _build_selector(self, network, checkpoint, device):
        if action_selector is _DEFAULT_ACTION_SELECTOR:
            return decision_selector(
                network,
                checkpoint,
                device,
                epsilon=self.epsilon,
                deterministic=self.deterministic,
            )
        legacy_fn = action_selector(network, checkpoint, device)

        def wrapped(obs, rng):
            out = legacy_fn(obs, rng)
            if isinstance(out, ActionDecision):
                return out
            return ActionDecision(action=int(out))

        return wrapped

    def _run(self, path, device, seed) -> None:
        try:
            network, config, checkpoint = load_checkpoint(path, device)
            select_decision = self._build_selector(network, checkpoint, device)
            algorithm, training_step = checkpoint["algorithm"], int(checkpoint["step"])
            self._algorithm = algorithm
            self._training_step = training_step
            del checkpoint
            if self._stop.is_set():
                return

            env = make_env(config, render_mode="rgb_array")
            try:
                base_fps = 60.0 / max(1, int(getattr(config, "frame_skip", 4)))
                self._play(env, select_decision, seed, algorithm, training_step, 1.0 / base_fps, base_fps)
            finally:
                env.close()
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            if not self._stop.is_set():
                self._publish(self._error)
        finally:
            self._done.set()

    def _open_episode_writer(self, episode: int, ep_seed: int, fps: float) -> tuple[Mp4Writer | None, Path | None]:
        if self.record_dir is None:
            return None, None
        video_path = self.record_dir / f"ep{episode:02d}_seed{ep_seed}.mp4"
        return Mp4Writer(video_path, 420, 532, fps=fps), video_path

    def _play(self, env, select_decision, seed, algorithm, training_step, base_interval, fps) -> None:
        rng = np.random.default_rng(seed)
        episode = 1
        ep_seed = seed
        obs, _ = env.reset(seed=ep_seed)
        score, last_score = 0.0, None
        ep_steps = 0
        ep_start_frames = _safe_frame_count(env)
        initial_lives = _safe_lives(env)
        max_lives = initial_lives if (initial_lives is not None and initial_lives > 0) else 5
        lives = initial_lives
        life_losses: list[LifeLossEvent] = []
        action_counts = {name: 0 for name in ACTION_NAMES}
        exploratory_steps = 0

        writer, video_path = self._open_episode_writer(episode, ep_seed, fps)

        init_frame = Frame(
            rgb=env.render(),
            algorithm=algorithm,
            training_step=training_step,
            episode=episode,
            score=score,
            last_score=last_score,
            step=0,
            frame=0,
            lives=lives,
            max_lives=max_lives,
            target_episodes=self.max_episodes,
            completed_episodes=0,
            finished=False,
            run_label=self._run_label,
        )
        self._publish(init_frame)
        if self._on_frame is not None:
            self._on_frame(init_frame)

        try:
            while not self._stop.is_set():
                self._playing.wait()
                if self._stop.is_set():
                    break

                started = perf_counter()
                decision = select_decision(obs, rng)
                act = int(decision.action)
                obs, reward, terminated, truncated, info = env.step(act)
                ep_steps += 1
                ep_frames = max(0, _safe_frame_count(env) - ep_start_frames)
                score += float(reward)
                if 0 <= act < len(ACTION_NAMES):
                    action_counts[ACTION_NAMES[act]] += 1
                if decision.exploratory:
                    exploratory_steps += 1
                lives = _safe_lives(env)

                episode_finished = "episode" in info
                if episode_finished:
                    final_score = float(info["episode"]["r"])
                    score = final_score
                    if terminated and not truncated:
                        life_losses.append(
                            LifeLossEvent(
                                life_lost_index=len(life_losses) + 1,
                                lives_remaining=0 if lives is None else max(0, lives),
                                step=ep_steps,
                                frame=ep_frames,
                                score=final_score,
                            )
                        )
                elif terminated or truncated:
                    life_losses.append(
                        LifeLossEvent(
                            life_lost_index=len(life_losses) + 1,
                            lives_remaining=max(0, max_lives - len(life_losses) - 1) if lives is None else lives,
                            step=ep_steps,
                            frame=ep_frames,
                            score=score,
                        )
                    )

                rgb = env.render()
                completed_now = len(self._records) + (1 if episode_finished else 0)
                reached_target = bool(episode_finished and self.max_episodes is not None and completed_now >= self.max_episodes)

                cur_frame = Frame(
                    rgb=rgb,
                    algorithm=algorithm,
                    training_step=training_step,
                    episode=episode,
                    score=score,
                    last_score=score if episode_finished else last_score,
                    action=act,
                    action_values=decision.values or None,
                    value_kind=decision.value_kind or None,
                    exploratory=decision.exploratory,
                    step=ep_steps,
                    frame=ep_frames,
                    lives=lives,
                    max_lives=max_lives,
                    target_episodes=self.max_episodes,
                    completed_episodes=completed_now,
                    finished=reached_target,
                    run_label=self._run_label,
                )

                if writer is not None and rgb is not None:
                    writer.write(render_card_frame(cur_frame))

                if episode_finished:
                    rel_video = None
                    if writer is not None:
                        writer.close()
                        writer = None
                    if video_path is not None:
                        if self.video_root is not None:
                            try:
                                rel_video = str(video_path.relative_to(self.video_root).as_posix())
                            except ValueError:
                                rel_video = str(video_path)
                        else:
                            rel_video = str(video_path)

                    record = EpisodeRecord(
                        episode=episode,
                        seed=ep_seed,
                        return_score=score,
                        steps=ep_steps,
                        frames=ep_frames,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        exploratory_steps=exploratory_steps,
                        action_counts=dict(action_counts),
                        life_losses=tuple(life_losses),
                        video=rel_video,
                    )
                    with self._lock:
                        self._records.append(record)

                    self._publish(cur_frame)
                    if self._on_frame is not None:
                        self._on_frame(cur_frame)
                    if self._on_episode_end is not None:
                        self._on_episode_end(record)

                    if reached_target:
                        if self._on_complete is not None:
                            self._on_complete(self.records)
                        break

                    if self._episode_barrier is not None and not self._stop.is_set():
                        try:
                            self._episode_barrier.wait()
                        except Exception:
                            if self._stop.is_set():
                                break

                    last_score = score
                    episode += 1
                    ep_seed = seed + (episode - 1)
                    obs, _ = env.reset(seed=ep_seed)
                    score = 0.0
                    ep_steps = 0
                    ep_start_frames = _safe_frame_count(env)
                    lives = _safe_lives(env)
                    life_losses = []
                    action_counts = {name: 0 for name in ACTION_NAMES}
                    exploratory_steps = 0
                    writer, video_path = self._open_episode_writer(episode, ep_seed, fps)
                else:
                    if terminated or truncated:
                        obs, _ = env.reset()
                    self._publish(cur_frame)
                    if self._on_frame is not None:
                        self._on_frame(cur_frame)

                speed = self._current_speed()
                if speed > 0:
                    delay = (base_interval / speed) - (perf_counter() - started)
                    if delay > 0:
                        self._stop.wait(delay)
        finally:
            if writer is not None:
                writer.close()
