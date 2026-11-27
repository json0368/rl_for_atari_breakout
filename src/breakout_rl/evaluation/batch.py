"""Coordinated multi-model unified evaluation with Plan A + Plan B recording and export."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Callable, Sequence

from .playback import Frame, Playback
from .recording import (
    EpisodeRecord,
    Mp4Writer,
    export_evaluation_results,
    render_grid_frame,
    summarize_model_records,
)


def _slug(path: Path, index: int, used: set[str]) -> str:
    base = path.parent.name or path.stem or f"model_{index + 1}"
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in base).strip("_") or f"model_{index + 1}"
    name = safe
    suffix = 2
    while name in used:
        name = f"{safe}_{suffix}"
        suffix += 1
    used.add(name)
    return name


class BatchEvaluationSession:
    """Run N models on identical seeds [base_seed .. base_seed+episodes-1], record A+B MP4s, and export JSON/CSV."""

    def __init__(
        self,
        checkpoints: Sequence[Path],
        *,
        device,
        output_dir: Path | None = None,
        base_seed: int = 30_000,
        episodes: int = 10,
        speed_multiplier: float = 0.0,
        record_individual: bool = True,
        record_comparison: bool = True,
        epsilon: float = 0.01,
        deterministic: bool = False,
        fps: float = 15.0,
        on_finish: Callable[[dict], None] | None = None,
    ):
        if not checkpoints:
            raise ValueError("checkpoints must be non-empty")
        if episodes < 1:
            raise ValueError("episodes must be positive")

        self.checkpoints = [Path(p) for p in checkpoints]
        self.device = device
        self.base_seed = int(base_seed)
        self.episodes = int(episodes)
        self.record_individual = bool(record_individual)
        self.record_comparison = bool(record_comparison)
        self.epsilon = float(epsilon)
        self.deterministic = bool(deterministic)
        self.fps = float(fps)
        self._on_finish = on_finish

        if output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = Path.cwd() / "outputs" / "evaluations" / f"viewer-eval-{stamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._stop = Event()
        self._finished = Event()
        self._result: dict | None = None
        self._comparison_videos: list[str] = []
        self._slugs: list[str] = []
        self._queues: list[Queue] = []
        self.playbacks: list[Playback] = []

        used_slugs: set[str] = set()
        for idx, ckpt in enumerate(self.checkpoints):
            slug = _slug(ckpt, idx, used_slugs)
            self._slugs.append(slug)
            q: Queue = Queue(maxsize=6)
            self._queues.append(q)

            rec_dir = (self.output_dir / slug / "videos") if self.record_individual else None

            def make_on_frame(queue_ref: Queue):
                def _push(frame: Frame):
                    if not self.record_comparison:
                        return
                    while not self._stop.is_set():
                        try:
                            queue_ref.put(("frame", frame), timeout=0.05)
                            return
                        except Full:
                            continue
                return _push

            def make_on_ep_end(queue_ref: Queue):
                def _end(rec: EpisodeRecord):
                    if not self.record_comparison:
                        return
                    while not self._stop.is_set():
                        try:
                            queue_ref.put(("ep_end", rec), timeout=0.05)
                            return
                        except Full:
                            continue
                return _end

            def make_on_complete(queue_ref: Queue):
                def _done(_records):
                    if not self.record_comparison:
                        return
                    while not self._stop.is_set():
                        try:
                            queue_ref.put(("done", None), timeout=0.05)
                            return
                        except Full:
                            continue
                return _done

            pb = Playback(
                ckpt,
                device,
                self.base_seed,
                max_episodes=self.episodes,
                speed_multiplier=speed_multiplier,
                record_dir=rec_dir,
                video_root=self.output_dir,
                epsilon=self.epsilon,
                deterministic=self.deterministic,
                on_frame=make_on_frame(q),
                on_episode_end=make_on_ep_end(q),
                on_complete=make_on_complete(q),
            )
            self.playbacks.append(pb)

        self._composer_thread: Thread | None = None
        if self.record_comparison:
            self._composer_thread = Thread(
                target=self._compose_comparison_videos,
                name="breakout-comparison-composer",
                daemon=True,
            )
            self._composer_thread.start()

        self._monitor_thread = Thread(
            target=self._wait_and_export,
            name="breakout-eval-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def start(self) -> None:
        for pb in self.playbacks:
            pb.set_paused(False)

    def pause(self) -> None:
        for pb in self.playbacks:
            pb.set_paused(True)

    def set_speed(self, speed_multiplier: float) -> None:
        for pb in self.playbacks:
            pb.set_speed(speed_multiplier)

    def stop(self) -> None:
        self._stop.set()
        for pb in self.playbacks:
            pb.close()

    def wait(self, timeout: float | None = None) -> dict | None:
        self._finished.wait(timeout=timeout)
        return self._result

    @property
    def result(self) -> dict | None:
        return self._result

    def _compose_comparison_videos(self) -> None:
        """Consume frames episode-by-episode in lockstep across all models to write Plan B MP4s."""
        n = len(self.playbacks)
        comp_dir = self.output_dir / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)

        # Pull initial frames (episode 1 reset frames)
        latest: list[Frame | None] = [None] * n
        for i in range(n):
            while not self._stop.is_set() and latest[i] is None:
                if self.playbacks[i].is_done() and self._queues[i].empty():
                    break
                try:
                    kind, payload = self._queues[i].get(timeout=0.05)
                    if kind == "frame":
                        latest[i] = payload
                except Empty:
                    continue

        if self._stop.is_set() or any(f is None for f in latest):
            return

        for ep in range(1, self.episodes + 1):
            if self._stop.is_set():
                break
            ep_seed = self.base_seed + (ep - 1)
            video_path = comp_dir / f"comparison_ep{ep:02d}_seed{ep_seed}.mp4"
            ep_done = [False] * n
            ep_scores: list[float | None] = [None] * n
            writer: Mp4Writer | None = None

            try:
                while not self._stop.is_set() and not all(ep_done):
                    progressed = False
                    for i in range(n):
                        if ep_done[i]:
                            continue
                        if self.playbacks[i].is_done() and self._queues[i].empty():
                            ep_done[i] = True
                            continue
                        try:
                            kind, payload = self._queues[i].get(timeout=0.05)
                        except Empty:
                            continue
                        progressed = True
                        if kind == "frame":
                            latest[i] = payload
                        elif kind == "ep_end":
                            ep_done[i] = True
                            ep_scores[i] = payload.return_score
                        elif kind == "done":
                            ep_done[i] = True

                    if not progressed:
                        if all(pb.is_done() for pb in self.playbacks):
                            break
                        continue

                    overrides = [
                        f"本局结束 {ep_scores[i]:g} 分" if ep_done[i] and ep_scores[i] is not None else None
                        for i in range(n)
                    ]
                    grid = render_grid_frame(
                        latest,
                        header_title="Breakout 多模型统一评估对比",
                        header_subtitle=f"第 {ep}/{self.episodes} 局  ·  种子 {ep_seed}",
                        status_overrides=overrides,
                    )
                    if writer is None:
                        writer = Mp4Writer(video_path, grid.shape[1], grid.shape[0], fps=self.fps)
                    writer.write(grid)

                # Hold final frame briefly (~1.5s) at end of episode so viewer can read final scores
                if writer is not None and not self._stop.is_set():
                    overrides = [
                        f"本局结束 {ep_scores[i]:g} 分" if ep_scores[i] is not None else "本局结束"
                        for i in range(n)
                    ]
                    final_grid = render_grid_frame(
                        latest,
                        header_title="Breakout 多模型统一评估对比",
                        header_subtitle=f"第 {ep}/{self.episodes} 局  ·  种子 {ep_seed}  ·  全部完赛",
                        status_overrides=overrides,
                    )
                    for _ in range(max(1, int(round(self.fps * 1.5)))):
                        writer.write(final_grid)
                    rel = str(video_path.relative_to(self.output_dir).as_posix())
                    self._comparison_videos.append(rel)
            finally:
                if writer is not None:
                    writer.close()

    def _wait_and_export(self) -> None:
        for pb in self.playbacks:
            pb.join()
        if self._composer_thread is not None:
            self._composer_thread.join(timeout=30.0)

        summaries = []
        for pb in self.playbacks:
            summaries.append(
                summarize_model_records(
                    algorithm=pb.algorithm or "unknown",
                    run_label=pb.run_label,
                    checkpoint=pb.path.resolve(),
                    training_step=pb.training_step,
                    records=pb.records,
                )
            )

        files = export_evaluation_results(
            self.output_dir,
            summaries,
            base_seed=self.base_seed,
            target_episodes=self.episodes,
            epsilon=self.epsilon,
            deterministic=self.deterministic,
            comparison_videos=self._comparison_videos,
        )
        self._result = {
            "output_dir": str(self.output_dir),
            "files": {k: str(v) for k, v in files.items()},
            "comparison_videos": list(self._comparison_videos),
            "models": summaries,
            "aborted": self._stop.is_set(),
        }
        self._finished.set()
        if self._on_finish is not None:
            self._on_finish(self._result)
