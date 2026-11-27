"""HUD rendering, MP4 recording (Plan A & Plan B), and JSON/CSV evaluation export."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..infrastructure.artifacts import write_json
from .policy import ACTION_LABELS, ACTION_NAMES


ALGORITHMS = {
    "dqn": "DQN",
    "ddqn": "Double DQN",
    "dueling": "Dueling DQN",
    "ppo": "PPO",
    "pqn": "PQN",
}

# Match viewer.py visual theme
BG_RGB = (237, 242, 239)
PANEL_RGB = (255, 255, 255)
SOFT_RGB = (241, 246, 243)
BORDER_RGB = (214, 225, 218)
MATTE_RGB = (228, 235, 230)
TEXT_RGB = (22, 44, 36)
MUTED_RGB = (83, 103, 93)
ACCENT_RGB = (18, 100, 75)
ACCENT_SOFT_RGB = (218, 241, 230)
EXPLORE_BG_RGB = (251, 239, 216)
EXPLORE_FG_RGB = (140, 88, 16)
LIFE_ON_RGB = (204, 68, 75)
LIFE_OFF_RGB = (210, 218, 214)
BAR_BG_RGB = (224, 233, 228)
BAR_IDLE_RGB = (160, 182, 172)


@dataclass(frozen=True)
class LifeLossEvent:
    life_lost_index: int
    lives_remaining: int
    step: int
    frame: int
    score: float


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    seed: int
    return_score: float
    steps: int
    frames: int
    terminated: bool
    truncated: bool
    exploratory_steps: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    life_losses: tuple[LifeLossEvent, ...] = ()
    video: str | None = None

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "seed": self.seed,
            "return": self.return_score,
            "steps": self.steps,
            "frames": self.frames,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "exploratory_steps": self.exploratory_steps,
            "action_counts": dict(self.action_counts),
            "life_losses": [asdict(ev) for ev in self.life_losses],
            "video": self.video,
        }


FONT_CANDIDATES_REGULAR = (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

FONT_CANDIDATES_BOLD = (
    "/mnt/c/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


@lru_cache(maxsize=32)
def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _truncate_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    ellipsis = "…"
    while text and draw.textlength(text + ellipsis, font=font) > max_width:
        text = text[:-1]
    return text + ellipsis


def render_card_frame(
    frame,
    *,
    card_width: int = 420,
    card_height: int = 532,
    status_override: str | None = None,
) -> np.ndarray:
    """Render a single model card with Atari screen, score, lives, action, and Q/prob bars."""
    img = Image.new("RGB", (card_width, card_height), PANEL_RGB)
    draw = ImageDraw.Draw(img)

    # Outer border
    draw.rectangle((0, 0, card_width - 1, card_height - 1), outline=BORDER_RGB, width=1)

    f_title = _load_font(20, bold=True)
    f_sub = _load_font(12, bold=False)
    f_badge = _load_font(11, bold=True)
    f_score = _load_font(19, bold=True)
    f_meta = _load_font(12, bold=False)
    f_act = _load_font(12, bold=True)
    f_small = _load_font(11, bold=False)

    alg_name = ALGORITHMS.get(frame.algorithm, frame.algorithm.upper())
    draw.text((14, 10), alg_name, fill=TEXT_RGB, font=f_title)

    badge_text = status_override or ("已完成" if frame.finished else "运行中")
    badge_bg = SOFT_RGB if frame.finished else ACCENT_SOFT_RGB
    badge_fg = MUTED_RGB if frame.finished else ACCENT_RGB
    bw = int(draw.textlength(badge_text, font=f_badge)) + 16
    bx0 = card_width - 14 - bw
    draw.rounded_rectangle((bx0, 12, card_width - 14, 34), radius=4, fill=badge_bg)
    draw.text((bx0 + 8, 15), badge_text, fill=badge_fg, font=f_badge)

    sub_text = frame.run_label or f"{frame.training_step:,} 步"
    if frame.run_label and frame.training_step:
        sub_text = f"{frame.run_label}  ·  {frame.training_step:,} 步"
    sub_text = _truncate_text(draw, sub_text, f_sub, card_width - 28)
    draw.text((14, 38), sub_text, fill=MUTED_RGB, font=f_sub)

    # Game display (4:3 crisp nearest-neighbor)
    pad_x = 14
    screen_w = card_width - 2 * pad_x
    screen_h = 288
    sy0 = 60
    draw.rectangle((pad_x, sy0, pad_x + screen_w, sy0 + screen_h), fill=MATTE_RGB)
    if frame.rgb is not None and frame.rgb.size > 0:
        fit_w = min(screen_w, screen_h * 4 // 3)
        fit_h = max(1, fit_w * 3 // 4)
        resized = cv2.resize(frame.rgb, (fit_w, fit_h), interpolation=cv2.INTER_NEAREST)
        gx0 = pad_x + (screen_w - fit_w) // 2
        gy0 = sy0 + (screen_h - fit_h) // 2
        img.paste(Image.fromarray(resized), (gx0, gy0))

    # Score + episode + last score strip
    fy0 = sy0 + screen_h + 8
    fh = 42
    draw.rectangle((pad_x, fy0, card_width - pad_x, fy0 + fh), fill=SOFT_RGB)
    ep_text = (
        f"第 {frame.episode}/{frame.target_episodes} 局"
        if frame.target_episodes
        else f"第 {frame.episode} 局"
    )
    draw.text((pad_x + 10, fy0 + 13), ep_text, fill=MUTED_RGB, font=f_meta)
    ep_w = int(draw.textlength(ep_text, font=f_meta))
    draw.text((pad_x + 20 + ep_w, fy0 + 8), f"{frame.score:g} 分", fill=ACCENT_RGB, font=f_score)

    last_text = "上一局 —" if frame.last_score is None else f"上一局 {frame.last_score:g} 分"
    lw = int(draw.textlength(last_text, font=f_meta))
    draw.text((card_width - pad_x - 10 - lw, fy0 + 13), last_text, fill=MUTED_RGB, font=f_meta)

    # Lives + step + current action line
    ay0 = fy0 + fh + 8
    act_idx = frame.action
    if act_idx is not None and 0 <= act_idx < len(ACTION_NAMES):
        act_str = f"动作: {ACTION_LABELS[act_idx]} ({ACTION_NAMES[act_idx]})"
    else:
        act_str = "动作: —"
    draw.text((pad_x + 2, ay0), act_str, fill=TEXT_RGB, font=f_act)

    if frame.exploratory:
        ex_x = pad_x + int(draw.textlength(act_str, font=f_act)) + 10
        draw.rounded_rectangle((ex_x, ay0 - 1, ex_x + 36, ay0 + 18), radius=3, fill=EXPLORE_BG_RGB)
        draw.text((ex_x + 6, ay0), "探索", fill=EXPLORE_FG_RGB, font=f_small)

    # Lives dots + steps on right
    step_str = f"步数 {frame.step:,}"
    sw = int(draw.textlength(step_str, font=f_small))
    draw.text((card_width - pad_x - sw, ay0 + 1), step_str, fill=MUTED_RGB, font=f_small)

    if frame.lives is not None:
        max_lives = max(1, int(frame.max_lives))
        lives = max(0, min(max_lives, int(frame.lives)))
        dot_gap = 11
        dots_total = max_lives * dot_gap
        dx_start = card_width - pad_x - sw - 12 - dots_total
        life_lbl = "生命"
        llw = int(draw.textlength(life_lbl, font=f_small))
        draw.text((dx_start - llw - 5, ay0 + 1), life_lbl, fill=MUTED_RGB, font=f_small)
        for i in range(max_lives):
            cx = dx_start + i * dot_gap + 4
            cy = ay0 + 8
            col = LIFE_ON_RGB if i < lives else LIFE_OFF_RGB
            draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=col)

    # Action Q-value / probability bars (4 compact horizontal bars, 2x2 grid)
    by0 = ay0 + 25
    values = frame.action_values or ()
    kind = frame.value_kind or ""
    if kind == "prob" and len(values) == 4:
        ratios = [max(0.0, min(1.0, float(v))) for v in values]
        val_texts = [f"{v * 100:4.1f}%" for v in values]
    elif len(values) == 4:
        vmin, vmax = min(values), max(values)
        span = vmax - vmin
        ratios = [0.15 + 0.85 * ((v - vmin) / span) if span > 1e-6 else 0.5 for v in values]
        val_texts = [f"{v:5.2f}" for v in values]
    else:
        ratios = [0.0, 0.0, 0.0, 0.0]
        val_texts = ["—", "—", "—", "—"]

    col_gap = 12
    col_w = (card_width - 2 * pad_x - col_gap) // 2
    row_h = 26
    for idx in range(4):
        row, col = divmod(idx, 2)
        x0 = pad_x + col * (col_w + col_gap)
        y0 = by0 + row * row_h
        is_chosen = act_idx == idx
        lbl_col = ACCENT_RGB if is_chosen else MUTED_RGB
        draw.text((x0, y0 + 2), ACTION_LABELS[idx], fill=lbl_col, font=f_act if is_chosen else f_small)
        bar_x0 = x0 + 32
        bar_x1 = x0 + col_w - 44
        draw.rounded_rectangle((bar_x0, y0 + 5, bar_x1, y0 + 17), radius=3, fill=BAR_BG_RGB)
        fill_w = int(round((bar_x1 - bar_x0) * ratios[idx]))
        if fill_w > 0:
            fill_col = ACCENT_RGB if is_chosen else BAR_IDLE_RGB
            draw.rounded_rectangle((bar_x0, y0 + 5, bar_x0 + max(3, fill_w), y0 + 17), radius=3, fill=fill_col)
        vt = val_texts[idx]
        vw = int(draw.textlength(vt, font=f_small))
        draw.text((x0 + col_w - vw, y0 + 2), vt, fill=TEXT_RGB if is_chosen else MUTED_RGB, font=f_small)

    # Subtle bottom caption for value kind
    kind_hint = "策略动作概率分布 (PPO)" if kind == "prob" else ("动作估计价值 Q(s, a)" if kind == "q" else "")
    if kind_hint:
        draw.text((pad_x, card_height - 20), kind_hint, fill=MUTED_RGB, font=_load_font(10))

    return np.asarray(img, dtype=np.uint8)


def render_grid_frame(
    frames: Sequence,
    *,
    header_title: str = "Breakout 多模型统一评估对比",
    header_subtitle: str = "",
    card_width: int = 400,
    card_height: int = 516,
    status_overrides: Sequence[str | None] | None = None,
) -> np.ndarray:
    """Compose multiple model cards into one balanced side-by-side grid frame (Plan B)."""
    n = len(frames)
    if n == 0:
        raise ValueError("frames must be non-empty")

    columns = 3 if n in (5, 6) else min(4, n)
    rows = (n + columns - 1) // columns
    gap = 14
    pad = 18
    header_h = 56

    grid_w = columns * card_width + (columns - 1) * gap + 2 * pad
    grid_h = header_h + rows * card_height + (rows - 1) * gap + 2 * pad
    # Ensure even dimensions for H.264 yuv420p compatibility
    grid_w += grid_w % 2
    grid_h += grid_h % 2

    canvas = Image.new("RGB", (grid_w, grid_h), BG_RGB)
    draw = ImageDraw.Draw(canvas)

    # Top header banner
    draw.rectangle((0, 0, grid_w, header_h), fill=PANEL_RGB)
    draw.line((0, header_h, grid_w, header_h), fill=BORDER_RGB, width=1)
    f_h = _load_font(20, bold=True)
    f_s = _load_font(12, bold=False)
    draw.text((pad, 14), header_title, fill=TEXT_RGB, font=f_h)
    if header_subtitle:
        sw = int(draw.textlength(header_subtitle, font=f_s))
        draw.text((grid_w - pad - sw, 20), header_subtitle, fill=MUTED_RGB, font=f_s)

    for idx, frame in enumerate(frames):
        row, col = divmod(idx, columns)
        row_items = min(columns, n - row * columns)
        row_w = row_items * card_width + (row_items - 1) * gap
        x0 = (grid_w - row_w) // 2 + col * (card_width + gap)
        y0 = header_h + pad + row * (card_height + gap)
        override = status_overrides[idx] if status_overrides else None
        card_arr = render_card_frame(
            frame,
            card_width=card_width,
            card_height=card_height,
            status_override=override,
        )
        canvas.paste(Image.fromarray(card_arr), (x0, y0))

    return np.asarray(canvas, dtype=np.uint8)


class Mp4Writer:
    """Streaming H.264 MP4 writer using imageio-ffmpeg with OpenCV fallback."""

    def __init__(self, path: Path, width: int, height: int, *, fps: float = 15.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self.fps = float(fps)
        self._gen = None
        self._cv_writer = None
        self._closed = False
        self._open()

    def _open(self):
        try:
            import imageio_ffmpeg

            self._gen = imageio_ffmpeg.write_frames(
                str(self.path),
                size=(self.width, self.height),
                fps=self.fps,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                quality=7,
                macro_block_size=2,
            )
            self._gen.send(None)
        except Exception:
            self._gen = None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._cv_writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.width, self.height))

    def write(self, rgb: np.ndarray) -> None:
        if self._closed:
            return
        h, w = rgb.shape[:2]
        if w != self.width or h != self.height:
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if self._gen is not None:
            self._gen.send(rgb.tobytes())
        elif self._cv_writer is not None:
            self._cv_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._gen is not None:
            try:
                self._gen.close()
            except Exception:
                pass
            self._gen = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def summarize_model_records(
    *,
    algorithm: str,
    run_label: str,
    checkpoint: Path | str,
    training_step: int,
    records: Sequence[EpisodeRecord],
) -> dict:
    returns = [r.return_score for r in records]
    steps = [r.steps for r in records]
    frames = [r.frames for r in records]
    low_lt300 = sum(1 for v in returns if v < 300.0)
    return {
        "algorithm": algorithm,
        "algorithm_name": ALGORITHMS.get(algorithm, algorithm),
        "run_label": run_label,
        "checkpoint": str(checkpoint),
        "training_step": int(training_step),
        "episodes_completed": len(records),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "median_return": float(np.median(returns)) if returns else 0.0,
        "std_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "min_return": float(min(returns)) if returns else 0.0,
        "max_return": float(max(returns)) if returns else 0.0,
        "low_score_lt300_count": low_lt300,
        "low_score_lt300_rate": (low_lt300 / len(returns)) if returns else 0.0,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "mean_frames": float(np.mean(frames)) if frames else 0.0,
        "episodes": [r.to_dict() for r in records],
    }


def export_evaluation_results(
    output_dir: Path,
    model_summaries: Sequence[dict],
    *,
    base_seed: int,
    target_episodes: int,
    epsilon: float = 0.01,
    deterministic: bool = False,
    comparison_videos: Sequence[str] = (),
) -> dict[str, Path]:
    """Export unified evaluation results to summary.json, summary.csv, episodes.csv, and life_losses.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_seed": int(base_seed),
        "target_episodes": int(target_episodes),
        "epsilon": float(epsilon),
        "ppo_deterministic": bool(deterministic),
        "comparison_videos": list(comparison_videos),
        "models": list(model_summaries),
    }

    summary_json = output_dir / "summary.json"
    write_json(summary_json, payload)

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "algorithm_name", "run_label", "training_step",
            "episodes", "mean_return", "median_return", "std_return",
            "min_return", "max_return", "low_lt300_count", "low_lt300_rate",
            "mean_steps", "mean_frames", "checkpoint",
        ])
        for m in model_summaries:
            writer.writerow([
                m["algorithm"], m["algorithm_name"], m["run_label"], m["training_step"],
                m["episodes_completed"], f"{m['mean_return']:.2f}", f"{m['median_return']:.2f}",
                f"{m['std_return']:.2f}", f"{m['min_return']:g}", f"{m['max_return']:g}",
                m["low_score_lt300_count"], f"{m['low_score_lt300_rate']:.3f}",
                f"{m['mean_steps']:.1f}", f"{m['mean_frames']:.1f}", m["checkpoint"],
            ])

    episodes_csv = output_dir / "episodes.csv"
    with episodes_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "algorithm_name", "run_label", "episode", "seed",
            "return", "steps", "frames", "terminated", "truncated",
            "exploratory_steps", "action_noop", "action_fire", "action_right", "action_left",
            "life_loss_count", "life_losses_summary", "video",
        ])
        for m in model_summaries:
            for ep in m["episodes"]:
                counts = ep.get("action_counts", {})
                losses = ep.get("life_losses", [])
                loss_summary = "; ".join(
                    f"L{ev['life_lost_index']}@step{ev['step']}/frame{ev['frame']}/score{ev['score']:g}"
                    for ev in losses
                )
                writer.writerow([
                    m["algorithm"], m["algorithm_name"], m["run_label"],
                    ep["episode"], ep["seed"], f"{ep['return']:g}",
                    ep["steps"], ep["frames"], ep["terminated"], ep["truncated"],
                    ep.get("exploratory_steps", 0),
                    counts.get("NOOP", 0), counts.get("FIRE", 0),
                    counts.get("RIGHT", 0), counts.get("LEFT", 0),
                    len(losses), loss_summary, ep.get("video") or "",
                ])

    life_losses_csv = output_dir / "life_losses.csv"
    with life_losses_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "algorithm_name", "run_label", "episode", "seed",
            "life_lost_index", "lives_remaining", "step", "frame",
            "score_at_loss", "final_return",
        ])
        for m in model_summaries:
            for ep in m["episodes"]:
                for ev in ep.get("life_losses", []):
                    writer.writerow([
                        m["algorithm"], m["algorithm_name"], m["run_label"],
                        ep["episode"], ep["seed"],
                        ev["life_lost_index"], ev["lives_remaining"],
                        ev["step"], ev["frame"], f"{ev['score']:g}", f"{ep['return']:g}",
                    ])

    return {
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "episodes_csv": episodes_csv,
        "life_losses_csv": life_losses_csv,
    }
