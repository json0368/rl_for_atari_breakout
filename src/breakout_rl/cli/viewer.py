"""Desktop viewer and unified batch evaluator for multiple Breakout policies."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ..evaluation.batch import BatchEvaluationSession
from ..evaluation.playback import Frame, Playback
from ..evaluation.policy import ACTION_LABELS, ACTION_NAMES
from ..evaluation.recording import ALGORITHMS
from ..infrastructure.runtime import setup


BACKGROUND = "#edf2ef"
PANEL = "#ffffff"
SOFT = "#f1f6f3"
SUBBAR = "#f6faf8"
BORDER = "#d6e1da"
MATTE = "#e4ebe6"
TEXT = "#162c24"
MUTED = "#53675d"
ACCENT = "#12644b"
ACCENT_SOFT = "#daf1e6"
WARN_BG = "#fbefd8"
WARN_FG = "#785319"
ERROR = "#952c36"
LIFE_ON = "#cc444b"
LIFE_OFF = "#d2dad6"
BAR_BG = "#e0e9e4"
BAR_IDLE = "#a0b6ac"
FONT = "Microsoft YaHei UI"
WEIGHT_TYPES = [("PyTorch 权重", "*.pt"), ("所有文件", "*.*")]
SPEED_OPTIONS = [
    ("1× 实时", 1.0),
    ("2×", 2.0),
    ("4×", 4.0),
    ("全速", 0.0),
]


def label(parent, text="", *, bg=PANEL, fg=TEXT, size=10, bold=False, **options):
    font = (FONT, size, "bold" if bold else "normal")
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font, anchor="w", **options)


class ModelCard(tk.Frame):
    def __init__(self, parent, viewer, path):
        super().__init__(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        self.viewer = viewer
        self.playback = None
        self.owns_playback = True
        self.latest_frame = None
        self.photo = None
        self.screen_height_limit = 290

        header = tk.Frame(self, bg=PANEL)
        header.pack(fill="x", padx=14, pady=(10, 0))
        self.status = label(header, bg=SOFT, fg=MUTED, size=8, bold=True, padx=8, pady=3)
        self.status.pack(side="right")
        self.title = label(header, size=15, bold=True, width=1)
        self.title.pack(side="left", fill="x", expand=True)
        self.weight = label(self, fg=MUTED, size=9, width=1)
        self.weight.pack(fill="x", padx=14, pady=(0, 6))

        self.screen = tk.Canvas(self, bg=MATTE, height=200, highlightthickness=0)
        self.screen.pack(fill="x", padx=12)
        self.image = self.screen.create_image(0, 0)
        self.message = self.screen.create_text(0, 0, fill=MUTED, font=(FONT, 10), justify="center")
        self.screen.bind("<Configure>", self.resize_screen)

        # Score & episode bar
        footer = tk.Frame(self, bg=SOFT, padx=8, pady=2)
        footer.pack(fill="x", padx=12, pady=(6, 0))
        self.episode = label(footer, bg=SOFT, fg=MUTED, size=9)
        self.episode.pack(side="left", padx=(0, 8))
        self.score = label(footer, bg=SOFT, fg=ACCENT, size=14, bold=True)
        self.score.pack(side="left")
        self.last_score = label(footer, bg=SOFT, fg=MUTED, size=9)
        self.last_score.pack(side="right")

        # Live action + lives + steps strip
        telemetry = tk.Frame(self, bg=PANEL)
        telemetry.pack(fill="x", padx=12, pady=(5, 0))
        self.action_label = label(telemetry, text="动作: —", fg=TEXT, size=9, bold=True)
        self.action_label.pack(side="left")
        self.explore_badge = label(telemetry, text="", bg=PANEL, fg=WARN_FG, size=8, bold=True, padx=5, pady=1)
        self.explore_badge.pack(side="left", padx=(6, 0))
        self.meta_right = label(telemetry, text="生命 —  ·  步数 0", fg=MUTED, size=8)
        self.meta_right.pack(side="right")

        # 4-action Q-value / probability bar canvas
        self.bars_canvas = tk.Canvas(self, bg=PANEL, height=48, highlightthickness=0)
        self.bars_canvas.pack(fill="x", padx=12, pady=(3, 0))
        self.bars_canvas.bind("<Configure>", lambda _e: self.redraw_bars())

        actions = tk.Frame(self, bg=PANEL)
        actions.pack(fill="x", padx=12, pady=(6, 9))
        self.play_button = ttk.Button(actions, text="开始", style="Play.TButton", command=self.toggle_playback)
        self.play_button.pack(side="left", padx=(0, 6))
        self.swap_button = ttk.Button(actions, text="更换权重", command=self.choose_weight)
        self.swap_button.pack(side="left")
        self.remove_button = ttk.Button(
            actions, text="移除", style="Quiet.TButton",
            command=lambda: self.viewer.remove_card(self),
        )
        self.remove_button.pack(side="right")
        self.load(path)

    def choose_weight(self):
        if self.viewer.is_evaluating():
            return
        path = filedialog.askopenfilename(
            parent=self.viewer.root, title="更换模型权重", initialdir=self.path.parent, filetypes=WEIGHT_TYPES,
        )
        if path:
            self.load(Path(path))

    def load(self, path):
        self.close()
        self.path = Path(path)
        self.owns_playback = True
        self._reset_visuals()
        self.playback = Playback(
            self.path,
            self.viewer.device,
            self.viewer.current_seed(),
            speed_multiplier=self.viewer.current_speed(),
        )
        self.update_controls()

    def attach_batch_playback(self, playback: Playback):
        self.close()
        self.owns_playback = False
        self._reset_visuals()
        self.playback = playback
        self.play_state = "running"
        self.update_controls()

    def _reset_visuals(self):
        self.play_state = "ready"
        self.latest_frame = None
        self.photo = None
        self.title.configure(text=self.path.parent.name or self.path.stem)
        self.weight.configure(text=self.path.name)
        self.status.configure(text="加载中", bg=SOFT, fg=MUTED)
        self.episode.configure(text="本局")
        self.score.configure(text="—")
        self.last_score.configure(text="上一局 —")
        self.action_label.configure(text="动作: —", fg=TEXT)
        self.explore_badge.configure(text="", bg=PANEL)
        self.meta_right.configure(text="生命 —  ·  步数 0")
        self.screen.itemconfigure(self.image, image="")
        self.screen.itemconfigure(self.message, text="正在加载权重…", fill=MUTED)
        self.redraw()
        self.redraw_bars()

    def close(self):
        if self.playback is not None and self.owns_playback:
            self.playback.close()
        self.playback = None

    def set_speed(self, speed_multiplier: float):
        if self.playback is not None:
            self.playback.set_speed(speed_multiplier)

    def toggle_playback(self):
        self.pause() if self.play_state == "running" else self.start()

    def start(self):
        if self.play_state in ("ready", "paused") and self.playback is not None:
            self.play_state = "running"
            self.playback.set_paused(False)
            self.update_controls()

    def pause(self):
        if self.play_state == "running" and self.playback is not None:
            self.play_state = "paused"
            self.playback.set_paused(True)
            self.update_controls()

    def update_controls(self):
        status, command, tint, color = {
            "ready": ("待开始", "开始", SOFT, MUTED),
            "running": ("运行中", "暂停", ACCENT_SOFT, ACCENT),
            "paused": ("已暂停", "继续", WARN_BG, WARN_FG),
            "finished": ("已完成", "已完成", ACCENT_SOFT, ACCENT),
            "failed": ("运行失败", "开始", "#fae5e8", ERROR),
        }[self.play_state]
        style = "TButton" if self.play_state == "running" else "Play.TButton"
        eval_locked = self.viewer.is_evaluating()
        btn_disabled = self.play_state in ("failed", "finished")
        self.play_button.configure(text=command, style=style, state="disabled" if btn_disabled else "normal")
        self.swap_button.configure(state="disabled" if eval_locked else "normal")
        self.remove_button.configure(state="disabled" if eval_locked else "normal")
        if self.latest_frame is not None or self.play_state in ("failed", "finished"):
            self.status.configure(text=status, bg=tint, fg=color)
        self.viewer.update_playback_buttons()

    def refresh(self):
        if self.playback is None:
            return
        update = self.playback.take_latest()
        if isinstance(update, str):
            self.play_state = "failed"
            self.latest_frame = None
            self.photo = None
            self.screen.itemconfigure(self.image, image="")
            self.screen.itemconfigure(self.message, text=update[:260], fill=ERROR)
            self.update_controls()
        elif isinstance(update, Frame):
            first_frame = self.latest_frame is None
            self.latest_frame = update
            if first_frame:
                self.title.configure(text=ALGORITHMS.get(update.algorithm, update.algorithm.upper()))
                self.weight.configure(text=f"{self.path.parent.name} / {self.path.name}  ·  {update.training_step:,} 步")
                self.update_controls()
            if update.finished and self.play_state != "finished":
                self.play_state = "finished"
                self.update_controls()

            ep_text = (
                f"第 {update.episode}/{update.target_episodes} 局"
                if update.target_episodes
                else f"第 {update.episode} 局"
            )
            self.episode.configure(text=ep_text)
            self.score.configure(text=f"{update.score:g} 分")

            if update.target_episodes and self.playback.records:
                mean_ret = sum(r.return_score for r in self.playback.records) / len(self.playback.records)
                self.last_score.configure(text=f"均分 {mean_ret:.1f} ({len(self.playback.records)} 局)")
            else:
                last = "—" if update.last_score is None else f"{update.last_score:g} 分"
                self.last_score.configure(text=f"上一局 {last}")

            if update.action is not None and 0 <= update.action < len(ACTION_NAMES):
                self.action_label.configure(
                    text=f"动作: {ACTION_LABELS[update.action]} ({ACTION_NAMES[update.action]})",
                    fg=ACCENT,
                )
            else:
                self.action_label.configure(text="动作: —", fg=TEXT)

            if update.exploratory:
                self.explore_badge.configure(text="探索", bg=WARN_BG, fg=WARN_FG)
            else:
                self.explore_badge.configure(text="", bg=PANEL)

            if update.lives is not None:
                max_l = max(1, int(update.max_lives))
                cur_l = max(0, min(max_l, int(update.lives)))
                hearts = "●" * cur_l + "○" * (max_l - cur_l)
                self.meta_right.configure(text=f"生命 {hearts}  ·  步数 {update.step:,}")
            else:
                self.meta_right.configure(text=f"步数 {update.step:,}")

            self.screen.itemconfigure(self.message, text="")
            self.redraw()
            self.redraw_bars()

    def resize_screen(self, _event=None):
        target_height = max(140, min(self.screen_height_limit, self.screen.winfo_width() * 3 // 4))
        if int(self.screen.cget("height")) != target_height:
            self.screen.configure(height=target_height)
        self.redraw()

    def redraw(self):
        width, height = self.screen.winfo_width(), self.screen.winfo_height()
        for item in (self.image, self.message):
            self.screen.coords(item, width // 2, height // 2)
        self.screen.itemconfigure(self.message, width=max(40, width - 32))
        if self.latest_frame is None or self.latest_frame.rgb is None or min(width, height) < 2:
            return

        photo_width = min(width, height * 4 // 3)
        photo_height = max(1, photo_width * 3 // 4)
        rgb = cv2.resize(self.latest_frame.rgb, (photo_width, photo_height), interpolation=cv2.INTER_NEAREST)
        ppm = f"P6\n{photo_width} {photo_height}\n255\n".encode("ascii") + rgb.tobytes()
        self.photo = tk.PhotoImage(master=self.screen, data=ppm, format="PPM")
        self.screen.itemconfigure(self.image, image=self.photo)

    def redraw_bars(self):
        c = self.bars_canvas
        c.delete("all")
        width = c.winfo_width()
        if width < 80:
            return

        frame = self.latest_frame
        values = frame.action_values if (frame and frame.action_values) else ()
        kind = (frame.value_kind or "") if frame else ""
        chosen = frame.action if frame else None

        if kind == "prob" and len(values) == 4:
            ratios = [max(0.0, min(1.0, float(v))) for v in values]
            texts = [f"{v * 100:4.1f}%" for v in values]
        elif len(values) == 4:
            vmin, vmax = min(values), max(values)
            span = vmax - vmin
            ratios = [0.15 + 0.85 * ((v - vmin) / span) if span > 1e-6 else 0.5 for v in values]
            texts = [f"{v:5.2f}" for v in values]
        else:
            ratios = [0.0, 0.0, 0.0, 0.0]
            texts = ["—", "—", "—", "—"]

        col_gap = 12
        col_w = (width - col_gap) // 2
        row_h = 22
        for idx in range(4):
            row, col = divmod(idx, 2)
            x0 = col * (col_w + col_gap)
            y0 = 2 + row * row_h
            is_chosen = chosen == idx
            lbl_fg = ACCENT if is_chosen else MUTED
            c.create_text(
                x0, y0 + 9, text=ACTION_LABELS[idx], anchor="w",
                fill=lbl_fg, font=(FONT, 8, "bold" if is_chosen else "normal"),
            )
            bx0 = x0 + 30
            bx1 = x0 + col_w - 42
            if bx1 > bx0 + 8:
                c.create_rectangle(bx0, y0 + 4, bx1, y0 + 14, fill=BAR_BG, outline="")
                fw = int(round((bx1 - bx0) * ratios[idx]))
                if fw > 0:
                    c.create_rectangle(
                        bx0, y0 + 4, bx0 + max(2, fw), y0 + 14,
                        fill=ACCENT if is_chosen else BAR_IDLE, outline="",
                    )
            c.create_text(
                x0 + col_w, y0 + 9, text=texts[idx], anchor="e",
                fill=TEXT if is_chosen else MUTED, font=(FONT, 8),
            )


class Viewer:
    def __init__(self, root, device, seed, checkpoints):
        self.root = root
        self.device = device
        self.default_seed = int(seed)
        self.cards: list[ModelCard] = []
        self.columns = 0
        self.batch_session: BatchEvaluationSession | None = None

        cwd = Path.cwd()
        self.batch_directory = next((path for path in (cwd / "runs" / "runs", cwd / "outputs" / "runs") if path.is_dir()), cwd)
        self.directory = self.batch_directory
        self.eval_base_dir = cwd / "outputs" / "evaluations"

        root.title("Breakout 模型演示与统一复评")
        root.configure(bg=BACKGROUND)
        root.geometry(f"{min(1340, root.winfo_screenwidth() - 60)}x{min(920, root.winfo_screenheight() - 80)}")
        root.minsize(560, 480)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-o>", lambda _event: self.add_weights())

        style = ttk.Style(root)
        style.theme_use("clam")
        for name in ("TButton", "TMenubutton"):
            style.configure(name, background=PANEL, foreground=TEXT, borderwidth=1, bordercolor=BORDER,
                            lightcolor=PANEL, darkcolor=PANEL, focuscolor=ACCENT, width=0,
                            font=(FONT, 9, "bold"), padding=(10, 5))
            style.map(name, background=[("disabled", SOFT), ("active", ACCENT_SOFT)],
                      foreground=[("disabled", MUTED)])
        style.configure("TMenubutton", arrowcolor=ACCENT, arrowsize=7)
        style.configure("Primary.TButton", background=ACCENT, foreground=PANEL, padding=(14, 6),
                        bordercolor=ACCENT, darkcolor=ACCENT, lightcolor=ACCENT, focuscolor=ACCENT_SOFT)
        style.map("Primary.TButton", background=[("disabled", SOFT), ("active", "#0c543e")],
                  bordercolor=[("disabled", BORDER)])
        style.configure("Danger.TButton", background="#fae5e8", foreground=ERROR, bordercolor="#e8c3c8", padding=(12, 6))
        style.map("Danger.TButton", background=[("disabled", SOFT), ("active", "#f4cfd5")],
                  foreground=[("disabled", MUTED)])
        style.configure("Play.TButton", background=ACCENT_SOFT, foreground=ACCENT, bordercolor=ACCENT_SOFT)
        style.map("Play.TButton", background=[("disabled", SOFT), ("active", "#bfebd6")])
        style.configure("Quiet.TButton", foreground=MUTED, bordercolor=PANEL, padding=(8, 5))
        style.map("Quiet.TButton", background=[("active", "#fae5e8")], foreground=[("active", ERROR)])
        style.configure("TCheckbutton", background=SUBBAR, foreground=TEXT, font=(FONT, 9))
        style.map("TCheckbutton", background=[("active", SUBBAR)])
        style.configure("Vertical.TScrollbar", background=BORDER, troughcolor=BACKGROUND, arrowcolor=MUTED,
                        bordercolor=BACKGROUND, darkcolor=BACKGROUND, lightcolor=BACKGROUND, arrowsize=10)

        header = tk.Frame(root, bg=PANEL, padx=22, pady=11)
        header.pack(fill="x")
        title_row = tk.Frame(header, bg=PANEL)
        title_row.pack(fill="x")
        label(title_row, "Breakout 模型演示与统一复评", size=20, bold=True).pack(side="left")
        self.count = label(title_row, bg=SOFT, fg=MUTED, size=9, bold=True, padx=10, pady=4)
        self.count.pack(side="right")

        # Row 1 toolbar: model management + live playback controls
        toolbar = tk.Frame(header, bg=PANEL)
        toolbar.pack(fill="x", pady=(8, 0))
        self.add_button = ttk.Button(toolbar, text="添加权重", command=self.add_weights)
        self.add_button.pack(side="left")
        self.batch_button = ttk.Menubutton(toolbar, text="批量导入")
        batch_menu = tk.Menu(self.batch_button, tearoff=False, font=(FONT, 10))
        batch_menu.add_command(label="从目录导入 best.pt", command=lambda: self.import_folder("best.pt"))
        batch_menu.add_command(label="从目录导入 final.pt", command=lambda: self.import_folder("final.pt"))
        batch_menu.add_command(label="从目录导入所有 .pt", command=lambda: self.import_folder("*.pt"))
        self.batch_button.configure(menu=batch_menu)
        self.batch_button.pack(side="left", padx=7)

        label(toolbar, "倍速:", fg=MUTED, size=9).pack(side="left", padx=(14, 4))
        self.speed_var = tk.StringVar(value=SPEED_OPTIONS[0][0])
        self.speed_box = ttk.Combobox(
            toolbar, textvariable=self.speed_var, width=8, state="readonly",
            values=[name for name, _ in SPEED_OPTIONS], font=(FONT, 9),
        )
        self.speed_box.pack(side="left")
        self.speed_box.bind("<<ComboboxSelected>>", self.on_speed_change)

        self.start_button = ttk.Button(toolbar, text="全部开始", style="Primary.TButton", command=self.start_all)
        self.start_button.pack(side="right")
        self.pause_button = ttk.Button(toolbar, text="全部暂停", command=self.pause_all)
        self.pause_button.pack(side="right", padx=7)

        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        # Row 2 sub-bar: unified batch evaluation & A+B recording controls
        eval_bar = tk.Frame(root, bg=SUBBAR, padx=22, pady=8)
        eval_bar.pack(fill="x")
        label(eval_bar, "统一复评", bg=SUBBAR, fg=ACCENT, size=9, bold=True).pack(side="left", padx=(0, 10))

        label(eval_bar, "起始种子:", bg=SUBBAR, fg=MUTED, size=9).pack(side="left")
        self.seed_var = tk.StringVar(value=str(self.default_seed))
        self.seed_entry = ttk.Entry(eval_bar, textvariable=self.seed_var, width=8, font=(FONT, 9))
        self.seed_entry.pack(side="left", padx=(4, 12))

        label(eval_bar, "复评局数:", bg=SUBBAR, fg=MUTED, size=9).pack(side="left")
        self.episodes_var = tk.StringVar(value="10")
        self.episodes_spin = ttk.Spinbox(eval_bar, from_=1, to=100, textvariable=self.episodes_var, width=5, font=(FONT, 9))
        self.episodes_spin.pack(side="left", padx=(4, 14))

        self.rec_a_var = tk.BooleanVar(value=True)
        self.rec_b_var = tk.BooleanVar(value=True)
        self.rec_a_chk = ttk.Checkbutton(eval_bar, text="单模型HUD录像(A)", variable=self.rec_a_var)
        self.rec_a_chk.pack(side="left", padx=(0, 10))
        self.rec_b_chk = ttk.Checkbutton(eval_bar, text="并排对比录像(B)", variable=self.rec_b_var)
        self.rec_b_chk.pack(side="left", padx=(0, 12))

        self.dir_button = ttk.Button(eval_bar, text="输出目录", command=self.choose_eval_dir)
        self.dir_button.pack(side="left")

        self.stop_eval_button = ttk.Button(
            eval_bar, text="停止复评", style="Danger.TButton", command=self.stop_batch_eval, state="disabled",
        )
        self.stop_eval_button.pack(side="right", padx=(7, 0))
        self.eval_button = ttk.Button(
            eval_bar, text="开始统一复评", style="Primary.TButton", command=self.start_batch_eval,
        )
        self.eval_button.pack(side="right")

        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(root, bg=BACKGROUND)
        body.pack(fill="both", expand=True, padx=(16, 12), pady=(10, 8))
        self.canvas = tk.Canvas(body, bg=BACKGROUND, highlightthickness=0)
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=scroll.set)
        self.gallery = tk.Frame(self.canvas, bg=BACKGROUND)
        self.gallery_window = self.canvas.create_window((0, 0), anchor="nw", window=self.gallery)
        self.empty = label(self.gallery, "选择 .pt 权重文件，或从目录批量导入。",
                           fg=MUTED, padx=20, pady=20, highlightthickness=1, highlightbackground=BORDER)
        self.gallery.bind("<Configure>", self.update_scroll_region)
        self.canvas.bind("<Configure>", self.layout_cards)
        root.bind("<MouseWheel>", lambda event: self.canvas.yview_scroll(-int(event.delta / 120), "units"))
        root.bind("<Button-4>", lambda _event: self.canvas.yview_scroll(-1, "units"))
        root.bind("<Button-5>", lambda _event: self.canvas.yview_scroll(1, "units"))

        self.add_checkpoints(checkpoints)
        self.timer = root.after(33, self.refresh)

    def is_evaluating(self) -> bool:
        return self.batch_session is not None

    def current_seed(self) -> int:
        try:
            return int(self.seed_var.get().strip())
        except ValueError:
            return self.default_seed

    def current_speed(self) -> float:
        selected = self.speed_var.get()
        for name, val in SPEED_OPTIONS:
            if name == selected:
                return val
        return 1.0

    def on_speed_change(self, _event=None):
        speed = self.current_speed()
        if self.batch_session is not None:
            self.batch_session.set_speed(speed)
        for card in self.cards:
            card.set_speed(speed)

    def choose_eval_dir(self):
        directory = filedialog.askdirectory(
            parent=self.root, title="选择复评结果保存根目录",
            initialdir=self.eval_base_dir if self.eval_base_dir.exists() else Path.cwd(),
        )
        if directory:
            self.eval_base_dir = Path(directory)

    def start_batch_eval(self):
        if not self.cards:
            messagebox.showwarning("统一复评", "请先导入至少一个模型权重。", parent=self.root)
            return
        try:
            base_seed = int(self.seed_var.get().strip())
            episodes = int(self.episodes_var.get().strip())
            if episodes < 1:
                raise ValueError("局数必须大于 0")
        except ValueError as exc:
            messagebox.showerror("参数错误", f"请检查起始种子和复评局数：{exc}", parent=self.root)
            return

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = self.eval_base_dir / f"viewer-eval-{stamp}"
        checkpoints = [card.path for card in self.cards]

        self.batch_session = BatchEvaluationSession(
            checkpoints,
            device=self.device,
            output_dir=out_dir,
            base_seed=base_seed,
            episodes=episodes,
            speed_multiplier=self.current_speed(),
            record_individual=self.rec_a_var.get(),
            record_comparison=self.rec_b_var.get(),
            on_finish=lambda res: self.root.after(0, self._on_batch_eval_finished, res),
        )
        for card, pb in zip(self.cards, self.batch_session.playbacks):
            card.attach_batch_playback(pb)
        self.batch_session.start()
        self._update_eval_toolbar_state()

    def stop_batch_eval(self):
        if self.batch_session is not None:
            self.batch_session.stop()
            self.stop_eval_button.configure(state="disabled")

    def _on_batch_eval_finished(self, result: dict):
        self.batch_session = None
        self._update_eval_toolbar_state()
        for card in self.cards:
            card.update_controls()

        lines = [
            f"状态: {'已手动停止（已保存已完成局）' if result.get('aborted') else '全部完成'}",
            f"保存目录: {result['output_dir']}",
            "",
            "各模型复评汇总：",
        ]
        for m in result.get("models", []):
            lines.append(
                f"  · {m['algorithm_name']} ({m['episodes_completed']} 局): "
                f"均值 {m['mean_return']:.1f} | 中位数 {m['median_return']:.1f} | "
                f"标准差 {m['std_return']:.1f} | 区间 [{m['min_return']:g}, {m['max_return']:g}]"
            )
        comp = result.get("comparison_videos", [])
        if comp:
            lines.append("")
            lines.append(f"已生成 {len(comp)} 份并排对比录像（comparison/）。")
        lines.append("已自动导出 summary.json、summary.csv、episodes.csv、life_losses.csv。")
        messagebox.showinfo("统一复评完成", "\n".join(lines), parent=self.root)

    def _update_eval_toolbar_state(self):
        running = self.is_evaluating()
        state_idle = "disabled" if running else "normal"
        self.add_button.configure(state=state_idle)
        self.batch_button.configure(state=state_idle)
        self.seed_entry.configure(state=state_idle)
        self.episodes_spin.configure(state=state_idle)
        self.rec_a_chk.configure(state=state_idle)
        self.rec_b_chk.configure(state=state_idle)
        self.dir_button.configure(state=state_idle)
        self.eval_button.configure(state="disabled" if (running or not self.cards) else "normal")
        self.stop_eval_button.configure(state="normal" if running else "disabled")
        self._refresh_header_badge()

    def _refresh_header_badge(self):
        card_count = len(self.cards)
        if self.batch_session is not None:
            done_eps = [len(pb.records) for pb in self.batch_session.playbacks]
            min_ep = min(done_eps) if done_eps else 0
            self.count.configure(
                text=f"复评中 ({min_ep}/{self.batch_session.episodes} 局)  ·  种子 {self.batch_session.base_seed}  ·  {str(self.device).upper()}",
                bg=ACCENT_SOFT, fg=ACCENT,
            )
        else:
            self.count.configure(text=f"{card_count} 个模型  ·  {str(self.device).upper()}", bg=SOFT, fg=MUTED)

    def add_weights(self):
        if self.is_evaluating():
            return
        paths = filedialog.askopenfilenames(
            parent=self.root, title="添加模型权重", initialdir=self.directory, filetypes=WEIGHT_TYPES,
        )
        if paths:
            self.directory = Path(paths[-1]).parent
            self.add_checkpoints(map(Path, paths))

    def import_folder(self, pattern):
        if self.is_evaluating():
            return
        directory = filedialog.askdirectory(
            parent=self.root, title=f"选择权重目录（{pattern}）", initialdir=self.batch_directory, mustexist=True,
        )
        if not directory:
            return

        self.batch_directory = Path(directory)
        try:
            paths = sorted(path for path in self.batch_directory.rglob(pattern) if path.is_file())
            loaded = {card.path.resolve() for card in self.cards}
            new_paths = [path for path in paths if path.resolve() not in loaded]
        except OSError as exc:
            messagebox.showerror("导入失败", str(exc), parent=self.root)
            return
        if not new_paths:
            text = "匹配的权重已经添加。" if paths else f"这个目录和子目录中没有找到 {pattern}。"
            messagebox.showinfo("批量导入", text, parent=self.root)
            return
        self.add_checkpoints(new_paths)

    def add_checkpoints(self, paths):
        for path in paths:
            self.cards.append(ModelCard(self.gallery, self, path))
        self.layout_cards()

    def remove_card(self, card):
        if self.is_evaluating():
            return
        card.close()
        self.cards.remove(card)
        card.destroy()
        self.layout_cards()

    def update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), self.gallery.winfo_height()))

    def layout_cards(self, _event=None):
        card_count = len(self.cards)
        width = self.canvas.winfo_width()
        preferred_columns = 3 if card_count in (5, 6) else 4
        columns = max(1, min(preferred_columns, card_count, width // 300))
        rows = (card_count + columns - 1) // columns
        screen_height_limit = max(140, self.canvas.winfo_height() // 2 - 236) if rows == 2 else 290
        content_width = min(width, columns * 450)
        self.canvas.itemconfigure(self.gallery_window, width=content_width)
        self.canvas.coords(self.gallery_window, (width - content_width) // 2, 0)
        for column in range(2 * max(columns, self.columns)):
            active = column < 2 * columns
            self.gallery.columnconfigure(column, weight=int(active), uniform="models" if active else "")
        self.columns = columns
        self.empty.grid_remove()
        if not self.cards:
            self.empty.grid(row=0, column=0, columnspan=2, padx=8, pady=24, sticky="ew")
        for index, card in enumerate(self.cards):
            row, column = divmod(index, columns)
            row_size = min(columns, card_count - row * columns)
            card.grid(row=row, column=2 * column + columns - row_size, columnspan=2,
                      padx=8, pady=(0, 12), sticky="nsew")
            card.screen_height_limit = screen_height_limit
            card.resize_screen()
        self.update_scroll_region()
        self._update_eval_toolbar_state()
        self.update_playback_buttons()

    def update_playback_buttons(self):
        can_start = any(card.play_state in ("ready", "paused") for card in self.cards)
        can_pause = any(card.play_state == "running" for card in self.cards)
        self.start_button.configure(state="normal" if can_start else "disabled")
        self.pause_button.configure(state="normal" if can_pause else "disabled")

    def start_all(self):
        for card in self.cards:
            card.start()

    def pause_all(self):
        for card in self.cards:
            card.pause()

    def refresh(self):
        for card in self.cards:
            card.refresh()
        if self.batch_session is not None:
            self._refresh_header_badge()
        self.timer = self.root.after(33, self.refresh)

    def close(self):
        self.root.after_cancel(self.timer)
        if self.batch_session is not None:
            self.batch_session.stop()
        for card in self.cards:
            card.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", type=Path, help="optional .pt files to open at launch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--batch-eval", action="store_true", help="run unified batch evaluation headlessly")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per model for --batch-eval")
    parser.add_argument("--output-dir", type=Path, help="output directory for --batch-eval")
    parser.add_argument("--no-video-a", action="store_true", help="disable Plan A per-model HUD videos")
    parser.add_argument("--no-video-b", action="store_true", help="disable Plan B side-by-side comparison videos")
    args = parser.parse_args()
    device = setup(args.seed, args.device, threads=1)

    if args.batch_eval:
        if not args.checkpoints:
            parser.error("--batch-eval requires at least one checkpoint path")
        session = BatchEvaluationSession(
            args.checkpoints,
            device=device,
            output_dir=args.output_dir,
            base_seed=args.seed,
            episodes=args.episodes,
            speed_multiplier=0.0,
            record_individual=not args.no_video_a,
            record_comparison=not args.no_video_b,
        )
        session.start()
        result = session.wait()
        print(f"Saved evaluation results to: {result['output_dir']}", flush=True)
        for m in result["models"]:
            print(
                f"{m['algorithm_name']:12s} episodes={m['episodes_completed']} "
                f"mean={m['mean_return']:.2f} median={m['median_return']:.2f} "
                f"std={m['std_return']:.2f} range=[{m['min_return']:g}, {m['max_return']:g}]",
                flush=True,
            )
        return

    root = tk.Tk()
    Viewer(root, device, args.seed, args.checkpoints)
    root.mainloop()


if __name__ == "__main__":
    main()
