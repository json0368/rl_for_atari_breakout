"""Desktop viewer for multiple Breakout policies."""

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ..evaluation.playback import Frame, Playback
from ..infrastructure.runtime import setup


BACKGROUND = "#edf2ef"
PANEL = "#ffffff"
SOFT = "#f1f6f3"
BORDER = "#d6e1da"
MATTE = "#e4ebe6"
TEXT = "#162c24"
MUTED = "#53675d"
ACCENT = "#12644b"
ACCENT_SOFT = "#daf1e6"
ERROR = "#952c36"
FONT = "Microsoft YaHei UI"
ALGORITHMS = {"dqn": "DQN", "ddqn": "Double DQN", "dueling": "Dueling DQN", "ppo": "PPO", "pqn": "PQN"}
WEIGHT_TYPES = [("PyTorch 权重", "*.pt"), ("所有文件", "*.*")]


def label(parent, text="", *, bg=PANEL, fg=TEXT, size=10, bold=False, **options):
    font = (FONT, size, "bold" if bold else "normal")
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font, anchor="w", **options)


class ModelCard(tk.Frame):
    def __init__(self, parent, viewer, path):
        super().__init__(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        self.viewer = viewer
        self.playback = None
        self.latest_frame = None
        self.photo = None
        self.screen_height_limit = 320

        header = tk.Frame(self, bg=PANEL)
        header.pack(fill="x", padx=14, pady=(10, 0))
        self.status = label(header, bg=SOFT, fg=MUTED, size=8, bold=True, padx=8, pady=3)
        self.status.pack(side="right")
        self.title = label(header, size=16, bold=True, width=1)
        self.title.pack(side="left", fill="x", expand=True)
        self.weight = label(self, fg=MUTED, size=9, width=1)
        self.weight.pack(fill="x", padx=14, pady=(0, 8))

        self.screen = tk.Canvas(self, bg=MATTE, height=220, highlightthickness=0)
        self.screen.pack(fill="x", padx=12)
        self.image = self.screen.create_image(0, 0)
        self.message = self.screen.create_text(0, 0, fill=MUTED, font=(FONT, 10), justify="center")
        self.screen.bind("<Configure>", self.resize_screen)

        footer = tk.Frame(self, bg=SOFT, padx=8, pady=2)
        footer.pack(fill="x", padx=12, pady=(6, 0))
        self.episode = label(footer, bg=SOFT, fg=MUTED, size=9)
        self.episode.pack(side="left", padx=(0, 8))
        self.score = label(footer, bg=SOFT, fg=ACCENT, size=14, bold=True)
        self.score.pack(side="left")
        self.last_score = label(footer, bg=SOFT, fg=MUTED, size=9)
        self.last_score.pack(side="right")

        actions = tk.Frame(self, bg=PANEL)
        actions.pack(fill="x", padx=12, pady=(7, 9))
        self.play_button = ttk.Button(actions, text="开始", style="Play.TButton", command=self.toggle_playback)
        self.play_button.pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="更换权重", command=self.choose_weight).pack(side="left")
        ttk.Button(actions, text="移除", style="Quiet.TButton",
                   command=lambda: self.viewer.remove_card(self)).pack(side="right")
        self.load(path)

    def choose_weight(self):
        path = filedialog.askopenfilename(
            parent=self.viewer.root, title="更换模型权重", initialdir=self.path.parent, filetypes=WEIGHT_TYPES,
        )
        if path:
            self.load(Path(path))

    def load(self, path):
        self.close()
        self.path = path
        self.play_state = "ready"
        self.latest_frame = None
        self.photo = None
        self.title.configure(text=path.parent.name or path.stem)
        self.weight.configure(text=path.name)
        self.status.configure(text="加载中", bg=SOFT, fg=MUTED)
        self.episode.configure(text="本局")
        self.score.configure(text="—")
        self.last_score.configure(text="上一局 —")
        self.screen.itemconfigure(self.image, image="")
        self.screen.itemconfigure(self.message, text="正在加载权重…", fill=MUTED)
        self.redraw()
        self.playback = Playback(path, self.viewer.device, self.viewer.seed)
        self.update_controls()

    def close(self):
        if self.playback is not None:
            self.playback.close()

    def toggle_playback(self):
        self.pause() if self.play_state == "running" else self.start()

    def start(self):
        if self.play_state in ("ready", "paused"):
            self.play_state = "running"
            self.playback.set_paused(False)
            self.update_controls()

    def pause(self):
        if self.play_state == "running":
            self.play_state = "paused"
            self.playback.set_paused(True)
            self.update_controls()

    def update_controls(self):
        status, command, tint, color = {
            "ready": ("待开始", "开始", SOFT, MUTED),
            "running": ("运行中", "暂停", ACCENT_SOFT, ACCENT),
            "paused": ("已暂停", "继续", "#fbefd8", "#785319"),
            "failed": ("运行失败", "开始", "#fae5e8", ERROR),
        }[self.play_state]
        style = "TButton" if self.play_state == "running" else "Play.TButton"
        self.play_button.configure(text=command, style=style,
                                   state="disabled" if self.play_state == "failed" else "normal")
        if self.latest_frame is not None or self.play_state == "failed":
            self.status.configure(text=status, bg=tint, fg=color)
        self.viewer.update_playback_buttons()

    def refresh(self):
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
                self.title.configure(text=ALGORITHMS[update.algorithm])
                self.weight.configure(text=f"{self.path.parent.name} / {self.path.name}  ·  {update.training_step:,} 步")
                self.update_controls()
            self.episode.configure(text=f"第 {update.episode} 局")
            self.score.configure(text=f"{update.score:g} 分")
            last = "—" if update.last_score is None else f"{update.last_score:g} 分"
            self.last_score.configure(text=f"上一局 {last}")
            self.screen.itemconfigure(self.message, text="")
            self.redraw()

    def resize_screen(self, _event=None):
        target_height = max(150, min(self.screen_height_limit, self.screen.winfo_width() * 3 // 4))
        if int(self.screen.cget("height")) != target_height:
            self.screen.configure(height=target_height)
        self.redraw()

    def redraw(self):
        width, height = self.screen.winfo_width(), self.screen.winfo_height()
        for item in (self.image, self.message):
            self.screen.coords(item, width // 2, height // 2)
        self.screen.itemconfigure(self.message, width=max(40, width - 32))
        if self.latest_frame is None or min(width, height) < 2:
            return

        # Fit the Atari display with its 4:3 aspect ratio. Model inputs stay 84×84.
        photo_width = min(width, height * 4 // 3)
        photo_height = max(1, photo_width * 3 // 4)
        rgb = cv2.resize(self.latest_frame.rgb, (photo_width, photo_height), interpolation=cv2.INTER_NEAREST)
        ppm = f"P6\n{photo_width} {photo_height}\n255\n".encode("ascii") + rgb.tobytes()
        self.photo = tk.PhotoImage(master=self.screen, data=ppm, format="PPM")
        self.screen.itemconfigure(self.image, image=self.photo)


class Viewer:
    def __init__(self, root, device, seed, checkpoints):
        self.root = root
        self.device = device
        self.seed = seed
        self.cards = []
        self.columns = 0
        cwd = Path.cwd()
        self.batch_directory = next((path for path in (cwd / "runs" / "runs", cwd / "outputs" / "runs") if path.is_dir()), cwd)
        self.directory = self.batch_directory

        root.title("Breakout 模型演示")
        root.configure(bg=BACKGROUND)
        root.geometry(f"{min(1280, root.winfo_screenwidth() - 80)}x{min(850, root.winfo_screenheight() - 100)}")
        root.minsize(440, 420)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-o>", lambda _event: self.add_weights())

        style = ttk.Style(root)
        style.theme_use("clam")
        for name in ("TButton", "TMenubutton"):
            style.configure(name, background=PANEL, foreground=TEXT, borderwidth=1, bordercolor=BORDER,
                            lightcolor=PANEL, darkcolor=PANEL, focuscolor=ACCENT, width=0,
                            font=(FONT, 9, "bold"), padding=(11, 6))
            style.map(name, background=[("disabled", SOFT), ("active", ACCENT_SOFT)],
                      foreground=[("disabled", MUTED)])
        style.configure("TMenubutton", arrowcolor=ACCENT, arrowsize=7)
        style.configure("Primary.TButton", background=ACCENT, foreground=PANEL, padding=(16, 7),
                        bordercolor=ACCENT, darkcolor=ACCENT, lightcolor=ACCENT, focuscolor=ACCENT_SOFT)
        style.map("Primary.TButton", background=[("disabled", SOFT), ("active", "#0c543e")],
                  bordercolor=[("disabled", BORDER)])
        style.configure("Play.TButton", background=ACCENT_SOFT, foreground=ACCENT, bordercolor=ACCENT_SOFT)
        style.map("Play.TButton", background=[("disabled", SOFT), ("active", "#bfebd6")])
        style.configure("Quiet.TButton", foreground=MUTED, bordercolor=PANEL, padding=(8, 6))
        style.map("Quiet.TButton", background=[("active", "#fae5e8")], foreground=[("active", ERROR)])
        style.configure("Vertical.TScrollbar", background=BORDER, troughcolor=BACKGROUND, arrowcolor=MUTED,
                        bordercolor=BACKGROUND, darkcolor=BACKGROUND, lightcolor=BACKGROUND, arrowsize=10)

        header = tk.Frame(root, bg=PANEL, padx=24, pady=13)
        header.pack(fill="x")
        title_row = tk.Frame(header, bg=PANEL)
        title_row.pack(fill="x")
        label(title_row, "Breakout 模型演示", size=23, bold=True).pack(side="left")
        self.count = label(title_row, bg=SOFT, fg=MUTED, size=9, bold=True, padx=10, pady=4)
        self.count.pack(side="right")
        toolbar = tk.Frame(header, bg=PANEL)
        toolbar.pack(fill="x", pady=(9, 0))
        ttk.Button(toolbar, text="添加权重", command=self.add_weights).pack(side="left")
        batch_button = ttk.Menubutton(toolbar, text="批量导入")
        batch_menu = tk.Menu(batch_button, tearoff=False, font=(FONT, 10))
        batch_menu.add_command(label="从目录导入 best.pt", command=lambda: self.import_folder("best.pt"))
        batch_menu.add_command(label="从目录导入 final.pt", command=lambda: self.import_folder("final.pt"))
        batch_menu.add_command(label="从目录导入所有 .pt", command=lambda: self.import_folder("*.pt"))
        batch_button.configure(menu=batch_menu)
        batch_button.pack(side="left", padx=7)
        self.start_button = ttk.Button(toolbar, text="全部开始", style="Primary.TButton", command=self.start_all)
        self.start_button.pack(side="right")
        self.pause_button = ttk.Button(toolbar, text="全部暂停", command=self.pause_all)
        self.pause_button.pack(side="right", padx=7)
        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(root, bg=BACKGROUND)
        body.pack(fill="both", expand=True, padx=(16, 12), pady=(12, 8))
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

    def add_weights(self):
        paths = filedialog.askopenfilenames(
            parent=self.root, title="添加模型权重", initialdir=self.directory, filetypes=WEIGHT_TYPES,
        )
        if paths:
            self.directory = Path(paths[-1]).parent
            self.add_checkpoints(map(Path, paths))

    def import_folder(self, pattern):
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
        card.close()
        self.cards.remove(card)
        card.destroy()
        self.layout_cards()

    def update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), self.gallery.winfo_height()))

    def layout_cards(self, _event=None):
        card_count = len(self.cards)
        width = self.canvas.winfo_width()
        # Use two balanced rows for the five baseline models when the window is wide enough.
        preferred_columns = 3 if card_count in (5, 6) else 4
        columns = max(1, min(preferred_columns, card_count, width // 284))
        rows = (card_count + columns - 1) // columns
        screen_height_limit = max(150, self.canvas.winfo_height() // 2 - 188) if rows == 2 else 320
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
        self.count.configure(text=f"{card_count} 个模型  ·  {str(self.device).upper()}")
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
        self.timer = self.root.after(33, self.refresh)

    def close(self):
        self.root.after_cancel(self.timer)
        for card in self.cards:
            card.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", type=Path, help="optional .pt files to open at launch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()
    device = setup(args.seed, args.device, threads=1)
    root = tk.Tk()
    Viewer(root, device, args.seed, args.checkpoints)
    root.mainloop()


if __name__ == "__main__":
    main()
