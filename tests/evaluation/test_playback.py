from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock
import csv
import json

import cv2
import numpy as np
import torch

from breakout_rl.evaluation import batch, playback, policy, recording


def test_playback_waits_for_explicit_start(monkeypatch):
    initialized = Event()
    stepped = Event()
    closed = Event()
    env = Mock()
    env.render.side_effect = initialized.set
    env.reset.return_value = (None, {})
    env.step.return_value = (None, 0.0, False, False, {})
    env.close.side_effect = closed.set

    def select_action(_obs, _rng):
        stepped.set()
        return 0

    checkpoint = {"algorithm": "dqn", "step": 10}
    monkeypatch.setattr(playback, "load_checkpoint", lambda *_: (None, SimpleNamespace(frame_skip=4), checkpoint))
    monkeypatch.setattr(playback, "action_selector", lambda *_: select_action)
    monkeypatch.setattr(playback, "make_env", lambda *_args, **_kwargs: env)

    worker = playback.Playback(Path("best.pt"), "cpu", 10_000)
    try:
        assert initialized.wait(2)
        assert not stepped.wait(0.15)
        env.step.assert_not_called()
        worker.set_paused(False)
        assert stepped.wait(2)
    finally:
        worker.close()
    assert closed.wait(2)
    env.step.assert_called_with(0)


def test_decision_selector_matches_action_selector_for_dqn_and_ppo():
    class FakeQNet:
        def __call__(self, x):
            return torch.tensor([[1.0, 2.5, 4.2, 3.1]], dtype=torch.float32)

    obs = np.zeros((4, 84, 84), dtype=np.uint8)
    ckpt = {"algorithm": "dqn", "action_count": 4}
    sel_act = policy.action_selector(FakeQNet(), ckpt, "cpu", epsilon=0.5)
    sel_dec = policy.decision_selector(FakeQNet(), ckpt, "cpu", epsilon=0.5)

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    saw_explore = False
    saw_greedy = False
    for _ in range(30):
        a = sel_act(obs, rng1)
        d = sel_dec(obs, rng2)
        assert d.action == a
        assert d.value_kind == "q"
        assert len(d.values) == 4
        saw_explore |= d.exploratory
        saw_greedy |= not d.exploratory
    assert saw_explore and saw_greedy

    class FakePPONet:
        def distribution_and_value(self, x):
            logits = torch.tensor([[0.1, 0.2, 1.5, -0.3]], dtype=torch.float32)
            return torch.distributions.Categorical(logits=logits), torch.tensor([1.0])

    ppo_ckpt = {"algorithm": "ppo"}
    ppo_act = policy.action_selector(FakePPONet(), ppo_ckpt, "cpu", deterministic=False)
    ppo_dec = policy.decision_selector(FakePPONet(), ppo_ckpt, "cpu", deterministic=False)
    rng3 = np.random.default_rng(77)
    rng4 = np.random.default_rng(77)
    for _ in range(20):
        a = ppo_act(obs, rng3)
        d = ppo_dec(obs, rng4)
        assert d.action == a
        assert d.value_kind == "prob"
        assert abs(sum(d.values) - 1.0) < 1e-5
        assert not d.exploratory


def test_batch_evaluation_tracks_life_losses_records_ab_videos_and_exports(monkeypatch, tmp_path):
    class FakeALE:
        def __init__(self):
            self._lives = 2
        def lives(self):
            return self._lives

    class FakeEnv:
        def __init__(self, steps_per_ep):
            self.steps_per_ep = steps_per_ep
            self.total_frames = 0
            self.ep_step = 0
            self.unwrapped = SimpleNamespace(ale=FakeALE())

        def get_wrapper_attr(self, name):
            assert name == "total_frames"
            return self.total_frames

        def reset(self, seed=None, **_kwargs):
            if seed is not None:
                self.ep_step = 0
                self.unwrapped.ale._lives = 2
            return np.zeros((4, 84, 84), dtype=np.uint8), {}

        def step(self, action):
            self.ep_step += 1
            self.total_frames += 4
            if self.ep_step == 2:
                # First life lost
                self.unwrapped.ale._lives = 1
                return np.zeros((4, 84, 84), dtype=np.uint8), 3.0, True, False, {}
            if self.ep_step >= self.steps_per_ep:
                # Final life lost -> game over
                self.unwrapped.ale._lives = 0
                total_score = float(self.steps_per_ep * 2)
                return np.zeros((4, 84, 84), dtype=np.uint8), 2.0, True, False, {"episode": {"r": total_score}}
            return np.zeros((4, 84, 84), dtype=np.uint8), 1.0, False, False, {}

        def render(self):
            img = np.zeros((210, 160, 3), dtype=np.uint8)
            img[:, :, 1] = min(255, self.ep_step * 40)
            return img

        def close(self):
            pass

    class FakeQNet:
        def __call__(self, x):
            return torch.tensor([[0.5, 1.0, 3.2, 2.0]], dtype=torch.float32)

    class FakePPONet:
        def distribution_and_value(self, x):
            logits = torch.tensor([[0.0, 0.2, 1.8, 0.4]], dtype=torch.float32)
            return torch.distributions.Categorical(logits=logits), torch.tensor([0.5])

    ckpt_dqn = tmp_path / "dqn-run" / "best.pt"
    ckpt_ppo = tmp_path / "ppo-run" / "best.pt"
    ckpt_dqn.parent.mkdir(parents=True)
    ckpt_ppo.parent.mkdir(parents=True)
    ckpt_dqn.write_bytes(b"fake")
    ckpt_ppo.write_bytes(b"fake")

    def fake_load(path, _device):
        if "dqn" in str(path):
            return FakeQNet(), SimpleNamespace(frame_skip=4, steps_per_ep=4), {"algorithm": "dqn", "step": 1000, "action_count": 4}
        return FakePPONet(), SimpleNamespace(frame_skip=4, steps_per_ep=6), {"algorithm": "ppo", "step": 2000}

    monkeypatch.setattr(playback, "load_checkpoint", fake_load)
    monkeypatch.setattr(playback, "make_env", lambda cfg, **_kwargs: FakeEnv(cfg.steps_per_ep))

    out_dir = tmp_path / "eval_out"
    session = batch.BatchEvaluationSession(
        [ckpt_dqn, ckpt_ppo],
        device="cpu",
        output_dir=out_dir,
        base_seed=30_000,
        episodes=2,
        speed_multiplier=0.0,
        record_individual=True,
        record_comparison=True,
        epsilon=0.0,
    )
    session.start()
    result = session.wait(timeout=30.0)
    assert result is not None
    assert not result["aborted"]
    assert len(result["models"]) == 2

    # Verify JSON and CSV exports
    summary_json = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["base_seed"] == 30_000
    assert summary_json["target_episodes"] == 2
    assert len(summary_json["comparison_videos"]) == 2

    dqn_summary = summary_json["models"][0]
    assert dqn_summary["algorithm"] == "dqn"
    assert dqn_summary["episodes_completed"] == 2
    assert dqn_summary["mean_return"] == 8.0
    assert len(dqn_summary["episodes"][0]["life_losses"]) == 2
    assert dqn_summary["episodes"][0]["life_losses"][0]["step"] == 2
    assert dqn_summary["episodes"][0]["life_losses"][0]["frame"] == 8

    for csv_name in ("summary.csv", "episodes.csv", "life_losses.csv"):
        csv_path = out_dir / csv_name
        assert csv_path.is_file()
        rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
        assert len(rows) >= 3

    # Verify Plan A per-model HUD videos and Plan B comparison videos exist and are valid video files
    plan_a_videos = [out_dir / ep["video"] for m in summary_json["models"] for ep in m["episodes"]]
    plan_b_videos = [out_dir / rel for rel in summary_json["comparison_videos"]]
    assert len(plan_a_videos) == 4
    assert len(plan_b_videos) == 2
    for vp in plan_a_videos + plan_b_videos:
        assert vp.is_file() and vp.stat().st_size > 1000
        cap = cv2.VideoCapture(str(vp))
        ret, frame = cap.read()
        cap.release()
        assert ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0
