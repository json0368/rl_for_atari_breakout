from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from breakout_rl.evaluation import playback


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
