import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from breakout_rl.algorithms.dqn import trainer
from breakout_rl.algorithms.dqn.config import parser, validate


@pytest.mark.parametrize(
    "scores,patience,eval_frequency,expected_step,counts",
    [
        ([5, 5, 4, 5], 3, 2, 8, [0, 1, 2, 3]),
        ([5, 4, 6, 6, 5, 6], 3, 2, 12, [0, 1, 0, 1, 2, 3]),
        ([5, 5], 1, 2, 4, [0, 1]),
        ([5] * 10, 0, 2, 20, list(range(10))),
        ([], 3, 0, 20, []),
    ],
)
def test_training_early_stopping(
    tmp_path, monkeypatch, scores, patience, eval_frequency, expected_step, counts
):
    args = parser().parse_args([
        "--output-dir", str(tmp_path), "--run-name", "test",
        "--total-timesteps", "20", "--learning-starts", "20",
        "--eval-frequency", str(eval_frequency),
        "--early-stopping-patience", str(patience),
        "--save-frequency", "100", "--log-frequency", "100",
        "--start-e", "1", "--end-e", "1", "--swanlab-mode", "disabled",
    ])
    obs = np.zeros((4, 2, 2), dtype=np.uint8)
    env = SimpleNamespace(
        action_space=SimpleNamespace(n=2, sample=lambda: 0, seed=lambda seed: None),
        observation_space=SimpleNamespace(shape=obs.shape),
        reset=Mock(return_value=(obs, {})),
        step=Mock(return_value=(obs, 0.0, False, False, {})),
        close=Mock(),
    )
    replay = Mock(image_nbytes=0, size=0)
    evaluate = Mock(side_effect=[{"mean_return": s, "std_return": 0.0} for s in scores])
    save = Mock()
    log = Mock()
    finish = Mock()
    monkeypatch.setattr(trainer, "setup", lambda *a: torch.device("cpu"))
    monkeypatch.setattr(trainer, "make_env", lambda *a, **kw: env)
    monkeypatch.setattr(trainer, "frame_count", lambda env: env.step.call_count * 4)
    monkeypatch.setattr(trainer, "QNetwork", lambda *a: torch.nn.Linear(1, 1))
    monkeypatch.setattr(trainer, "ReplayBuffer", lambda *a: replay)
    monkeypatch.setattr(trainer, "evaluate", evaluate)
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    monkeypatch.setattr(trainer.swanlab, "init", Mock())
    monkeypatch.setattr(trainer.swanlab, "log", log)
    monkeypatch.setattr(trainer.swanlab, "finish", finish)

    run_dir = trainer.train(args)

    assert env.step.call_count == expected_step
    assert evaluate.call_count == len(scores)
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    eval_records = [r for r in records if "eval/mean_return" in r]
    assert [r["eval/no_improvement_count"] for r in eval_records] == counts
    assert [r["eval/early_stopped"] for r in eval_records] == (
        [0] * (len(scores) - 1) + [1] if expected_step < 20 else [0] * len(scores)
    )
    assert records[-1]["step"] == expected_step
    assert [call.kwargs["step"] for call in log.call_args_list] == [r["step"] for r in records]
    assert [call.args[0] for call in log.call_args_list] == [
        {key: value for key, value in r.items() if key != "step"} for r in records
    ]
    best_steps = [call.args[5] for call in save.call_args_list if call.args[0].name == "best.pt"]
    expected_best = [
        (i + 1) * eval_frequency for i, score in enumerate(scores)
        if score > max(scores[:i], default=-float("inf"))
    ]
    assert best_steps == expected_best
    final = save.call_args_list[-1]
    assert final.args[0].name == "final.pt"
    assert final.args[5] == expected_step
    assert json.loads((run_dir / "config.json").read_text())["train"]["early_stopping_patience"] == patience
    env.close.assert_called_once()
    finish.assert_called_once()


def test_early_stopping_default():
    args = parser().parse_args([])
    assert args.early_stopping_patience == 3
    validate(args)


def test_negative_patience_rejected():
    args = parser().parse_args(["--early-stopping-patience", "-1"])
    with pytest.raises(ValueError, match="early-stopping-patience"):
        validate(args)
