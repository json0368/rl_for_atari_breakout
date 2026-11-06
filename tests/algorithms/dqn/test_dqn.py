import numpy as np
import pytest
import torch

from breakout_rl.algorithms.dqn.checkpoints import load_checkpoint
from breakout_rl.infrastructure.artifacts import save_checkpoint
from breakout_rl.algorithms.dqn.network import QNetwork
from breakout_rl.algorithms.dqn.replay import ReplayBuffer
from breakout_rl.algorithms.dqn.optimization import epsilon_at, td_target, update
from breakout_rl.envs import EnvConfig


def test_truncation_bootstraps_but_game_over_does_not():
    result = td_target(torch.tensor([1., 1.]), torch.tensor([False, True]),
                       torch.tensor([10., 10.]), 0.9)
    torch.testing.assert_close(result, torch.tensor([10., 1.]))


@pytest.mark.parametrize("capacity", [2, 3, 8, 50])
def test_replay_preserves_stacks_across_resets_and_overwrites(capacity):
    replay = ReplayBuffer(capacity, (4, 2, 2), seed=1)
    expected = {}
    obs = np.full((4,2,2), 1, np.uint8)
    for i in range(100):
        next_obs=np.concatenate((obs[1:],np.full((1,2,2),i+2,np.uint8)))
        terminal = i % 7 == 6
        truncated = i % 11 == 10 and not terminal
        replay.add(obs,next_obs,i,float(i),terminal,truncated)
        expected[i]=(obs.copy(),next_obs.copy(),terminal)
        if terminal or truncated:
            obs=np.full((4,2,2),150+i,np.uint8)
        else:
            obs=next_obs
        states, next_states, actions, rewards, dones=replay.sample(25,"cpu")
        for j, action in enumerate(actions.tolist()):
            np.testing.assert_array_equal(states[j],expected[action][0])
            np.testing.assert_array_equal(next_states[j],expected[action][1])
            assert rewards[j].item()==action
            assert dones[j].item()==expected[action][2]
    assert replay.size==capacity
    assert replay.image_nbytes==capacity*2*2


def test_epsilon_schedule_reaches_floor():
    assert epsilon_at(0, 1., 0.01, 100) == 1.
    assert epsilon_at(1000, 1., 0.01, 100) == pytest.approx(0.01)


def test_update_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    model, target = QNetwork(), QNetwork()
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model.network[-1].weight.detach().clone()
    obs = torch.randint(256, (2, 4, 84, 84), dtype=torch.uint8)
    batch = (obs, obs, torch.tensor([0, 1]), torch.tensor([1., -1.]), torch.tensor([True, False]))
    with torch.no_grad():
        values=model(obs).gather(1,batch[2][:,None]).squeeze(1)
        expected=td_target(batch[3],batch[4],target(obs).max(1).values,0.99)
        expected_mse=torch.nn.functional.mse_loss(values,expected).item()
    metrics = update(model, target, optimizer, batch, 0.99)
    assert metrics["losses/td_loss"] == pytest.approx(expected_mse)
    assert np.isfinite(metrics["losses/td_loss"])
    assert not torch.equal(before, model.network[-1].weight)
    assert all(p.grad is None for p in target.parameters())
    path = tmp_path / "test.pt"
    save_checkpoint(path, model, optimizer, EnvConfig(), {"seed": 1}, 123, 500)
    loaded, config, checkpoint = load_checkpoint(path, "cpu")
    torch.testing.assert_close(loaded(obs), model(obs))
    assert config == EnvConfig()
    assert checkpoint["step"] == 123
