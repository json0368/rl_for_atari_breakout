import json
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from breakout_rl.algorithms.ppo.config import parser, validate
from breakout_rl.algorithms.ppo.network import ActorCritic
from breakout_rl.algorithms.ppo.optimization import advantages_and_returns, objective
from breakout_rl.algorithms.ppo.policy import act
from breakout_rl.algorithms.checkpoints import load_checkpoint
from breakout_rl.algorithms.ppo import trainer
from breakout_rl.envs import EnvConfig
from breakout_rl.envs import vector
from breakout_rl.infrastructure.artifacts import save_checkpoint


class ToyAtari(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
    action_space = gym.spaces.Discrete(4)

    def __init__(self, *, timeout=False):
        self.timeout = timeout
        self.steps = 0
        self.total_steps = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.full(self.observation_space.shape, 200, np.uint8), {}

    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        end = self.steps == 2
        info = {"episode": {"r": 7.0, "l": 8}} if end else {}
        return np.full(self.observation_space.shape, 10, np.uint8), 1.0, end and not self.timeout, end and self.timeout, info

    def close(self):
        self.closed = True


def test_defaults():
    a = parser().parse_args([])
    validate(a)
    assert (a.num_envs, a.num_steps, a.num_minibatches, a.update_epochs) == (8,128,4,4)
    assert (a.learning_rate, a.gamma, a.gae_lambda, a.clip_coef) == (2.5e-4,.99,.95,.1)
    assert (a.ent_coef, a.vf_coef, a.max_grad_norm, a.early_stopping_patience) == (.01,.5,.5,3)
    assert a.total_timesteps == 10_000_000 and a.anneal_lr and a.norm_adv and a.clip_vloss


@pytest.mark.parametrize("flag,value", [("--num-envs","0"),("--early-stopping-patience","-1"),
    ("--total-timesteps","9"),("--num-minibatches","1024"),("--gae-lambda","2")])
def test_invalid_configuration(flag, value):
    with pytest.raises(ValueError):
        validate(parser().parse_args([flag,value]))


def test_gae_matches_hand_calculation_and_cuts_at_boundaries():
    r = torch.tensor([[1.],[5.],[2.]])
    v = torch.tensor([[1.],[2.],[3.]])
    adv, returns = advantages_and_returns(r, v, torch.tensor([[False],[True],[False]]), torch.tensor([4.]), .9, .95)
    torch.testing.assert_close(adv.flatten(), torch.tensor([4.365,3.,2.6]))
    torch.testing.assert_close(returns.flatten(), torch.tensor([5.365,5.,5.6]))


def test_timeout_bootstraps_final_state_and_never_next_episode():
    # Final-state value is 10; next episode has value 1000 and must not leak in.
    rewards = torch.tensor([[5. + .9 * 10.], [1.]])
    values = torch.tensor([[2.], [1000.]])
    adv, returns = advantages_and_returns(rewards, values, torch.tensor([[True],[False]]), torch.tensor([1000.]), .9, .95)
    assert returns[0].item() == pytest.approx(14.)
    assert adv[0].item() == pytest.approx(12.)


def test_clipped_objective_matches_hand_calculation():
    a=parser().parse_args(["--no-norm-adv"])
    loss, metrics = objective(torch.tensor([1.5,.5]).log(), torch.zeros(2), torch.full((2,),.5),
        torch.tensor([4.,-4.]), torch.zeros(2), torch.tensor([2.,-2.]), torch.tensor([1.,-1.]), a)
    assert metrics["losses/policy_loss"] == pytest.approx(-.2)
    assert metrics["losses/value_loss"] == pytest.approx(4.5)
    assert metrics["losses/clipfrac"] == 1
    assert loss.item() == pytest.approx(2.045)
    _, metrics = objective(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.full((2,),4.),
                          torch.zeros(2), torch.ones(2), torch.full((2,),5.), a)
    assert metrics["losses/value_loss"] == pytest.approx(12.005)


def test_same_step_vector_reset_preserves_terminal_data(monkeypatch):
    monkeypatch.setattr(vector, "make_env", lambda *a, **kw: ToyAtari(timeout=True))
    envs=vector.make_vector_env(EnvConfig(),2)
    try:
        envs.reset(seed=1)
        envs.step(np.zeros(2,dtype=int))
        obs,reward,term,trunc,infos=envs.step(np.zeros(2,dtype=int))
        indices, final_obs=vector.timeout_observations(infos,term,trunc)
        assert indices.tolist()==[0,1]
        assert (final_obs==10).all() and (obs==200).all()
        assert list(vector.episode_records(infos))==[(7.,8),(7.,8)]
        _,reward,term,trunc,_=envs.step(np.zeros(2,dtype=int))
        assert reward.tolist()==[1.,1.] and not (term|trunc).any()
    finally:
        envs.close()


def test_checkpoint_and_sampling_roundtrip(tmp_path):
    torch.set_num_threads(1)
    model=ActorCritic()
    optimizer=torch.optim.Adam(model.parameters(),lr=2.5e-4,eps=1e-5)
    path=tmp_path/'ppo.pt'
    save_checkpoint(path,model,optimizer,EnvConfig(),{},np.int64(128),np.int64(512),
                    algorithm='ppo',action_count=np.int64(4))
    loaded,config,payload=load_checkpoint(path,'cpu')
    obs=np.full((4,84,84),100,np.uint8)
    inputs=torch.as_tensor(obs).unsqueeze(0)
    torch.testing.assert_close(model.get_value(inputs),loaded.get_value(inputs))
    assert config==EnvConfig() and payload['algorithm']=='ppo'
    rng1,rng2=np.random.default_rng(3),np.random.default_rng(3)
    state=torch.get_rng_state()
    assert [act(loaded,obs,rng1,'cpu') for _ in range(8)] == [act(loaded,obs,rng2,'cpu') for _ in range(8)]
    assert torch.equal(state,torch.get_rng_state())
    distribution,_=loaded.distribution_and_value(inputs)
    assert act(loaded,obs,rng1,'cpu',deterministic=True)==distribution.probs.argmax().item()


@pytest.mark.parametrize('scores,patience,total,frequency,expected_step,counts',[
    ([5,5,4,5],3,40,8,32,[0,1,2,3]),
    ([5,4,6,6,5,6],3,64,8,48,[0,1,0,1,2,3]),
    ([5]*5,0,40,8,40,[0,1,2,3,4]),
    ([],3,20,0,20,[]),
])
def test_training_updates_logs_saves_and_stops(tmp_path,monkeypatch,scores,patience,total,frequency,expected_step,counts):
    args=parser().parse_args(['--num-envs','2','--num-steps','4','--num-minibatches','2','--update-epochs','1',
        '--total-timesteps',str(total),'--eval-frequency',str(frequency),'--early-stopping-patience',str(patience),
        '--output-dir',str(tmp_path),'--run-name','ppo-test','--device','cpu','--torch-threads','1','--swanlab-mode','disabled'])
    envs=gym.vector.SyncVectorEnv([ToyAtari,ToyAtari],autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    monkeypatch.setattr(trainer,'make_vector_env',lambda *a:envs)
    monkeypatch.setattr(trainer,'total_frames',lambda envs:sum(e.total_steps*4 for e in envs.envs))
    evaluation=Mock(side_effect=[{'mean_return':s,'std_return':0.} for s in scores])
    monkeypatch.setattr(trainer,'evaluate',evaluation)
    saved=Mock(wraps=save_checkpoint) if scores==[5,5,4,5] else Mock()
    monkeypatch.setattr(trainer,'save_checkpoint',saved)
    for name in ('init','log','finish'):
        monkeypatch.setattr(trainer.swanlab,name,Mock())
    networks=[]
    def build(*args):
        agent=ActorCritic(*args)
        networks.append((agent,agent.actor.weight.detach().clone()))
        return agent
    monkeypatch.setattr(trainer,'ActorCritic',build)
    result=trainer.train(args)
    records=[json.loads(line) for line in (result/'metrics.jsonl').read_text().splitlines()]
    evaluation_records=[r for r in records if 'eval/mean_return' in r]
    assert [r['eval/no_improvement_count'] for r in evaluation_records]==counts
    assert evaluation.call_count==len(scores)
    assert records[-1]['step']==expected_step
    assert sum(e.total_steps for e in envs.envs)==expected_step
    summary=json.loads((result/'training_summary.json').read_text())
    assert summary['early_stopped']==(expected_step<total)
    assert not torch.equal(networks[0][0].actor.weight,networks[0][1])
    assert all(e.closed for e in envs.envs)
    assert all(np.isfinite(records[-1][k]) for k in ('losses/policy_loss','losses/value_loss','losses/grad_norm'))
    assert any('train/episodic_return' in r for r in records)
    assert len(trainer.swanlab.log.call_args_list)==len(records)
    assert saved.call_args_list[-1].args[0].name=='final.pt'
    assert saved.call_args_list[-1].args[5]==expected_step
    assert all(c.kwargs['algorithm']=='ppo' for c in saved.call_args_list)
    expected_best=[(i+1)*frequency for i,s in enumerate(scores) if s>max(scores[:i],default=-float('inf'))]
    assert [c.args[5] for c in saved.call_args_list if c.args[0].name=='best.pt']==expected_best
    if expected_step<total:
        assert records[-1]['eval/early_stopped']==1
    trainer.swanlab.finish.assert_called_once()
    if scores==[5,5,4,5]:
        loaded,_,payload=load_checkpoint(result/'final.pt','cpu')
        assert payload['step']==32 and payload['optimizer']['state']
        best,_,payload=load_checkpoint(result/'best.pt','cpu')
        assert payload['step']==8


@pytest.mark.parametrize("algorithm", ["dqn", "ppo", "pqn"])
def test_evaluate_cli_selects_checkpoint_policy(tmp_path, monkeypatch, algorithm):
    from breakout_rl.cli import evaluate as command
    from breakout_rl.algorithms.dqn.network import QNetwork
    torch.set_num_threads(1)
    from breakout_rl.algorithms.pqn.network import QNetwork as PQNetwork
    model = {"ppo": ActorCritic, "dqn": QNetwork, "pqn": PQNetwork}[algorithm]()
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = tmp_path / f"{algorithm}.pt"
    save_checkpoint(checkpoint, model, optimizer, EnvConfig(), {}, 8, 32, algorithm=algorithm)
    output = tmp_path / "result.json"
    def fake_evaluate(select_action, config, **kwargs):
        action = select_action(np.zeros((4,84,84),np.uint8), np.random.default_rng(2))
        assert 0 <= action < 4
        assert kwargs["render_mode"] == "human"
        return {"mean_return":7.0,"std_return":0.0}
    monkeypatch.setattr(command, "evaluate", fake_evaluate)
    args = command.parser().parse_args(["--checkpoint",str(checkpoint),"--output",str(output),
                                       "--device","cpu","--deterministic"])
    command.run(args, human=True)
    result=json.loads(output.read_text())
    assert result["algorithm"]==algorithm
    assert (result["deterministic"] is True) if algorithm=="ppo" else (result["epsilon"]==0.01)


def test_keyboard_interrupt_saves_ppo_snapshot(tmp_path, monkeypatch):
    args=parser().parse_args(["--num-envs","2","--num-steps","4","--num-minibatches","2",
        "--total-timesteps","16","--output-dir",str(tmp_path),"--run-name","interrupt",
        "--device","cpu","--torch-threads","1","--swanlab-mode","disabled"])
    envs=gym.vector.SyncVectorEnv([ToyAtari,ToyAtari],autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    original_step=envs.step
    calls=0
    def step(actions):
        nonlocal calls
        calls += 1
        if calls==3:
            raise KeyboardInterrupt
        return original_step(actions)
    monkeypatch.setattr(envs,"step",step)
    monkeypatch.setattr(trainer,"make_vector_env",lambda *a:envs)
    monkeypatch.setattr(trainer,"total_frames",lambda envs:sum(e.total_steps*4 for e in envs.envs))
    for name in ("init","log","finish"):
        monkeypatch.setattr(trainer.swanlab,name,Mock())
    run=trainer.train(args)
    model,config,payload=load_checkpoint(run/"interrupted.pt","cpu")
    assert payload["algorithm"]=="ppo" and payload["step"]==4
    assert not (run/"final.pt").exists()
    assert all(e.closed for e in envs.envs)
    trainer.swanlab.finish.assert_called_once()
