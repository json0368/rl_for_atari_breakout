import json
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from breakout_rl.algorithms.pqn.config import parser, validate
from breakout_rl.algorithms.pqn.network import QNetwork
from breakout_rl.algorithms.pqn.optimization import q_lambda_returns, update
from breakout_rl.algorithms.pqn.policy import act, sample_actions, epsilon_at
from breakout_rl.algorithms.checkpoints import load_checkpoint
from breakout_rl.algorithms.pqn import trainer
from breakout_rl.envs import EnvConfig
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



def test_cleanrl_defaults_and_layer_norm():
    args=parser().parse_args([])
    validate(args)
    assert (args.total_timesteps,args.num_envs,args.num_steps)==(10_000_000,8,128)
    assert (args.learning_rate,args.q_lambda,args.gamma)==(2.5e-4,.65,.99)
    assert (args.num_minibatches,args.update_epochs,args.max_grad_norm)==(4,4,10.)
    assert args.early_stopping_patience==3 and args.anneal_lr
    torch.set_num_threads(1)
    model=QNetwork()
    assert [m.normalized_shape for m in model.modules() if isinstance(m,torch.nn.LayerNorm)] == [(32,20,20),(64,9,9),(64,7,7),(512,)]
    assert model(torch.zeros((2,4,84,84),dtype=torch.uint8)).shape==(2,4)


@pytest.mark.parametrize('flag,value',[("--q-lambda","2"),("--num-envs","0"),
    ("--early-stopping-patience","-1"),("--total-timesteps","9"),("--num-minibatches","2048"),
    ("--end-e","2"),("--exploration-fraction","0")])
def test_invalid_configuration(flag,value):
    with pytest.raises(ValueError): validate(parser().parse_args([flag,value]))


@pytest.mark.parametrize('lam,expected',[(0.,[6.4,9.2,12.]),(.5,[8.65,11.,12.]),(1.,[12.52,12.8,12.])])
def test_q_lambda_matches_hand_calculation(lam,expected):
    targets=q_lambda_returns(torch.tensor([[1.],[2.],[3.]]),torch.tensor([[4.],[6.],[8.]],requires_grad=True),
        torch.zeros((3,1),dtype=torch.bool),torch.tensor([10.],requires_grad=True),.9,lam)
    torch.testing.assert_close(targets.flatten(),torch.tensor(expected))
    assert not targets.requires_grad


def test_terminal_cuts_recursion_and_timeout_bootstraps_final_state():
    targets=q_lambda_returns(torch.tensor([[1.],[2.],[3.]]),torch.tensor([[4.],[6.],[8.]]),
        torch.tensor([[False],[True],[False]]),torch.tensor([10.]),.9,.5)
    torch.testing.assert_close(targets.flatten(),torch.tensor([4.6,2.,12.]))
    # First column is terminal; second column times out with final-state Q=10.
    # The next episode's Q=1000 must not enter either target.
    targets=q_lambda_returns(torch.tensor([[1.,1.+.9*10.]]),torch.zeros((1,2)),
        torch.ones((1,2),dtype=torch.bool),torch.full((2,),1000.),.9,.65)
    torch.testing.assert_close(targets,torch.tensor([[1.,10.]]))


def test_epsilon_schedule_and_independent_exploration():
    assert epsilon_at(0,1.,.01,100)==1.
    assert epsilon_at(1000,1.,.01,100)==pytest.approx(.01)
    q=torch.tensor([[0.,1.,2.,3.]]).expand(256,-1)
    assert (sample_actions(q,0.)==3).all()
    torch.manual_seed(3)
    random=sample_actions(q,1.)
    assert set(random.tolist())=={0,1,2,3}


def test_radam_update_keeps_targets_frozen_and_checkpoint_loads(tmp_path):
    torch.set_num_threads(1)
    model=QNetwork()
    optimizer=torch.optim.RAdam(model.parameters(),lr=2.5e-4)
    args=parser().parse_args(['--num-minibatches','1','--update-epochs','1'])
    obs=torch.randint(256,(2,4,84,84),dtype=torch.uint8)
    actions=torch.tensor([0,2])
    targets=torch.tensor([2.,-1.],requires_grad=True)
    with torch.no_grad(): expected=torch.nn.functional.mse_loss(model(obs).gather(1,actions[:,None]).squeeze(-1),targets).item()
    before=model.network[-1].weight.detach().clone()
    metrics=update(model,optimizer,(obs,actions,targets),args)
    assert metrics['losses/td_loss']==pytest.approx(expected)
    assert not torch.equal(model.network[-1].weight,before)
    assert targets.grad is None
    assert optimizer.param_groups[0]['eps']==1e-8
    path=tmp_path/'pqn.pt'
    save_checkpoint(path,model,optimizer,EnvConfig(),{},8,32,algorithm='pqn',action_count=np.int64(4))
    loaded,config,payload=load_checkpoint(path,'cpu')
    torch.testing.assert_close(loaded(obs),model(obs))
    assert payload['algorithm']=='pqn' and config==EnvConfig()
    rng1,rng2=np.random.default_rng(9),np.random.default_rng(9)
    state=torch.get_rng_state()
    assert [act(loaded,obs[0],.5,rng1,'cpu') for _ in range(8)] == [act(loaded,obs[0],.5,rng2,'cpu') for _ in range(8)]
    assert torch.equal(state,torch.get_rng_state())


@pytest.mark.parametrize('timeout,expected_last',[(False,1.),(True,1.9)])
def test_collection_uses_final_observation_for_timeout(tmp_path,monkeypatch,timeout,expected_last):
    args=parser().parse_args(['--num-envs','1','--num-steps','2','--num-minibatches','1','--total-timesteps','2',
        '--gamma','.9','--eval-frequency','0','--output-dir',str(tmp_path),'--run-name','boundary',
        '--device','cpu','--torch-threads','1','--swanlab-mode','disabled'])
    envs=gym.vector.SyncVectorEnv([lambda:ToyAtari(timeout=timeout)],autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    class KnownQ(torch.nn.Module):
        def __init__(self,*args):
            super().__init__()
            self.bias=torch.nn.Parameter(torch.zeros(4))
        def forward(self,obs):
            return obs.float().flatten(1).mean(1)[:,None]/10+self.bias[None,:]
    monkeypatch.setattr(trainer,'QNetwork',KnownQ)
    monkeypatch.setattr(trainer,'make_vector_env',lambda *args:envs)
    monkeypatch.setattr(trainer,'total_frames',lambda envs:sum(e.total_steps*4 for e in envs.envs))
    for name in ('init','log','finish'): monkeypatch.setattr(trainer.swanlab,name,Mock())
    captured=[]
    def fake_update(model,optimizer,rollout,args):
        captured.append(rollout[2].clone())
        return {'losses/td_loss':0.,'losses/q_mean':0.,'losses/grad_norm':0.}
    monkeypatch.setattr(trainer,'update',fake_update)
    trainer.train(args)
    assert captured[0][-1].item()==pytest.approx(expected_last)
    expected_first=1.+.9*(.65*expected_last+.35*1.)
    assert captured[0][0].item()==pytest.approx(expected_first)


@pytest.mark.parametrize('scores,patience,total,frequency,expected_step,counts',[
    ([5,5,4,5],3,40,8,32,[0,1,2,3]),
    ([5,4,6,6,5,6],3,64,8,48,[0,1,0,1,2,3]),
    ([5]*5,0,40,8,40,[0,1,2,3,4]),
    ([],3,20,0,20,[]),
])
def test_training_updates_logs_saves_and_stops(tmp_path,monkeypatch,scores,patience,total,frequency,expected_step,counts):
    args=parser().parse_args(['--num-envs','2','--num-steps','4','--num-minibatches','2','--update-epochs','1',
        '--total-timesteps',str(total),'--eval-frequency',str(frequency),'--early-stopping-patience',str(patience),
        '--output-dir',str(tmp_path),'--run-name','pqn-test','--device','cpu','--torch-threads','1','--swanlab-mode','disabled'])
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
        agent=QNetwork(*args)
        networks.append((agent,agent.network[-1].weight.detach().clone()))
        return agent
    monkeypatch.setattr(trainer,'QNetwork',build)
    result=trainer.train(args)
    records=[json.loads(line) for line in (result/'metrics.jsonl').read_text().splitlines()]
    evaluation_records=[r for r in records if 'eval/mean_return' in r]
    assert [r['eval/no_improvement_count'] for r in evaluation_records]==counts
    assert evaluation.call_count==len(scores)
    assert records[-1]['step']==expected_step
    assert sum(e.total_steps for e in envs.envs)==expected_step
    summary=json.loads((result/'training_summary.json').read_text())
    assert summary['early_stopped']==(expected_step<total)
    assert not torch.equal(networks[0][0].network[-1].weight,networks[0][1])
    assert all(e.closed for e in envs.envs)
    assert all(np.isfinite(records[-1][k]) for k in ('losses/td_loss','losses/q_mean','losses/grad_norm'))
    assert any('train/episodic_return' in r for r in records)
    assert len(trainer.swanlab.log.call_args_list)==len(records)
    assert saved.call_args_list[-1].args[0].name=='final.pt'
    assert saved.call_args_list[-1].args[5]==expected_step
    assert all(c.kwargs['algorithm']=='pqn' for c in saved.call_args_list)
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


def test_keyboard_interrupt_saves_pqn_snapshot(tmp_path, monkeypatch):
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
    assert payload["algorithm"]=="pqn" and payload["step"]==4
    assert not (run/"final.pt").exists()
    assert all(e.closed for e in envs.envs)
    trainer.swanlab.finish.assert_called_once()
