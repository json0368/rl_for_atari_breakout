import json
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from breakout_rl.algorithms.dueling.config import parser
from breakout_rl.algorithms.dueling.network import QNetwork
from breakout_rl.algorithms.dueling import trainer
from breakout_rl.algorithms.dqn import trainer as common
from breakout_rl.algorithms.dqn.config import parser as dqn_parser
from breakout_rl.algorithms.dqn.optimization import update
from breakout_rl.algorithms.checkpoints import load_checkpoint
from breakout_rl.infrastructure.artifacts import save_checkpoint


def test_value_advantage_aggregation_is_centered_and_shift_invariant():
    torch.set_num_threads(1)
    model=QNetwork()
    with torch.no_grad():
        model.value[-1].weight.zero_()
        model.value[-1].bias.fill_(2.)
        model.advantage[-1].weight.zero_()
        model.advantage[-1].bias.copy_(torch.tensor([1.,2.,3.,4.]))
    obs=torch.randint(256,(2,4,84,84),dtype=torch.uint8)
    q=model(obs)
    torch.testing.assert_close(q,torch.tensor([[.5,1.5,2.5,3.5]]).expand(2,-1))
    torch.testing.assert_close(q.mean(1),torch.full((2,),2.))
    with torch.no_grad(): model.advantage[-1].bias.add_(10.)
    torch.testing.assert_close(model(obs),q)
    # Input conversion is identical for uint8 and floating-point pixel values.
    torch.testing.assert_close(model(obs),model(obs.float()))


def test_configuration_matches_dqn():
    args=vars(parser().parse_args([]))
    assert args.pop('algorithm')=='dueling'
    assert args==vars(dqn_parser().parse_args([]))


def test_update_changes_both_heads_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    model,target=QNetwork(),QNetwork()
    target.load_state_dict(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    obs=torch.randint(256,(2,4,84,84),dtype=torch.uint8)
    actions=torch.tensor([0,1]);rewards=torch.tensor([1.,-1.]);terminated=torch.tensor([False,True])
    value_before=model.value[-1].weight.detach().clone()
    advantage_before=model.advantage[-1].weight.detach().clone()
    with torch.no_grad():
        expected=rewards+.99*(~terminated)*target(obs).max(1).values
        expected_loss=torch.nn.functional.mse_loss(model(obs).gather(1,actions[:,None]).squeeze(1),expected).item()
    metrics=update(model,target,optimizer,(obs,obs,actions,rewards,terminated),.99)
    assert metrics['losses/td_loss']==pytest.approx(expected_loss)
    assert not torch.equal(model.value[-1].weight,value_before)
    assert not torch.equal(model.advantage[-1].weight,advantage_before)
    assert all(p.grad is None for p in target.parameters())
    path=tmp_path/'dueling.pt'
    from breakout_rl.envs import EnvConfig
    save_checkpoint(path,model,optimizer,EnvConfig(),{},8,32,algorithm='dueling',action_count=np.int64(4))
    loaded,_,payload=load_checkpoint(path,'cpu')
    assert isinstance(loaded,QNetwork) and payload['algorithm']=='dueling'
    torch.testing.assert_close(loaded(obs),model(obs))


class ToyAtari(gym.Env):
    observation_space=gym.spaces.Box(0,255,(4,84,84),np.uint8)
    action_space=gym.spaces.Discrete(4)

    def __init__(self,interrupt_at=None):
        self.steps=0
        self.total_steps=0
        self.closed=False
        self.interrupt_at=interrupt_at

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps=0
        return np.zeros(self.observation_space.shape,np.uint8),{}

    def step(self,action):
        if self.total_steps==self.interrupt_at: raise KeyboardInterrupt
        self.steps+=1;self.total_steps+=1
        done=self.steps==3
        info={'episode':{'r':3.,'l':12}} if done else {}
        return np.full(self.observation_space.shape,self.steps,np.uint8),1.,done,False,info

    def close(self): self.closed=True


@pytest.mark.parametrize('patience,total,expected_step,scores',[(3,20,16,[5,5,4,5]),(0,20,20,[5]*5)])
def test_training_builds_dueling_online_and_target_and_preserves_early_stopping(tmp_path,monkeypatch,patience,total,expected_step,scores):
    args=parser().parse_args(['--output-dir',str(tmp_path),'--run-name','dueling-test','--total-timesteps',str(total),
        '--buffer-size','32','--batch-size','2','--learning-starts','1','--train-frequency','1',
        '--target-network-frequency','4','--eval-frequency','4','--early-stopping-patience',str(patience),
        '--save-frequency','100','--log-frequency','1','--device','cpu','--torch-threads','1','--swanlab-mode','disabled'])
    env=ToyAtari()
    monkeypatch.setattr(common,'make_env',lambda *a,**kw:env)
    monkeypatch.setattr(common,'frame_count',lambda env:env.total_steps*4)
    for name in ('init','log','finish'): monkeypatch.setattr(common.swanlab,name,Mock())
    evaluation=Mock(side_effect=[{'mean_return':s,'std_return':0.} for s in scores])
    monkeypatch.setattr(common,'evaluate',evaluation)
    optimize=Mock(wraps=update);monkeypatch.setattr(common,'update',optimize)
    networks=[]
    def build(*args):
        model=QNetwork(*args)
        networks.append((model,model.value[-1].weight.detach().clone(),model.advantage[-1].weight.detach().clone()))
        return model
    monkeypatch.setattr(trainer,'QNetwork',build)
    saved=Mock(wraps=save_checkpoint) if patience==3 else Mock()
    monkeypatch.setattr(common,'save_checkpoint',saved)
    run=trainer.train(args)
    assert len(networks)==2 and all(isinstance(n[0],QNetwork) for n in networks)
    assert optimize.call_count>0 and not any('next_value_fn' in c.kwargs for c in optimize.call_args_list)
    assert env.total_steps==expected_step and env.closed
    assert not torch.equal(networks[0][0].value[-1].weight,networks[0][1])
    assert not torch.equal(networks[0][0].advantage[-1].weight,networks[0][2])
    assert all(p.grad is None for p in networks[1][0].parameters())
    assert json.loads((run/'config.json').read_text())['train']['algorithm']=='dueling'
    records=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    evaluations=[r for r in records if 'eval/mean_return' in r]
    assert [r['eval/no_improvement_count'] for r in evaluations]==list(range(len(scores)))
    assert records[-1]['step']==expected_step
    assert records[-1]['eval/early_stopped']==int(expected_step<total)
    assert np.isfinite(records[-1]['losses/td_loss'])
    assert any('train/episodic_return' in r for r in records)
    assert all(c.kwargs['algorithm']=='dueling' for c in saved.call_args_list)
    assert saved.call_args_list[-1].args[0].name=='final.pt'
    assert saved.call_args_list[-1].args[5]==expected_step
    assert len(common.swanlab.log.call_args_list)==len(records)
    common.swanlab.finish.assert_called_once()
    if patience==3:
        loaded,_,payload=load_checkpoint(run/'final.pt','cpu')
        assert isinstance(loaded,QNetwork) and payload['step']==16 and payload['optimizer']['state']
        _,_,payload=load_checkpoint(run/'best.pt','cpu')
        assert payload['algorithm']=='dueling' and payload['step']==4


def test_interrupt_saves_dueling_network(tmp_path,monkeypatch):
    args=parser().parse_args(['--output-dir',str(tmp_path),'--run-name','interrupt',
        '--total-timesteps','10','--buffer-size','4','--batch-size','2','--learning-starts','10',
        '--eval-frequency','0','--device','cpu','--torch-threads','1','--swanlab-mode','disabled'])
    env=ToyAtari(interrupt_at=3)
    monkeypatch.setattr(common,'make_env',lambda *a,**kw:env)
    monkeypatch.setattr(common,'frame_count',lambda env:env.total_steps*4)
    for name in ('init','log','finish'): monkeypatch.setattr(common.swanlab,name,Mock())
    run=trainer.train(args)
    model,_,payload=load_checkpoint(run/'interrupted.pt','cpu')
    assert isinstance(model,QNetwork) and payload['algorithm']=='dueling'
    assert env.closed and not (run/'final.pt').exists()
