import json
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from breakout_rl.algorithms.ddqn.config import parser
from breakout_rl.algorithms.ddqn.optimization import next_q_values, update
from breakout_rl.algorithms.ddqn import trainer
from breakout_rl.algorithms.dqn import trainer as common
from breakout_rl.algorithms.dqn.config import parser as dqn_parser
from breakout_rl.algorithms.dqn.network import QNetwork
from breakout_rl.algorithms.checkpoints import load_checkpoint
from breakout_rl.infrastructure.artifacts import save_checkpoint


class TableNetwork(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table=torch.nn.Parameter(torch.tensor(table,dtype=torch.float32))

    def forward(self, indices):
        return self.table[indices]


def test_online_selects_and_target_evaluates_different_actions():
    online=TableNetwork([[0.,0.],[10.,1.]])
    target=TableNetwork([[0.,0.],[1.,100.]])
    following=torch.tensor([1,1])
    values=next_q_values(online,target,following)
    torch.testing.assert_close(values,torch.ones(2))
    assert not values.requires_grad
    assert target(following).max(1).values.tolist()==[100.,100.]
    batch=(torch.tensor([0,0]),following,torch.tensor([0,1]),torch.ones(2),torch.tensor([False,True]))
    optimizer=torch.optim.Adam(online.parameters(),lr=1e-4)
    metrics=update(online,target,optimizer,batch,.9)
    # One continuing/time-limited transition: 1 + .9 * 1; one terminal: 1.
    assert metrics['losses/td_loss']==pytest.approx((1.9**2+1)/2)
    assert target.table.grad is None
    torch.testing.assert_close(online.table.grad[1],torch.zeros(2))
    assert not torch.equal(online.table[0],torch.zeros(2))


def test_configuration_matches_dqn_baseline():
    ddqn=vars(parser().parse_args([]))
    assert ddqn.pop('algorithm')=='ddqn'
    assert ddqn==vars(dqn_parser().parse_args([]))


class ToyAtari(gym.Env):
    observation_space=gym.spaces.Box(0,255,(4,84,84),np.uint8)
    action_space=gym.spaces.Discrete(4)

    def __init__(self, interrupt_at=None):
        self.steps=0
        self.total_steps=0
        self.closed=False
        self.interrupt_at=interrupt_at

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps=0
        return np.zeros(self.observation_space.shape,np.uint8),{}

    def step(self, action):
        if self.total_steps==self.interrupt_at:
            raise KeyboardInterrupt
        self.steps+=1
        self.total_steps+=1
        end=self.steps==3
        info={'episode':{'r':3.,'l':12}} if end else {}
        return np.full(self.observation_space.shape,self.steps,np.uint8),1.,end,False,info

    def close(self):
        self.closed=True


@pytest.mark.parametrize('scores,patience,total,expected_step,counts',[
    ([5,5,4,5],3,20,16,[0,1,2,3]),
    ([5,4,6,6,5,6],3,28,24,[0,1,0,1,2,3]),
    ([5]*5,0,20,20,[0,1,2,3,4]),
])
def test_shared_training_uses_ddqn_updates_and_saves_identity(tmp_path,monkeypatch,scores,patience,total,expected_step,counts):
    args=parser().parse_args(['--output-dir',str(tmp_path),'--run-name','ddqn-test','--total-timesteps',str(total),
        '--buffer-size','32','--batch-size','2','--learning-starts','1','--train-frequency','1',
        '--target-network-frequency','4','--eval-frequency','4','--early-stopping-patience',str(patience),
        '--save-frequency','100','--log-frequency','1','--device','cpu','--torch-threads','1','--swanlab-mode','disabled'])
    env=ToyAtari()
    monkeypatch.setattr(common,'make_env',lambda *a,**kw:env)
    monkeypatch.setattr(common,'frame_count',lambda env:env.total_steps*4)
    for name in ('init','log','finish'): monkeypatch.setattr(common.swanlab,name,Mock())
    evaluation=Mock(side_effect=[{'mean_return':s,'std_return':0.} for s in scores])
    monkeypatch.setattr(common,'evaluate',evaluation)
    optimize=Mock(wraps=update);monkeypatch.setattr(trainer,'update',optimize)
    networks=[]
    def build(*args):
        model=QNetwork(*args)
        networks.append((model,model.network[-1].weight.detach().clone()))
        return model
    monkeypatch.setattr(common,'QNetwork',build)
    saved=Mock(wraps=save_checkpoint) if total==20 and patience==3 else Mock()
    monkeypatch.setattr(common,'save_checkpoint',saved)
    run=trainer.train(args)
    assert env.total_steps==expected_step and env.closed
    assert optimize.call_count>0 and evaluation.call_count==len(scores)
    assert not torch.equal(networks[0][0].network[-1].weight,networks[0][1])
    assert all(p.grad is None for p in networks[1][0].parameters())
    config=json.loads((run/'config.json').read_text())
    assert config['train']['algorithm']=='ddqn'
    records=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    evaluations=[r for r in records if 'eval/mean_return' in r]
    assert [r['eval/no_improvement_count'] for r in evaluations]==counts
    assert records[-1]['step']==expected_step
    assert records[-1]['eval/early_stopped']==int(expected_step<total)
    assert any('train/episodic_return' in r for r in records)
    assert np.isfinite(records[-1]['losses/td_loss'])
    assert all(c.kwargs['algorithm']=='ddqn' for c in saved.call_args_list)
    assert saved.call_args_list[-1].args[0].name=='final.pt'
    assert saved.call_args_list[-1].args[5]==expected_step
    assert len(common.swanlab.log.call_args_list)==len(records)
    common.swanlab.finish.assert_called_once()
    if total==20 and patience==3:
        loaded,_,payload=load_checkpoint(run/'final.pt','cpu')
        assert payload['algorithm']=='ddqn' and payload['step']==16
        assert payload['optimizer']['state']
        loaded,_,payload=load_checkpoint(run/'best.pt','cpu')
        assert payload['algorithm']=='ddqn' and payload['step']==4


def test_interrupt_checkpoint_retains_ddqn_identity(tmp_path,monkeypatch):
    args=parser().parse_args(['--output-dir',str(tmp_path),'--run-name','interrupt',
        '--total-timesteps','10','--buffer-size','4','--batch-size','2','--learning-starts','10',
        '--eval-frequency','0','--device','cpu','--torch-threads','1','--swanlab-mode','disabled'])
    env=ToyAtari(interrupt_at=3)
    monkeypatch.setattr(common,'make_env',lambda *a,**kw:env)
    monkeypatch.setattr(common,'frame_count',lambda env:env.total_steps*4)
    for name in ('init','log','finish'): monkeypatch.setattr(common.swanlab,name,Mock())
    run=trainer.train(args)
    model,_,payload=load_checkpoint(run/'interrupted.pt','cpu')
    assert payload['algorithm']=='ddqn' and payload['action_count']==4
    assert env.closed and not (run/'final.pt').exists()
