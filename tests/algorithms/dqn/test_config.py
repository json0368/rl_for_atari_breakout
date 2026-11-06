from breakout_rl.algorithms.dqn.config import parser
from breakout_rl.envs import EnvConfig
from breakout_rl.algorithms.dqn.checkpoints import load_checkpoint


def test_cleanrl_defaults():
    a=parser().parse_args([])
    assert (a.total_timesteps,a.buffer_size,a.learning_starts)==(10000000,1000000,80000)
    assert (a.learning_rate,a.batch_size,a.gamma)==(1e-4,32,0.99)
    assert (a.train_frequency,a.target_network_frequency)==(4,1000)
    assert (a.start_e,a.end_e,a.exploration_fraction)==(1.,0.01,0.1)
    assert a.max_grad_norm is None and a.sticky_action_probability==0.
    assert EnvConfig().protocol=='cleanrl'


def test_existing_checkpoint_uses_legacy_protocol(tmp_path):
    import torch
    from breakout_rl.algorithms.dqn.network import QNetwork
    from breakout_rl.infrastructure.artifacts import save_checkpoint
    model=QNetwork()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    path=tmp_path/'legacy.pt'
    old_env=EnvConfig(env_id='ALE/Breakout-v5',repeat_action_probability=0.25,protocol='legacy')
    save_checkpoint(path,model,optimizer,old_env,{},1,4)
    payload=torch.load(path,weights_only=True)
    del payload['env_config']['protocol']
    torch.save(payload,path)
    _,env,_=load_checkpoint(path,'cpu')
    assert env.protocol=='legacy' and env.repeat_action_probability==0.25
