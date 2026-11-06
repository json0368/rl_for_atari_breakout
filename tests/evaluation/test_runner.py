from breakout_rl.envs import EnvConfig
from breakout_rl.evaluation import runner


def test_generic_evaluator_continues_after_life_loss_and_reports_raw_score(monkeypatch):
    class FakeEnv:
        frames=0
        resets=0
        closed=False
        def reset(self,**kwargs):
            self.resets+=1
            return self.frames,{}
        def step(self,action):
            self.frames+=1
            terminal=self.frames in (1,3)
            info={'episode':{'r':7.0}} if self.frames==3 else {}
            return self.frames,1.0,terminal,False,info
        def close(self): self.closed=True
    env=FakeEnv()
    monkeypatch.setattr(runner,'make_env',lambda *a,**k:env)
    monkeypatch.setattr(runner,'frame_count',lambda env:env.frames)
    seen=[]
    result=runner.evaluate(lambda obs,rng:seen.append(obs) or 0,EnvConfig(),episodes=1)
    assert result['mean_return']==7.0
    assert result['episodes'][0]['steps']==3
    assert env.resets==2 and env.closed
    assert len(seen)==3
