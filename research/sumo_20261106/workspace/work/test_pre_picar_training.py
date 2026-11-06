import numpy as np
import torch
import pre_picar_training as training
from pre_picar_training import init_agent, OLD
from pre_picar_study import LatchedGuard
from src.train import resolved_config
from src.inference import Policy


def test_expanded_model_preserves_original_q_values():
    base,cfg=resolved_config('full',algorithm='rainbow_lite',control_mode='direct')
    cfg.update(seed=42,device='cpu',state_dim=cfg['state_dim']+4)
    agent=init_agent(base,cfg)
    original=Policy(OLD)
    rng=np.random.default_rng(123)
    x=torch.tensor(rng.random((12,cfg['state_dim']-4)),dtype=torch.float32)
    extra=torch.tensor(rng.random((12,4)),dtype=torch.float32)
    with torch.inference_mode():
        before=original.model(x)
        after=agent.online(torch.cat((x,extra),dim=1))
    torch.testing.assert_close(before,after,rtol=0,atol=1e-6)


def test_replay_records_proposed_action_through_guard(tmp_path,monkeypatch):
    class ForceOverride(LatchedGuard):
        def choose(self,preferred,mask,phase,transitioning,queues,pending):
            choices=[i for i,legal in enumerate(mask) if legal and i!=preferred]
            return (choices[0],choices[0]) if choices and not transitioning else (preferred,None)

    base,cfg=resolved_config('full',algorithm='rainbow_lite',control_mode='direct')
    base['simulation'].update(backend='libsumo',horizon_seconds=900,audit_vehicle_safety=True)
    cfg.update(seed=42,device='cpu',state_dim=cfg['state_dim']+4,learning_start=100000)
    agent=init_agent(base,cfg); proposed=[]; applied=[]; stored=[]
    def select(_state,mask,explore=True):
        action=int(np.flatnonzero(mask)[0]);proposed.append(action)
        return action,np.zeros(3)
    original_choose=ForceOverride.choose
    def choose(self,*args):
        result=original_choose(self,*args);applied.append(result[0]);return result
    original_add=agent.replay.add
    def add(state,action,*args,**kwargs):
        stored.append(action);return original_add(state,action,*args,**kwargs)
    monkeypatch.setattr(ForceOverride,'choose',choose)
    monkeypatch.setattr(training,'LatchedGuard',ForceOverride)
    monkeypatch.setattr(agent,'select',select);monkeypatch.setattr(agent.replay,'add',add)
    training.episode(agent,base,'medium',10060,'train0','limited_clean',tmp_path,True)
    assert stored==proposed and len(stored)==180
    assert any(p!=a for p,a in zip(proposed,applied))
