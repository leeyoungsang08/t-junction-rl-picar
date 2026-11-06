import numpy as np
import torch
import pre_picar_study
from anchored_dqn import AnchoredDQN
from pre_picar_training import OLD
from src.dqn_agent import DQNAgent
from src.train import resolved_config


def test_zero_anchor_matches_original_dqn_update():
    _,cfg=resolved_config('full',algorithm='rainbow_lite',control_mode='direct')
    cfg.update(seed=123,device='cpu',state_dim=cfg['state_dim']+4,learning_start=8,batch_size=8,replay_capacity=32)
    original=DQNAgent(cfg);anchored=AnchoredDQN(cfg,OLD,0.)
    anchored.online.load_state_dict(original.online.state_dict());anchored.target.load_state_dict(original.target.state_dict())
    rng=np.random.default_rng(456)
    for i in range(12):
        s=rng.random(cfg['state_dim'],dtype=np.float32);ns=rng.random(cfg['state_dim'],dtype=np.float32)
        for agent in (original,anchored):agent.replay.add(s,i%3,float(i)/10,ns,False,False,[True,True,True],duration_steps=1.)
    original.global_step=anchored.global_step=12
    original.update();anchored.update()
    for k,v in original.online.state_dict().items():torch.testing.assert_close(v,anchored.online.state_dict()[k],rtol=0,atol=0)
