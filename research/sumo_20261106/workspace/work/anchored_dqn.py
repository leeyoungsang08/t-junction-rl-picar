"""Experimental DQN with a soft baseline Q-advantage regularizer.

This is a heuristic, not SPIBB, not a formal policy-improvement guarantee.
"""
import numpy as np
import torch
from src.dqn_agent import DQNAgent, masked_bootstrap
from src.inference import Policy


class AnchoredDQN(DQNAgent):
    def __init__(self,config,teacher_path,anchor_weight):
        super().__init__(config)
        self.anchor_weight=float(anchor_weight)
        self.teacher=Policy(teacher_path,device=str(self.device)).model
        for p in self.teacher.parameters():p.requires_grad_(False)
        self.teacher_dim=self.teacher.layers[0].in_features

    def update(self):
        c=self.config
        if self.replay.size<max(c['batch_size'],c['learning_start']) or self.global_step%c['update_every']:
            return None
        bstart=c.get('per_beta_start',.4);bend=c.get('per_beta_end',1.);bsteps=max(int(c.get('per_beta_steps',100000)),1)
        beta=bstart+(bend-bstart)*min(1.,self.global_step/bsteps)
        batch=self.replay.sample(c['batch_size'],self.rng,beta=beta)
        b={k:torch.as_tensor(v,device=self.device) for k,v in batch.items() if k!='indices'}
        all_q=self.online(b['states']);q=all_q.gather(1,b['actions'][:,None]).squeeze(1)
        with torch.no_grad():
            done=b['terminated'] if c['bootstrap_truncated'] else b['terminated']|b['truncated']
            live=~done;target=b['rewards'].clone()
            if live.any():
                ns=b['next_states'][live];target_v=self.target(ns)
                online_v=self.online(ns) if c['double_dqn'] else target_v
                next_q=masked_bootstrap(online_v,target_v,b['next_masks'][live],c['double_dqn'])
                target[live]+=c['gamma']**b['duration_steps'][live]*next_q
            teacher_q=self.teacher(b['states'][:,:self.teacher_dim])
            teacher_adv=teacher_q-teacher_q.mean(dim=1,keepdim=True)
        td=target-q
        td_loss=(b['weights']*torch.nn.functional.smooth_l1_loss(q,target,reduction='none')).mean()
        advantage=all_q-all_q.mean(dim=1,keepdim=True)
        anchor_loss=torch.nn.functional.smooth_l1_loss(advantage,teacher_adv)
        loss=td_loss+self.anchor_weight*anchor_loss
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.online.parameters(),c['gradient_clip'])
        self.optimizer.step();self.replay.update_priorities(batch['indices'],td.detach().abs().cpu().numpy())
        self.updates+=1
        if self.updates%c['target_update_steps']==0:self.target.load_state_dict(self.online.state_dict())
        return {'loss':float(loss.item()),'td_loss':float(td_loss.item()),'anchor_loss':float(anchor_loss.item()),'gradient_norm':float(norm)}
