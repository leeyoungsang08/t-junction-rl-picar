"""Exploratory simulator-only fine-tuning with sensor-accessible guard state.

Fresh replay and optimizer; old model preserved. Simulated training reward uses
SUMO truth, so this is not an online PiCar trainer. Proposed action is learned
through a deterministic safeguard/fallback wrapper; guard ages are in state.
"""
import os, sys, json, time, argparse, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from pre_picar_study import PROJECT, HERE, LimitedSensor, LatchedGuard, atomic_json, versions
sys.path.insert(0,str(PROJECT))
from src.environment import TJunctionEnv
from src.dqn_agent import DQNAgent, atomic_torch_save
from src.train import resolved_config
from src.traffic_generator import generate_traffic
from src.baselines import direct_phase_baseline
from src.utils import sha256

OUT=HERE/'outputs'/'SUMO_PiCar_전_균형선택_재학습'
OLD=HERE/'work/robust_retrain/old_seed42.pt'


def vector(adapter, sensed, guard, sensor):
    # Four explicit controller-owned state features; age is based only on sensed queues.
    extra=np.asarray([min(x/80,3) for x in guard.ages]+[min(sensor.outage_steps/4,1)],dtype=np.float32)
    return np.concatenate((adapter.encode(sensed),extra)).astype(np.float32)


def init_agent(base, cfg):
    agent=DQNAgent(cfg)
    payload=torch.load(OLD,map_location=agent.device,weights_only=True)
    old=payload['state_dict']; new=agent.online.state_dict()
    for key,value in old.items():
        if key=='layers.0.weight':
            assert new[key].shape[1]==value.shape[1]+4
            new[key].zero_(); new[key][:,:value.shape[1]]=value
        else:
            assert new[key].shape==value.shape
            new[key].copy_(value)
    agent.online.load_state_dict(new); agent.target.load_state_dict(new)
    return agent


def episode(agent,base,scenario,seed,mapname,sensor_mode,output,training):
    route=generate_traffic(scenario,seed,900,'train' if training else 'validation',directory=OUT/'routes')
    net=PROJECT/'maps'/'experiment'/mapname/'map.net.xml'
    guard=LatchedGuard(80); sensor=LimitedSensor(sensor_mode,seed)
    env=TJunctionEnv(base,route,seed,output,network_file=net,service_observer=guard)
    total_reward=0.; overrides=0; updates=0
    try:
        _,info=env.reset(seed=seed)
        sensed,_=sensor.read(info['raw_state'],0); guard.set_observation(sensed)
        state=vector(env.adapter,sensed,guard,sensor)
        while True:
            proposed,_=agent.select(state,info['action_mask'],explore=training)
            if sensor.outage_steps>=3:
                preferred=direct_phase_baseline('fixed_time',{**info,'raw_state':sensed},base)
            else: preferred=proposed
            raw=info['raw_state']
            action,_=guard.choose(preferred,info['action_mask'],raw['phase'],raw['transitioning'],sensed['queues'],[0,0,0])
            overrides+=int(action!=proposed)
            _,reward,terminated,truncated,next_info=env.step(action)
            next_sensed,_=sensor.read(next_info['raw_state'],env.time//5)
            guard.set_observation(next_sensed)
            next_state=vector(env.adapter,next_sensed,guard,sensor)
            if training:
                duration=next_info['duration_seconds']; interval=base['simulation']['decision_seconds']
                discounted=sum(agent.config['gamma']**(i/interval)*row['reward'] for i,row in enumerate(env.timeseries[-duration:]))
                # The wrapper transforms proposed -> executed. Store the proposed action.
                agent.replay.add(state,proposed,discounted,next_state,terminated,truncated,next_info['action_mask'],duration_steps=duration/interval)
                agent.global_step+=1
                updates+=int(agent.update() is not None)
            total_reward+=reward
            state,info,sensed=next_state,next_info,next_sensed
            if terminated or truncated: break
        return {'scenario':scenario,'seed':seed,'map':mapname,'sensor':sensor_mode,'reward':total_reward,
                'guard_or_fallback_overrides':overrides,'episode_updates':updates,**info['metrics']}
    finally: env.close()


def validate(agent,base,run,ep):
    rows=[]
    for scenario in ('medium','imbalanced','south_val_stress'):
      for seed in (30050,30051):
       for mapname in ('validation0','validation1'):
        for sensor in ('limited_clean','limited_delay_miss'):
            rows.append(episode(agent,base,scenario,seed,mapname,sensor,run/'validation'/f'e{ep}_{scenario}_{seed}_{mapname}_{sensor}',False))
    scores=np.sort([r['cohort_delay_s'] for r in rows]); tail=scores[-max(1,int(np.ceil(.2*len(scores)))):]
    score=.5*float(np.mean(scores))+.5*float(np.mean(tail))
    return score,rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    p.add_argument('--episodes',type=int,default=100); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    base,cfg=resolved_config('full',algorithm='rainbow_lite',control_mode='direct')
    base['simulation'].update(backend='libsumo',horizon_seconds=900,audit_vehicle_safety=True)
    base['signal']['fixed_green']=15
    cfg.update(seed=a.seed,device='cpu',state_dim=cfg['state_dim']+4,episodes=a.episodes,
               learning_rate=.00003,epsilon_start=.2,epsilon_end=.05,epsilon_decay_steps=18000)
    if a.smoke:
        cfg.update(learning_start=16,batch_size=16)
    run=OUT/(f'smoke{a.seed}' if a.smoke else f'seed{a.seed}')
    run.mkdir(parents=True,exist_ok=True)
    plan={'seed':a.seed,'episodes':a.episodes,'old_sha256':sha256(OLD),'driver_sha256':sha256(Path(__file__)),
          'sensor_driver_sha256':sha256(HERE/'work/pre_picar_study.py'),'base':base,'training':cfg,'software':versions(),
          'note':'fine-tuning; validation selection requires non-regression in clean and delayed-miss groups; same initial weights, fresh optimizer and replay; simulator-only'}
    manifest=run/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf8'))!=plan: raise RuntimeError('Run plan mismatch')
    if not manifest.exists(): atomic_json(manifest,plan)
    agent=init_agent(base,cfg); checkpoint=run/'training_checkpoint.pt'; rows=[]; validation_rows=[]; best=float('inf'); next_ep=0
    if checkpoint.exists():
        c=agent.restore(checkpoint); e=c['extra']; next_ep=c['next_episode']; rows=e['rows']; validation_rows=e['validation_rows']; best=e['best']
    elif not a.smoke:
        best,vr=validate(agent,base,run,'initial'); validation_rows += [{'episode':0,**v} for v in vr]
        atomic_torch_save({'state_dict':agent.online.state_dict(),'base':base,'cfg':cfg,'episode':0,'validation_score':best},run/'best.pt')
        print('INITIAL VALIDATION',a.seed,best,flush=True)
    scenarios=cfg['train_scenarios']; start=time.monotonic(); limit=min(a.episodes,2) if a.smoke else a.episodes
    for ep in range(next_ep,limit):
        seed=10000+(a.seed-42)*600+ep
        scenario=scenarios[(ep+a.seed-42)%len(scenarios)]
        mapname=f'train{(ep+a.seed-42)%5}'
        sensor=('limited_clean','limited_delay_miss','limited_outage')[(ep//5+a.seed)%3]
        row=episode(agent,base,scenario,seed,mapname,sensor,run/'episodes'/f'{ep:04d}',True)
        rows.append({'episode':ep+1,'epsilon':agent.epsilon,'updates':agent.updates,**row})
        if not a.smoke and ((ep+1)%20==0 or ep+1==limit):
            score,vr=validate(agent,base,run,ep+1); validation_rows += [{'episode':ep+1,**v} for v in vr]
            initial=[r for r in validation_rows if r['episode']==0]
            initial_mean={s:float(np.mean([r['cohort_delay_s'] for r in initial if r['sensor']==s])) for s in ('limited_clean','limited_delay_miss')}
            current_mean={s:float(np.mean([r['cohort_delay_s'] for r in vr if r['sensor']==s])) for s in initial_mean}
            eligible=all(current_mean[s] <= initial_mean[s] for s in initial_mean)
            if eligible and score<best:
                best=score; atomic_torch_save({'state_dict':agent.online.state_dict(),'base':base,'cfg':cfg,'episode':ep+1,'validation_score':best},run/'best.pt')
            print('VALIDATE',a.seed,'ep',ep+1,'score',round(score,4),'eligible',eligible,'means',current_mean,'best',round(best,4),flush=True)
        if (ep+1)%5==0 or ep+1==limit:
            atomic_torch_save(agent.checkpoint(ep+1,{'rows':rows,'validation_rows':validation_rows,'best':best}),checkpoint)
            pd.DataFrame(rows).to_csv(run/'training.csv',index=False)
            pd.DataFrame(validation_rows).to_csv(run/'validation.csv',index=False)
            atomic_json(run/'status.json',{'completed_episodes':ep+1,'planned_episodes':limit,'updates':agent.updates,'complete':ep+1==limit})
        print('TRAIN',a.seed,ep+1,'updates',agent.updates,'elapsed',round(time.monotonic()-start),flush=True)


if __name__=='__main__': main()
