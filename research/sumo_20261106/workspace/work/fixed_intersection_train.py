"""Paired robust fine-tuning on one frozen SUMO intersection.

Validation scenarios/seeds are disjoint from training. Test set is not opened.
Anchor is heuristic; no formal safety or non-regression guarantee is claimed.
"""
import argparse, json, os, shutil, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
import pre_picar_training as common
from pre_picar_study import HERE, PROJECT, atomic_json, versions
from anchored_dqn import AnchoredDQN
sys.path.insert(0,str(PROJECT))
from src.dqn_agent import atomic_torch_save
from src.train import resolved_config
from src.utils import sha256

OUT=HERE/'outputs'/'SUMO_balanceA_안정화_20261105'
OLD=HERE/'work/robust_retrain/old_seed42.pt'
SOURCE_MAP=HERE/'outputs/SUMO_PiCar_전_균형선택_새시험/inputs/maps/balanceA'
SCENARIOS=('low','medium','high','imbalanced','east_bias','south_bias','rush_hour','rotating','south_train_stress')
VAL_SCENARIOS=('medium','high','imbalanced','east_bias','rotating','south_val_stress')
VAL_SEEDS=(30080,30081)
SENSORS=('limited_clean','limited_delay_miss','limited_outage')
N_EPISODES=180
GROUP_TOLERANCE_S=.5
COMPLETION_TOLERANCE=.005
MAX_WAIT_REGRESSION_S=15
MIN_AVG_IMPROVEMENT_S=.25


def prepare():
    inputs=OUT/'inputs';marker=OUT/'manifest.json'
    if marker.exists():
        manifest=json.loads(marker.read_text(encoding='utf8'))
        for rel,h in manifest['inputs'].items():
            if sha256(OUT/rel)!=h:raise RuntimeError(f'Frozen input mismatch: {rel}')
        if manifest['software']!=versions():raise RuntimeError('Software version mismatch')
        for name in ('fixed_intersection_train.py','anchored_dqn.py','pre_picar_training.py','pre_picar_study.py'):
            if sha256(HERE/'work'/name)!=sha256(inputs/name):raise RuntimeError(f'Code changed: {name}')
        return manifest
    if inputs.exists() and any(inputs.iterdir()):raise RuntimeError('Incomplete previous inputs')
    inputs.mkdir(parents=True,exist_ok=True)
    for name in ('fixed_intersection_train.py','anchored_dqn.py','pre_picar_training.py','pre_picar_study.py'):
        shutil.copy2(HERE/'work'/name,inputs/name)
    shutil.copy2(OLD,inputs/'old.pt');shutil.copytree(SOURCE_MAP,inputs/'target_map')
    with zipfile.ZipFile(inputs/'project_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for part in ('src','maps','configs'):
            for f in (PROJECT/part).rglob('*'):
                if f.is_file() and '__pycache__' not in f.parts and 'experiment' not in f.parts:z.write(f,f.relative_to(PROJECT))
    manifest={'target_map_sha256':sha256(inputs/'target_map/map.net.xml'),'software':versions(),
      'training_scenarios':SCENARIOS,'validation_scenarios':VAL_SCENARIOS,'validation_seeds':VAL_SEEDS,'sensor_modes':SENSORS,
      'training_episodes_per_run':N_EPISODES,'validation_every':30,'anchor_weights':[0,.1],'training_seeds':[42,43,44],
      'selection':{'mean_delay_improvement_at_least_s':MIN_AVG_IMPROVEMENT_S,'max_group_mean_regression_s':GROUP_TOLERANCE_S,
       'completion_regression_tolerance':COMPLETION_TOLERANCE,'worst_wait_regression_tolerance_s':MAX_WAIT_REGRESSION_S},
      'notice':'Exploratory fixed intersection; reference anchored DQN heuristic; existing model retained; no universal guarantee.',
      'inputs':{str(f.relative_to(OUT)):sha256(f) for f in sorted(inputs.rglob('*')) if f.is_file()}}
    atomic_json(marker,manifest);return manifest


def initialize(base,cfg,weight):
    agent=AnchoredDQN(cfg,OUT/'inputs/old.pt',weight)
    p=torch.load(OUT/'inputs/old.pt',map_location=agent.device,weights_only=True);new=agent.online.state_dict()
    for k,v in p['state_dict'].items():
        if k=='layers.0.weight':
            assert new[k].shape[1]==v.shape[1]+4
            new[k].zero_();new[k][:,:v.shape[1]]=v
        else:
            assert new[k].shape==v.shape;new[k].copy_(v)
    agent.online.load_state_dict(new);agent.target.load_state_dict(new)
    return agent


def validate(agent,base,run,tag):
    items=[];target=(OUT/'inputs/target_map').resolve()
    for sc in VAL_SCENARIOS:
      for seed in VAL_SEEDS:
       for sensor in SENSORS:
        item=common.episode(agent,base,sc,seed,str(target),sensor,run/'validation'/f'{tag}_{sc}_{seed}_{sensor}',False)
        item['map']='balanceA';items.append(item)
    return items


def aggregate(rows):
    frame=pd.DataFrame(rows)
    return {'mean_delay':float(frame.cohort_delay_s.mean()),'mean_completion':float(frame.completion_rate.mean()),
            'max_wait':float(frame.max_consecutive_wait_s.max()),
            'groups':{f'{a}|{b}':float(g.cohort_delay_s.mean()) for (a,b),g in frame.groupby(['scenario','sensor'])}}


def run(seed,weight,smoke=False):
    prepare(); label='anchor' if weight else 'plain';run=OUT/('smoke' if smoke else f'{label}_seed{seed}')
    run.mkdir(parents=True,exist_ok=True);common.OUT=run
    base,cfg=resolved_config('full',algorithm='rainbow_lite',control_mode='direct')
    base['simulation'].update(backend='libsumo',horizon_seconds=900,audit_vehicle_safety=True)
    base['signal']['fixed_green']=15
    cfg.update(seed=seed,device='cpu',state_dim=cfg['state_dim']+4,episodes=N_EPISODES,
               learning_rate=.00003,epsilon_start=.2,epsilon_end=.05,epsilon_decay_steps=N_EPISODES*180)
    if smoke:cfg.update(learning_start=16,batch_size=16)
    plan={'seed':seed,'anchor_weight':weight,'cfg':cfg,'base':base,'study_manifest_sha256':sha256(OUT/'manifest.json'),'smoke':smoke}
    plan_file=run/'run_manifest.json'
    if plan_file.exists() and json.loads(plan_file.read_text(encoding='utf8'))!=plan:raise RuntimeError('Run differs from previous manifest')
    if not plan_file.exists():atomic_json(plan_file,plan)
    agent=initialize(base,cfg,weight); checkpoint=run/'training_checkpoint.pt'; rows=[]; valrows=[]; start_ep=0;best_score=float('inf')
    if checkpoint.exists():
        ck=agent.restore(checkpoint); start_ep=ck['next_episode']; info=ck['extra']
        rows=info['rows'];valrows=info['validation_rows'];best_score=info['best_score'];baseline=info['baseline']
    elif not smoke:
        baseline_rows=validate(agent,base,run,'initial');baseline=aggregate(baseline_rows)
        valrows.extend([{'episode':0,**r} for r in baseline_rows]);best_score=baseline['mean_delay']
        atomic_torch_save({'state_dict':agent.online.state_dict(),'base':base,'cfg':cfg,'episode':0,'validation_score':best_score,'anchor_weight':weight},run/'best.pt')
        atomic_json(run/'baseline.json',baseline)
        print('BASELINE',label,seed,baseline['mean_delay'],flush=True)
    else:baseline=None
    losses=[];orig_update=agent.update
    def track_update():
        result=orig_update()
        if result is not None:losses.append(result)
        return result
    agent.update=track_update;start=time.monotonic();limit=2 if smoke else N_EPISODES
    for ep in range(start_ep,limit):
        data_seed=13000+(seed-42)*300+ep
        scenario=SCENARIOS[(ep+(seed-42))%len(SCENARIOS)]
        sensor=SENSORS[(ep//len(SCENARIOS)+(seed-42))%len(SENSORS)]
        target=str((OUT/'inputs/target_map').resolve());losses.clear()
        row=common.episode(agent,base,scenario,data_seed,target,sensor,run/'episodes'/f'{ep:04d}',True)
        row['map']='balanceA'; row.update(episode=ep+1,epsilon=agent.epsilon,updates=agent.updates,
            td_loss=float(np.mean([x['td_loss'] for x in losses])) if losses else None,
            anchor_loss=float(np.mean([x['anchor_loss'] for x in losses])) if losses else None)
        rows.append(row)
        if not smoke and (ep+1)%30==0:
            current_rows=validate(agent,base,run,ep+1);valrows.extend([{'episode':ep+1,**r} for r in current_rows]);current=aggregate(current_rows)
            group_worst=max(current['groups'][k]-baseline['groups'][k] for k in baseline['groups'])
            eligible=(current['mean_delay']<=baseline['mean_delay']-MIN_AVG_IMPROVEMENT_S and
                group_worst<=GROUP_TOLERANCE_S and current['mean_completion']>=baseline['mean_completion']-COMPLETION_TOLERANCE and
                current['max_wait']<=baseline['max_wait']+MAX_WAIT_REGRESSION_S)
            atomic_json(run/f'validation_{ep+1}.json',{'current':current,'baseline':baseline,'worst_group_regression_s':group_worst,'eligible':eligible})
            if eligible and current['mean_delay']<best_score:
                best_score=current['mean_delay']
                atomic_torch_save({'state_dict':agent.online.state_dict(),'base':base,'cfg':cfg,'episode':ep+1,'validation_score':best_score,'anchor_weight':weight},run/'best.pt')
            # Retain all validation models so a failed criterion is auditable, not selected on final test.
            atomic_torch_save({'state_dict':agent.online.state_dict(),'base':base,'cfg':cfg,'episode':ep+1,'validation_score':current['mean_delay'],'anchor_weight':weight},run/f'candidate_{ep+1}.pt')
            print('VAL',label,seed,ep+1,'mean',round(current['mean_delay'],3),'worst_group',round(group_worst,3),'eligible',eligible,flush=True)
        if (ep+1)%5==0 or ep+1==limit:
            extra={'rows':rows,'validation_rows':valrows,'best_score':best_score,'baseline':baseline}
            atomic_torch_save(agent.checkpoint(ep+1,extra),checkpoint)
            pd.DataFrame(rows).to_csv(run/'training.csv',index=False);pd.DataFrame(valrows).to_csv(run/'validation.csv',index=False)
            atomic_json(run/'status.json',{'completed_episodes':ep+1,'planned_episodes':limit,'updates':agent.updates,'complete':ep+1==limit})
        if (ep+1)%10==0 or ep+1==limit:print('TRAIN',label,seed,ep+1,'updates',agent.updates,'seconds',round(time.monotonic()-start),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    p.add_argument('--weight',type=float,choices=(0.,.1),default=0.);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.seed,a.weight,a.smoke)
