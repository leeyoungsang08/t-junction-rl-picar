"""Test periodic direction-service watchdog with earlier learned policy.

This is a simulation experiment, not a guarantee of individual waiting time or
an implementation of physical traffic safety equipment.
"""
import argparse,json,os,shutil,sys,tempfile,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from pre_picar_study import HERE,PROJECT,LimitedSensor,LatchedGuard,atomic_json,versions
from pre_picar_balanced_evaluate import NewPolicy
from fixed_intersection_evaluate import SevereSensor
sys.path.insert(0,str(PROJECT))
from src.environment import TJunctionEnv
from src.inference import Policy
from src.baselines import direct_phase_baseline
from src.traffic_generator import generate_traffic
from src.utils import sha256

SOURCE=HERE/'outputs/SUMO_balanceA_독립시험_20261105'
OUT=HERE/'outputs/SUMO_balanceA_장시간대기_추가시험'
SCENARIOS=('low','medium','high','imbalanced','heldout_reversal','south_oversaturated')
VSCENARIOS=('medium','high','imbalanced','east_bias','rotating','south_val_stress')
VS=(30084,30085);TS=(70993,70994,70995)
VSENSORS=('limited_clean','limited_delay_miss','limited_outage')
TSENSORS=(*VSENSORS,'severe_unseen')
VAL_CONTROLLERS=('old','prior43','watch70','watch90')
TEST_CONTROLLERS=(*VAL_CONTROLLERS,'selected')


class Watchdog(LatchedGuard):
    def __init__(self,threshold):
        self.watch_threshold=threshold;super().__init__(80)
    def reset(self):
        super().reset();self.red_age=[0,0,0]
    def observe_second(self,phase,stage,queues,pending):
        super().observe_second(phase,stage,queues,pending)
        self.red_age=[0 if stage=='green' and d==phase else a+1 for d,a in enumerate(self.red_age)]
    def request(self,preferred,mask,phase,transitioning):
        if self.watch_threshold is None:return preferred
        possible=[d for d in range(3) if d!=phase and mask[d] and self.red_age[d]>=self.watch_threshold]
        return max(possible,key=lambda d:(self.red_age[d],-((d-phase)%3))) if possible and not transitioning else preferred


def setup():
    inp=OUT/'inputs';marker=OUT/'manifest.json'
    if marker.exists():
        m=json.loads(marker.read_text(encoding='utf8'))
        if m['software']!=versions() or m['driver_sha256']!=sha256(Path(__file__)):raise RuntimeError('Code/software changed')
        for rel,h in m['inputs'].items():
            if sha256(OUT/rel)!=h:raise RuntimeError(f'Input mismatch {rel}')
        return m
    if inp.exists() and any(inp.iterdir()):raise RuntimeError('Incomplete input directory')
    source=json.loads((SOURCE/'manifest.json').read_text(encoding='utf8'))
    for rel,h in source['inputs'].items():
        if sha256(SOURCE/rel)!=h:raise RuntimeError(f'Prior input mismatch {rel}')
    inp.mkdir(parents=True,exist_ok=True)
    for name in ('fixed_intersection_watchdog.py','fixed_intersection_evaluate.py','fixed_intersection_gate.py',
                 'pre_picar_study.py','pre_picar_training.py','pre_picar_balanced_evaluate.py'):
        shutil.copy2(HERE/'work'/name,inp/name)
    for name in ('old.pt','prior43.pt','config.yaml','project_source.zip'):shutil.copy2(SOURCE/'inputs'/name,inp/name)
    shutil.copytree(SOURCE/'inputs/target_map',inp/'target_map')
    for sc in VSCENARIOS:
      for seed in VS:generate_traffic(sc,seed,900,'validation',directory=inp/'routes')
    for sc in SCENARIOS:
      for seed in TS:generate_traffic(sc,seed,900,'final',directory=inp/'routes')
    m={'driver_sha256':sha256(Path(__file__)),'software':versions(),'validation_scenarios':VSCENARIOS,'validation_seeds':VS,
       'validation_sensors':VSENSORS,'test_scenarios':SCENARIOS,'test_seeds':TS,'test_sensors':TSENSORS,
       'val_controllers':VAL_CONTROLLERS,'test_controllers':TEST_CONTROLLERS,
       'selection':'versus old on validation: mean improves 0.25s, every scenario/sensor mean regression <=0.5s, individual regression <=1s, completion >= old-.005 and worst observed wait <=old',
       'watchdog':'request oldest red direction at 70/90s regardless of sensed traffic, legal signal mask/clearance remain active',
       'scope':'synthetic one-junction study; this does not guarantee vehicle wait or hardware safety',
       'inputs':{str(f.relative_to(OUT)):sha256(f) for f in sorted(inp.rglob('*')) if f.is_file()}}
    atomic_json(marker,m);return m


def run_case(phase,scenario,seed,sensor_mode,label,selected,models,cfg):
    dest=OUT/'runs'/phase/scenario/str(seed)/sensor_mode/label;rec=dest/'record.json';rawzip=dest/'raw.zip';inp=OUT/'inputs'
    if rec.exists() and rawzip.exists():
        r=json.loads(rec.read_text(encoding='utf8'))
        if sha256(rawzip)!=r['raw_sha256']:raise RuntimeError('Saved raw file changed')
        return r
    if rec.exists() or rawzip.exists():raise RuntimeError('Partial result; inspect')
    applied=(selected if selected is not None else 'old') if label=='selected' else label
    threshold={'watch70':70,'watch90':90}.get(applied)
    net=inp/'target_map/map.net.xml';split='validation' if phase=='validation' else 'final'
    route=inp/'routes'/split/f'{scenario}_seed{seed}_t900.rou.xml'
    guard=Watchdog(threshold);sensor=SevereSensor(seed) if sensor_mode=='severe_unseen' else LimitedSensor(sensor_mode,seed)
    changes=watch_changes=0;decisions=[]
    with tempfile.TemporaryDirectory(prefix='watchdog_study_') as tmp:
        env=TJunctionEnv(cfg,route,seed,Path(tmp),network_file=net,service_observer=guard)
        try:
            _,info=env.reset(seed=seed)
            while True:
                raw=info['raw_state'];sensed,outage=sensor.read(raw,env.time//5);guard.set_observation(sensed)
                fallback=sensor.outage_steps>=3
                if fallback:pref=direct_phase_baseline('fixed_time',{**info,'raw_state':sensed},cfg)
                elif applied=='old':pref=models['old'].predict(sensed)['action_id']
                else:pref=models['prior43'].predict(env.adapter,sensed,guard,sensor,info['action_mask'])
                action,_=guard.choose(pref,info['action_mask'],raw['phase'],raw['transitioning'],sensed['queues'],[0,0,0])
                proposed_watch=guard.request(action,info['action_mask'],raw['phase'],raw['transitioning'])
                watch_changes+=int(proposed_watch!=action);action=proposed_watch;changes+=int(action!=pref)
                decisions.append({'time':env.time,'preferred':pref,'action':action,'fallback':fallback,'outage':outage,'red_age':guard.red_age})
                _,_,_,done,info=env.step(action)
                if done:break
            dest.mkdir(parents=True,exist_ok=True);(Path(tmp)/'watch_decisions.json').write_text(json.dumps(decisions),encoding='utf8')
            temp=dest/'raw.zip.tmp'
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                for name in ('timeseries.csv','decisions.csv','tripinfo.xml','metrics.json','watch_decisions.json'):z.write(Path(tmp)/name,name)
            os.replace(temp,rawzip)
            r={'phase':phase,'scenario':scenario,'seed':seed,'sensor':sensor_mode,'controller':label,'effective_controller':applied,
               'map_sha256':sha256(net),'route_sha256':sha256(route),'raw_sha256':sha256(rawzip),
               'guard_interventions':changes,'watchdog_interventions':watch_changes,**info['metrics']}
            atomic_json(rec,r);return r
        finally:env.close()


def phase_run(phase,scenario):
    setup();selected=None
    if phase=='test':
        s=json.loads((OUT/'validation_selection.json').read_text(encoding='utf8'));selected=s['selected']
    inp=OUT/'inputs';cfg=yaml.safe_load((inp/'config.yaml').read_text(encoding='utf8'))
    models={'old':Policy(inp/'old.pt'),'prior43':NewPolicy(inp/'prior43.pt')}
    seeds=VS if phase=='validation' else TS;sensors=VSENSORS if phase=='validation' else TSENSORS
    labels=VAL_CONTROLLERS if phase=='validation' else TEST_CONTROLLERS
    rows=[];start=time.monotonic()
    for seed in seeds:
      for sensor in sensors:
       for label in labels:rows.append(run_case(phase,scenario,seed,sensor,label,selected,models,cfg))
      (OUT/'partial'/phase).mkdir(parents=True,exist_ok=True)
      pd.DataFrame(rows).to_csv(OUT/'partial'/phase/f'{scenario}.csv',index=False)
      print('WATCH',phase,scenario,seed,len(rows),'seconds',round(time.monotonic()-start),flush=True)


def combine(phase):
    setup();scenarios=VSCENARIOS if phase=='validation' else SCENARIOS
    seeds=VS if phase=='validation' else TS;sensors=VSENSORS if phase=='validation' else TSENSORS
    labels=VAL_CONTROLLERS if phase=='validation' else TEST_CONTROLLERS
    files=[OUT/'partial'/phase/f'{sc}.csv' for sc in scenarios]
    if not all(p.exists() for p in files):raise RuntimeError('Missing scenario output')
    df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    expected=len(scenarios)*len(seeds)*len(sensors)*len(labels)
    if len(df)!=expected or df.duplicated(['scenario','seed','sensor','controller']).any():raise RuntimeError('Missing/duplicate runs')
    if not (df.groupby(['scenario','seed']).route_sha256.nunique()==1).all():raise RuntimeError('Different routes')
    for r in df.itertuples(index=False):
        raw=OUT/'runs'/phase/r.scenario/str(r.seed)/r.sensor/r.controller/'raw.zip'
        if sha256(raw)!=r.raw_sha256:raise RuntimeError('Raw hash mismatch')
    df.to_csv(OUT/f'{phase}_results.csv',index=False)
    if phase=='validation':
        ref=df[df.controller=='old'];evaluations=[]
        for label in ('prior43','watch70','watch90'):
            cand=df[df.controller==label];p=ref.merge(cand,on=['scenario','seed','sensor'],suffixes=('_old','_new'),validate='one_to_one')
            p['delta']=p.cohort_delay_s_new-p.cohort_delay_s_old
            avg=float(p.delta.mean());worst_group=float(p.groupby(['scenario','sensor']).delta.mean().max());worst_run=float(p.delta.max())
            comp=float((p.completion_rate_new-p.completion_rate_old).mean());wait=float(p.max_consecutive_wait_s_new.max()-p.max_consecutive_wait_s_old.max())
            eligible=avg<=-.25 and worst_group<=.5 and worst_run<=1 and comp>=-.005 and wait<=0
            evaluations.append({'label':label,'mean_delta':avg,'worst_group_delta':worst_group,'worst_run_delta':worst_run,
                                'completion_delta':comp,'max_wait_delta':wait,'eligible':bool(eligible)})
        passing=[r for r in evaluations if r['eligible']];chosen=min(passing,key=lambda r:r['mean_delta'])['label'] if passing else None
        atomic_json(OUT/'validation_selection.json',{'selected':chosen,'results':evaluations,'note':'None retains old; no guarantee'})
        print('WATCH_SELECTED',chosen,evaluations,flush=True)
    atomic_json(OUT/f'{phase}_status.json',{'complete':True,'rows':len(df),'raw_hashes_verified':True})
    print('WATCH_COMPLETE',phase,len(df),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--combine',choices=('validation','test'))
    p.add_argument('--phase',choices=('validation','test'));p.add_argument('--scenario');a=p.parse_args()
    if a.prepare:setup()
    elif a.combine:combine(a.combine)
    elif a.phase and a.scenario:
        if a.scenario not in (VSCENARIOS if a.phase=='validation' else SCENARIOS):p.error('Invalid scenario')
        phase_run(a.phase,a.scenario)
    else:p.error('Specify --prepare, --combine or --phase with --scenario')
