"""Validation-only baseline/candidate action gate; not a formal guarantee.

All candidate/threshold choices are made with separate validation seeds, never
with the held-out demand seeds. The baseline has priority in uncertain states.
"""
import argparse,json,os,shutil,sys,tempfile,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from pre_picar_study import HERE,PROJECT,LimitedSensor,LatchedGuard,atomic_json,versions
from pre_picar_balanced_evaluate import NewPolicy
sys.path.insert(0,str(PROJECT))
from src.environment import TJunctionEnv
from src.inference import Policy
from src.baselines import direct_phase_baseline
from src.traffic_generator import generate_traffic
from src.utils import sha256

TRAIN=HERE/'outputs/SUMO_balanceA_안정화_20261105'
OUT=HERE/'outputs/SUMO_balanceA_기준정책_안전판_검증'
SCENARIOS=('medium','high','imbalanced','east_bias','rotating','south_val_stress')
SEEDS=(30082,30083);SENSORS=('limited_clean','limited_delay_miss','limited_outage')
THRESHOLDS=(.05,.2,.5)


def choose_by_teacher_gap(reference,candidate,threshold):
    best=int(reference['action_id']);candidate=int(candidate)
    if threshold<0 or not np.isfinite(threshold) or candidate<0 or candidate>=len(reference['q_values']) or not reference['valid_actions'][candidate]:
        raise ValueError('Invalid gate candidate or threshold')
    gap=float(reference['q_values'][best]-reference['q_values'][candidate])
    if not np.isfinite(gap):raise ValueError('Non-finite teacher value')
    return (candidate if gap<=threshold else best),gap


def setup():
    inp=OUT/'inputs';marker=OUT/'manifest.json'
    if marker.exists():
        plan=json.loads(marker.read_text(encoding='utf8'))
        if plan['driver_sha256']!=sha256(Path(__file__)) or plan['software']!=versions():raise RuntimeError('Code or software changed')
        for rel,h in plan['inputs'].items():
            if sha256(OUT/rel)!=h:raise RuntimeError(f'Frozen input mismatch {rel}')
        return plan
    if inp.exists() and any(inp.iterdir()):raise RuntimeError('Incomplete inputs')
    selected={}
    for kind in ('plain','anchor'):
      for seed in (42,43,44):
        run=TRAIN/f'{kind}_seed{seed}';status=json.loads((run/'status.json').read_text(encoding='utf8'))
        if not status['complete'] or status['completed_episodes']!=180:raise RuntimeError(f'Training incomplete {kind} {seed}')
        options=[]
        for ep in (30,60,90,120,150,180):
            d=json.loads((run/f'validation_{ep}.json').read_text(encoding='utf8'));b=d['baseline'];c=d['current']
            if c['mean_delay']<=b['mean_delay']-.1 and c['mean_completion']>=b['mean_completion']-.005 and c['max_wait']<=b['max_wait']+15:
                options.append((d['worst_group_regression_s'],c['mean_delay'],ep))
        if options:
            worst,mean,ep=min(options);selected[f'{kind}{seed}']={'episode':ep,'validation_mean':mean,'worst_group_regression':worst}
    inp.mkdir(parents=True,exist_ok=True)
    for name in ('fixed_intersection_gate.py','pre_picar_study.py','pre_picar_training.py','pre_picar_balanced_evaluate.py'):
        shutil.copy2(HERE/'work'/name,inp/name)
    shutil.copy2(TRAIN/'inputs/old.pt',inp/'old.pt');shutil.copytree(TRAIN/'inputs/target_map',inp/'target_map')
    shutil.copy2(TRAIN/'inputs/project_source.zip',inp/'project_source.zip')
    cfg=json.loads((TRAIN/'plain_seed42/run_manifest.json').read_text(encoding='utf8'))['base']
    (inp/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
    for label,d in selected.items():
        kind='plain' if label.startswith('plain') else 'anchor';seed=int(label[-2:])
        shutil.copy2(TRAIN/f'{kind}_seed{seed}'/f"candidate_{d['episode']}.pt",inp/f'{label}.pt')
    for sc in SCENARIOS:
      for seed in SEEDS:generate_traffic(sc,seed,900,'validation',directory=inp/'routes')
    plan={'driver_sha256':sha256(Path(__file__)),'software':versions(),'candidate_selection':'minimum worst group regression among validation checkpoints whose mean improves 0.1s with completion/wait bounds',
       'selected_candidates':selected,'thresholds':THRESHOLDS,'scenarios':SCENARIOS,'validation_seeds':SEEDS,'sensors':SENSORS,
       'selection_gate':'mean improvement >=0.25s; each scenario/sensor mean regression <=0.5s; individual regression <=1s; completion >= baseline -0.005; max wait <= baseline+15s',
       'scope':'validation only; teacher Q-gap heuristic; no formal safety improvement guarantee',
       'inputs':{str(p.relative_to(OUT)):sha256(p) for p in sorted(inp.rglob('*')) if p.is_file()}}
    atomic_json(marker,plan);return plan


def run_case(sc,seed,sensor_mode,label,threshold,old,models,cfg):
    tag='old' if label=='old' else f'{label}_t{threshold:.2f}'
    dest=OUT/'runs'/sc/str(seed)/sensor_mode/tag;rec=dest/'record.json';rawzip=dest/'raw.zip';inp=OUT/'inputs'
    if rec.exists() and rawzip.exists():
        r=json.loads(rec.read_text(encoding='utf8'))
        if r['raw_sha256']!=sha256(rawzip):raise RuntimeError(f'Raw hash mismatch {rawzip}')
        return r
    if rec.exists() or rawzip.exists():raise RuntimeError('Partial run; inspect')
    route=inp/'routes/validation'/f'{sc}_seed{seed}_t900.rou.xml';net=inp/'target_map/map.net.xml'
    guard=LatchedGuard(80);sensor=LimitedSensor(sensor_mode,seed);decisions=[];changes=0;candidate_allowed=0;candidate_diff=0
    with tempfile.TemporaryDirectory(prefix='gated_validation_') as tmp:
        env=TJunctionEnv(cfg,route,seed,Path(tmp),network_file=net,service_observer=guard)
        try:
            _,info=env.reset(seed=seed)
            while True:
                raw=info['raw_state'];sensed,outage=sensor.read(raw,env.time//5);guard.set_observation(sensed)
                fallback=sensor.outage_steps>=3;gap=None;used=False
                if fallback:pref=direct_phase_baseline('fixed_time',{**info,'raw_state':sensed},cfg)
                else:
                    ref=old.predict(sensed);pref=ref['action_id']
                    if label!='old':
                        candidate=models[label].predict(env.adapter,sensed,guard,sensor,info['action_mask'])
                        candidate_diff+=int(candidate!=pref)
                        chosen,gap=choose_by_teacher_gap(ref,candidate,threshold)
                        used=chosen!=pref;pref=chosen;candidate_allowed+=int(used)
                action,_=guard.choose(pref,info['action_mask'],raw['phase'],raw['transitioning'],sensed['queues'],[0,0,0])
                changes+=int(action!=pref);decisions.append({'time':env.time,'preferred':pref,'executed':action,'fallback':fallback,'used_candidate':used,'teacher_gap':gap})
                _,_,_,done,info=env.step(action)
                if done:break
            dest.mkdir(parents=True,exist_ok=True);(Path(tmp)/'sensor_decisions.json').write_text(json.dumps(decisions),encoding='utf8')
            temp=dest/'raw.zip.tmp'
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                for n in ('timeseries.csv','decisions.csv','tripinfo.xml','metrics.json','sensor_decisions.json'):z.write(Path(tmp)/n,n)
            os.replace(temp,rawzip)
            r={'scenario':sc,'seed':seed,'sensor':sensor_mode,'controller':tag,'map_sha256':sha256(net),'route_sha256':sha256(route),
               'raw_sha256':sha256(rawzip),'guard_interventions':changes,'candidate_different':candidate_diff,'candidate_allowed':candidate_allowed,**info['metrics']}
            atomic_json(rec,r);return r
        finally:env.close()


def select(rows,plan):
    df=pd.DataFrame(rows);baseline=df[df.controller=='old'];summaries=[]
    for label in plan['selected_candidates']:
      for threshold in THRESHOLDS:
        tag=f'{label}_t{threshold:.2f}';candidate=df[df.controller==tag]
        p=baseline.merge(candidate,on=['scenario','seed','sensor'],suffixes=('_old','_candidate'),validate='one_to_one')
        if len(p)!=len(SCENARIOS)*len(SEEDS)*len(SENSORS):raise RuntimeError('Incomplete validation')
        if not (p.route_sha256_old==p.route_sha256_candidate).all():raise RuntimeError('Demand mismatch')
        p['diff']=p.cohort_delay_s_candidate-p.cohort_delay_s_old
        group=p.groupby(['scenario','sensor'])['diff'].mean()
        avg=float(p['diff'].mean());worst=float(group.max());completion=float((p.completion_rate_candidate-p.completion_rate_old).mean())
        wait=float(p.max_consecutive_wait_s_candidate.max()-p.max_consecutive_wait_s_old.max())
        max_individual=float(p['diff'].max())
        eligible=avg<=-.25 and worst<=.5 and max_individual<=1. and completion>=-.005 and wait<=15
        summaries.append({'label':label,'threshold':threshold,'mean_delta':avg,'worst_group_delta':worst,'max_individual_delta':max_individual,'completion_delta':completion,
                          'max_wait_delta':wait,'eligible':bool(eligible),'wins':int((p['diff']< -1e-9).sum()),
                          'losses':int((p['diff']>1e-9).sum()),'ties':int((p['diff'].abs()<=1e-9).sum()),
                          'candidate_allowed':int(p.candidate_allowed_candidate.sum())})
    passing=[x for x in summaries if x['eligible']]
    chosen=min(passing,key=lambda x:(x['mean_delta'],x['worst_group_delta'])) if passing else None
    result={'selected_gate':chosen,'all_candidates':summaries,'note':'Validation only. None means retain baseline; no guarantee on unseen conditions.'}
    atomic_json(OUT/'selection.json',result);return result


def main(scenario):
    plan=setup();inp=OUT/'inputs';cfg=yaml.safe_load((inp/'config.yaml').read_text(encoding='utf8'))
    old=Policy(inp/'old.pt');models={n:NewPolicy(inp/f'{n}.pt') for n in plan['selected_candidates']}
    rows=[];start=time.monotonic()
    for sc in (scenario,):
     for seed in SEEDS:
      for sensor in SENSORS:
       rows.append(run_case(sc,seed,sensor,'old',0,old,models,cfg))
       for label in models:
        for threshold in THRESHOLDS:rows.append(run_case(sc,seed,sensor,label,threshold,old,models,cfg))
      (OUT/'partial').mkdir(parents=True,exist_ok=True)
      pd.DataFrame(rows).to_csv(OUT/'partial'/f'{sc}.csv',index=False)
      print('GATE_VALIDATE',sc,seed,len(rows),'seconds',round(time.monotonic()-start),flush=True)


def combine():
    plan=setup();files=[OUT/'partial'/f'{sc}.csv' for sc in SCENARIOS]
    if not all(f.exists() for f in files):raise RuntimeError('Not all validation scenarios complete')
    df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    expected=len(SCENARIOS)*len(SEEDS)*len(SENSORS)*(1+len(plan['selected_candidates'])*len(THRESHOLDS))
    if len(df)!=expected or df.duplicated(['scenario','seed','sensor','controller']).any():raise RuntimeError('Incomplete or duplicate validation')
    for r in df.itertuples(index=False):
        raw=OUT/'runs'/r.scenario/str(r.seed)/r.sensor/r.controller/'raw.zip'
        if sha256(raw)!=r.raw_sha256:raise RuntimeError(f'Raw mismatch {raw}')
    df.to_csv(OUT/'results.csv',index=False)
    result=select(df.to_dict('records'),plan);atomic_json(OUT/'status.json',{'complete':True,'rows':len(df),'raw_hashes_verified':True})
    print('GATE_SELECTED',result['selected_gate'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--combine',action='store_true')
    p.add_argument('--scenario',choices=SCENARIOS);a=p.parse_args()
    if a.prepare:setup()
    elif a.combine:combine()
    elif a.scenario:main(a.scenario)
    else:p.error('Specify --prepare, --scenario or --combine')
