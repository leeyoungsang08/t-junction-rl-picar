"""Frozen held-out evaluation for the fixed balanceA intersection.

Models were selected by a separate validation gate before these routes exist.
"""
import argparse,json,os,shutil,sys,tempfile,time,zipfile
from collections import deque
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from pre_picar_study import HERE,PROJECT,LimitedSensor,LatchedGuard,atomic_json,versions
from pre_picar_balanced_evaluate import NewPolicy
from fixed_intersection_gate import choose_by_teacher_gap
sys.path.insert(0,str(PROJECT))
from src.environment import TJunctionEnv
from src.inference import Policy
from src.baselines import direct_phase_baseline
from src.traffic_generator import generate_traffic
from src.utils import sha256

TRAIN=HERE/'outputs/SUMO_balanceA_안정화_20261105'
OUT=HERE/'outputs/SUMO_balanceA_독립시험_20261105'
PRIOR=HERE/'outputs/SUMO_PiCar_전_균형선택_재학습/seed43/best.pt'
GATE=HERE/'outputs/SUMO_balanceA_기준정책_안전판_검증'
SCENARIOS=('low','medium','high','imbalanced','heldout_reversal','south_oversaturated')
SEEDS=(70990,70991,70992)
SENSORS=('limited_clean','limited_delay_miss','limited_outage','severe_unseen')
BASE_LABELS=('old','prior43','plain42','plain43','plain44','anchor42','anchor43','anchor44','gated','queue','fixed_s30')


class SevereSensor(LimitedSensor):
    """Unseen assumption: 60% detection, four-step delay, 30 s outages."""
    def __init__(self,seed):
        super().__init__('severe_unseen',seed);self.history=deque(maxlen=5)
    def read(self,raw,step):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,step,17171]))
        v=np.asarray(raw['vehicles'],dtype=int);q=np.minimum(v,np.asarray(raw['queues'],dtype=int))
        sq=rng.binomial(q,.6);sv=sq+rng.binomial(v-q,.6)
        sample=(sq.astype(float),sv.astype(float));self.history.append(sample)
        selected=self.history[0] if len(self.history)>1 else sample
        outage=60<=(step*5)%120<90
        self.outage_steps=self.outage_steps+1 if outage else 0
        if outage and self.last is not None:selected=self.last
        else:self.last=tuple(x.copy() for x in selected)
        q,v=(x.copy() for x in selected);self.age=np.where(q>0,self.age+5,0)
        self.recent.append(np.maximum(v-self.prev,0));self.prev=v
        state=deepcopy(raw)
        state.update(vehicles=v.tolist(),queues=q.tolist(),pending=[0.,0.,0.],
          mean_waiting_seconds=self.age.tolist(),longest_wait=self.age.tolist(),
          recent_requests=np.sum(self.recent,axis=0).tolist(),internal_vehicles=0.)
        return state,outage


def setup():
    inputs=OUT/'inputs';marker=OUT/'manifest.json'
    if marker.exists():
        plan=json.loads(marker.read_text(encoding='utf8'))
        if plan['driver_sha256']!=sha256(Path(__file__)) or plan['software']!=versions():raise RuntimeError('Code or software changed')
        for path,h in plan['inputs'].items():
            if sha256(OUT/path)!=h:raise RuntimeError(f'Input mismatch {path}')
        return plan
    if inputs.exists() and any(inputs.iterdir()):raise RuntimeError('Incomplete existing input; inspect')
    for kind in ('plain','anchor'):
      for seed in (42,43,44):
        status=json.loads((TRAIN/f'{kind}_seed{seed}/status.json').read_text(encoding='utf8'))
        if not status['complete'] or status['completed_episodes']!=180:raise RuntimeError(f'Training incomplete: {kind} {seed}')
    gate_status=json.loads((GATE/'status.json').read_text(encoding='utf8'))
    if not gate_status['complete']:raise RuntimeError('Gate validation incomplete')
    gate_selection=json.loads((GATE/'selection.json').read_text(encoding='utf8'))['selected_gate']
    gate_plan=json.loads((GATE/'manifest.json').read_text(encoding='utf8'))
    for rel,h in gate_plan['inputs'].items():
        if sha256(GATE/rel)!=h:raise RuntimeError(f'Gate input mismatch: {rel}')
    exploratory=tuple('explore_'+label for label in sorted(gate_plan['selected_candidates']))
    inputs.mkdir(parents=True,exist_ok=True)
    for name in ('fixed_intersection_evaluate.py','fixed_intersection_train.py','fixed_intersection_gate.py','anchored_dqn.py',
                 'pre_picar_study.py','pre_picar_training.py','pre_picar_balanced_evaluate.py'):
        shutil.copy2(HERE/'work'/name,inputs/name)
    shutil.copy2(TRAIN/'inputs/old.pt',inputs/'old.pt');shutil.copy2(PRIOR,inputs/'prior43.pt')
    shutil.copytree(TRAIN/'inputs/target_map',inputs/'target_map')
    shutil.copy2(TRAIN/'inputs/project_source.zip',inputs/'project_source.zip')
    for kind in ('plain','anchor'):
      for seed in (42,43,44):shutil.copy2(TRAIN/f'{kind}_seed{seed}/best.pt',inputs/f'{kind}{seed}.pt')
    shutil.copy2(GATE/'selection.json',inputs/'gate_selection.json')
    if gate_selection is not None:shutil.copy2(GATE/'inputs'/f"{gate_selection['label']}.pt",inputs/'gated.pt')
    for label in gate_plan['selected_candidates']:
        shutil.copy2(GATE/'inputs'/f'{label}.pt',inputs/f'explore_{label}.pt')
    cfg=json.loads((TRAIN/'plain_seed42/run_manifest.json').read_text(encoding='utf8'))['base']
    (inputs/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
    for scenario in SCENARIOS:
      for seed in SEEDS:generate_traffic(scenario,seed,900,'final',directory=inputs/'routes')
    m={'driver_sha256':sha256(Path(__file__)),'software':versions(),'scenarios':SCENARIOS,'seeds':SEEDS,'sensors':SENSORS,'controllers':BASE_LABELS+exploratory,
       'horizon_s':900,'selection':'all six candidates and gate selected from validation only; no test selection','gate':gate_selection,
       'scope':'one previously studied target geometry, new demand seeds; known scenario types; severe sensor assumption unseen; explore_ models are validation-rejected and reported for diagnosis only',
       'inputs':{str(f.relative_to(OUT)):sha256(f) for f in sorted(inputs.rglob('*')) if f.is_file()}}
    atomic_json(marker,m);return m


def run_case(sc,seed,sensor_mode,label,models,cfg,gate):
    dest=OUT/'runs'/sc/str(seed)/sensor_mode/label;rec=dest/'record.json';rawzip=dest/'raw.zip';inp=OUT/'inputs'
    if rec.exists() and rawzip.exists():
        r=json.loads(rec.read_text(encoding='utf8'))
        if r['raw_sha256']!=sha256(rawzip):raise RuntimeError(f'Raw file mismatch {rawzip}')
        return r
    if rec.exists() or rawzip.exists():raise RuntimeError(f'Incomplete output {dest}')
    route=inp/'routes/final'/f'{sc}_seed{seed}_t900.rou.xml';net=inp/'target_map/map.net.xml'
    guard=LatchedGuard(80);sensor=SevereSensor(seed) if sensor_mode=='severe_unseen' else LimitedSensor(sensor_mode,seed)
    changed=fallbacks=0;decisions=[]
    with tempfile.TemporaryDirectory(prefix='balanceA_eval_') as tmp:
        env=TJunctionEnv(cfg,route,seed,Path(tmp),network_file=net,service_observer=guard)
        try:
            _,info=env.reset(seed=seed)
            while True:
                raw=info['raw_state'];sensed,outage=sensor.read(raw,env.time//5);guard.set_observation(sensed)
                fallback=sensor.outage_steps>=3;fallbacks+=int(fallback)
                if fallback:pref=direct_phase_baseline('fixed_time',{**info,'raw_state':sensed},cfg)
                elif label=='old':pref=models['old'].predict(sensed)['action_id']
                elif label=='gated':
                    ref=models['old'].predict(sensed);pref=ref['action_id']
                    if gate is not None:
                        candidate=models['gated'].predict(env.adapter,sensed,guard,sensor,info['action_mask'])
                        pref,_=choose_by_teacher_gap(ref,candidate,gate['threshold'])
                elif label not in ('queue','fixed_s30'):pref=models[label].predict(env.adapter,sensed,guard,sensor,info['action_mask'])
                elif label=='queue':pref=direct_phase_baseline('max_pressure_proxy',{**info,'raw_state':sensed},cfg)
                else:
                    mask=np.asarray(info['action_mask'],dtype=bool);current=int(raw['phase']);green=30 if current==2 else 15
                    if mask.sum()==1:pref=int(np.flatnonzero(mask)[0])
                    elif mask[current] and raw['green_elapsed']<green:pref=current
                    else:pref=(current+1)%3
                action,_=guard.choose(pref,info['action_mask'],raw['phase'],raw['transitioning'],sensed['queues'],[0,0,0])
                changed+=int(action!=pref);decisions.append({'time':env.time,'preferred':pref,'executed':action,'outage':outage,'fallback':fallback,'sensed_queues':sensed['queues']})
                _,_,_,done,info=env.step(action)
                if done:break
            dest.mkdir(parents=True,exist_ok=True);(Path(tmp)/'sensor_decisions.json').write_text(json.dumps(decisions),encoding='utf8')
            temp=dest/'raw.zip.tmp'
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                for name in ('timeseries.csv','decisions.csv','tripinfo.xml','metrics.json','sensor_decisions.json'):z.write(Path(tmp)/name,name)
            os.replace(temp,rawzip)
            r={'scenario':sc,'seed':seed,'sensor':sensor_mode,'controller':label,'map_sha256':sha256(net),'route_sha256':sha256(route),
               'raw_sha256':sha256(rawzip),'guard_interventions':changed,'fallback_decisions':fallbacks,**info['metrics']}
            atomic_json(rec,r);return r
        finally:env.close()


def main(scenario):
    plan=setup();inp=OUT/'inputs';cfg=yaml.safe_load((inp/'config.yaml').read_text(encoding='utf8'));gate=plan['gate']
    labels=plan['controllers']
    models={'old':Policy(inp/'old.pt'),**{n:NewPolicy(inp/f'{n}.pt') for n in labels if n not in ('old','queue','fixed_s30','gated')}}
    if gate is not None:models['gated']=NewPolicy(inp/'gated.pt')
    print('SELECTED',{n:models[n].metadata for n in models if n!='old'},flush=True)
    rows=[];start=time.monotonic()
    for sc in (scenario,):
      for seed in SEEDS:
       for sensor in SENSORS:
        for label in labels:rows.append(run_case(sc,seed,sensor,label,models,cfg,gate))
       (OUT/'partial').mkdir(parents=True,exist_ok=True)
       pd.DataFrame(rows).to_csv(OUT/'partial'/f'{sc}.csv',index=False)
       print('EVAL',sc,seed,len(rows),'seconds',round(time.monotonic()-start),flush=True)


def combine():
    plan=setup(); files=[OUT/'partial'/f'{sc}.csv' for sc in SCENARIOS]
    if not all(f.exists() for f in files):raise RuntimeError('Not all evaluation scenarios complete')
    df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    expected=len(SCENARIOS)*len(SEEDS)*len(SENSORS)*len(plan['controllers'])
    if len(df)!=expected or df.duplicated(['scenario','seed','sensor','controller']).any():raise RuntimeError('Incomplete or duplicate records')
    if not (df.groupby(['scenario','seed']).route_sha256.nunique()==1).all():raise RuntimeError('Different routes for same scenario/seed')
    for r in df.itertuples(index=False):
        raw=OUT/'runs'/r.scenario/str(r.seed)/r.sensor/r.controller/'raw.zip'
        if sha256(raw)!=r.raw_sha256:raise RuntimeError(f'Raw mismatch {raw}')
    df.to_csv(OUT/'results.csv',index=False)
    atomic_json(OUT/'status.json',{'complete':True,'rows':len(df),'raw_hashes_verified':True})
    print('EVALUATION_COMPLETE',len(df),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--combine',action='store_true')
    p.add_argument('--scenario',choices=SCENARIOS);a=p.parse_args()
    if a.prepare:setup()
    elif a.combine:combine()
    elif a.scenario:main(a.scenario)
    else:p.error('Specify --prepare, --scenario or --combine')
