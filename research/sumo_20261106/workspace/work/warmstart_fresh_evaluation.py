"""Frozen-input SUMO evaluation; run in the project environment, never overwrite inputs."""
import csv, json, os, shutil, tempfile, zipfile, platform
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch, libsumo, yaml
from maps.experiment_maps import build
from src.environment import TJunctionEnv
from src.inference import Policy
from src.service_guard import DirectionServiceGuard
from src.baselines import direct_phase_baseline
from src.traffic_generator import generate_traffic
from src.utils import ROOT, load_yaml, sha256

DRIVE = Path('/content/drive/MyDrive/SUMO_Robust_Retrain_20261104_v1')
OUT = DRIVE/'warmstart_finetune_v1/frozen_eval_v1'
INPUT = OUT/'inputs'
MAPS = {'freshA':((-295,-75),(335,45),(45,-205),(10.,12.,9.5)),
        'freshB':((-410,75),(220,-45),(-65,-250),(12.,10.5,9.5)),
        'freshC':((-180,-45),(325,65),(75,-165),(9.5,12.,8.5))}
SCENARIOS = ['medium','high','heldout_reversal','south_oversaturated']
SEEDS = [70880,70881,70882]
SENSORS = ['clean','noisy_stochastic']
LABELS = ['old','candidate42','candidate43','candidate44','queue','wait_rule','fixed']
NOISE_VERSION = 'independent uniform multiplicative 0.85-1.15, stochastic rounding for counts; integer error can exceed 15 percent'

def observe(raw, seed, step, noisy):
    if not noisy: return deepcopy(raw)
    state=deepcopy(raw); rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(step),97531]))
    for key in ('vehicles','queues','pending','recent_requests'):
        v=np.maximum(0,np.asarray(state[key],dtype=float)*rng.uniform(.85,1.15,3)); lo=np.floor(v)
        state[key]=(lo+(rng.random(3)<v-lo)).tolist()
    for key in ('mean_waiting_seconds','longest_wait'):
        state[key]=(np.asarray(state[key],dtype=float)*rng.uniform(.85,1.15,3)).tolist()
    state['internal_vehicles']=max(0.,float(state['internal_vehicles'])*float(rng.uniform(.85,1.15)))
    return state

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8'); os.replace(tmp,path)

def setup():
    for seed in (42,43,44):
        status = json.loads((DRIVE/f'warmstart_finetune_v1/seed{seed}/training_status.json').read_text())
        if not status.get('complete') or status.get('completed_episodes') != 400:
            raise RuntimeError(f'Incomplete fine-tuning for seed {seed}')
    OUT.mkdir(parents=True,exist_ok=True)
    marker=OUT/'manifest.json'
    if marker.exists():
        plan=json.loads(marker.read_text(encoding='utf8'))
        if sha256(Path(__file__).resolve()) != plan['inputs']['inputs/experiment_driver.py']:
            raise RuntimeError('Driver differs from frozen experiment; do not resume')
        if plan['software'] != {'python':platform.python_version(),'torch':torch.__version__,'sumo':str(libsumo.getVersion())}:
            raise RuntimeError('Runtime versions differ from frozen experiment; do not resume')
        for rel,digest in plan['inputs'].items():
            p=OUT/rel
            if not p.is_file() or sha256(p)!=digest: raise RuntimeError(f'Frozen input mismatch: {rel}')
        return plan
    if INPUT.exists() and any(INPUT.iterdir()): raise RuntimeError('Incomplete existing input directory; preserve and inspect it before retrying')
    INPUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), INPUT/'experiment_driver.py')
    for name,data in MAPS.items():
        build(name,data); shutil.copytree(ROOT/'maps/experiment'/name,INPUT/'maps'/name)
    for scenario in SCENARIOS:
        for seed in SEEDS:
            generate_traffic(scenario,seed,900,'final',directory=INPUT/'routes')
    (INPUT/'models').mkdir()
    shutil.copy2(ROOT/'models/colab_previous_seed42.pt',INPUT/'models/old.pt')
    for seed in (42,43,44):
        shutil.copy2(DRIVE/f'warmstart_finetune_v1/seed{seed}/checkpoints/best_model.pt',INPUT/f'models/candidate{seed}.pt')
    shutil.copytree(ROOT/'configs',INPUT/'configs')
    # Include the exact executable project source used, alongside this driver.
    with zipfile.ZipFile(INPUT/'project_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for part in ('src','maps','configs'):
            for p in (ROOT/part).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
                    z.write(p,p.relative_to(ROOT))
    cfg=load_yaml('base.yaml'); cfg['signal']['control_mode']='direct'; cfg['simulation'].update(backend='libsumo',horizon_seconds=900,audit_vehicle_safety=True)
    (INPUT/'effective_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
    plan={'maps':MAPS,'scenarios':SCENARIOS,'seeds':SEEDS,'sensors':SENSORS,'controllers':LABELS,'horizon_seconds':900,
          'noise':NOISE_VERSION,'guard':'DirectionServiceGuard(80): true simulator state; not a real-world safety guarantee',
          'scope':'single-lane T intersections; exploratory comparisons; no tuning on this evaluation',
          'software':{'python':platform.python_version(),'torch':torch.__version__,'sumo':str(libsumo.getVersion())},
          'inputs':{str(p.relative_to(OUT)):sha256(p) for p in sorted(INPUT.rglob('*')) if p.is_file()}}
    atomic_json(marker,plan); return plan

def run():
    plan=setup(); cfg=yaml.safe_load((INPUT/'effective_config.yaml').read_text(encoding='utf8')); rows=[]
    print('FROZEN INPUTS VERIFIED',len(plan['inputs']),'expected runs',len(MAPS)*len(SCENARIOS)*len(SEEDS)*len(SENSORS)*len(LABELS),flush=True)
    for mn in MAPS:
      net=INPUT/'maps'/mn/'map.net.xml'
      for sc in SCENARIOS:
       for seed in SEEDS:
        route=INPUT/'routes'/'final'/f'{sc}_seed{seed}_t900.rou.xml'
        for sensor in SENSORS:
         for label in LABELS:
          dest=OUT/'runs'/mn/sc/str(seed)/sensor/label; rec=dest/'record.json'; rawzip=dest/'raw.zip'
          expected={'map_sha256':sha256(net),'route_sha256':sha256(route),'model_sha256':sha256(INPUT/f'models/{label}.pt') if label in ('old','candidate42','candidate43','candidate44') else None}
          if rec.exists() and rawzip.exists():
              r=json.loads(rec.read_text(encoding='utf8'))
              if any(r.get(k)!=v for k,v in expected.items()) or r.get('raw_sha256')!=sha256(rawzip): raise RuntimeError(f'Existing result mismatch: {rec}')
              rows.append(r); continue
          if rec.exists() or rawzip.exists(): raise RuntimeError(f'Incomplete saved run; inspect before resuming: {dest}')
          guard=DirectionServiceGuard(80); model=Policy(INPUT/f'models/{label}.pt') if expected['model_sha256'] else None
          with tempfile.TemporaryDirectory(prefix='sumo_eval_') as tmp:
           env=TJunctionEnv(cfg,route,seed,Path(tmp),network_file=net,service_observer=guard); changes=0
           try:
            _,info=env.reset(seed=seed)
            while True:
             raw=info['raw_state']; sensed=observe(raw,seed,env.time//5,sensor!='clean')
             mode={'queue':'max_pressure_proxy','wait_rule':'rule_observed','fixed':'fixed_time'}.get(label)
             pref=model.predict(sensed)['action_id'] if model else direct_phase_baseline(mode,{**info,'raw_state':sensed},cfg)
             act,_=guard.choose(pref,info['action_mask'],raw['phase'],raw['transitioning'],raw['queues'],raw['pending']); changes+=act!=pref
             _,_,_,done,info=env.step(act)
             if done: break
            env.save_logs(); dest.mkdir(parents=True,exist_ok=True)
            temp=dest/'raw.zip.tmp'
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
             for n in ('timeseries.csv','decisions.csv','tripinfo.xml','metrics.json'): z.write(Path(tmp)/n,n)
            os.replace(temp,rawzip)
            r={'map':mn,'scenario':sc,'seed':seed,'sensor':sensor,'controller':label,**expected,'raw_sha256':sha256(rawzip),'guard_interventions':changes,**info['metrics']}
            atomic_json(rec,r); rows.append(r)
           finally: env.close()
        df=pd.DataFrame(rows); temp=OUT/'results.csv.tmp'; df.to_csv(temp,index=False); os.replace(temp,OUT/'results.csv')
        print('COMPLETED',mn,sc,seed,len(rows),flush=True)
    df=pd.DataFrame(rows); print('FINAL',len(df),flush=True)
    print(df.groupby('controller')[['cohort_delay_s','completion_rate','collision_events','teleport_events','guard_interventions']].mean().round(4).to_string(),flush=True)

if __name__=='__main__': run()
