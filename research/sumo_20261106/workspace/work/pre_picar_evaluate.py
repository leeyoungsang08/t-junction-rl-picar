"""Paired exploratory evaluation of all sensor-trained checkpoints.

New maps and seeds are never used for selecting training checkpoints. Scenario
types were studied previously; this is not an untouched final benchmark.
"""
import os, sys, json, time, shutil, tempfile, zipfile, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch, libsumo, yaml
from pre_picar_study import PROJECT,HERE,LimitedSensor,LatchedGuard,atomic_json,versions
from pre_picar_training import vector
sys.path.insert(0,str(PROJECT))
from maps.experiment_maps import build
from src.environment import TJunctionEnv
from src.inference import Policy
from src.dqn_network import QNetwork
from src.baselines import direct_phase_baseline
from src.traffic_generator import generate_traffic
from src.utils import sha256

OUT=HERE/'outputs'/'SUMO_PiCar_전_재학습_비교'
TRAIN=HERE/'outputs'/'SUMO_PiCar_전_센서재학습'
OLD=HERE/'work/robust_retrain/old_seed42.pt'
MAPS={'evalA':((-280,-25),(260,30),(45,-225),(10.5,11.5,9.)),
      'evalB':((-375,45),(230,-35),(-50,-245),(12.,10.5,9.5))}
SCENARIOS=('medium','high','heldout_reversal','south_oversaturated')
SEEDS=(70970,70971,70972)
SENSORS=('limited_clean','limited_delay_miss','limited_outage')
CONTROLLERS=('old','new42','new43','new44','queue','fixed_15')


class NewPolicy:
    def __init__(self,path):
        p=torch.load(path,map_location='cpu',weights_only=False)
        cfg=p['cfg']; self.model=QNetwork(cfg['hidden'],cfg['state_dim'],cfg['action_dim'],cfg['dueling'])
        self.model.load_state_dict(p['state_dict']); self.model.eval(); self.metadata={'episode':p['episode'],'score':p['validation_score']}
    def predict(self,adapter,sensed,guard,sensor,mask):
        with torch.inference_mode(): q=self.model(torch.as_tensor(vector(adapter,sensed,guard,sensor)).unsqueeze(0)).squeeze(0).numpy()
        return int(np.where(mask,q,-np.inf).argmax())


def setup():
    marker=OUT/'manifest.json'; inputs=OUT/'inputs'
    if marker.exists():
        m=json.loads(marker.read_text(encoding='utf8'))
        if m['driver_sha256']!=sha256(Path(__file__)) or m['software']!=versions(): raise RuntimeError('Driver/software changed')
        for name,h in m['inputs'].items():
            if sha256(OUT/name)!=h: raise RuntimeError(f'Input mismatch {name}')
        return m
    if inputs.exists() and any(inputs.iterdir()): raise RuntimeError('Incomplete prior input; inspect')
    for seed in (42,43,44):
        s=json.loads((TRAIN/f'seed{seed}'/'status.json').read_text(encoding='utf8'))
        if not s['complete'] or s['completed_episodes']!=100: raise RuntimeError(f'Training not complete: {seed}')
    inputs.mkdir(parents=True,exist_ok=True)
    for path in (Path(__file__),HERE/'work/pre_picar_training.py',HERE/'work/pre_picar_study.py'):
        shutil.copy2(path,inputs/path.name)
    shutil.copy2(OLD,inputs/'old.pt')
    for seed in (42,43,44): shutil.copy2(TRAIN/f'seed{seed}'/'best.pt',inputs/f'new{seed}.pt')
    for name,geometry in MAPS.items():
        build('pre_picar_eval_'+name,geometry)
        shutil.copytree(PROJECT/'maps/experiment'/('pre_picar_eval_'+name),inputs/'maps'/name)
    for scenario in SCENARIOS:
        for seed in SEEDS:
            generate_traffic(scenario,seed,900,'final',directory=inputs/'routes')
    cfg=yaml.safe_load((TRAIN/'seed42'/'manifest.json').read_text(encoding='utf8'))['base']
    (inputs/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
    with zipfile.ZipFile(inputs/'project_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for part in ('src','maps','configs'):
            for f in (PROJECT/part).rglob('*'):
                if f.is_file() and '__pycache__' not in f.parts and 'experiment' not in f.parts: z.write(f,f.relative_to(PROJECT))
    m={'driver_sha256':sha256(Path(__file__)),'software':versions(),'maps':MAPS,'scenarios':SCENARIOS,'seeds':SEEDS,
       'sensors':SENSORS,'controllers':CONTROLLERS,'fixed_green':15,'horizon':900,
       'scope':'new maps and seeds, known scenario types; synthetic sensor; same initial weights for new seeds',
       'inputs':{str(f.relative_to(OUT)):sha256(f) for f in sorted(inputs.rglob('*')) if f.is_file()}}
    atomic_json(marker,m); return m


def run_case(mapname,scenario,seed,sensor_mode,controller,models,cfg):
    dest=OUT/'runs'/mapname/scenario/str(seed)/sensor_mode/controller
    rec=dest/'record.json'; rawzip=dest/'raw.zip'
    if rec.exists() and rawzip.exists():
        r=json.loads(rec.read_text(encoding='utf8'))
        if r['raw_sha256']!=sha256(rawzip): raise RuntimeError(f'Raw file mismatch {rawzip}')
        return r
    if rec.exists() or rawzip.exists(): raise RuntimeError(f'Partial output {dest}')
    inputs=OUT/'inputs'; route=inputs/'routes/final'/f'{scenario}_seed{seed}_t900.rou.xml'; net=inputs/'maps'/mapname/'map.net.xml'
    guard=LatchedGuard(80); sensor=LimitedSensor(sensor_mode,seed); overrides=fallbacks=0; decisions=[]
    with tempfile.TemporaryDirectory(prefix='pre_picar_eval_') as tmp:
        env=TJunctionEnv(cfg,route,seed,Path(tmp),network_file=net,service_observer=guard)
        try:
            _,info=env.reset(seed=seed)
            while True:
                raw=info['raw_state']; sensed,outage=sensor.read(raw,env.time//5); guard.set_observation(sensed)
                fallback=sensor.outage_steps>=3; fallbacks+=int(fallback)
                if fallback: pref=direct_phase_baseline('fixed_time',{**info,'raw_state':sensed},cfg)
                elif controller=='old': pref=models['old'].predict(sensed)['action_id']
                elif controller.startswith('new'): pref=models[controller].predict(env.adapter,sensed,guard,sensor,info['action_mask'])
                else: pref=direct_phase_baseline('fixed_time' if controller=='fixed_15' else 'max_pressure_proxy',{**info,'raw_state':sensed},cfg)
                act,_=guard.choose(pref,info['action_mask'],raw['phase'],raw['transitioning'],sensed['queues'],[0,0,0])
                overrides+=int(act!=pref); decisions.append({'time':env.time,'preferred':pref,'executed':act,'fallback':fallback,'outage':outage,'queues':sensed['queues']})
                _,_,_,done,info=env.step(act)
                if done: break
            dest.mkdir(parents=True,exist_ok=True)
            (Path(tmp)/'sensor_decisions.json').write_text(json.dumps(decisions),encoding='utf8')
            temp=dest/'raw.zip.tmp'
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                for n in ('timeseries.csv','decisions.csv','tripinfo.xml','metrics.json','sensor_decisions.json'): z.write(Path(tmp)/n,n)
            os.replace(temp,rawzip)
            r={'map':mapname,'scenario':scenario,'seed':seed,'sensor':sensor_mode,'controller':controller,
               'map_sha256':sha256(net),'route_sha256':sha256(route),'raw_sha256':sha256(rawzip),
               'guard_interventions':overrides,'fallback_decisions':fallbacks,**info['metrics']}
            atomic_json(rec,r); return r
        finally: env.close()


def main():
    setup(); inputs=OUT/'inputs'; cfg=yaml.safe_load((inputs/'config.yaml').read_text(encoding='utf8'))
    models={'old':Policy(inputs/'old.pt'),**{f'new{s}':NewPolicy(inputs/f'new{s}.pt') for s in (42,43,44)}}
    print('SELECTED CHECKPOINTS',{s:models[f'new{s}'].metadata for s in (42,43,44)},flush=True)
    rows=[]; start=time.monotonic()
    for mapname in MAPS:
     for scenario in SCENARIOS:
      for seed in SEEDS:
       for sensor in SENSORS:
        for controller in CONTROLLERS:
            rows.append(run_case(mapname,scenario,seed,sensor,controller,models,cfg))
       pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
       print('EVAL',mapname,scenario,seed,len(rows),'elapsed',round(time.monotonic()-start),flush=True)
    atomic_json(OUT/'status.json',{'complete':True,'rows':len(rows),'seconds':round(time.monotonic()-start,1)})
    print('COMPLETE',len(rows),flush=True)


if __name__=='__main__': main()
