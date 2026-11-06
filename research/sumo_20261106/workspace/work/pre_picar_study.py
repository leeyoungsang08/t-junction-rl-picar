"""Controlled synthetic sensor / guard study. This is not a PiCar integration.

Run with the project's SUMO-enabled Python; inputs and outputs are preserved.
All randomness and experimental groupings below are deliberate and logged.
"""
import os, sys, json, time, hashlib, shutil, tempfile, zipfile, argparse, platform
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import libsumo
import yaml

PROJECT = Path(__file__).resolve().parent / 'robust_retrain/T_Junction_RL_reward_ablation'
sys.path.insert(0, str(PROJECT))
from maps.experiment_maps import build
from src.environment import TJunctionEnv
from src.inference import Policy
from src.service_guard import DirectionServiceGuard
from src.baselines import direct_phase_baseline
from src.traffic_generator import generate_traffic
from src.utils import load_yaml, sha256

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'outputs' / 'SUMO_PiCar_전_센서시험'
OLD = HERE / 'work/robust_retrain/old_seed42.pt'
HORIZON = 900
MAPS = {
    'calibration': ((-240, 10), (250, -10), (5, -220), (11., 11., 10.)),
    'probeA': ((-305, -35), (285, 20), (40, -205), (11., 11.5, 9.)),
    'probeB': ((-385, 55), (210, -40), (-40, -260), (12., 10., 10.)),
}
SCENARIOS = ('medium', 'high', 'heldout_reversal', 'south_oversaturated')
SEEDS = (70950, 70951, 70952)
SENSORS = ('limited_clean', 'limited_delay_miss', 'limited_outage')


class LimitedSensor:
    """Toy lane-wide count sensor, with no pending/true wait/internal oracle.

    Values derived from whole incoming lanes, not a calibrated PiCar camera.
    Queue occupancy age is a feature proxy, not the waiting age of each car.
    """
    def __init__(self, mode, seed):
        self.mode, self.seed = mode, seed
        self.history = deque(maxlen=3)
        self.last = None
        self.age = np.zeros(3)
        self.recent = deque(maxlen=6)
        self.prev = np.zeros(3)
        self.outage_steps = 0

    def read(self, raw, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, 17171]))
        p = 1.0 if self.mode == 'limited_clean' else .8
        vehicles = np.asarray(raw['vehicles'], dtype=int)
        queues = np.minimum(vehicles, np.asarray(raw['queues'], dtype=int))
        sensed_q = rng.binomial(queues, p)
        sensed_moving = rng.binomial(vehicles - queues, p)
        sample = (sensed_q.astype(float), (sensed_q + sensed_moving).astype(float))
        self.history.append(sample)
        selected = self.history[0] if self.mode != 'limited_clean' and len(self.history) > 1 else sample
        # One contiguous 20-second communication outage in each 120 seconds.
        outage = self.mode == 'limited_outage' and 60 <= (step * 5) % 120 < 80
        self.outage_steps = self.outage_steps + 1 if outage else 0
        if outage and self.last is not None:
            selected = self.last
        else:
            self.last = tuple(x.copy() for x in selected)
        q, v = (x.copy() for x in selected)
        self.age = np.where(q > 0, self.age + 5, 0)
        self.recent.append(np.maximum(v - self.prev, 0))
        self.prev = v
        state = deepcopy(raw)  # signal timing is known from the controller itself
        state.update(vehicles=v.tolist(), queues=q.tolist(), pending=[0.,0.,0.],
                     mean_waiting_seconds=self.age.tolist(), longest_wait=self.age.tolist(),
                     recent_requests=np.sum(self.recent, axis=0).tolist(), internal_vehicles=0.)
        return state, outage


class LatchedGuard(DirectionServiceGuard):
    """Only sees the last sensor count; never the true SUMO queues or pending."""
    def reset(self):
        super().reset()
        self.latched = [0., 0., 0.]

    def set_observation(self, state):
        self.latched = list(state['queues'])

    def observe_second(self, phase, stage, queues, pending):
        super().observe_second(phase, stage, self.latched, (0, 0, 0))


def atomic_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(t, p)


def setup():
    inputs = OUT / 'inputs'
    marker = OUT / 'manifest.json'
    if marker.exists():
        m = json.loads(marker.read_text(encoding='utf-8'))
        if sha256(Path(__file__)) != m['driver_sha256']:
            raise RuntimeError('Driver changed; preserve previous run and create new output directory')
        for name, h in m['inputs'].items():
            if sha256(OUT/name) != h: raise RuntimeError(f'Input mismatch: {name}')
        if m['software'] != versions(): raise RuntimeError('Software differs; do not resume')
        return m
    if inputs.exists() and any(inputs.iterdir()): raise RuntimeError('Incomplete existing inputs; inspect before retry')
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), inputs/'driver.py')
    shutil.copy2(OLD, inputs/'old.pt')
    for name, geometry in MAPS.items():
        build('pre_picar_'+name, geometry)
        shutil.copytree(PROJECT/'maps/experiment'/('pre_picar_'+name), inputs/'maps'/name)
    for scenario in SCENARIOS:
        for seed in (*SEEDS, 70960, 70961, 70962):
            generate_traffic(scenario, seed, HORIZON, 'final', directory=inputs/'routes')
    cfg=load_yaml('base.yaml')
    cfg['signal']['control_mode']='direct'
    cfg['simulation'].update(backend='libsumo',horizon_seconds=HORIZON,audit_vehicle_safety=True)
    (inputs/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    with zipfile.ZipFile(inputs/'project_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for part in ('src','maps','configs'):
            for p in (PROJECT/part).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and 'experiment' not in p.parts:
                    z.write(p,p.relative_to(PROJECT))
    m={'driver_sha256':sha256(Path(__file__)), 'maps':MAPS, 'scenarios':SCENARIOS,
       'evaluation_seeds':SEEDS,'calibration_seeds':[70960,70961,70962], 'sensor_modes':SENSORS,
       'horizon':HORIZON, 'software':versions(), 'scope':'exploratory synthetic sensor and safeguard experiment; no PiCar measurements',
       'inputs':{str(p.relative_to(OUT)):sha256(p) for p in sorted(inputs.rglob('*')) if p.is_file()}}
    atomic_json(marker,m)
    return m


def versions():
    return {'python':platform.python_version(),'sumo':str(libsumo.getVersion()),'torch':str(torch.__version__)}


def run_case(mapname, scenario, seed, sensor_mode, controller, fixed_green=20):
    root=OUT/'runs'/mapname/scenario/str(seed)/sensor_mode/controller
    if controller.startswith('fixed_'): root=OUT/'runs'/mapname/scenario/str(seed)/sensor_mode/controller
    result=root/'record.json'; rawzip=root/'raw.zip'
    if result.exists() and rawzip.exists():
        data=json.loads(result.read_text(encoding='utf-8'))
        if data['raw_sha256']!=sha256(rawzip): raise RuntimeError(f'Corrupt raw file {rawzip}')
        return data
    if result.exists() or rawzip.exists(): raise RuntimeError(f'Partial prior run {root}')
    inputs=OUT/'inputs'; cfg=yaml.safe_load((inputs/'config.yaml').read_text(encoding='utf-8'))
    cfg['signal']['fixed_green']=fixed_green
    route=inputs/'routes/final'/f'{scenario}_seed{seed}_t{HORIZON}.rou.xml'
    net=inputs/'maps'/mapname/'map.net.xml'
    oracle=sensor_mode=='oracle'
    guard=DirectionServiceGuard(80) if oracle else LatchedGuard(80)
    sensor=None if oracle else LimitedSensor(sensor_mode,seed)
    model=Policy(inputs/'old.pt') if controller=='model' else None
    overrides=0; fallbacks=0; missed=0; actions=[]
    with tempfile.TemporaryDirectory(prefix='sumo_pre_picar_') as tmp:
        env=TJunctionEnv(cfg,route,seed,Path(tmp),network_file=net,service_observer=guard)
        try:
            _,info=env.reset(seed=seed)
            while True:
                raw=info['raw_state']; step=env.time//5
                sensed, outage = (deepcopy(raw),False) if oracle else sensor.read(raw,step)
                if not oracle: guard.set_observation(sensed)
                missed += int(outage)
                fallback=bool(not oracle and sensor.outage_steps>=3)
                if fallback:
                    pref=direct_phase_baseline('fixed_time',{**info,'raw_state':sensed},cfg); fallbacks+=1
                elif model:
                    pref=model.predict(sensed)['action_id']
                else:
                    pref=direct_phase_baseline('fixed_time' if controller.startswith('fixed_') else 'max_pressure_proxy',
                                               {**info,'raw_state':sensed},cfg)
                q,p=(raw['queues'],raw['pending']) if oracle else (sensed['queues'],[0,0,0])
                act,_=guard.choose(pref,info['action_mask'],raw['phase'],raw['transitioning'],q,p)
                overrides += int(act!=pref)
                actions.append({'time':env.time,'preferred':pref,'action':act,'guard_override':act!=pref,
                                'sensor_outage':outage,'fallback':fallback,'sensed_queues':sensed['queues']})
                _,_,_,done,info=env.step(act)
                if done: break
            root.mkdir(parents=True,exist_ok=True)
            (Path(tmp)/'sensor_decisions.json').write_text(json.dumps(actions),encoding='utf-8')
            temp=root/'raw.zip.tmp'
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                for n in ('timeseries.csv','decisions.csv','tripinfo.xml','metrics.json','sensor_decisions.json'):
                    z.write(Path(tmp)/n,n)
            os.replace(temp,rawzip)
            data={'map':mapname,'scenario':scenario,'seed':seed,'sensor':sensor_mode,'controller':controller,
                  'map_sha256':sha256(net),'route_sha256':sha256(route),'model_sha256':sha256(inputs/'old.pt') if model else None,
                  'raw_sha256':sha256(rawzip),'guard_interventions':overrides,'fallback_decisions':fallbacks,
                  'outage_decisions':missed,**info['metrics']}
            atomic_json(result,data)
            return data
        finally: env.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    m=setup(); rows=[]; start=time.time()
    if args.smoke:
        for sensor in ('oracle','limited_clean','limited_delay_miss','limited_outage'):
            r=run_case('calibration','medium',70960,sensor,'model'); print('SMOKE',sensor,r['cohort_delay_s'],flush=True)
        return
    # Tune fixed timing on separate map/seeds, same guard and limited observations.
    for scenario in ('medium','high'):
      for seed in (70960,70961,70962):
       for fixed in (15,20,30,40):
        r=run_case('calibration',scenario,seed,'limited_clean',f'fixed_{fixed}',fixed); rows.append(r)
    tune=pd.DataFrame(rows).groupby('controller').cohort_delay_s.mean()
    best=min(tune.index,key=lambda n:(tune[n],int(n.split('_')[1])))
    fixed=int(best.split('_')[1]); atomic_json(OUT/'calibration.json',{'scores':tune.to_dict(),'selected_fixed_green':fixed})
    print('CALIBRATION', tune.to_dict(), 'selected',fixed,flush=True)
    # Test labels reflect *known*, previously explored scenario types, not a fresh final test.
    for mapname in ('probeA','probeB'):
     for scenario in SCENARIOS:
      for seed in SEEDS:
       for sensor in (*SENSORS,'oracle'):
        controllers=('model',) if sensor=='oracle' else ('model','queue',f'fixed_{fixed}')
        for controller in controllers:
            rows.append(run_case(mapname,scenario,seed,sensor,controller,fixed))
       pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
       print('DONE',mapname,scenario,seed,'rows',len(rows),'elapsed_s',round(time.time()-start),flush=True)
    atomic_json(OUT/'status.json',{'complete':True,'rows':len(rows),'seconds':round(time.time()-start,1)})
    print('COMPLETE',len(rows),flush=True)


if __name__=='__main__': main()
