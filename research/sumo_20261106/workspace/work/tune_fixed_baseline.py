"""Tune fixed phase splits on disjoint calibration inputs, then test frozen plan."""
import os,sys,json,shutil,tempfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from pre_picar_study import PROJECT,HERE,LimitedSensor,LatchedGuard,atomic_json,versions
sys.path.insert(0,str(PROJECT))
from src.environment import TJunctionEnv
from src.traffic_generator import generate_traffic
from src.utils import sha256

OUT=HERE/'outputs'/'SUMO_PiCar_전_고정신호_조정'
EVAL=HERE/'outputs'/'SUMO_PiCar_전_재학습_비교'
PLANS={'u15':(15,15,15),'u20':(20,20,20),'u30':(30,30,30),'s35':(10,10,35),
       's30':(15,15,30),'we25':(25,25,10),'w30':(30,15,10),'e30':(15,30,10)}
CALMAPS=('validation0','validation1'); CALSCENARIOS=('medium','high','imbalanced','south_val_stress'); CALSEEDS=(30070,30071)
TESTMAPS=('evalA','evalB'); TESTSCENARIOS=('medium','high','heldout_reversal','south_oversaturated'); TESTSEEDS=(70970,70971,70972)
SENSORS=('limited_clean','limited_delay_miss','limited_outage')


def setup():
    prior=json.loads((EVAL/'manifest.json').read_text(encoding='utf8'))
    for path,h in prior['inputs'].items():
        if sha256(EVAL/path)!=h: raise RuntimeError(f'Evaluation input mismatch {path}')
    marker=OUT/'manifest.json'; inputs=OUT/'inputs'
    if marker.exists():
        m=json.loads(marker.read_text(encoding='utf8'))
        if m['driver_sha256']!=sha256(Path(__file__)) or m['software']!=versions():raise RuntimeError('Plan changed')
        if m['evaluation_manifest_sha256']!=sha256(EVAL/'manifest.json'):raise RuntimeError('Evaluation inputs changed')
        for path,h in m['inputs'].items():
            if sha256(OUT/path)!=h:raise RuntimeError(f'Frozen calibration input mismatch {path}')
        return
    if inputs.exists() and any(inputs.iterdir()):raise RuntimeError('Incomplete prior inputs')
    inputs.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),inputs/'driver.py')
    for mapname in CALMAPS:shutil.copytree(PROJECT/'maps/experiment'/mapname,inputs/'maps'/mapname)
    for scenario in CALSCENARIOS:
        for seed in CALSEEDS:generate_traffic(scenario,seed,900,'validation',directory=inputs/'routes')
    shutil.copy2(EVAL/'inputs/config.yaml',inputs/'config.yaml')
    plan={'plans':PLANS,'calibration_maps':CALMAPS,'calibration_scenarios':CALSCENARIOS,'calibration_seeds':CALSEEDS,
          'test_maps':TESTMAPS,'test_scenarios':TESTSCENARIOS,'test_seeds':TESTSEEDS,'sensor_modes':SENSORS,
          'driver_sha256':sha256(Path(__file__)),'evaluation_manifest_sha256':sha256(EVAL/'manifest.json'),'software':versions(),
          'scope':'one global fixed plan selected using calibration mean delay; selection excludes evaluation inputs',
          'inputs':{str(f.relative_to(OUT)):sha256(f) for f in sorted(inputs.rglob('*')) if f.is_file()}}
    atomic_json(marker,plan)


def run_case(phase,mapname,scenario,seed,sensor_mode,planname):
    dest=OUT/'runs'/phase/mapname/scenario/str(seed)/sensor_mode/planname; rec=dest/'record.json'; rawzip=dest/'raw.zip'
    if rec.exists() and rawzip.exists():
        r=json.loads(rec.read_text(encoding='utf8'))
        if r['raw_sha256']!=sha256(rawzip):raise RuntimeError('Raw file changed')
        return r
    if rec.exists() or rawzip.exists():raise RuntimeError('Partial run')
    inp=OUT/'inputs' if phase=='calibration' else EVAL/'inputs'
    route=inp/'routes'/('validation' if phase=='calibration' else 'final')/f'{scenario}_seed{seed}_t900.rou.xml'
    net=inp/'maps'/mapname/'map.net.xml';cfg=yaml.safe_load((OUT/'inputs/config.yaml').read_text(encoding='utf8'))
    guard=LatchedGuard(80);sensor=LimitedSensor(sensor_mode,seed);plan=PLANS[planname];overrides=0;fallbacks=0
    with tempfile.TemporaryDirectory(prefix='fixed_tune_') as tmp:
        env=TJunctionEnv(cfg,route,seed,Path(tmp),network_file=net,service_observer=guard)
        try:
            _,info=env.reset(seed=seed)
            while True:
                raw=info['raw_state'];sensed,_=sensor.read(raw,env.time//5);guard.set_observation(sensed)
                phaseid=int(raw['phase']); mask=info['action_mask']
                # Under prolonged outage, use the same uniform 15-second fallback as all controllers.
                fallback=sensor.outage_steps>=3;fallbacks+=int(fallback)
                green=15 if fallback else plan[phaseid]
                if np.asarray(mask,dtype=bool).sum()==1:pref=int(np.flatnonzero(mask)[0])
                elif mask[phaseid] and raw['green_elapsed']<green:pref=phaseid
                else:pref=(phaseid+1)%3
                action,_=guard.choose(pref,mask,phaseid,raw['transitioning'],sensed['queues'],[0,0,0])
                overrides+=int(action!=pref)
                _,_,_,done,info=env.step(action)
                if done:break
            dest.mkdir(parents=True,exist_ok=True);temp=dest/'raw.zip.tmp'
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                for n in ('timeseries.csv','decisions.csv','tripinfo.xml','metrics.json'):z.write(Path(tmp)/n,n)
            os.replace(temp,rawzip)
            r={'phase':phase,'map':mapname,'scenario':scenario,'seed':seed,'sensor':sensor_mode,'plan':planname,
               'map_sha256':sha256(net),'route_sha256':sha256(route),'raw_sha256':sha256(rawzip),
               'guard_interventions':overrides,'fallback_decisions':fallbacks,**info['metrics']}
            atomic_json(rec,r);return r
        finally:env.close()


def main():
    setup();rows=[]
    for mapname in CALMAPS:
     for scenario in CALSCENARIOS:
      for seed in CALSEEDS:
       for plan in PLANS:rows.append(run_case('calibration',mapname,scenario,seed,'limited_delay_miss',plan))
       print('TUNE',mapname,scenario,seed,len(rows),flush=True)
    cal=pd.DataFrame(rows);scores=cal.groupby('plan').cohort_delay_s.mean();best=min(scores.index,key=lambda x:(scores[x],x))
    atomic_json(OUT/'selected_plan.json',{'selected':best,'phase_green_seconds':PLANS[best],'calibration_mean_delay':scores.to_dict()})
    print('SELECTED',best,scores.to_dict(),flush=True)
    for mapname in TESTMAPS:
     for scenario in TESTSCENARIOS:
      for seed in TESTSEEDS:
       for sensor in SENSORS:rows.append(run_case('evaluation',mapname,scenario,seed,sensor,best))
       pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
       print('TEST',mapname,scenario,seed,len(rows),flush=True)
    atomic_json(OUT/'status.json',{'complete':True,'rows':len(rows),'selected':best})


if __name__=='__main__':main()
