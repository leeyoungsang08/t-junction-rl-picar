"""Test a sensor-independent cyclic red-age watchdog with legal phase masking."""
import os, sys, json, shutil, tempfile, zipfile
from pathlib import Path
import pandas as pd
import yaml
from pre_picar_study import PROJECT,HERE,LimitedSensor,LatchedGuard,atomic_json,versions
sys.path.insert(0,str(PROJECT))
from src.environment import TJunctionEnv
from src.inference import Policy
from src.baselines import direct_phase_baseline
from src.utils import sha256

SOURCE=HERE/'outputs'/'SUMO_PiCar_전_센서시험'
OUT=HERE/'outputs'/'SUMO_PiCar_전_신호감시_시험'
MAPS=('probeA','probeB'); SCENARIOS=('medium','high','heldout_reversal','south_oversaturated'); SEEDS=(70950,70951,70952)


class BlindWatchdog(LatchedGuard):
    def reset(self):
        super().reset(); self.red_age=[0,0,0]

    def observe_second(self,phase,stage,queues,pending):
        super().observe_second(phase,stage,queues,pending)
        self.red_age=[0 if stage=='green' and d==phase else age+1 for d,age in enumerate(self.red_age)]

    def apply_watchdog(self,preferred,mask,phase,transitioning):
        eligible=[d for d in range(3) if d!=phase and mask[d] and self.red_age[d]>=70]
        return max(eligible,key=lambda d:(self.red_age[d],-((d-phase)%3))) if eligible and not transitioning else preferred


def setup():
    original=json.loads((SOURCE/'manifest.json').read_text(encoding='utf8'))
    for path,h in original['inputs'].items():
        if sha256(SOURCE/path)!=h: raise RuntimeError(f'Frozen source changed {path}')
    marker=OUT/'manifest.json'
    plan={'driver_sha256':sha256(Path(__file__)),'sensor_driver_sha256':sha256(HERE/'work/pre_picar_study.py'),
          'source_manifest_sha256':sha256(SOURCE/'manifest.json'),'software':versions(),
          'scope':'sensor-independent 70-second red-age request for every direction, regardless of detected demand; legal mask and clearance retained'}
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf8'))!=plan: raise RuntimeError('Plan changed')
    else:
        OUT.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),OUT/'driver.py');atomic_json(marker,plan)


def run(mapname,scenario,seed):
    dest=OUT/'runs'/mapname/scenario/str(seed); rec=dest/'record.json'; zipfile_path=dest/'raw.zip'
    if rec.exists() and zipfile_path.exists():
        r=json.loads(rec.read_text(encoding='utf8'))
        if sha256(zipfile_path)!=r['raw_sha256']: raise RuntimeError('Raw log changed')
        return r
    if rec.exists() or zipfile_path.exists(): raise RuntimeError('Incomplete run')
    inp=SOURCE/'inputs'; cfg=yaml.safe_load((inp/'config.yaml').read_text(encoding='utf8'));cfg['signal']['fixed_green']=15
    route=inp/'routes/final'/f'{scenario}_seed{seed}_t900.rou.xml';net=inp/'maps'/mapname/'map.net.xml'
    policy=Policy(inp/'old.pt'); guard=BlindWatchdog(80);sensor=LimitedSensor('limited_outage',seed)
    fallback_n=overrides=watchdog_overrides=0;decisions=[]
    with tempfile.TemporaryDirectory(prefix='fallback_eval_') as tmp:
        env=TJunctionEnv(cfg,route,seed,Path(tmp),network_file=net,service_observer=guard)
        try:
            _,info=env.reset(seed=seed)
            while True:
                raw=info['raw_state'];sensed,outage=sensor.read(raw,env.time//5);guard.set_observation(sensed)
                fallback=sensor.outage_steps>=3;fallback_n+=int(fallback)
                if fallback:
                    pref=direct_phase_baseline('fixed_time',{**info,'raw_state':sensed},cfg)
                    action,_=guard.choose(pref,info['action_mask'],raw['phase'],raw['transitioning'],sensed['queues'],[0,0,0])
                else:
                    pref=policy.predict(sensed)['action_id']
                    action,_=guard.choose(pref,info['action_mask'],raw['phase'],raw['transitioning'],sensed['queues'],[0,0,0])
                watchdog_action=guard.apply_watchdog(action,info['action_mask'],raw['phase'],raw['transitioning'])
                watchdog_overrides+=int(watchdog_action!=action);action=watchdog_action
                overrides+=int(action!=pref)
                decisions.append({'time':env.time,'fallback':fallback,'outage':outage,'preferred':pref,'action':action,'red_age':guard.red_age})
                _,_,_,done,info=env.step(action)
                if done: break
            dest.mkdir(parents=True,exist_ok=True)
            (Path(tmp)/'sensor_decisions.json').write_text(json.dumps(decisions),encoding='utf8')
            temp=dest/'raw.zip.tmp'
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
                for n in ('timeseries.csv','decisions.csv','tripinfo.xml','metrics.json','sensor_decisions.json'):z.write(Path(tmp)/n,n)
            os.replace(temp,zipfile_path)
            r={'map':mapname,'scenario':scenario,'seed':seed,'guard_interventions':overrides,'fallback_decisions':fallback_n,'watchdog_interventions':watchdog_overrides,
               'raw_sha256':sha256(zipfile_path),'route_sha256':sha256(route),'map_sha256':sha256(net),**info['metrics']}
            atomic_json(rec,r);return r
        finally:env.close()


def main():
    setup();rows=[]
    for mapname in MAPS:
      for scenario in SCENARIOS:
       for seed in SEEDS:
        rows.append(run(mapname,scenario,seed))
        print('WATCHDOG',mapname,scenario,seed,len(rows),flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'results.csv',index=False)
    old=pd.read_csv(SOURCE/'results.csv');old=old[(old.sensor=='limited_outage')&(old.controller=='model')]
    paired=df.merge(old,on=['map','scenario','seed'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(paired)==24 and (paired.route_sha256_new==paired.route_sha256_old).all()
    for r in paired.itertuples(index=False):assert sha256(OUT/'runs'/r.map/r.scenario/str(r.seed)/'raw.zip')==r.raw_sha256_new
    paired.to_csv(OUT/'paired.csv',index=False)
    d=paired.cohort_delay_s_new-paired.cohort_delay_s_old
    tol=1e-9
    summary={'complete':True,'cases':len(paired),'mean_delay_old':paired.cohort_delay_s_old.mean(),
             'mean_delay_new':paired.cohort_delay_s_new.mean(),'mean_delta':d.mean(),'wins':int((d< -tol).sum()),
             'losses':int((d>tol).sum()),'ties':int((d.abs()<=tol).sum()),'worst_regression':d.max(),
             'max_wait_old':paired.max_consecutive_wait_s_old.max(),'max_wait_new':paired.max_consecutive_wait_s_new.max(),
             'collisions_new':int(paired.collision_events_new.sum()),'watchdog_interventions':int(paired.watchdog_interventions.sum())}
    atomic_json(OUT/'summary.json',summary);print('SUMMARY',summary,flush=True)


if __name__=='__main__':main()
