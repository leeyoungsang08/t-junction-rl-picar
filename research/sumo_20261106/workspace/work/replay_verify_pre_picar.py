"""Re-run a few saved scenarios in the same software environment, no overwrites."""
import json, math, shutil, tempfile, zipfile
from pathlib import Path
import yaml
import pre_picar_balanced_evaluate as e
from pre_picar_study import atomic_json

original=e.OUT
cases=[('balanceA','medium',70980,'limited_clean','old'),
       ('balanceA','medium',70980,'limited_clean','new43'),
       ('balanceB','south_oversaturated',70982,'limited_outage','old'),
       ('balanceB','south_oversaturated',70982,'limited_outage','new43')]
names=('timeseries.csv','decisions.csv','metrics.json','sensor_decisions.json')
rows=[]
def metric_equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(metric_equal(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(metric_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)
    return a==b
with tempfile.TemporaryDirectory(prefix='sumo_replay_audit_') as tmp:
    e.OUT=Path(tmp);shutil.copytree(original/'inputs',e.OUT/'inputs')
    cfg=yaml.safe_load((e.OUT/'inputs/config.yaml').read_text(encoding='utf8'))
    models={'old':e.Policy(e.OUT/'inputs/old.pt'),'new43':e.NewPolicy(e.OUT/'inputs/new43.pt')}
    for case in cases:
        e.run_case(*case,models,cfg)
        rel=Path('runs')/case[0]/case[1]/str(case[2])/case[3]/case[4]/'raw.zip'
        with zipfile.ZipFile(original/rel) as old, zipfile.ZipFile(e.OUT/rel) as new:
            checks={name:old.read(name)==new.read(name) for name in names}
            first=json.loads(old.read('metrics.json'));second=json.loads(new.read('metrics.json'))
            checks['metrics_values_within_tolerance']=metric_equal(first,second)
            diffs={k:[first[k],second[k]] for k in first if not metric_equal(first[k],second[k])}
            rounded={k:[first[k],second[k]] for k in first if first[k]!=second[k] and metric_equal(first[k],second[k])}
            if diffs or rounded: print('METRIC_DIFFERENCES',diffs,'rounding',rounded,flush=True)
        rows.append({'case':case,'checks':checks,'metric_disagreements':diffs,'rounding_differences':rounded})
        print('REPLAY',case,checks,flush=True)
        if not all(checks[n] for n in ('timeseries.csv','decisions.csv','sensor_decisions.json','metrics_values_within_tolerance')):raise AssertionError(f'Replay mismatch: {case}')
result={'scope':'4 same-environment reruns; metrics compared with abs tolerance 1e-9, rel tolerance 1e-10; no independent environment or full 504-run reproduction',
        'all_match':True,'cases':rows}
atomic_json(original/'representative_rerun_verification.json',result)
