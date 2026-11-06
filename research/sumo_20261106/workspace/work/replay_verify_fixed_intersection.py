"""Re-run four selected cases in the same environment without altering inputs."""
import json,math,shutil,tempfile,zipfile
from pathlib import Path
import yaml
import fixed_intersection_evaluate as e
from pre_picar_study import atomic_json

original=e.OUT
cases=[('medium',70990,'limited_clean','old'),('medium',70990,'limited_clean','explore_plain44'),
       ('south_oversaturated',70992,'severe_unseen','old'),('south_oversaturated',70992,'severe_unseen','prior43')]


def equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)
    return a==b


rows=[]
with tempfile.TemporaryDirectory(prefix='balanceA_rerun_') as tmp:
    e.OUT=Path(tmp);shutil.copytree(original/'inputs',e.OUT/'inputs');inp=e.OUT/'inputs'
    cfg=yaml.safe_load((inp/'config.yaml').read_text(encoding='utf8'))
    models={'old':e.Policy(inp/'old.pt'),'explore_plain44':e.NewPolicy(inp/'explore_plain44.pt'),'prior43':e.NewPolicy(inp/'prior43.pt')}
    for case in cases:
        e.run_case(*case,models,cfg,None)
        path=Path('runs')/case[0]/str(case[1])/case[2]/case[3]/'raw.zip'
        with zipfile.ZipFile(original/path) as first,zipfile.ZipFile(e.OUT/path) as second:
            exact={n:first.read(n)==second.read(n) for n in ('timeseries.csv','decisions.csv','sensor_decisions.json')}
            metrics=equal(json.loads(first.read('metrics.json')),json.loads(second.read('metrics.json')))
        if not all(exact.values()) or not metrics:raise AssertionError(f'Rerun mismatch {case}: {exact} {metrics}')
        rows.append({'case':case,'exact_csv_and_decisions':exact,'metrics_within_tolerance':metrics})
        print('RERUN',case,exact,metrics,flush=True)
atomic_json(original/'representative_rerun.json',{'all_match':True,'scope':'four cases, same software environment; no full independent rerun',
    'numeric_tolerance':'abs 1e-9, rel 1e-10','cases':rows})
