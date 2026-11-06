import json,math,shutil,tempfile,zipfile
from pathlib import Path
import yaml
import fixed_intersection_watchdog as w
from pre_picar_study import atomic_json

original=w.OUT
cases=[('medium',70993,'limited_clean','old'),('medium',70993,'limited_clean','watch70'),
       ('south_oversaturated',70995,'severe_unseen','old'),('south_oversaturated',70995,'severe_unseen','watch90')]
def eq(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)
    return a==b
rows=[]
with tempfile.TemporaryDirectory(prefix='watchdog_rerun_') as tmp:
    w.OUT=Path(tmp);shutil.copytree(original/'inputs',w.OUT/'inputs');inp=w.OUT/'inputs'
    cfg=yaml.safe_load((inp/'config.yaml').read_text(encoding='utf8'))
    models={'old':w.Policy(inp/'old.pt'),'prior43':w.NewPolicy(inp/'prior43.pt')}
    for case in cases:
        w.run_case('test',*case,None,models,cfg)
        path=Path('runs/test')/case[0]/str(case[1])/case[2]/case[3]/'raw.zip'
        with zipfile.ZipFile(original/path) as a,zipfile.ZipFile(w.OUT/path) as b:
            exact={n:a.read(n)==b.read(n) for n in ('timeseries.csv','decisions.csv','watch_decisions.json')}
            metrics=eq(json.loads(a.read('metrics.json')),json.loads(b.read('metrics.json')))
        if not all(exact.values()) or not metrics:raise AssertionError(f'Rerun mismatch {case}: {exact} {metrics}')
        rows.append({'case':case,'decisions_and_timeseries_exact':exact,'metrics_within_tolerance':metrics})
        print('WATCH_RERUN',case,exact,metrics,flush=True)
atomic_json(original/'representative_rerun.json',{'all_match':True,'scope':'four test cases, same software environment, not the whole experiment',
    'numeric_tolerance':'abs 1e-9, rel 1e-10','cases':rows})
