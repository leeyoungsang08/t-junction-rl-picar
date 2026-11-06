"""Ensure pending-vehicle delay is invariant across Python hash seeds."""
import os,subprocess,sys
from pathlib import Path


def test_uninserted_delay_independent_of_hash_seed(tmp_path):
    root=Path(__file__).resolve().parents[1]
    metric=root/'outputs/SUMO_결과계산_결정성_수정/metrics.py'
    source=root/'work/robust_retrain/T_Junction_RL_reward_ablation'
    tripinfo=tmp_path/'tripinfo.xml';tripinfo.write_text('<tripinfos/>',encoding='utf8')
    code='''
import importlib.util,json,sys
spec=importlib.util.spec_from_file_location("patched_metrics",sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
vehicles={f"v{i:04d}":{"requested_depart":i*0.3191,"origin":"WES"[i%3]} for i in range(1500)}
timeseries=[{"queue_total":0,"max_consecutive_wait_s":0}]
r=m.episode_metrics(sys.argv[2],{"horizon_seconds":900,"vehicles":vehicles},timeseries,set(),set(),0,0,0,0)
print(json.dumps(r,sort_keys=True))
'''
    def evaluate(path):
        values=[]
        for seed in ('1','2','123'):
            env={**os.environ,'PYTHONHASHSEED':seed,'PYTHONPATH':str(source)}
            values.append(subprocess.check_output([sys.executable,'-c',code,str(path),str(tripinfo)],env=env,text=True))
        return values
    assert len(set(evaluate(metric)))==1
    assert len(set(evaluate(source/'src/metrics.py')))>1
