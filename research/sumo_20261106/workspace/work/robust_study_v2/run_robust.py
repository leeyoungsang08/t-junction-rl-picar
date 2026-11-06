import os,sys,json,csv
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import product,groupby
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/"colab_study/T_Junction_RL"
sys.path.insert(0,str(ROOT))
from src.utils import load_yaml,sha256,write_json
from src.traffic_generator import generate_traffic
MAPS={"development":("test0","test1"),"holdout":("stress2","stress3")}
SEEDS={"development":range(70530,70535),"holdout":range(70700,70710)}
SCENARIOS=("medium","high","heldout_reversal","south_oversaturated")
CANDIDATES=(.7,.8,.9)
MODEL=HERE.parent/"gui_test/colab_seed42.pt"

def run(task):
 from src.environment import TJunctionEnv
 from src.guarded_policy import GuardedDirectPolicy
 from src.service_guard import DirectionServiceGuard
 from src.baselines import direct_phase_baseline
 stage,mapname,scenario,seed,variants=task
 net=ROOT/f"maps/experiment/{mapname}/map.net.xml"
 route=generate_traffic(scenario,seed,900,"final",directory=HERE/"demand")
 config=load_yaml("base.yaml")
 config["signal"]["control_mode"]="direct"
 config["simulation"].update(backend="libsumo",horizon_seconds=900,audit_vehicle_safety=True)
 out=[]
 for kind in variants:
  pol=GuardedDirectPolicy(MODEL,80)
  env=TJunctionEnv(config,route,seed,HERE/"runs"/stage/mapname/scenario/str(seed)/kind,
                   network_file=net,service_observer=pol.guard)
  decisions=[]
  try:
   _,info=env.reset(seed=seed)
   while True:
    model=pol.predict(info)
    proxy=direct_phase_baseline("max_pressure_proxy",info,config)
    state=info["raw_state"]
    requests=state["recent_requests"]
    total=sum(requests)
    dominant=max(requests)/total if total else 0
    if kind=="rl":
     selected=model.preferred
     source="rl"
    elif kind=="proxy":
     selected=proxy
     source="proxy"
    else:
     threshold=float(kind.split("_")[1])
     source="rl" if total>=8 and dominant>=threshold else "proxy"
     selected=model.preferred if source=="rl" else proxy
    action,urgent=pol.guard.choose(selected,list(info["action_mask"]),state["phase"],state["transitioning"],state["queues"],state["pending"])
    record=dict(time=env.time,rl=model.preferred,proxy=proxy,selected=selected,action=action,source=source,
                dominance=dominant,recent_total=total,urgent=urgent)
    _,_,_,done,info=env.step(action)
    decisions.append(record)
    if done:break
   with (env.output_dir/"policy_sources.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=decisions[0]);w.writeheader();w.writerows(decisions)
   maxima=[]
   for i,d in enumerate("WES"):
    stream=((r["phase"]!=i or r["stage"]!="green") and r[f"queue_{d}"]+r[f"pending_{d}"]>0 for r in env.timeseries)
    maxima.append(max((sum(1 for _ in g) for yes,g in groupby(stream) if yes),default=0))
   out.append(dict(stage=stage,map=mapname,scenario=scenario,seed=seed,variant=kind,route_hash=sha256(route),net_hash=sha256(net),
                   max_backlog_red=max(maxima),policy_rl=sum(r["source"]=="rl" for r in decisions),
                   switches_due_guard=sum(r["action"]!=r["selected"] for r in decisions),**info["metrics"]))
  finally:env.close()
 return out

def stage_run(stage,chosen=None):
 variants=["rl","proxy"]+[f"hybrid_{v:.1f}" for v in (CANDIDATES if stage=="development" else [chosen])]
 tasks=list(product([stage],MAPS[stage],SCENARIOS,SEEDS[stage],[variants]))
 rows=[]
 with ProcessPoolExecutor(max_workers=8) as p:
  for i,result in enumerate(p.map(run,tasks),1):
   rows.extend(result)
   if i%10==0:print(f"{stage}: {len(rows)}/{len(tasks)*len(variants)}",flush=True)
 with (HERE/f'{stage}.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 return rows

def choose(rows):
 import pandas as pd
 d=pd.DataFrame(rows)
 pivot=d.pivot(index=["map","scenario","seed"],columns="variant",values="cohort_delay_s")
 summary={}
 for q in CANDIDATES:
  name=f"hybrid_{q:.1f}"
  x=((pivot.proxy-pivot[name])/pivot.proxy*100).reset_index(name="improvement")
  group=x.groupby(["map","scenario"]).improvement.mean()
  rr=d[d.variant==name]
  summary[name]={"worst_map_scenario_mean_gain_pct":group.min(),"average_gain_pct":x.improvement.mean(),
                 "max_backlog_red":int(rr.max_backlog_red.max()),"collisions":int(rr.collision_events.sum())}
 valid=[n for n,x in summary.items() if x["max_backlog_red"]<=115 and x["collisions"]==0]
 if not valid:raise RuntimeError(f"No candidate met safety gate: {summary}")
 selected=max(valid,key=lambda n:(summary[n]["worst_map_scenario_mean_gain_pct"],summary[n]["average_gain_pct"]))
 write_json(HERE/"selection.json",{"selected":selected,"criterion":"maximize worst map-scenario mean percentage gain vs guarded proxy, subject to observed backlog red <=115 and no collision","results":summary})
 print("selection",selected,summary,flush=True)
 return float(selected.split("_")[1])

if __name__=="__main__":
 dev=stage_run("development")
 selected=choose(dev)
 stage_run("holdout",selected)
