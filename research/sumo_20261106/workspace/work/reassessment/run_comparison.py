import csv, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/"fairness/study"
PROJECT=HERE.parent/"fairness/T_Junction_RL"
sys.path[:0]=[str(PROJECT),str(STUDY)]
from run_guard import read_plan, MAP, FULL, MODEL_KEYS, SPLITS
from src.utils import sha256

def task(item):
 import torch
 from src.inference import Policy
 from src.environment import TJunctionEnv
 from src.service_guard import DirectionServiceGuard
 from src.baselines import direct_phase_baseline
 scenario,seed=item
 _,cfg=read_plan()
 route=STUDY/f"demand/{scenario}_{seed}.rou.xml"
 rows=[]
 for variant in ("untrained","partial_sensor"):
  for ix,model in enumerate(MODEL_KEYS):
   policy=Policy(FULL/model/"checkpoints/best_model.pt")
   if variant=="untrained":
    from src.dqn_network import QNetwork
    torch.manual_seed(91000+ix)
    policy.model=QNetwork(policy.payload["hidden"],len(policy.adapter.features),3,True)
    policy.model.eval()
   guard=DirectionServiceGuard(80)
   # The reduced-sensor case uses only road-side stopped vehicles for fairness.
   if variant=="partial_sensor":
    original_observe=guard.observe_second
    def physical_only(phase,stage,queues,pending):
     return original_observe(phase,stage,queues,[0,0,0])
    guard.observe_second=physical_only
   directory=HERE/"runs"/scenario/str(seed)/variant/model
   env=TJunctionEnv(cfg,route,seed,directory,network_file=MAP,service_observer=guard)
   changes=0
   try:
    _,info=env.reset(seed=seed)
    while True:
     data=deepcopy(info)
     if variant=="partial_sensor":
      data["raw_state"]["pending"]=[0,0,0]
      data["raw_state"]["recent_requests"]=[0,0,0]
      # The original longest-wait feature also includes uninserted vehicles;
      # use observed max consecutive physical waiting as a rough substitute.
      data["raw_state"]["longest_wait"]=list(info["max_direction_wait"])
     state=data["raw_state"]
     p=policy.predict(state)
     assert p["valid_actions"]==info["action_mask"].tolist()
     action,target=guard.choose(p["action_id"],info["action_mask"],state["phase"],state["transitioning"],
                                state["queues"],state["pending"])
     changes += action!=p["action_id"]
     _,_,_,done,info=env.step(action)
     if done: break
    rows.append(dict(scenario=scenario,seed=seed,model=model,variant=variant,route_sha256=sha256(route),
                     guard_overrides=changes,**info["metrics"]))
   finally: env.close()
 return rows

def main():
 tasks=[(s,i) for s in ("balanced","reversal","south_pulse","double_peak") for i in SPLITS["holdout"]]
 rows=[]
 with ProcessPoolExecutor(max_workers=8) as pool:
  for n,batch in enumerate(pool.map(task,tasks),1):
   rows+=batch
   if n%8==0: print(f"SUMO {len(rows)}/288",flush=True)
 with (HERE/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=="__main__": main()
