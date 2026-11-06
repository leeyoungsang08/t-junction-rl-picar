import csv,json,os,sys,hashlib
from pathlib import Path
from copy import deepcopy
from itertools import groupby
ROOT=Path(__file__).resolve().parents[1]/"colab_study/T_Junction_RL"
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.evaluate import run_episode
from src.guarded_policy import GuardedDirectPolicy
from src.environment import TJunctionEnv
from src.traffic_generator import generate_traffic
from src.utils import load_yaml,sha256

def main():
 model=WORK/"colab_seed42.pt"
 plan=json.loads((WORK/"colab_plan.json").read_text())
 assert sha256(model)==plan["model_sha256"]["42_best_model"]
 net=ROOT/"maps/experiment/test0/map.net.xml"
 assert sha256(net)==plan["map_sha256"]["test0"]
 seed=70520
 assert seed not in plan["seeds"]
 route=generate_traffic("heldout_reversal",seed,900,"final",directory=WORK/"routes")
 config=load_yaml("base.yaml")
 config["signal"]["control_mode"]="direct"
 config["simulation"].update(backend="libsumo",horizon_seconds=900,audit_vehicle_safety=True)
 print("model",sha256(model),"net",sha256(net),"route",sha256(route),flush=True)
 results={}
 for baseline in ("fixed_time","max_pressure_proxy"):
  results[baseline]=run_episode(config,route,seed,baseline,WORK/"runs"/baseline,network_file=net)
  print("baseline",baseline,results[baseline]["cohort_delay_s"],flush=True)
 policy=GuardedDirectPolicy(model,80)
 config["simulation"]["backend"]="traci"
 env=TJunctionEnv(config,route,seed,WORK/"runs"/"trained",gui=True,delay_ms=35,capture=True,network_file=net,service_observer=policy.guard)
 try:
  _,info=env.reset(seed=seed)
  views=env.conn.gui.getIDList()
  print("GUI connected",views,flush=True)
  if views:
   env.conn.gui.setZoom(views[0],150)
   env.conn.gui.setOffset(views[0],0,0)
  while True:
   dec=policy.predict(info)
   _,_,_,done,info=env.step(dec.action)
   env.decisions[-1].update(model_preferred=dec.preferred,guard_target=dec.urgent,guard_override=dec.overridden,
                            **{f"age_{d}":x for d,x in zip("WES",dec.ages)})
   if env.time in (200,400,700) and views:
    env.conn.gui.screenshot(views[0], str(WORK/f"sumo_gui_{env.time}.png"))
   if env.time%100==0: print("GUI seconds",env.time,flush=True)
   if done: 
    env.save_logs()
    results["trained"]=info["metrics"]
    break
 finally: env.close()
 (WORK/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
 print("DONE",json.dumps({x:{k:v[k] for k in ("cohort_delay_s","completion_rate","collision_events","teleport_events")} for x,v in results.items()}),flush=True)
if __name__=="__main__":main()
