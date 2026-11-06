"""Final tests use new maps/demand. All controllers get identical noise + safety guard."""
import csv,json,os,shutil,zipfile
from pathlib import Path
from src.environment import TJunctionEnv
from src.inference import Policy
from src.service_guard import DirectionServiceGuard
from src.robust_observation import perturb_raw
from src.baselines import direct_phase_baseline
from src.traffic_generator import generate_traffic
from src.utils import ROOT,load_yaml,sha256,write_json

def main():
 drive=Path(os.environ["ROBUST_DRIVE"])
 quick=os.environ.get("ROBUST_EVAL_SMOKE")=="1"
 maps=["final_new0"] if quick else ["final_new0","final_new1"]
 scenarios=["medium"] if quick else ["medium","high","heldout_reversal","south_oversaturated"]
 seeds=[70800] if quick else list(range(70800,70808))
 run_seeds=[42] if quick else [42,43,44,45,46]
 models={f"robust_seed{s}":drive/f"seed{s}/checkpoints/best_model.pt" for s in run_seeds}
 models["previous_seed42"]=ROOT/"models/colab_previous_seed42.pt"
 assert all(x.is_file() for x in models.values())
 conditions=["clean","noise15"]
 policy_labels=list(models)+["queue_proxy"]
 dst=drive/("evaluation_smoke" if quick else "independent_evaluation")
 dst.mkdir(parents=True,exist_ok=True)
 mapinfo=json.loads((ROOT/"maps/experiment/manifest.json").read_text())
 plan={"test_maps":{m:mapinfo[m] for m in maps},"test_seeds":seeds,"scenarios":scenarios,
       "conditions":conditions,"models":{label:sha256(f) for label,f in models.items()},
       "baseline":"queue_proxy","guard_s":80,"simulation_seconds":900,
       "noise":"independent 15% multiplicative bounded jitter; full missing-sensor outages not simulated"}
 target=dst/"plan.json"
 if target.exists() and json.loads(target.read_text())!=plan:
  raise RuntimeError("Evaluation plan mismatch, use a fresh output location")
 write_json(target,plan)
 cfg=load_yaml("base.yaml")
 cfg["signal"]["control_mode"]="direct"
 cfg["simulation"].update(backend="libsumo",horizon_seconds=900,audit_vehicle_safety=True)
 records=[]
 for mapname in maps:
  net=ROOT/f"maps/experiment/{mapname}/map.net.xml"
  assert sha256(net)==mapinfo[mapname]
  for scenario in scenarios:
   for seed in seeds:
    route=generate_traffic(scenario,seed,900,"final",directory=ROOT/"outputs/robust_final_demand")
    for condition in conditions:
     for label in policy_labels:
      base=dst/"runs"/mapname/scenario/str(seed)/condition/label
      record=base/"record.json"; evidence=base/"raw.zip"
      if record.exists() and evidence.exists():
       result=json.loads(record.read_text())
      else:
       temp=ROOT/"outputs/robust_final_tmp"/mapname/scenario/str(seed)/condition/label
       observer=DirectionServiceGuard(80)
       env=TJunctionEnv(cfg,route,seed,temp,network_file=net,service_observer=observer)
       model=Policy(models[label]) if label in models else None
       guard_changes=0
       try:
        _,info=env.reset(seed=seed)
        while True:
         noisy=perturb_raw(info["raw_state"],seed,env.time//5,condition=="noise15")
         altered={**info,"raw_state":noisy}
         if model: preferred=model.predict(noisy)["action_id"]
         else: preferred=direct_phase_baseline("max_pressure_proxy",altered,cfg)
         raw=info["raw_state"]
         action,urgent=observer.choose(preferred,info["action_mask"],raw["phase"],raw["transitioning"],raw["queues"],raw["pending"])
         guard_changes+=action!=preferred
         _,_,_,done,info=env.step(action)
         if done:break
        result={"map":mapname,"scenario":scenario,"seed":seed,"sensor":condition,"controller":label,
                "route_sha256":sha256(route),"guard_interventions":guard_changes,**info["metrics"]}
       finally: env.close()
       base.mkdir(parents=True,exist_ok=True)
       tmp=base/"raw.zip.tmp"
       with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ("timeseries.csv","decisions.csv","tripinfo.xml","metrics.json"):
         z.write(temp/name,name)
        z.write(route,"demand.rou.xml")
        z.write(route.with_suffix(".json"),"demand.rou.json")
       os.replace(tmp,evidence)
       write_json(record,result)
       shutil.rmtree(temp)
      records.append(result)
     print("tested",mapname,scenario,seed,condition,flush=True)
     tempcsv=dst/"results.csv.tmp"
     with tempcsv.open("w",newline="",encoding="utf-8") as f:
      w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
     os.replace(tempcsv,dst/"results.csv")
 print("COMPLETE",len(records),flush=True)
if __name__=="__main__":main()
