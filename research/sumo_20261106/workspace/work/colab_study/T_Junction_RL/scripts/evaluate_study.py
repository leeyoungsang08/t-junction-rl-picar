import csv,json,os,shutil
from copy import deepcopy
from pathlib import Path
from src.evaluate import run_episode
from src.guarded_policy import GuardedDirectPolicy
from src.traffic_generator import generate_traffic
from src.utils import ROOT,load_yaml,sha256,write_json

SCENARIOS=("medium","high","heldout_reversal","south_oversaturated")
MAPS=("test0","test1")
SEEDS=tuple(range(70400,70410))
MODELS=(42,43,44,45,46)

def main():
 root=Path(os.environ["STUDY_DRIVE"]).resolve()
 output=root/"final_evaluation"
 output.mkdir(parents=True,exist_ok=True)
 cfg=load_yaml("base.yaml")
 cfg["signal"]["control_mode"]="direct"
 cfg["simulation"].update(backend="libsumo",horizon_seconds=900,audit_vehicle_safety=True)
 plan={"maps":MAPS,"scenarios":SCENARIOS,"seeds":SEEDS,"models":MODELS,"checkpoints":["best_at_300","best_model"],
       "guard_threshold":80,"horizon":900,"note":"fixed evaluation after all training; 80s is an intervention trigger"}
 for model in MODELS:
  for checkpoint in plan["checkpoints"]:
   path=root/f"seed{model}/checkpoints/{checkpoint}.pt"
   if not path.is_file(): raise FileNotFoundError(path)
   plan.setdefault("model_sha256",{})[f"{model}_{checkpoint}"]=sha256(path)
 mapmanifest=json.loads((ROOT/"maps/experiment/manifest.json").read_text())
 plan["map_sha256"]={x:mapmanifest[x] for x in MAPS}
 planned=output/"plan.json"
 if planned.exists() and json.loads(planned.read_text())!=json.loads(json.dumps(plan)):
  raise RuntimeError("Evaluation plan changed; start a new output directory")
 write_json(planned,plan)
 records=[]
 for mapname in MAPS:
  net=ROOT/f"maps/experiment/{mapname}/map.net.xml"
  assert sha256(net)==mapmanifest[mapname]
  for scenario in SCENARIOS:
   for seed in SEEDS:
    route=generate_traffic(scenario,seed,900,"final",directory=ROOT/"outputs/final_routes")
    specs=[(f"seed{s}_{ck}",root/f"seed{s}/checkpoints/{ck}.pt",None) for s in MODELS for ck in plan["checkpoints"]]
    specs += [(v,None,v) for v in ("fixed_time","rule_based","max_pressure_proxy")]
    for label,model,baseline in specs:
     directory=ROOT/"outputs/final_eval"/mapname/scenario/str(seed)/label
     saved=output/"runs"/mapname/scenario/str(seed)/label/"metrics.json"
     if saved.is_file():
      result=json.loads(saved.read_text())
     else:
      guarded=GuardedDirectPolicy(model,80) if model else None
      result=run_episode(cfg,route,seed,"dqn" if guarded else baseline,directory,network_file=net,guarded_policy=guarded)
      saved.parent.mkdir(parents=True,exist_ok=True)
      for filename in ("metrics.json","timeseries.csv","decisions.csv","tripinfo.xml"):
       shutil.copy2(directory/filename,saved.parent/filename)
      result=json.loads(saved.read_text())
     records.append({"map":mapname,"scenario":scenario,"seed":seed,"controller":label,
                     "route_sha256":sha256(route),**result})
    print(f"evaluated {mapname} {scenario} {seed}",flush=True)
    with (output/"results.csv.tmp").open("w",newline="",encoding="utf-8") as f:
     w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    os.replace(output/"results.csv.tmp",output/"results.csv")
 print("Evaluation complete",len(records))
if __name__=="__main__": main()
