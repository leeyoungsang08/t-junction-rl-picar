from pathlib import Path
from collections import Counter
from itertools import groupby
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import sys
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK.parent/"colab_study/T_Junction_RL"))
from src.utils import sha256
manifest=next((WORK/"routes/final").glob("*.json"))
req=json.loads(manifest.read_text())["vehicles"]
results=json.loads((WORK/"results.json").read_text())
phase=[]
for controller in ("trained","max_pressure_proxy","fixed_time"):
 d=WORK/"runs"/controller
 ts=pd.read_csv(d/"timeseries.csv")
 assert len(ts)==900 and ts.time.tolist()==list(range(1,901))
 xml={e.get('id'):e.attrib for e in ET.parse(d/"tripinfo.xml").findall("tripinfo")}
 data=results[controller]
 assert len(xml)==data["departed_vehicles"]
 delay=sum(float(v["waitingTime"])+float(v["departDelay"]) for v in xml.values())+sum(900-x["requested_depart"] for k,x in req.items() if k not in xml)
 assert abs(delay/len(req)-data["cohort_delay_s"])<1e-9
 assert sum(float(x["arrival"])>=0 for x in xml.values())==data["completed_vehicles"]
 runs=[(p,s,len(list(g))) for (p,s),g in groupby(zip(ts.phase,ts.stage))]
 for j,(p,s,n) in enumerate(runs):
  if j==len(runs)-1: continue
  p2,s2,_=runs[j+1]
  if s=="green": assert 10<=n<=45 and p==p2 and s2=="yellow"
  elif s=="yellow": assert n==3 and p==p2 and s2=="all_red"
  else: assert n==1 and p!=p2 and s2=="green"
 for j,(lo,hi) in enumerate([(0,198),(198,495),(495,900)]):
  w=ts.iloc[lo:hi]
  r=Counter(v["origin"] for v in req.values() if lo <= v["requested_depart"] <hi)
  phase.append({"controller":controller,"start":lo,"end":hi,"requests_W":r["W"],"requests_E":r["E"],"requests_S":r["S"],
    **{f"green_{d}_pct":float(((w.phase==i)&(w.stage=="green")).mean()*100) for i,d in enumerate("WES")},
    "switches":int(w.switch_events.sum()),"avg_queue":w.queue_total.mean()})
out=pd.DataFrame(phase)
out.to_csv(WORK/"phase_summary.csv",index=False)
print(out.round(2).to_string(index=False))
print("RESULTS")
print(pd.DataFrame(results).T[["requested_vehicles","completed_vehicles","uninserted_vehicles","completion_rate","cohort_delay_s","p95_cohort_delay_s","worst_direction_cohort_delay_s","signal_switches","collision_events","teleport_events"]].round(3).to_string())
decision=pd.read_csv(WORK/"runs/trained/decisions.csv")
print("guard override",decision.guard_override.sum(),"requests",sum(~decision.requested_switch_target.isna()))
print("direct versus proxy percent",100*(1-results["trained"]["cohort_delay_s"]/results["max_pressure_proxy"]["cohort_delay_s"]))
first=pd.read_csv(WORK/"runs_first/trained/timeseries.csv")
second=pd.read_csv(WORK/"runs/trained/timeseries.csv")
print("first vs second timeseries identical:",first.equals(second))
print("Tripinfo raw sums and full signal clearances verified for 3 controllers")
