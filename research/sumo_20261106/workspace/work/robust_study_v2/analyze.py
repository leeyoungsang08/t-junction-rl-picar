import csv,json
from pathlib import Path
from itertools import groupby
import xml.etree.ElementTree as ET
import numpy as np,pandas as pd
from scipy.stats import t
HERE=Path("R:/robust_study_v2")
PROJECT=Path("R:/colab_study/T_Junction_RL")
sys_path=__import__('sys').path
sys_path.insert(0,str(PROJECT))
from src.utils import sha256

def audit_row(r):
 path=HERE/"runs"/r.stage/r.map/r.scenario/str(r.seed)/r.variant
 routes=HERE/f"demand/final/{r.scenario}_seed{r.seed}_t900.rou.xml"
 assert sha256(routes)==r.route_hash
 assert sha256(PROJECT/f"maps/experiment/{r.map}/map.net.xml")==r.net_hash
 req=json.loads(routes.with_suffix('.json').read_text())["vehicles"]
 tree=ET.parse(path/"tripinfo.xml")
 trips={e.get("id"):e.attrib for e in tree.findall('tripinfo')}
 assert len(req)==r.requested_vehicles and len(trips)==r.departed_vehicles
 delay=sum(float(x["waitingTime"])+float(x["departDelay"]) for x in trips.values())
 delay+=sum(900-v["requested_depart"] for k,v in req.items() if k not in trips)
 assert abs(delay/len(req)-r.cohort_delay_s)<1e-9
 assert sum(float(x["arrival"])>=0 for x in trips.values())==r.completed_vehicles
 ts=pd.read_csv(path/"timeseries.csv")
 assert len(ts)==900 and ts.time.tolist()==list(range(1,901))
 assert ts.cumulative_arrived.iloc[-1]==r.completed_vehicles
 segments=[(p,s,len(list(g))) for (p,s),g in groupby(zip(ts.phase,ts.stage))]
 for i,(p,s,n) in enumerate(segments[:-1]):
  p2,s2,_=segments[i+1]
  assert (s=="green" and 10<=n<=45 and p2==p and s2=="yellow") or (s=="yellow" and n==3 and p2==p and s2=="all_red") or (s=="all_red" and n==1 and p2!=p and s2=="green")
 maxes=[]
 for i,d in enumerate('WES'):
  ticks=((ts.phase!=i)|(ts.stage!="green"))&((ts[f"queue_{d}"]+ts[f"pending_{d}"])>0)
  maxes.append(max((sum(1 for _ in g) for yes,g in groupby(ticks) if yes),default=0))
 assert max(maxes)==r.max_backlog_red
 p=pd.read_csv(path/"policy_sources.csv")
 decisions=pd.read_csv(path/"decisions.csv")
 assert len(p)==len(decisions)==180
 assert all(p.action==decisions.action)
 assert (p.source=="rl").sum()==r.policy_rl
 assert (p.action!=p.selected).sum()==r.switches_due_guard
 return None

def main():
 chosen=json.loads((HERE/"selection.json").read_text())["selected"]
 a=pd.read_csv(HERE/"development.csv");b=pd.read_csv(HERE/"holdout.csv")
 assert len(a)==200 and len(b)==240
 assert not set(a.seed)&set(b.seed) and not set(a['map'])&set(b['map'])
 assert not pd.concat([a,b]).duplicated(["stage","map","scenario","seed","variant"]).any()
 for x in pd.concat([a,b]).itertuples(index=False): audit_row(x)
 print("audited native runs",len(a)+len(b))
 print("selection",chosen)
 fields=['cohort_delay_s','completion_rate','p95_cohort_delay_s','worst_direction_cohort_delay_s','signal_switches','max_backlog_red','policy_rl','switches_due_guard','collision_events','teleport_events']
 summary=b.groupby(["map","scenario","variant"])[fields].mean()
 summary.to_csv(HERE/"holdout_summary.csv")
 print(summary[fields[:6]].round(3).to_string())
 print("macro",b.groupby('variant')[fields].mean().round(3).to_string())
 pivot=b.pivot(index=["map","scenario","seed"],columns="variant",values="cohort_delay_s")
 gain=(pivot.proxy-pivot[chosen])/pivot.proxy*100
 print("individual gain",gain.describe().round(3).to_string(),"negative",int((gain<0).sum()))
 print("scenario means",gain.groupby(level=["map","scenario"]).agg(['mean','min','max',lambda x:sum(x<0)]).round(3).to_string())
 print("max fairness and events",b.groupby('variant')[["max_backlog_red","collision_events","teleport_events"]].max().to_string())
if __name__ == "__main__": main()
