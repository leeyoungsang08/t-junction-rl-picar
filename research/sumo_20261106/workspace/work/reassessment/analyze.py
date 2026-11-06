from pathlib import Path
import pandas as pd, numpy as np,json
from scipy.stats import t
from xml.etree import ElementTree as ET
base=Path("R:/reassessment")
study=Path("R:/fairness/study")
df=pd.read_csv(base/"comparison.csv")
held=pd.read_csv(study/"holdout.csv")
trained=held[held.variant=="guard80"].copy()
assert len(df)==288 and len(trained)==144
assert not df.duplicated(["scenario","seed","model","variant"]).any()
for r in df.itertuples(index=False):
 path=base/"runs"/r.scenario/str(r.seed)/r.variant/r.model
 manifest=json.loads((study/f"demand/{r.scenario}_{r.seed}.rou.json").read_text())
 xml={x.get("id"):x for x in ET.parse(path/"tripinfo.xml").findall("tripinfo")}
 requests=manifest["vehicles"]
 assert len(xml)==r.departed_vehicles
 ds=[float(xml[k].get("waitingTime"))+float(xml[k].get("departDelay")) if k in xml else 1200-v["requested_depart"] for k,v in requests.items()]
 assert abs(np.mean(ds)-r.cohort_delay_s)<1e-10
 assert sum(float(x.get("arrival"))>=0 for x in xml.values())==r.completed_vehicles
 assert (path/"timeseries.csv").exists()
 ref=held[(held.scenario==r.scenario)&(held.seed==r.seed)&(held.model==r.model)&(held.variant=="guard80")].iloc[0]
 assert ref.route_sha256==r.route_sha256
df["group"]=df.variant
trained["group"]="trained"
both=pd.concat([df,trained],ignore_index=True)
metrics=["cohort_delay_s","completion_rate","p95_cohort_delay_s","worst_direction_cohort_delay_s","cohort_delay_W_s","cohort_delay_E_s","cohort_delay_S_s","collision_events","teleport_events"]
out=both.groupby(["scenario","group"])[metrics].mean()
out.to_csv(base/"summary.csv")
print(out.round(3).to_string())
print("\nmacro\n",both.groupby("group")[metrics].mean().round(3).to_string())
averages=both.groupby(["scenario","seed","group"])[metrics].mean().reset_index()
res=[]
for s,g in averages.groupby("scenario"):
 w=g.pivot(index="seed",columns="group",values="cohort_delay_s")
 for variant in ("untrained","partial_sensor"):
  d=w[variant]-w.trained
  h=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
  res.append(dict(scenario=s,comparison=variant+" minus trained",n=len(d),mean=d.mean(),lower=d.mean()-h,upper=d.mean()+h))
p=pd.DataFrame(res)
p.to_csv(base/"paired.csv",index=False)
print("\npaired\n",p.round(3).to_string(index=False))
print("288 native trip files audited")
