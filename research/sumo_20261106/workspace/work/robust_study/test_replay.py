import json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,"R:/colab_study/T_Junction_RL")
from src.adaptive_policy import AdaptiveDirectPolicy
from src.utils import load_yaml
home=Path("R:/robust_study")
chosen=json.loads((home/"selection.json").read_text())["selected"]
rows=pd.read_csv(home/"holdout.csv").query("variant == @chosen")
cfg=load_yaml("base.yaml");cfg["signal"]["control_mode"]="direct"
for r in rows.itertuples(index=False):
 path=home/"runs"/r.stage/r.map/r.scenario/str(r.seed)/r.variant
 times=pd.read_csv(path/"timeseries.csv")
 dec=pd.read_csv(path/"decisions.csv")
 source=pd.read_csv(path/"policy_sources.csv")
 pol=AdaptiveDirectPolicy("R:/gui_test/colab_seed42.pt",cfg)
 for k,row in enumerate(dec.itertuples(index=False)):
  state=json.loads(row.state); mask=json.loads(row.action_mask)
  info={"raw_state":state,"action_mask":mask}
  prediction=pol.predict(info)
  assert prediction.action==int(row.action)==int(source.iloc[k].action)
  assert prediction.selected_source==source.iloc[k].source
  assert prediction.proposed==source.iloc[k].selected
  for tick in times.iloc[k*5:(k+1)*5].itertuples(index=False):
   pol.guard.observe_second(tick.phase,tick.stage,[getattr(tick,f'queue_{d}') for d in "WES"],
                            [getattr(tick,f'pending_{d}') for d in "WES"])
print("Portable policy replay matches",len(rows),"episodes,",len(rows)*180,"decisions")
