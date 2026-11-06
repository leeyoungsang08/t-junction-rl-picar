from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import t
p=Path('R:/colab_analysis')
d=pd.read_csv(p/'results.csv')
assert len(d)==1040 and not d.duplicated(['map','scenario','seed','controller']).any()
assert (d.groupby(['map','scenario','seed']).route_sha256.nunique()==1).all()
m=d[d.controller.str.startswith('seed')].copy()
m['budget']=np.where(m.controller.str.endswith('best_at_300'),'300','600')
m['train_seed']=m.controller.str.extract(r'seed(\d+)').astype(int)
cols=['cohort_delay_s','completion_rate','p95_cohort_delay_s','worst_direction_cohort_delay_s','mean_queue_veh','signal_switches','collision_events','teleport_events','requested_vehicles','completed_vehicles']
group=m.groupby(['map','scenario','seed','budget'])[cols].mean().reset_index()
b=d[~d.controller.str.startswith('seed')].copy().rename(columns={'controller':'budget'})
combined=pd.concat([group,b],ignore_index=True)
summ=combined.groupby(['scenario','budget'])[cols].mean()
summ.to_csv(p/'scenario_summary.csv')
print("SCENARIOS")
print(summ.round(3).to_string())
print("BY MAP", combined.groupby(['map','budget'])[cols[:4]].mean().round(3).to_string())
print("MACRO",combined.groupby('budget')[cols[:4]].mean().round(4).to_string())
print("seed_training",m.groupby(['train_seed','budget']).cohort_delay_s.mean().round(4).to_string())
paired=[]
for scen,g in combined.groupby('scenario'):
 for bname in ('300','max_pressure_proxy','rule_based','fixed_time'):
  wide=g.pivot_table(index=['map','seed'],columns='budget',values='cohort_delay_s')
  diff=(wide['600']-wide[bname]).reset_index().groupby('seed')[0].mean()
  half=t.ppf(.975,9)*diff.std(ddof=1)/np.sqrt(10)
  paired.append(dict(scenario=scen,compare=f'600-{bname}',n_seed=len(diff),mean=diff.mean(),lo=diff.mean()-half,hi=diff.mean()+half,better=int((diff<0).sum())))
pframe=pd.DataFrame(paired)
pframe.to_csv(p/'paired.csv',index=False)
print("PAIRED",pframe.round(3).to_string(index=False))
print("integrity nonfinite",int(d.select_dtypes('number').isna().sum().sum()),"collisions",int(d.collision_events.sum()),"teleports",int(d.teleport_events.sum()))
