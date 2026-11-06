import json,sys
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'robust_retrain/T_Junction_RL_reward_ablation'))
from src.utils import sha256
ROOT=P.parent/'outputs'; D=ROOT/'SUMO_PiCar_전_고정신호_조정'; E=ROOT/'SUMO_PiCar_전_재학습_비교'
m=json.loads((D/'manifest.json').read_text(encoding='utf8'))
assert sha256(E/'manifest.json')==m['evaluation_manifest_sha256']
for rel,h in m['inputs'].items():assert sha256(D/rel)==h
df=pd.read_csv(D/'results.csv');assert len(df)==200
for r in df.itertuples(index=False):
    raw=D/'runs'/r.phase/r.map/r.scenario/str(r.seed)/r.sensor/r.plan/'raw.zip'
    assert sha256(raw)==r.raw_sha256
assert len(df[df.phase=='calibration'])==128 and len(df[df.phase=='evaluation'])==72
x=df[df.phase=='evaluation'].copy();old=pd.read_csv(E/'results.csv');old=old[old.controller=='old']
p=x.merge(old,on=['map','scenario','seed','sensor'],suffixes=('_fixed','_old'),validate='one_to_one')
assert len(p)==72 and (p.route_sha256_fixed==p.route_sha256_old).all() and (p.map_sha256_fixed==p.map_sha256_old).all()
p.to_csv(D/'paired_with_old.csv',index=False)
lines=['VERIFIED 200 runs and every frozen input and raw file hash','SELECTED '+(D/'selected_plan.json').read_text(encoding='utf8')]
for sensor,sub in p.groupby('sensor'):
    delta=sub.cohort_delay_s_fixed-sub.cohort_delay_s_old
    lines.append(f'{sensor}: fixed {sub.cohort_delay_s_fixed.mean():.4f}, old {sub.cohort_delay_s_old.mean():.4f}, fixed_minus_old {delta.mean():.4f}, fixed_wins {int((delta< -1e-9).sum())}, fixed_losses {int((delta>1e-9).sum())}, ties {int((delta.abs()<=1e-9).sum())}')
lines.append('SCENARIOS\n'+p.groupby(['sensor','scenario'])[['cohort_delay_s_fixed','cohort_delay_s_old','completion_rate_fixed','completion_rate_old']].mean().round(4).to_string())
text='\n'.join(lines)+'\n';(D/'verified_summary.txt').write_text(text,encoding='utf8');print(text)
