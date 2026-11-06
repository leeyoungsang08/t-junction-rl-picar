"""Verify all watchdog experiment inputs and runs, summarize matched outcomes."""
import json,sys
from pathlib import Path
import pandas as pd
from pre_picar_study import HERE,PROJECT
sys.path.insert(0,str(PROJECT))
from src.utils import sha256

OUT=HERE/'outputs/SUMO_balanceA_장시간대기_추가시험'
m=json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
for rel,h in m['inputs'].items():assert sha256(OUT/rel)==h,rel
lines=[f'INPUTS VERIFIED {len(m["inputs"])}']
for phase,expected in (('validation',144),('test',360)):
    df=pd.read_csv(OUT/f'{phase}_results.csv')
    assert len(df)==expected and not df.duplicated(['scenario','seed','sensor','controller']).any()
    assert (df.groupby(['scenario','seed']).route_sha256.nunique()==1).all()
    for r in df.itertuples(index=False):
        raw=OUT/'runs'/phase/r.scenario/str(r.seed)/r.sensor/r.controller/'raw.zip'
        assert sha256(raw)==r.raw_sha256
    lines.append(f'\n{phase.upper()} VERIFIED {len(df)} runs')
    lines.append('\nMEANS\n'+df.groupby('controller')[['cohort_delay_s','completion_rate','max_consecutive_wait_s','watchdog_interventions']].mean().round(4).to_string())
    lines.append('\nMAX\n'+df.groupby('controller')[['max_consecutive_wait_s','collision_events','teleport_events']].max().to_string())
    p=df.pivot(index=['scenario','seed','sensor'],columns='controller',values='cohort_delay_s')
    for c in p.columns:
        if c=='old':continue
        delta=p[c]-p.old
        lines.append(f'PAIRED {c} minus old mean {delta.mean():.4f}, wins {int((delta< -1e-9).sum())}, losses {int((delta>1e-9).sum())}, ties {int((delta.abs()<=1e-9).sum())}, worst {delta.max():.4f}')
    lines.append('\nSENSOR\n'+df.pivot_table(index='sensor',columns='controller',values='cohort_delay_s').round(4).to_string())
text='\n'.join(lines)+'\n';(OUT/'verified_summary.txt').write_text(text,encoding='utf8');print(text)
