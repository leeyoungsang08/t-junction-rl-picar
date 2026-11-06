"""Integrity and paired summaries for the two pre-PiCar studies."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent/'robust_retrain/T_Junction_RL_reward_ablation'))
from src.utils import sha256

OUT=Path(__file__).resolve().parents[1]/'outputs'


def analyze(folder, expected, calibration=False):
    root=OUT/folder; df=pd.read_csv(root/'results.csv')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    for name,h in manifest['inputs'].items():
        assert sha256(root/name)==h, name
    assert len(df)==expected,(folder,len(df))
    cols=['map','scenario','seed','sensor','controller']
    assert not df.duplicated(cols).any()
    assert (df.groupby(['map','scenario','seed']).route_sha256.nunique()==1).all()
    for r in df.itertuples(index=False):
        raw=root/'runs'/r.map/r.scenario/str(r.seed)/r.sensor/r.controller/'raw.zip'
        assert sha256(raw)==r.raw_sha256,str(raw)
    report=[f'{folder}: {len(df)} runs; {len(manifest["inputs"])} frozen inputs and all raw ZIP hashes verified']
    if calibration: df=df[df['map']!='calibration']
    metrics=['cohort_delay_s','completion_rate','guard_interventions','fallback_decisions','collision_events','teleport_events','max_consecutive_wait_s']
    report.append('\nMEANS\n'+df.groupby(['sensor','controller'])[metrics].mean().round(4).to_string())
    report.append('\nMAXIMUM\n'+df.groupby(['sensor','controller'])[['max_consecutive_wait_s','collision_events','teleport_events']].max().round(3).to_string())
    for sensor,part in df.groupby('sensor'):
        p=part.pivot(index=['map','scenario','seed'],columns='controller',values='cohort_delay_s')
        if 'old' in p:
            for c in ('new42','new43','new44','queue','fixed_15','fixed_s30'):
                if c in p:
                    d=p[c]-p['old']; report.append(f'PAIRED {sensor} {c} minus old: {d.mean():.4f}, wins {int((d<0).sum())} losses {int((d>0).sum())} ties {int((d==0).sum())}, max regression {d.max():.4f}')
        if 'model' in p and 'queue' in p:
            d=p['model']-p['queue'];report.append(f'PAIRED {sensor} model minus queue: {d.mean():.4f}, wins {int((d<0).sum())} losses {int((d>0).sum())} ties {int((d==0).sum())}, max regression {d.max():.4f}')
    report.append('\nSCENARIOS\n'+df.pivot_table(index=['sensor','scenario'],columns='controller',values='cohort_delay_s').round(3).to_string())
    text='\n'.join(report)+'\n'; (root/'verified_summary.txt').write_text(text,encoding='utf8');print(text)


if __name__=='__main__':
    analyze('SUMO_PiCar_전_센서시험',264,True)
    if (OUT/'SUMO_PiCar_전_재학습_비교'/'status.json').exists(): analyze('SUMO_PiCar_전_재학습_비교',432)
    if (OUT/'SUMO_PiCar_전_균형선택_새시험'/'status.json').exists(): analyze('SUMO_PiCar_전_균형선택_새시험',504)
