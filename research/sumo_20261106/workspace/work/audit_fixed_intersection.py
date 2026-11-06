"""Independently check frozen inputs, validation gate and held-out pairings."""
import json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pre_picar_study import HERE,PROJECT,atomic_json
sys.path.insert(0,str(PROJECT))
from src.utils import sha256

TRAIN=HERE/'outputs/SUMO_balanceA_안정화_20261105'
GATE=HERE/'outputs/SUMO_balanceA_기준정책_안전판_검증'
TEST=HERE/'outputs/SUMO_balanceA_독립시험_20261105'


def audit_train():
    m=json.loads((TRAIN/'manifest.json').read_text(encoding='utf8'))
    for rel,h in m['inputs'].items():assert sha256(TRAIN/rel)==h,rel
    output=[]
    for kind in ('plain','anchor'):
      for seed in (42,43,44):
        run=TRAIN/f'{kind}_seed{seed}';status=json.loads((run/'status.json').read_text(encoding='utf8'))
        assert status['complete'] and status['completed_episodes']==180 and status['updates']>0
        training=pd.read_csv(run/'training.csv')
        assert len(training)==180 and training.episode.tolist()==list(range(1,181))
        assert np.isfinite(training.reward).all()
        after=training[training.episode_updates>0]
        assert len(after)>0 and np.isfinite(after.td_loss).all() and np.isfinite(after.anchor_loss).all()
        assert set(training.scenario)==set(m['training_scenarios'])
        data=pd.read_csv(run/'validation.csv');base=data[data.episode==0]
        assert len(base)==36
        baseline=base.groupby(['scenario','sensor']).cohort_delay_s.mean();passed=[]
        for ep in (30,60,90,120,150,180):
            cur=data[data.episode==ep];assert len(cur)==36
            group=cur.groupby(['scenario','sensor']).cohort_delay_s.mean();worst=float((group-baseline).max())
            eligible=(cur.cohort_delay_s.mean()<=base.cohort_delay_s.mean()-.25 and worst<=.5 and
                      cur.completion_rate.mean()>=base.completion_rate.mean()-.005 and
                      cur.max_consecutive_wait_s.max()<=base.max_consecutive_wait_s.max()+15)
            saved=json.loads((run/f'validation_{ep}.json').read_text(encoding='utf8'))
            assert saved['eligible']==bool(eligible)
            assert math.isclose(saved['worst_group_regression_s'],worst,abs_tol=1e-9)
            if eligible:passed.append((float(cur.cohort_delay_s.mean()),ep))
        selected=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        expected=min(passed)[1] if passed else 0
        assert selected['episode']==expected,(run,selected['episode'],expected)
        output.append({'method':kind,'seed':seed,'selected_episode':expected,'passing_checkpoints':len(passed),'updates':status['updates']})
    atomic_json(TRAIN/'audit_summary.json',{'frozen_inputs_verified':len(m['inputs']),'runs':output})
    print('TRAIN_AUDIT',output,flush=True)


def audit_test():
    m=json.loads((TEST/'manifest.json').read_text(encoding='utf8'))
    for rel,h in m['inputs'].items():assert sha256(TEST/rel)==h,rel
    df=pd.read_csv(TEST/'results.csv')
    expected=len(m['scenarios'])*len(m['seeds'])*len(m['sensors'])*len(m['controllers'])
    assert len(df)==expected and not df.duplicated(['scenario','seed','sensor','controller']).any()
    assert (df.groupby(['scenario','seed']).route_sha256.nunique()==1).all()
    for r in df.itertuples(index=False):assert sha256(TEST/'runs'/r.scenario/str(r.seed)/r.sensor/r.controller/'raw.zip')==r.raw_sha256
    mean=df.groupby(['sensor','controller'])[['cohort_delay_s','completion_rate','max_consecutive_wait_s']].mean()
    pivot=df.pivot(index=['scenario','seed','sensor'],columns='controller',values='cohort_delay_s')
    lines=[f'VERIFIED {len(df)} runs, {len(m["inputs"])} frozen inputs, each raw ZIP hash',
           '\nMEANS\n'+mean.round(4).to_string()]
    for sensor,group in df.groupby('sensor'):
        p=group.pivot(index=['scenario','seed'],columns='controller',values='cohort_delay_s')
        for c in [x for x in p.columns if x!='old']:
            d=p[c]-p.old
            lines.append(f'PAIRED {sensor} {c} minus old mean {d.mean():.4f}, wins {int((d< -1e-9).sum())}, losses {int((d>1e-9).sum())}, ties {int((d.abs()<=1e-9).sum())}, worst {d.max():.4f}')
    lines.append('\nSCENARIOS\n'+df.pivot_table(index=['sensor','scenario'],columns='controller',values='cohort_delay_s').round(3).to_string())
    text='\n'.join(lines)+'\n';(TEST/'verified_summary.txt').write_text(text,encoding='utf8');print(text,flush=True)


if __name__=='__main__':
    audit_train()
    if (TEST/'status.json').exists():audit_test()
