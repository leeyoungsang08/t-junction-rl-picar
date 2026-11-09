"""Where do long waits at J come from? Splits every stopped second of each request into causes:
other_phase (policy served another approach), min_green_lock, clearance (yellow/all-red/box clearing),
exit_full (own approach green but next segment full), green_queue_or_startup, fault.

Uses TRAINING-range seeds by default so the test split stays untouched.
python -m picar_sim.diagnose --policy runs/picar_v3_direct/seed43/best_policy.npz --out runs/diag
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from picar_sim.dqn import Policy
from picar_sim.env import PicarEnv
from picar_sim.map_builder import load_config
from picar_sim.train import controller_act

CAUSES = ["other_phase", "min_green_lock", "clearance", "exit_full", "green_queue_or_startup", "fault"]


def collect(policy_paths, seeds, scenarios, mode):
    rows = []
    cfg = load_config()
    env = PicarEnv(cfg, control_mode=mode)
    env.diagnose = True
    ctrls = {"queue_rule": controller_act(env, "queue_rule")}
    for p in policy_paths:
        pol = Policy(p)
        ctrls[f"dqn_{Path(p).parent.parent.name}_{Path(p).parent.name}"] = (lambda pol: lambda o, m: pol.act(o, m))(pol)
    for name, act in ctrls.items():
        for cars, bias in scenarios:
            for seed in seeds:
                obs, info = env.reset(seed=seed, cars=cars, bias=bias)
                done = False
                while not done:
                    obs, r, te, tr, info = env.step(act(obs, info["mask"]))
                    done = te or tr
                for rec in env.diag_records:
                    rows.append({"controller": name, "scenario": f"{bias}_{cars}", "seed": seed, **rec})
    df = pd.DataFrame(rows).fillna(0.0)
    for c in CAUSES:
        if c not in df:
            df[c] = 0.0
    return df


def summarize(df, top_q=0.95):
    out = {}
    for name, g in df.groupby("controller"):
        thr = g.delay.quantile(top_q)
        top = g[g.delay >= thr]
        share = lambda x: (x[CAUSES].sum() / max(x[CAUSES].sum().sum(), 1e-9)).round(3).to_dict()
        out[name] = {"requests": len(g), "mean_delay_s": round(g.delay.mean(), 2), "p95_threshold_s": round(thr, 1),
                     "worst_approach_counts": top.approach.value_counts().to_dict(),
                     "share_all_requests": share(g), "share_worst_5pct": share(top)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", nargs="*", default=[])
    ap.add_argument("--out", default="runs/diag")
    ap.add_argument("--mode", default="direct")
    ap.add_argument("--seeds", type=int, default=15)
    a = ap.parse_args()
    seeds = list(range(1700, 1700 + a.seeds))            # inside the training seed range, not test
    df = collect(a.policy, seeds, [(6, "uniform"), (6, "connector_heavy"), (5, "west_heavy")], a.mode)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "wait_breakdown.csv", index=False)
    s = summarize(df)
    (Path(a.out) / "wait_breakdown_summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(json.dumps(s, indent=1, ensure_ascii=False))
