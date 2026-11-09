"""Paired evaluation on held-out test scenarios: same seeds, same initial placement, same turn choices.

python -m picar_sim.evaluate --runs runs/picar_v1_cycle --out runs/eval_v1 --seeds-per 10
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from picar_sim.dqn import Policy
from picar_sim.env import PicarEnv
from picar_sim.map_builder import load_config
from picar_sim.train import controller_act, run_episode, scenario_list

KEY = ["mean_delay_s", "p95_delay_s", "completion_rate", "max_wait_s", "max_wait_W_s", "max_wait_E_s",
       "max_wait_N_s", "throughput_per_min", "switches", "collisions", "conflict_steps", "stopline_violations",
       "deadlock", "fault_events"]


def evaluate(runs, out, seeds_per=10, conditions=("nominal", "noisy"), split="test", policy_name="best_policy.npz"):
    cfg = load_config()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    policies = {f"dqn_{p.parent.name}": Policy(p) for r in runs for p in sorted(Path(r).glob(f"seed*/{policy_name}"))}
    modes = {pol.meta["control_mode"] for pol in policies.values()} or {cfg["signal"]["control_mode"]}
    if len(modes) != 1:
        raise ValueError("evaluate one control mode at a time")
    mode = modes.pop()
    rows = []
    base = cfg["seeds"][split][0]
    for cond in conditions:
        env = PicarEnv(load_config(), randomize=(cond == "noisy"), control_mode=mode)
        acts = {"fixed_time": controller_act(env, "fixed_time"), "queue_rule": controller_act(env, "queue_rule")}
        for name, pol in policies.items():
            acts[name] = (lambda pol: lambda o, m: pol.act(o, m))(pol)
        for j, (cars, bias) in enumerate(scenario_list(cfg, split)):
            for s in range(seeds_per):
                seed = base + 100 * j + s
                for name, act in acts.items():
                    m = run_episode(env, seed, cars, bias, act)
                    rows.append({"condition": cond, "scenario": f"{bias}_{cars}cars", "controller": name, **m})
    df = pd.DataFrame(rows)
    df.to_csv(out / "seed_level_results.csv", index=False)
    df["family"] = df.controller.where(~df.controller.str.startswith("dqn_"), "dqn_mean_of_seeds")
    summ = df.groupby(["condition", "scenario", "controller"])[KEY].mean().round(3)
    summ.to_csv(out / "summary_by_controller.csv")
    overall = df.groupby(["condition", "controller"])[KEY].mean().round(3)
    overall.to_csv(out / "summary_overall.csv")
    # paired DQN-seed vs queue_rule on identical seeds (CI over evaluation seeds; seed spread reported separately)
    paired = []
    for (cond, sc), g in df.groupby(["condition", "scenario"]):
        ref = g[g.controller == "queue_rule"].set_index("seed")["mean_delay_s"]
        for name in [c for c in g.controller.unique() if c.startswith("dqn_")]:
            d = (g[g.controller == name].set_index("seed")["mean_delay_s"] - ref).dropna()
            ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d)) if len(d) > 1 and d.std() > 0 else (d.mean(), d.mean())
            paired.append({"condition": cond, "scenario": sc, "controller": name, "n": len(d),
                           "delay_diff_vs_queue_rule_s": round(d.mean(), 3), "ci95_low": round(ci[0], 3), "ci95_high": round(ci[1], 3)})
    pd.DataFrame(paired).to_csv(out / "paired_vs_queue_rule.csv", index=False)
    seed_spread = (df[df.controller.str.startswith("dqn_")].groupby(["condition", "controller"])["mean_delay_s"].mean()
                   .groupby("condition").agg(["mean", "std", "min", "max"]).round(3))
    seed_spread.to_csv(out / "dqn_training_seed_spread.csv")
    (out / "eval_manifest.json").write_text(json.dumps({"runs": [str(r) for r in runs], "policies": list(policies),
                                                         "control_mode": mode, "split": split, "seeds_per": seeds_per,
                                                         "conditions": list(conditions), "episodes": len(df)}, indent=1))
    plot(df, out / "comparison.png")
    return overall, seed_spread


def plot(df, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("mean_delay_s", "mean delay per request [s]"), ("completion_rate", "completion rate"),
               ("max_wait_s", "worst wait [s]")]
    conds = list(df.condition.unique())
    ctrls = list(df.controller.unique())
    fig, axes = plt.subplots(len(conds), 3, figsize=(14, 3.6 * len(conds)), squeeze=False)
    for i, cond in enumerate(conds):
        g = df[df.condition == cond]
        for k, (col, label) in enumerate(metrics):
            ax = axes[i][k]
            means = g.groupby(["scenario", "controller"])[col].mean().unstack()[ctrls]
            means.plot.bar(ax=ax, rot=0, width=.8, legend=(k == 0))
            ax.set_title(f"{cond}: {label}", fontsize=10)
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=8)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds-per", type=int, default=10)
    p.add_argument("--conditions", nargs="+", default=["nominal", "noisy"])
    p.add_argument("--split", default="test", choices=["validation", "test"])
    a = p.parse_args()
    overall, spread = evaluate(a.runs, a.out, a.seeds_per, a.conditions, a.split)
    print(overall.to_string())
    print(spread.to_string())
