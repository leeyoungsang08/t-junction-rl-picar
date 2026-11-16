"""Stress test beyond the real fleet: more cars and skewed demand than the 6 PiCars can produce.

Same paired protocol as evaluate.py (same seed -> same placement and turn choices for every controller),
plus queue length per approach and intersection occupancy at J (ground truth from SUMO).
Each --runs directory is one DQN family (its seed*/best_policy.npz are averaged per evaluation seed).

python -m picar_sim.stress_test --runs runs/picar_v15_obsage_tail/s8w1 --map real --cars 6 8 10 12 \
    --biases uniform connector_heavy west_heavy --seeds 5 --out runs/stress_v15
python -m picar_sim.stress_test --runs runs/picar_v15_obsage_tail/s8w1 runs/picar_v17_mix_seg --map large \
    --cars 10 20 30 40 50 --biases uniform connector_heavy west_heavy alternating --seeds 8 --out runs/stress_large
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["mean_delay_s", "p95_delay_s", "max_wait_s", "throughput_per_min", "completion_rate",
       "mean_queue_total", "max_queue", "mean_queue_W", "mean_queue_E", "mean_queue_N",
       "max_queue_W", "max_queue_E", "max_queue_N", "mean_occupancy", "max_occupancy",
       "switches", "guard_switches", "collisions", "conflict_steps", "stopline_violations", "deadlock"]
RULES = ("oldest_first", "queue_rule", "fixed_time")
SAFETY = ("collisions", "conflict_steps", "stopline_violations")


def family_name(run):
    r = Path(run)
    return r.parent.name if r.name.startswith("s") and r.name[1:2].isdigit() else r.name   # .../v15/s8w1 -> v15


def _job(args):
    policies, map_name, cars, bias, seed, horizon = args
    from picar_sim.dqn import Policy
    from picar_sim.env import PicarEnv
    from picar_sim.map_builder import load_config
    from picar_sim.observation import Encoder
    from picar_sim.train import controller_act, run_episode
    cfg = load_config()
    pols = {name: Policy(p) for name, (fam, p) in policies.items()}
    env = PicarEnv(cfg, randomize=False, control_mode="direct", map_name=map_name)
    acts = {name: controller_act(env, name) for name in RULES}
    for name, pol in pols.items():
        if pol.meta["control_mode"] != "direct":
            raise ValueError(f"{name}: control mode {pol.meta['control_mode']}")
        enc = Encoder(cfg["signal"]["max_green_s"], cfg["signal"]["clear_timeout_s"], pol.meta.get("schema", "picar_v1"))
        acts[name] = (lambda pol, enc: lambda o, m: pol.act(env._obs(enc), m))(pol, enc)
    rows = []
    for name, act in acts.items():
        m = run_episode(env, seed, cars, bias, act, horizon=horizon)
        fam = policies[name][0] if name in policies else name
        rows.append({"map": map_name, "cars": cars, "bias": bias, "seed": seed, "controller": name, "family": fam,
                     **{k: m[k] for k in KEY}})
    env.close()
    return rows


def paired(df, ref_rule="best"):
    """Per (cars): each DQN family vs the best rule at that load (lowest mean delay), paired over seeds."""
    from scipy import stats
    out = []
    per_seed = df.groupby(["cars", "bias", "seed", "family"])[["mean_delay_s", "max_wait_s", "throughput_per_min"]].mean()
    for cars, g in per_seed.groupby(level=0):
        rules = g[g.index.get_level_values("family").isin(RULES)]
        best = rules.groupby(level="family").mean_delay_s.mean().idxmin()
        ref = g.xs(best, level="family")
        for fam in sorted(set(g.index.get_level_values("family")) - set(RULES)):
            d = (g.xs(fam, level="family") - ref).dropna()
            row = {"cars": cars, "family": fam, "vs": best, "n": len(d)}
            for col in ("mean_delay_s", "max_wait_s", "throughput_per_min"):
                x = d[col]
                lo, hi = stats.t.interval(0.95, len(x) - 1, x.mean(), stats.sem(x)) if x.std() > 0 else (x.mean(), x.mean())
                row.update({f"{col}_diff": round(x.mean(), 3), f"{col}_ci_lo": round(lo, 3), f"{col}_ci_hi": round(hi, 3)})
            row["dqn_better_share"] = round(float((d.mean_delay_s < 0).mean()), 3)
            out.append(row)
    return pd.DataFrame(out)


def main(runs, map_name, cars, biases, seeds, out, workers, horizon, seed_base):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    policies = {f"{family_name(r)}_{p.parent.name}": (family_name(r), str(p))
                for r in runs for p in sorted(Path(r).glob("seed*/best_policy.npz"))}
    combos = [(n, b) for n in cars for b in biases]
    jobs = [(policies, map_name, n, b, seed_base + 100 * i + s, horizon) for i, (n, b) in enumerate(combos)
            for s in range(seeds)]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_job, jobs), 1):
            rows += r
            if k % 10 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "seed_level.csv", index=False)
    df.groupby(["cars", "bias", "family"])[KEY].mean().round(3).to_csv(out / "summary.csv")
    by_cars = df.groupby(["cars", "family"])[KEY].mean().round(3)
    by_cars.to_csv(out / "summary_by_cars.csv")
    pr = paired(df)
    pr.to_csv(out / "paired_vs_best_rule.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"policies": policies, "map": map_name, "cars": cars,
                                                   "biases": biases, "seeds": seeds, "seed_base": seed_base,
                                                   "horizon_s": horizon, "episodes": len(df),
                                                   "safety_events": int(df[list(SAFETY)].sum().sum())}, indent=1))
    plot(df, out / "stress.png", map_name)
    return by_cars, pr


def plot(df, png, map_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("mean_delay_s", "mean delay [s]"), ("max_wait_s", "worst wait [s]"),
               ("throughput_per_min", "throughput [cars/min]"), ("mean_queue_total", "mean queue at J [cars]"),
               ("max_queue", "max queue on one approach [cars]"), ("switches", "phase switches / episode")]
    g = df.groupby(["cars", "family"])
    fams = [f for f in sorted(df.family.unique()) if f not in RULES] + [r for r in RULES if r in set(df.family)]
    palette = ["#d62728", "#9467bd", "#8c564b", "#e377c2"]
    colors = {f: palette[i % len(palette)] for i, f in enumerate(f for f in fams if f not in RULES)}
    colors.update({"oldest_first": "#2ca02c", "queue_rule": "#ff7f0e", "fixed_time": "#1f77b4"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, label) in zip(axes.flat, metrics):
        mean, sd = g[col].mean().unstack(), g[col].std().unstack()
        for fam in fams:
            ls = "-" if fam not in RULES else "--"
            ax.errorbar(mean.index, mean[fam], yerr=sd[fam], marker="o", capsize=3, label=fam, color=colors[fam], ls=ls)
        ax.set_title(label)
        ax.set_xlabel("cars on course")
        if map_name == "real":
            ax.axvline(6, color="gray", ls=":", lw=1)
        ax.grid(alpha=.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Junction J, {map_name} map" + (" (dotted line = real fleet, 6 PiCars)" if map_name == "real" else ""))
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--map", default="real", choices=["real", "large"])
    p.add_argument("--cars", nargs="+", type=int, default=[6, 8, 10, 12])
    p.add_argument("--biases", nargs="+", default=["uniform", "connector_heavy", "west_heavy"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-base", type=int, default=3500, help="test seeds (train 1000-1999, validation 2000-2099)")
    p.add_argument("--horizon", type=float, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    by_cars, pr = main(a.runs, a.map, a.cars, a.biases, a.seeds, a.out, a.workers, a.horizon, a.seed_base)
    print(by_cars[["mean_delay_s", "max_wait_s", "throughput_per_min", "mean_queue_total", "max_queue"]].to_string())
    print(pr.to_string())
