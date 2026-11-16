"""Stress test beyond the real fleet: more cars and skewed demand than the 6 PiCars can produce.

Same paired protocol as evaluate.py (same seed -> same placement and turn choices for every controller),
plus queue length per approach and intersection occupancy at J (ground truth from SUMO).

python -m picar_sim.stress_test --runs runs/picar_v15_obsage_tail/s8w1 --cars 6 8 10 12 \
    --biases uniform connector_heavy west_heavy --seeds 5 --out runs/stress_v15
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

KEY = ["mean_delay_s", "p95_delay_s", "max_wait_s", "throughput_per_min", "completion_rate",
       "mean_queue_total", "max_queue", "mean_queue_W", "mean_queue_E", "mean_queue_N",
       "max_queue_W", "max_queue_E", "max_queue_N", "mean_occupancy", "max_occupancy",
       "switches", "collisions", "conflict_steps", "stopline_violations", "deadlock"]
CONTROLLERS = ("oldest_first", "queue_rule", "fixed_time")


def _job(args):
    policies, cars, bias, seed, horizon = args
    from picar_sim.dqn import Policy
    from picar_sim.env import PicarEnv
    from picar_sim.map_builder import load_config
    from picar_sim.observation import Encoder
    from picar_sim.train import controller_act, run_episode
    cfg = load_config()
    pols = {name: Policy(p) for name, p in policies.items()}
    modes = {p.meta["control_mode"] for p in pols.values()} or {cfg["signal"]["control_mode"]}
    env = PicarEnv(cfg, randomize=False, control_mode=modes.pop())
    acts = {name: controller_act(env, name) for name in CONTROLLERS}
    for name, pol in pols.items():
        enc = Encoder(cfg["signal"]["max_green_s"], cfg["signal"]["clear_timeout_s"], pol.meta.get("schema", "picar_v1"))
        acts[name] = (lambda pol, enc: lambda o, m: pol.act(env._obs(enc), m))(pol, enc)
    rows = []
    for name, act in acts.items():
        try:
            m = run_episode(env, seed, cars, bias, act, horizon=horizon)
        except Exception as e:            # e.g. more cars than the course can place
            m = {"error": repr(e)[:200]}
        rows.append({"cars": cars, "bias": bias, "seed": seed, "controller": name, **m})
    env.close()
    return rows


def main(runs, cars, biases, seeds, out, workers, horizon):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    policies = {f"dqn_{p.parent.name}": str(p) for r in runs for p in sorted(Path(r).glob("seed*/best_policy.npz"))}
    jobs = [(policies, n, b, 3500 + 100 * i + s, horizon)
            for i, (n, b) in enumerate((n, b) for n in cars for b in biases) for s in range(seeds)]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_job, jobs), 1):
            rows += r
            print(f"[{k}/{len(jobs)}] cars={r[0]['cars']} bias={r[0]['bias']} seed={r[0]['seed']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "seed_level.csv", index=False)
    ok = df[df.get("error").isna()] if "error" in df else df
    ok = ok.assign(family=ok.controller.where(~ok.controller.str.startswith("dqn_"), "dqn"))
    summ = ok.groupby(["cars", "bias", "family"])[KEY].mean().round(3)
    summ.to_csv(out / "summary.csv")
    by_cars = ok.groupby(["cars", "family"])[KEY].mean().round(3)
    by_cars.to_csv(out / "summary_by_cars.csv")
    (out / "manifest.json").write_text(json.dumps({"policies": policies, "cars": cars, "biases": biases,
                                                   "seeds": seeds, "horizon_s": horizon}, indent=1))
    plot(ok, out / "stress.png")
    return by_cars


def plot(df, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("mean_delay_s", "mean delay [s]"), ("max_wait_s", "worst wait [s]"),
               ("throughput_per_min", "throughput [cars/min]"), ("mean_queue_total", "mean queue at J [cars]"),
               ("max_queue", "max queue on one approach [cars]"), ("mean_occupancy", "mean cars inside J")]
    g = df.groupby(["cars", "family"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {"dqn": "#d62728", "oldest_first": "#2ca02c", "queue_rule": "#ff7f0e", "fixed_time": "#1f77b4"}
    for ax, (col, label) in zip(axes.flat, metrics):
        mean, sd = g[col].mean().unstack(), g[col].std().unstack()
        for fam in [f for f in colors if f in mean]:
            ax.errorbar(mean.index, mean[fam], yerr=sd[fam], marker="o", capsize=3, label=fam, color=colors[fam])
        ax.set_title(label)
        ax.set_xlabel("cars on course")
        ax.axvline(6, color="gray", ls=":", lw=1)
        ax.grid(alpha=.3)
    axes[0][0].legend()
    fig.suptitle("Stress test at junction J (dotted line = real fleet size, 6 PiCars)")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--cars", nargs="+", type=int, default=[6, 8, 10, 12])
    p.add_argument("--biases", nargs="+", default=["uniform", "connector_heavy", "west_heavy"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--horizon", type=float, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(main(a.runs, a.cars, a.biases, a.seeds, a.out, a.workers, a.horizon).to_string())
