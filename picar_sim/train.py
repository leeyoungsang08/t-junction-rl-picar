"""Train one DQN seed on the PiCar course. Resumable; runs on numpy (CPU) or torch (CPU/GPU).

python -m picar_sim.train --seed 42 --episodes 800 --out runs/picar_v1
python -m picar_sim.train --seed 42 --episodes 800 --out runs/picar_v1 --resume      # continue
"""
import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

from picar_sim.baselines import CONTROLLERS
from picar_sim.dqn import Replay, make_agent, save_policy
from picar_sim.env import PicarEnv
from picar_sim.map_builder import load_config

AGENT_DEFAULTS = {"hidden": [128, 128], "lr": 5e-4, "gamma": 0.98, "batch": 64, "replay": 100_000,
                  "learning_start": 2_000, "target_sync": 1_000, "grad_clip": 10.0, "double": True,
                  "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frac": 0.4, "device": "auto",
                  "validation_every": 25, "validation_seeds_per": 2}


def scenario_list(cfg, split):
    out = []
    for s in cfg["scenarios"][split]:
        out += [(n, s["bias"]) for n in s["cars"]]
    return out


def validation_score(m):
    """Lower is better: delay, plus heavy penalties for unserved demand and any safety event."""
    unsafe = m["collisions"] + m["conflict_steps"] + m["stopline_violations"]
    return (m["mean_delay_s"] + 0.5 * m["max_wait_s"] / 10 + 100 * (1 - m["completion_rate"])
            + 1000 * (unsafe > 0) + 500 * m["deadlock"])


def run_episode(env, seed, cars, bias, act, horizon=None):
    obs, info = env.reset(seed=seed, cars=cars, bias=bias, horizon=horizon)
    done, ret = False, 0.0
    while not done:
        a = act(obs, info["mask"])
        obs, r, term, trunc, info = env.step(a)
        ret += r
        done = term or trunc
    m = info["metrics"]
    m["return"] = ret
    return m


def controller_act(env, name):
    fn = CONTROLLERS[name]
    return lambda obs, mask: fn(env.ledger.snapshot(env.t), env.sup.status(), mask, env.sup.mode)


def validate(env, cfg, act, seeds_per=2):
    vals = []
    base = cfg["seeds"]["validation"][0]
    for j, (cars, bias) in enumerate(scenario_list(cfg, "validation")):
        for s in range(seeds_per):
            vals.append(run_episode(env, base + 10 * j + s, cars, bias, act))
    return float(np.mean([validation_score(m) for m in vals])), vals


def train(seed, episodes, out, resume=False, backend="auto", control_mode=None, randomize=False,
          time_budget_min=None, agent_overrides=None, reward_overrides=None):
    cfg = load_config()
    cfg["reward"].update(reward_overrides or {})
    if control_mode:
        cfg["signal"]["control_mode"] = control_mode
    acfg = {**AGENT_DEFAULTS, **(agent_overrides or {})}
    out = Path(out) / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    env = PicarEnv(cfg, randomize=randomize)
    rng = np.random.default_rng(seed)
    agent = make_agent(env.obs_size, env.n_actions, acfg, rng, backend)
    replay = Replay(acfg["replay"], env.obs_size, env.n_actions, rng)
    total_steps = int(episodes * cfg["simulation"]["horizon_s"] / cfg["simulation"]["decision_s"])
    decay = acfg["eps_decay_frac"] * total_steps
    ep0, steps, best, history = 0, 0, float("inf"), []
    ckpt = out / "checkpoint.pkl"
    if resume and ckpt.exists():
        with ckpt.open("rb") as f:
            s = pickle.load(f)
        agent.load_state(s["agent"])
        replay.__dict__.update(s["replay"])
        replay.rng = rng
        rng.bit_generator.state = s["rng"]
        ep0, steps, best, history = s["episode"], s["steps"], s["best"], s["history"]
        print(f"resumed seed {seed} at episode {ep0}", flush=True)
    meta = {"schema": "picar_v1", "features": __import__("picar_sim.observation", fromlist=["FEATURES"]).FEATURES,
            "control_mode": cfg["signal"]["control_mode"], "signal": cfg["signal"], "seed": seed,
            "n_actions": env.n_actions, "randomize": randomize, "agent": acfg, "reward": cfg["reward"],
            "backend": str(agent.device)}
    (out / "config_snapshot.json").write_text(json.dumps({"course": cfg, "agent": acfg, "meta": meta}, indent=1, default=str))
    train_sc = scenario_list(cfg, "train")
    lo, hi = cfg["seeds"]["train"]
    log_path = out / "training_log.csv"
    t_start = time.time()
    loss = float("nan")
    for ep in range(ep0, episodes):
        cars, bias = train_sc[rng.integers(len(train_sc))]
        ep_seed = int(rng.integers(lo, hi + 1))
        obs, info = env.reset(seed=ep_seed, cars=cars, bias=bias)
        done, ret, losses, t0 = False, 0.0, [], time.time()
        while not done:
            eps = max(acfg["eps_end"], acfg["eps_start"] - (acfg["eps_start"] - acfg["eps_end"]) * steps / decay)
            mask = info["mask"]
            if rng.random() < eps:
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                a = int(np.where(mask, agent.q(obs), -np.inf).argmax())
            obs2, r, term, trunc, info = env.step(a)
            # time-limit truncation bootstraps; only a deadlock is a true terminal
            replay.add(obs, a, r, obs2, float(term), info["mask"])
            obs, ret, done, steps = obs2, ret + r, term or trunc, steps + 1
            if replay.n >= acfg["learning_start"]:
                losses.append(agent.update(replay.sample(acfg["batch"])))
                if agent.t % acfg["target_sync"] == 0:
                    agent.sync()
        m = info["metrics"]
        loss = float(np.mean(losses)) if losses else float("nan")
        row = {"episode": ep + 1, "seed": ep_seed, "cars": cars, "bias": bias, "return": round(ret, 3),
               "mean_delay_s": round(m["mean_delay_s"], 3), "completion": round(m["completion_rate"], 4),
               "max_wait_s": round(m["max_wait_s"], 2), "switches": m["switches"], "deadlock": m["deadlock"],
               "unsafe": m["collisions"] + m["conflict_steps"] + m["stopline_violations"],
               "epsilon": round(eps, 4), "loss": round(loss, 5), "updates": agent.t,
               "wall_s": round(time.time() - t0, 2), "val_score": ""}
        if (ep + 1) % acfg["validation_every"] == 0 or ep + 1 == episodes:
            greedy = lambda o, mk: int(np.where(mk, agent.q(o), -np.inf).argmax())
            score, _ = validate(env, cfg, greedy, acfg["validation_seeds_per"])
            row["val_score"] = round(score, 3)
            save_policy(out / "last_policy.npz", agent.weights(), {**meta, "episode": ep + 1, "val_score": score})
            save_policy(out / "policies" / f"ep{ep + 1:05d}.npz", agent.weights(),      # for post-hoc re-selection
                        {**meta, "episode": ep + 1, "val_score": score})
            if score < best:
                best = score
                save_policy(out / "best_policy.npz", agent.weights(), {**meta, "episode": ep + 1, "val_score": score})
            with ckpt.with_suffix(".tmp").open("wb") as f:
                pickle.dump({"agent": agent.state(), "replay": {k: v for k, v in replay.__dict__.items() if k != "rng"},
                             "rng": rng.bit_generator.state, "episode": ep + 1, "steps": steps, "best": best,
                             "history": history + [row]}, f)
            ckpt.with_suffix(".tmp").replace(ckpt)
            print(f"seed {seed} ep {ep+1}/{episodes} eps {eps:.3f} val {score:.2f} best {best:.2f} "
                  f"delay {m['mean_delay_s']:.1f} loss {loss:.4f}", flush=True)
        history.append(row)
        new = not log_path.exists() or ep == 0
        with log_path.open("w" if new and ep == 0 else "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        if time_budget_min and time.time() - t_start > 60 * time_budget_min:
            print(f"time budget reached at episode {ep+1}; resume with --resume", flush=True)
            break
    (out / "training_status.json").write_text(json.dumps({"episodes_done": len(history), "planned": episodes,
                                                          "best_val_score": best, "updates": agent.t,
                                                          "backend": str(agent.device)}, indent=1))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--out", default="runs/picar_v1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch"])
    p.add_argument("--control-mode", choices=["cycle", "direct"])
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--time-budget-min", type=float)
    p.add_argument("--starvation-s", type=float, help="reward: wait above this is penalised (default config 20)")
    p.add_argument("--starvation-weight", type=float, help="reward: long-wait penalty weight (default config 0.5)")
    p.add_argument("--switch-weight", type=float, help="reward: cost per phase switch (default config 0.1)")
    p.add_argument("--val-seeds-per", type=int, default=2, help="validation episodes per scenario")
    a = p.parse_args()
    rew = {k: v for k, v in (("starvation_s", a.starvation_s), ("starvation_weight", a.starvation_weight),
                             ("switch_weight", a.switch_weight)) if v is not None}
    train(a.seed, a.episodes, a.out, a.resume, a.backend, a.control_mode, a.randomize, a.time_budget_min,
          {"validation_seeds_per": a.val_seeds_per}, rew)
