import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
import shutil
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import yaml
from maps.generate_map import generate
from src.dqn_agent import DQNAgent, atomic_torch_save
from src.environment import TJunctionEnv
from src.state_adapter import StateAdapter
from src.evaluate import run_episode
from src.inference import export_model
from src.traffic_generator import generate_traffic
from src.utils import ROOT, atomic_text, load_yaml, sha256, version_info, write_json


def resolved_config(mode, double=False, reward_mode=None):
    base = load_yaml("base.yaml")
    cfg = load_yaml("training.yaml")
    cfg.update(cfg["modes"][mode])
    del cfg["modes"]
    cfg["mode"] = mode
    cfg["state_dim"] = len(StateAdapter(base["normalization"], base["signal"], base.get("observation_schema", 1)).features)
    cfg["double_dqn"] = double
    base["simulation"]["horizon_seconds"] = cfg["horizon_seconds"]
    if reward_mode:
        base["reward"]["mode"] = reward_mode
    return base, cfg


def train(mode="quick", output=None, resume=None, double=False, reward_mode=None, stop_after=None, mirror=None, training_seed=None):
    output = Path(output or ROOT / "outputs/run").resolve()
    if resume:
        saved = yaml.safe_load((output / "experiment_config.yaml").read_text())
        base, cfg = saved["environment"], saved["training"]
    else:
        base, cfg = resolved_config(mode, double, reward_mode)
        if training_seed is not None:
            cfg["seed"] = training_seed
        if (output / "checkpoints/training_checkpoint.pt").exists():
            raise FileExistsError("Existing run. Use --resume or a fresh --output directory")
    if mirror and (Path(mirror).resolve() == output or output in Path(mirror).resolve().parents):
        raise ValueError("Mirror must be outside the run directory")
    generate(base)
    for d in ("checkpoints", "logs", "validation", "training_episodes"):
        (output / d).mkdir(parents=True, exist_ok=True)
    atomic_text(output / "experiment_config.yaml", yaml.safe_dump({"environment": base, "training": cfg}, sort_keys=False))
    versions = version_info()
    agent = DQNAgent(cfg)
    next_episode, best_score = 0, float("inf")
    rows, updates, validation_rows = [], [], []
    if resume:
        checkpoint = agent.restore(resume)
        next_episode = checkpoint["next_episode"]
        if checkpoint["extra"].get("base_config") != base:
            raise ValueError("Environment configuration differs from checkpoint")
        if checkpoint["extra"]["net_sha256"] != sha256(ROOT / "maps/t_junction.net.xml"):
            raise ValueError("SUMO map differs from checkpoint")
        best_score = checkpoint["extra"]["best_score"]
        rows = checkpoint["extra"]["rows"]
        updates = checkpoint["extra"]["update_rows"]
        validation_rows = checkpoint["extra"]["validation_rows"]
        print(f"Resuming at episode {next_episode}; only completed checkpoint episodes are retained.", flush=True)
    start = time.perf_counter()
    def sync_mirror():
        if not mirror:
            return
        destination = Path(mirror).resolve()
        for source in output.rglob("*"):
            if not source.is_file() or source.suffix == ".tmp":
                continue
            target = destination / source.relative_to(output)
            if target.is_file() and (target.stat().st_size, target.stat().st_mtime_ns) == (source.stat().st_size, source.stat().st_mtime_ns):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
    def persist(episode):
        extra = {"best_score": best_score, "rows": rows, "update_rows": updates, "validation_rows": validation_rows,
                 "base_config": base, "net_sha256": sha256(ROOT / "maps/t_junction.net.xml")}
        atomic_torch_save(agent.checkpoint(episode + 1, extra), output / "checkpoints/training_checkpoint.pt")
        atomic_text(output / "logs/training.csv", pd.DataFrame(rows).to_csv(index=False))
        atomic_text(output / "logs/updates.csv", pd.DataFrame(updates).to_csv(index=False))
        atomic_text(output / "logs/validation.csv", pd.DataFrame(validation_rows).to_csv(index=False))
        sync_mirror()

    for episode in range(next_episode, cfg["episodes"]):
        episode_start = time.perf_counter()
        seed = load_yaml("scenarios.yaml")["splits"]["train"]["start"] + episode
        scenario = cfg["train_scenarios"][episode % len(cfg["train_scenarios"])]
        route = generate_traffic(scenario, seed, cfg["horizon_seconds"], "train", directory=output / "routes")
        env = TJunctionEnv(base, route, seed, output / "training_episodes" / f"ep{episode:04d}")
        state, info = env.reset(seed=seed)
        total_reward, losses, qs = 0.0, [], []
        try:
            while True:
                action, q = agent.select(state, info["action_mask"], explore=True)
                next_state, reward, terminated, truncated, next_info = env.step(action)
                agent.replay.add(state, next_info["executed_action"], reward, next_state, terminated, truncated, next_info["action_mask"])
                agent.global_step += 1
                update = agent.update()
                if update:
                    updates.append(update)
                    losses.append(update["loss"])
                env.decisions[-1].update(q_keep=float(q[0]), q_switch=float(q[1]))
                qs.append(q)
                total_reward += reward
                state, info = next_state, next_info
                if terminated or truncated:
                    env.save_logs()
                    break
        except BaseException:
            write_json(output / "interrupted.json", {"episode": episode, "resume_policy": "Restart from last completed checkpoint; SUMO mid-episode state is not saved"})
            raise
        finally:
            env.close()
        metric = info["metrics"]
        validation_score = None
        do_validate = episode == 0 or (episode + 1) % cfg["validation_every"] == 0 or episode == cfg["episodes"] - 1
        metadata = {"episode": episode, "global_step": agent.global_step, "updates": agent.updates,
                    "training_mode": cfg["mode"], "horizon_seconds": cfg["horizon_seconds"], "double_dqn": cfg["double_dqn"],
                    "reward_mode": base["reward"]["mode"], "versions": versions, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "selection_metric": "validation mean cohort_delay_s (waiting + insertion delay per scheduled vehicle)",
                    "performance_claim": "No improvement guarantee; inspect held-out evaluation"}
        if do_validate:
            this_validation = []
            for name in cfg["validation_scenarios"]:
                for i in range(cfg["validation_seeds"]):
                    vseed = load_yaml("scenarios.yaml")["splits"]["validation"]["start"] + i
                    vroute = generate_traffic(name, vseed, cfg["horizon_seconds"], "validation", directory=output / "routes")
                    vm = run_episode(base, vroute, vseed, "dqn", output / "validation" / f"ep{episode:04d}_{name}_{vseed}", agent=agent)
                    vr = {"episode": episode, "scenario": name, "seed": vseed, **vm}
                    validation_rows.append(vr)
                    this_validation.append(vm)
            validation_score = float(np.mean([v["cohort_delay_s"] for v in this_validation]))
            if validation_score < best_score:
                best_score = validation_score
                export_model(agent, base, output / "checkpoints/best_model.pt", {**metadata, "validation_score": validation_score, "artifact_role": "best_validation"})
        export_model(agent, base, output / "checkpoints/last_model.pt", {**metadata, "artifact_role": "last_episode"})
        rows.append({"episode": episode, "scenario": scenario, "seed": seed, "global_step": agent.global_step,
                     "episode_reward": total_reward, "epsilon": agent.epsilon, "loss": float(np.mean(losses)) if losses else None,
                     "q_keep_mean": float(np.mean(np.asarray(qs)[:, 0])), "q_switch_mean": float(np.mean(np.asarray(qs)[:, 1])),
                     "mean_waiting_s": metric["mean_waiting_s"], "mean_queue_veh": metric["mean_queue_veh"],
                     "cohort_delay_s": metric["cohort_delay_s"], "validation_score": validation_score,
                     "wall_seconds": time.perf_counter() - episode_start, "elapsed_seconds": time.perf_counter() - start})
        if do_validate or (episode + 1) % cfg["checkpoint_every"] == 0 or episode == cfg["episodes"] - 1 or stop_after == episode + 1:
            persist(episode)
        elapsed = time.perf_counter() - start
        eta = elapsed / (episode - next_episode + 1) * (cfg["episodes"] - episode - 1)
        print(f"ep {episode + 1}/{cfg['episodes']} steps={agent.global_step} reward={total_reward:.2f} wait={metric['mean_waiting_s']} queue={metric['mean_queue_veh']:.2f} eps={agent.epsilon:.3f} loss={rows[-1]['loss']} val={validation_score} elapsed={elapsed:.1f}s ETA~{eta:.1f}s", flush=True)
        if stop_after and episode + 1 >= stop_after:
            break
    atomic_text(output / "logs/training_summary.csv", pd.DataFrame(rows).to_csv(index=False))
    write_json(output / "training_status.json", {"completed_episodes": len(rows), "planned_episodes": cfg["episodes"],
               "mode": cfg["mode"], "complete": len(rows) == cfg["episodes"], "updates": agent.updates,
               "best_validation_cohort_delay_s": best_score, "elapsed_this_invocation_seconds": time.perf_counter() - start})
    sync_mirror()
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "quick", "revision", "full"], default="quick")
    p.add_argument("--output", default="outputs/run")
    p.add_argument("--resume")
    p.add_argument("--double-dqn", action="store_true")
    p.add_argument("--reward", choices=["simple", "composite"])
    p.add_argument("--stop-after", type=int)
    p.add_argument("--training-seed", type=int)
    p.add_argument("--mirror", help="Periodically copy completed checkpoints and logs to Drive or another disk")
    a = p.parse_args()
    train(a.mode, a.output, a.resume, a.double_dqn, a.reward, a.stop_after, a.mirror, a.training_seed)


if __name__ == "__main__":
    main()
