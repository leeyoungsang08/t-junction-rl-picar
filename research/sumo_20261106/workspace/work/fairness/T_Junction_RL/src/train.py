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
from maps.generate_map import generate, validate_map
from src.dqn_agent import DQNAgent, atomic_torch_save
from src.environment import TJunctionEnv
from src.state_adapter import StateAdapter
from src.evaluate import run_episode
from src.inference import export_model
from src.traffic_generator import generate_traffic
from src.utils import ROOT, atomic_text, load_yaml, sha256, version_info, write_json


def resolved_config(mode, double=False, reward_mode=None, algorithm=None, control_mode=None):
    base = load_yaml("base.yaml")
    cfg = load_yaml("training.yaml")
    cfg.update(cfg["modes"][mode])
    del cfg["modes"]
    cfg["mode"] = mode
    cfg["algorithm"] = algorithm or ("double" if double else "dqn")
    cfg["reward_discount_version"] = 1
    cfg["state_dim"] = len(StateAdapter(base["normalization"], base["signal"], base.get("observation_schema", 1)).features)
    algorithms = {"dqn": (False, False, False), "double": (True, False, False),
                  "dueling_double": (True, True, False), "rainbow_lite": (True, True, True)}
    if algorithm is not None and algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if algorithm is not None:
        cfg["double_dqn"], cfg["dueling"], cfg["prioritized_replay"] = algorithms[algorithm]
    else:
        cfg["double_dqn"] = double
    if control_mode is not None:
        if control_mode not in ("cycle", "direct"):
            raise ValueError("Unknown control mode")
        base["signal"]["control_mode"] = control_mode
    cfg["action_dim"] = 3 if base["signal"].get("control_mode", "cycle") == "direct" else 2
    base["simulation"]["horizon_seconds"] = cfg["horizon_seconds"]
    if reward_mode:
        base["reward"]["mode"] = reward_mode
    return base, cfg


def train(mode="quick", output=None, resume=None, double=False, reward_mode=None, stop_after=None, mirror=None, training_seed=None, algorithm=None, control_mode=None, use_existing_map=False, device=None, backend=None):
    output = Path(output or ROOT / "outputs/run").resolve()
    if resume:
        if device is not None:
            raise ValueError("Resume uses the original device setting")
        saved = yaml.safe_load((output / "experiment_config.yaml").read_text())
        base, cfg = saved["environment"], saved["training"]
        if backend is not None and backend != base["simulation"]["backend"]:
            raise ValueError("Resume uses the original SUMO backend")
    else:
        base, cfg = resolved_config(mode, double, reward_mode, algorithm, control_mode)
        if device is not None:
            cfg["device"] = device
        if backend is not None:
            base["simulation"]["backend"] = backend
        if training_seed is not None:
            cfg["seed"] = training_seed
        if (output / "checkpoints/training_checkpoint.pt").exists():
            raise FileExistsError("Existing run. Use --resume or a fresh --output directory")
    if mirror and (Path(mirror).resolve() == output or output in Path(mirror).resolve().parents):
        raise ValueError("Mirror must be outside the run directory")
    if use_existing_map:
        validate_map(ROOT / "maps/t_junction.net.xml")
    else:
        generate(base)
    for d in ("checkpoints", "logs", "validation", "training_episodes"):
        (output / d).mkdir(parents=True, exist_ok=True)
    atomic_text(output / "experiment_config.yaml", yaml.safe_dump({"environment": base, "training": cfg}, sort_keys=False))
    versions = version_info()
    net_digest = sha256(ROOT / "maps/t_junction.net.xml")
    best_payload = None
    best_path = output / "checkpoints/best_model.pt"
    agent = DQNAgent(cfg)
    next_episode, best_score = 0, float("inf")
    rows, updates, validation_rows = [], [], []
    if resume:
        checkpoint = agent.restore(resume)
        next_episode = checkpoint["next_episode"]
        if checkpoint["extra"].get("base_config") != base:
            raise ValueError("Environment configuration differs from checkpoint")
        if checkpoint["extra"]["net_sha256"] != net_digest:
            raise ValueError("SUMO map differs from checkpoint")
        best_score = checkpoint["extra"]["best_score"]
        best_payload = checkpoint["extra"].get("best_payload")
        if best_payload is not None:
            atomic_torch_save(best_payload, best_path)
            write_json(best_path.with_suffix(".metadata.json"), best_payload["metadata"])
            write_json(best_path.parent / "model_metadata.json", best_payload["metadata"])
        rows = checkpoint["extra"]["rows"]
        updates = checkpoint["extra"]["update_rows"]
        validation_rows = checkpoint["extra"]["validation_rows"]
        print(f"Resuming at episode {next_episode}; only completed checkpoint episodes are retained.", flush=True)
    algorithm = cfg.get("algorithm")
    if algorithm is None:
        if cfg.get("prioritized_replay"):
            algorithm = "rainbow_lite"
        elif cfg.get("dueling"):
            algorithm = "dueling_double"
        else:
            algorithm = "double" if cfg.get("double_dqn") else "dqn"
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
                 "base_config": base, "net_sha256": net_digest, "best_payload": best_payload}
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
        total_reward, losses, qs = 0.0, [], []
        try:
            state, info = env.reset(seed=seed)
            while True:
                action, q = agent.select(state, info["action_mask"], explore=True)
                next_state, reward, terminated, truncated, next_info = env.step(action)
                duration = next_info["duration_seconds"]
                interval = base["simulation"]["decision_seconds"]
                if cfg.get("reward_discount_version") == 1:
                    reward_for_replay = sum(cfg["gamma"] ** (offset / interval) * row["reward"]
                                            for offset, row in enumerate(env.timeseries[-duration:]))
                    duration_steps = duration / interval
                else:
                    reward_for_replay, duration_steps = reward, 1.0
                agent.replay.add(state, action, reward_for_replay, next_state, terminated, truncated,
                                 next_info["action_mask"], duration_steps=duration_steps)
                agent.global_step += 1
                update = agent.update()
                if update:
                    updates.append(update)
                    losses.append(update["loss"])
                env.decisions[-1].update(**{f"q_action_{i}": float(value) for i, value in enumerate(q)})
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
                    "training_seed": cfg["seed"], "net_sha256": net_digest,
                    "simulation_backend": base["simulation"]["backend"],
                    "training_mode": cfg["mode"], "horizon_seconds": cfg["horizon_seconds"], "double_dqn": cfg["double_dqn"],
                    "reward_mode": base["reward"]["mode"], "algorithm": algorithm,
                    "control_mode": base["signal"].get("control_mode", "cycle"), "versions": versions, "created_utc": datetime.now(timezone.utc).isoformat(),
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
                best_payload = export_model(agent, base, best_path, {**metadata, "validation_score": validation_score,
                                                                     "artifact_role": "best_validation"})
        export_model(agent, base, output / "checkpoints/last_model.pt", {**metadata, "artifact_role": "last_episode"})
        rows.append({"episode": episode, "scenario": scenario, "seed": seed, "global_step": agent.global_step,
                     "episode_reward": total_reward, "epsilon": agent.epsilon, "loss": float(np.mean(losses)) if losses else None,
                     "q_action_0_mean": float(np.mean(np.asarray(qs)[:, 0])), "q_action_1_mean": float(np.mean(np.asarray(qs)[:, 1])),
                     "q_action_2_mean": float(np.mean(np.asarray(qs)[:, 2])) if agent.actions == 3 else None,
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
    p.add_argument("--algorithm", choices=["dqn", "double", "dueling_double", "rainbow_lite"])
    p.add_argument("--control-mode", choices=["cycle", "direct"])
    p.add_argument("--backend", choices=["traci", "libsumo"])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--use-existing-map", action="store_true", help="Use the previously generated project map")
    p.add_argument("--mirror", help="Periodically copy completed checkpoints and logs to Drive or another disk")
    a = p.parse_args()
    train(a.mode, a.output, a.resume, a.double_dqn, a.reward, a.stop_after, a.mirror, a.training_seed, a.algorithm, a.control_mode, a.use_existing_map, a.device, a.backend)


if __name__ == "__main__":
    main()
