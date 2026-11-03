import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import pandas as pd
import yaml
from src.baselines import baseline_action
from src.environment import TJunctionEnv
from src.inference import Policy
from src.metrics import compare_results
from src.traffic_generator import generate_traffic
from src.utils import ROOT, atomic_text, load_yaml, sha256, version_info, write_json


def run_episode(config, route, seed, controller, directory, policy=None, agent=None, capture=False, gui=False, delay_ms=0):
    if controller == "fixed_time":
        config = deepcopy(config)
        config["simulation"]["decision_seconds"] = 1
    env = TJunctionEnv(config, route, seed, directory, capture=capture, gui=gui, delay_ms=delay_ms)
    state, info = env.reset(seed=seed)
    total_reward = 0
    started = time.perf_counter()
    try:
        while True:
            q = None
            if controller == "dqn":
                if agent is not None:
                    action, q = agent.select(state, info["action_mask"], explore=False)
                elif policy is not None:
                    prediction = policy.predict(info["raw_state"])
                    action, q = prediction["action_id"], prediction["q_values"]
                else:
                    raise ValueError("DQN requires a trained policy")
            else:
                action = baseline_action(controller, info, config)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if q is not None:
                env.decisions[-1].update(q_keep=float(q[0]), q_switch=float(q[1]))
            if terminated or truncated:
                env.save_logs()
                break
        return {**info["metrics"], "episode_reward": total_reward, "wall_seconds": time.perf_counter() - started}
    finally:
        env.close()


def evaluate(model_path, output, mode="quick", seeds=None, horizon=None, scenarios=None, controllers=None, resume=False):
    output = Path(output).resolve()
    results_path = output / "seed_level_results.csv"
    if results_path.exists() and not resume:
        raise FileExistsError(f"Evaluation already exists: {output}; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    ec = load_yaml("evaluation.yaml")
    sc = load_yaml("scenarios.yaml")
    count = seeds or ec[mode]["seeds"]
    horizon = horizon or ec[mode]["horizon_seconds"]
    names = scenarios or ec["scenarios"]
    controllers = controllers or ec["controllers"]
    cfg = load_yaml("base.yaml")
    cfg["simulation"]["horizon_seconds"] = horizon
    policy = Policy(model_path) if "dqn" in controllers else None
    if policy and (policy.payload["signal"] != cfg["signal"]):
        raise ValueError("Evaluation and saved model use different signal settings")
    if model_path and policy and policy.adapter.schema == cfg.get("observation_schema", 1):
        candidate_config = Path(model_path).parent.parent / "experiment_config.yaml"
        if candidate_config.exists():
            trained_base = yaml.safe_load(candidate_config.read_text())["environment"]
            if trained_base["reward"] != cfg["reward"]:
                cfg["reward"] = trained_base["reward"]
    plan = {"scenarios": names, "controllers": controllers, "seeds": count, "horizon": horizon,
            "environment": cfg, "traffic": sc, "mode": mode,
            "model_sha256": sha256(model_path) if model_path else None,
            "net_sha256": sha256(ROOT / "maps/t_junction.net.xml")}
    plan_path = output / "evaluation_plan.json"
    if resume and plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Evaluation plan differs from the saved run")
    if results_path.exists() and not plan_path.exists():
        raise ValueError("Existing evaluation has no resumable plan; use a fresh directory")
    write_json(plan_path, plan)
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    completed = {(r["scenario"], r["seed"], r["controller"]): r for r in rows}
    if len(completed) != len(rows):
        raise ValueError("Duplicate rows in evaluation checkpoint")
    routes = []
    for name in names:
        for index in range(count):
            seed = sc["splits"]["test"]["start"] + index
            route = generate_traffic(name, seed, horizon, "test", directory=output / "routes")
            digest = sha256(route)
            routes.append({"scenario": name, "seed": seed, "route": os.path.relpath(route, ROOT), "sha256": digest})
            for controller in controllers:
                run_dir = output / "episodes" / f"{name}_{seed}_{controller}"
                old = completed.get((name, seed, controller))
                if old and old["route_sha256"] == digest and all(
                    (run_dir / f).is_file() for f in ("tripinfo.xml", "metrics.json", "decisions.csv", "timeseries.csv")
                ):
                    continue
                if old:
                    rows.remove(old)
                metrics = run_episode(cfg, route, seed, controller, run_dir, policy=policy)
                assert sha256(route) == digest
                rows.append({"scenario": name, "seed": seed, "controller": controller, "horizon_seconds": horizon,
                             "route_sha256": digest, "episode_dir": os.path.relpath(run_dir, ROOT), **metrics})
                atomic_text(output / "seed_level_results.csv", pd.DataFrame(rows).to_csv(index=False))
                print(f"eval {name} seed={seed} {controller}: wait={metrics['mean_waiting_s']} cohort={metrics['cohort_delay_s']:.2f} completed={metrics['completed_vehicles']}", flush=True)
    frame = pd.DataFrame(rows)
    summary, paired = compare_results(frame)
    atomic_text(output / "final_results.csv", summary.to_csv(index=False))
    atomic_text(output / "scenario_comparison.csv", paired.to_csv(index=False))
    experiment = {"environment": cfg, "evaluation": {"mode": mode, "seeds": count, "horizon_seconds": horizon,
                  "scenarios": names, "controllers": controllers}, "traffic_config": sc}
    atomic_text(output / "experiment_config.yaml", yaml.safe_dump(experiment, sort_keys=False))
    write_json(output / "experiment_manifest.json", {"versions": version_info(), "routes": routes,
               "net_sha256": sha256(ROOT / "maps/t_junction.net.xml"), "model_sha256": sha256(model_path) if model_path else None,
               "model_metadata": policy.payload["metadata"] if policy else None,
               "evidence_scope": "short-horizon verification" if mode == "quick" else "full-horizon evaluation",
               "test_used_for_training": False})
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="outputs/evaluation")
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--seeds", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    evaluate(args.model, args.output, args.mode, args.seeds, args.horizon, resume=args.resume)


if __name__ == "__main__":
    main()
