import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import run_episode
from src.inference import Policy
from src.traffic_generator import generate_traffic
from src.utils import load_yaml
from src.visualization import render_trajectories


def play(model, controller, scenario, seed, horizon, output, gui=False, delay_ms=30):
    cfg = load_yaml("base.yaml")
    cfg["simulation"]["horizon_seconds"] = horizon
    policy = Policy(model) if controller == "dqn" else None
    if policy:
        cfg["signal"] = policy.payload["signal"]
    route = generate_traffic(scenario, seed, horizon, "test", Path(output) / "routes")
    result = run_episode(cfg, route, seed, controller, output, policy=policy, capture=True, gui=gui, delay_ms=delay_ms)
    render_trajectories([Path(output) / "trajectory.jsonl"], [controller], Path(output) / "playback.mp4", seconds=min(horizon, 600))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--controller", choices=["dqn", "fixed_time", "rule_based"], default="dqn")
    p.add_argument("--scenario", choices=list(load_yaml("scenarios.yaml")["scenarios"]), default="high")
    p.add_argument("--seed", type=int, default=3050)
    p.add_argument("--horizon", type=int, default=900)
    p.add_argument("--output", default="outputs/playground")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--delay-ms", type=int, default=30)
    a = p.parse_args()
    print(play(a.model, a.controller, a.scenario, a.seed, a.horizon, a.output, a.gui, a.delay_ms))
