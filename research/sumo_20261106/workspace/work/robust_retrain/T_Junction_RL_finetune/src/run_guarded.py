import argparse
import json
from pathlib import Path
from src.evaluate import run_episode
from src.guarded_policy import GuardedDirectPolicy
from src.utils import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Run frozen Direct Rainbow-lite with a direction service guard in SUMO.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True, help="SUMO route with a sibling .rou.json manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threshold", type=int, default=80, help="Backlog age in seconds to prioritize (default: 80)")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    cfg = load_yaml("base.yaml")
    policy = GuardedDirectPolicy(args.model, args.threshold)
    cfg["observation_schema"] = policy.base.adapter.schema
    cfg["signal"] = policy.base.payload["signal"]
    cfg["normalization"] = policy.base.payload["normalization"]
    demand = json.loads(args.route.with_suffix(".json").read_text(encoding="utf-8"))
    cfg["simulation"].update(backend="libsumo", horizon_seconds=demand["horizon_seconds"],
                             audit_vehicle_safety=True)
    result = run_episode(cfg, args.route, args.seed, "dqn", args.output,
                         network_file=args.network, guarded_policy=policy)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
