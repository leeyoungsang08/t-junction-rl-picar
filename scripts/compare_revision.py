"""Compare the archived V1 model, three independent V2 fits and two baselines."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from src.evaluate import run_episode
from src.inference import Policy
from src.traffic_generator import generate_traffic
from src.utils import atomic_text, load_yaml, sha256, version_info, write_json


def evaluate_pair(job):
    scenario, seed, horizon, output, models, model_hashes = job
    cfg = load_yaml("base.yaml")
    cfg["simulation"]["horizon_seconds"] = horizon
    route = generate_traffic(scenario, seed, horizon, "test", directory=output / "routes")
    rows = []
    route_digest = sha256(route)
    for label, path in [("fixed_time", None), ("rule_based", None), ("rule_observed", None), *models.items()]:
        directory = output / "episodes" / f"{scenario}_{seed}_{label}"
        identity = {"scenario": scenario, "seed": seed, "controller": label,
                    "training_seed": int(label.split("_")[1]) if label.startswith("v2_") else None,
                    "model_sha256": model_hashes.get(label), "model_path": path.as_posix() if path else None,
                    "route_file": route.relative_to(output).as_posix(), "route_sha256": route_digest,
                    "horizon_seconds": horizon, "episode_dir": directory.relative_to(output).as_posix()}
        receipt = directory / "result_row.json"
        if receipt.exists() and all((directory / f).is_file() for f in
                                    ("tripinfo.xml", "metrics.json", "decisions.csv", "timeseries.csv")):
            row = json.loads(receipt.read_text(encoding="utf-8"))
            if all(row.get(key) == value for key, value in identity.items()):
                rows.append(row)
                continue
        receipt.unlink(missing_ok=True)
        policy = Policy(path) if path else None
        if policy and policy.payload["signal"] != cfg["signal"]:
            raise ValueError(f"Signal configuration differs for {label}")
        metrics = run_episode(cfg, route, seed, "dqn" if policy else label, directory, policy=policy)
        if sha256(route) != route_digest:
            raise ValueError(f"Route changed during evaluation: {route}")
        row = {**identity, **metrics}
        write_json(receipt, row)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/revision_comparison"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--training-root", type=Path, default=Path("outputs/revision"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    models = {"v1": Path("outputs/reference/legacy_v1.pt")}
    models.update({f"v2_{s}": args.training_root / f"seed{s}/checkpoints/best_model.pt" for s in (42, 43, 44)})
    scenarios = ["medium", "imbalanced", "east_bias", "south_bias", "unseen", "heldout_reversal"]
    seeds = list(range(3120, 3124))
    horizon = 1800
    manifest = {"models": {k: sha256(p) for k, p in models.items()},
                "horizon": horizon, "seeds": seeds, "scenarios": scenarios, "versions": version_info(),
                "model_selection": "Lowest validation cohort delay within each training seed, before test",
                "net_sha256": sha256(Path("maps/t_junction.net.xml")),
                "configuration": {name: load_yaml(name) for name in ("base.yaml", "scenarios.yaml", "training.yaml")}}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    plan = args.output / "manifest.json"
    if args.resume and plan.exists():
        if json.loads(plan.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Comparison plan differs from the saved run")
    elif any(args.output.iterdir()):
        raise ValueError("Existing comparison has no resumable plan")
    else:
        write_json(plan, manifest)
    jobs = [(name, seed, horizon, args.output, models, manifest["models"]) for name in scenarios for seed in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch in pool.map(evaluate_pair, jobs):
            rows.extend(batch)
            print(batch[0]["scenario"], batch[0]["seed"], flush=True)
    result = pd.DataFrame(rows)
    atomic_text(args.output / "seed_level_results.csv", result.to_csv(index=False))
    columns = ["cohort_delay_s", "completion_rate", "uninserted_vehicles", "collision_events", "teleport_events"]
    result.groupby(["scenario", "controller"])[columns].mean().to_csv(args.output / "scenario_summary.csv")


if __name__ == "__main__":
    main()
