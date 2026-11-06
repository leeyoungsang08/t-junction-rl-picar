import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "T_Junction_RL"
WORK = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/v3_research/full"
FINAL = SOURCE / "final_eval"
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

OPTIONS = {
    "original": [],
    "no_teleport": ["--time-to-teleport", "-1"],
    "junction_check": ["--collision.check-junctions", "true"],
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_demand(task):
    from src.evaluate import run_episode
    from src.inference import Policy
    import libsumo

    mode, scenario, seed, route, configs, model_hashes = task
    results = []
    start = libsumo.start
    try:
        def start_logged(command, *args, **kwargs):
            return start([*command, *OPTIONS[mode]], *args, **kwargs)
        libsumo.start = start_logged
        for controller, cfg in configs.items():
            model_path = SOURCE / controller / "checkpoints/best_model.pt"
            policy = Policy(model_path) if controller in model_hashes else None
            path = WORK / "live" / mode / f"{scenario}_{seed}_{controller}"
            result = run_episode(cfg, route, seed, "dqn" if policy else controller, path, policy=policy)
            header = (path / "tripinfo.xml").read_text(encoding="utf-8").split("-->", 1)[0]
            assert "Eclipse SUMO libsumo 1.28.0" in header
            if mode == "no_teleport":
                assert result["teleport_events"] == 0 and '<time-to-teleport value="-1"' in header
            if mode == "junction_check":
                assert '<collision.check-junctions value="true"' in header
            results.append(dict(mode=mode, scenario=scenario, seed=seed, controller=controller, **result))
    finally:
        libsumo.start = start
    return results

def main():
    plan = json.loads((FINAL / "evaluation_plan.json").read_text())
    assert sha256(ROOT / "maps/t_junction.net.xml") == plan["net_sha256"]
    for controller, digest in plan["model_sha256"].items():
        assert sha256(SOURCE / controller / "checkpoints/best_model.pt") == digest
    routes = json.loads((FINAL / "demand_manifest.json").read_text())
    tasks = []
    for mode in OPTIONS:
        scenarios = {"high", "heldout_reversal", "south_oversaturated"} if mode == "original" else {"south_oversaturated"}
        for row in routes:
            if row["scenario"] not in scenarios:
                continue
            route = FINAL / row["route"]
            assert sha256(route) == row["sha256"]
            tasks.append((mode, row["scenario"], row["seed"], route, plan["environment"], plan["model_sha256"]))

    original = {(row["scenario"], int(row["seed"]), row["controller"]): row for row in
                csv.DictReader((FINAL / "seed_level_results.csv").open(newline=""))}
    rows = []
    with ProcessPoolExecutor(max_workers=8) as workers:
        for i, result in enumerate(workers.map(run_demand, tasks), start=1):
            rows.extend(result)
            if i % 10 == 0:
                print(f"live SUMO: {len(rows)}/{len(tasks) * 9}", flush=True)

    differences = []
    max_delta = {}
    for row in rows:
        if row["mode"] != "original":
            continue
        old = original[row["scenario"], row["seed"], row["controller"]]
        for field, value in row.items():
            if field in {"mode", "scenario", "seed", "controller", "wall_seconds"}:
                continue
            if value is None and old[field] == "":
                continue
            delta = abs(float(value) - float(old[field]))
            max_delta[field] = max(max_delta.get(field, 0), delta)
            if not math.isfinite(delta) or delta > 1e-9:
                differences.append((row["scenario"], row["seed"], row["controller"], field, delta))
    with (WORK / "live_sumo_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"counts": {mode: sum(r["mode"] == mode for r in rows) for mode in OPTIONS},
               "options": OPTIONS, "maximum_absolute_difference": max_delta, "differences_over_1e_9": differences}
    (WORK / "live_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    assert not differences

if __name__ == "__main__":
    main()
