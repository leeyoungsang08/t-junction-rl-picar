"""Continue the verified runs and compare validation-selected weights on fresh demand."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd
import torch
import yaml

from scripts.compare_revision import evaluate_pair
from src.dqn_agent import atomic_torch_save
from src.utils import atomic_text, load_yaml, sha256, version_info, write_json

RUN = Path("outputs/budget_check_20261103")
SOURCE = Path("outputs/verify_20261103/training")
SEEDS = (42, 43, 44)
EPISODES = 240
SCENARIOS = ["medium", "south_bias", "unseen", "heldout_reversal"]
DEMAND_SEEDS = list(range(3400, 3408))
HORIZON = 1800


def extend(seed):
    old = SOURCE / f"seed{seed}"
    new = RUN / "training" / f"seed{seed}"
    original = old / "checkpoints/training_checkpoint.pt"
    saved = new / "checkpoints/training_checkpoint.pt"
    if not saved.exists():
        temporary = new.with_name(new.name + "_staging")
        assert not new.exists() and not temporary.exists()
        shutil.copytree(old, temporary)
        cfg = yaml.safe_load((old / "experiment_config.yaml").read_text())
        state = torch.load(original, map_location="cpu", weights_only=False)
        assert state["config"] == cfg["training"] and state["next_episode"] == 80
        cfg["training"]["episodes"] = EPISODES
        state["config"]["episodes"] = EPISODES
        atomic_torch_save(state, temporary / "checkpoints/training_checkpoint.pt")
        atomic_text(temporary / "experiment_config.yaml", yaml.safe_dump(cfg, sort_keys=False))
        write_json(temporary / "continuation.json", {"source": original.as_posix(), "source_sha256": sha256(original),
                   "first_new_episode": 81, "total_episodes": EPISODES, "changed_config_keys": ["episodes"]})
        temporary.replace(new)
    return new, saved


def train():
    children = []
    for seed in SEEDS:
        new, saved = extend(seed)
        status_file = new / "training_status.json"
        if status_file.is_file():
            status = json.loads(status_file.read_text())
            if status["complete"] and status["completed_episodes"] == EPISODES:
                continue
        with (RUN / f"train_{seed}.log").open("a", encoding="utf-8") as log:
            cmd = [sys.executable, "-m", "scripts.resume_training", "--output", str(new), "--resume", str(saved)]
            print(subprocess.list2cmdline(cmd), file=log, flush=True)
            children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(5)
    codes = [child.wait() for child in children]
    assert all(code == 0 for code in codes), codes
    for seed in SEEDS:
        status = json.loads((RUN / "training" / f"seed{seed}" / "training_status.json").read_text())
        assert status["complete"] and status["completed_episodes"] == EPISODES and status["updates"] == 28673
        print(seed, status, flush=True)


def evaluate():
    out = RUN / "evaluation"
    out.mkdir(exist_ok=True)
    models = {"v1": Path("outputs/reference/legacy_v1.pt")}
    for seed in SEEDS:
        for budget, root in ((80, SOURCE), (240, RUN / "training")):
            models[f"v2_{seed}_{budget}"] = root / f"seed{seed}/checkpoints/best_model.pt"
    digests = {label: sha256(path) for label, path in models.items()}
    manifest = {"models": digests, "scenarios": SCENARIOS, "traffic_seeds": DEMAND_SEEDS,
                "horizon_seconds": HORIZON, "selection": "best validation at or before the training budget",
                "criterion": "all three fits at a budget must beat both rules in mean delay in both stress scenarios and be within 2 percentage points of the better rule in completion rate",
                "configuration": {name: load_yaml(name) for name in ("base.yaml", "scenarios.yaml", "training.yaml")},
                "map_sha256": sha256(Path("maps/t_junction.net.xml")), "versions": version_info()}
    plan = out / "manifest.json"
    if plan.exists():
        assert json.loads(plan.read_text()) == manifest
    else:
        write_json(plan, manifest)
    jobs = [(s, d, HORIZON, out, models, digests) for s in SCENARIOS for d in DEMAND_SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for group in pool.map(evaluate_pair, jobs):
            rows.extend(group)
            print(group[0]["scenario"], group[0]["seed"], len(rows), flush=True)
    table = pd.DataFrame(rows)
    assert len(table) == 320 and not table.duplicated(["scenario", "seed", "controller"]).any()
    assert table.groupby(["scenario", "seed"]).route_sha256.nunique().eq(1).all()
    assert table.groupby(["scenario", "seed"]).controller.nunique().eq(10).all()
    atomic_text(out / "seed_level_results.csv", table.to_csv(index=False))


if __name__ == "__main__":
    train()
    evaluate()

