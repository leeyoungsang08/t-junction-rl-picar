from pathlib import Path
import argparse
import json

from src.evaluate import run_episode
from src.inference import Policy
from src.utils import load_yaml, sha256

parser = argparse.ArgumentParser(description="Reproduce the bundled seed 43 SUMO evaluation on CPU.")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
run = Path("outputs/budget_check_20261103")
eval_dir = run / "evaluation"
episode = eval_dir / "episodes/heldout_reversal_3405_v2_43_240"
row = json.loads((episode / "result_row.json").read_text(encoding="utf-8"))
route = eval_dir / row["route_file"]
model = run / "training/seed43/checkpoints/best_model.pt"
assert sha256(route) == row["route_sha256"] and sha256(model) == row["model_sha256"]
for seed in (42, 43, 44):
    candidate = Policy(run / f"training/seed{seed}/checkpoints/best_model.pt", device="cpu")
    assert candidate.adapter.schema == 2 and candidate.model.layers[0].in_features == 25
base = load_yaml("base.yaml")
base["simulation"]["horizon_seconds"] = row["horizon_seconds"]
args.output.mkdir(parents=True, exist_ok=False)
result = run_episode(base, route, 3405, "dqn", args.output, policy=Policy(model, device="cpu"))
expected = json.loads((episode / "metrics.json").read_text(encoding="utf-8"))
assert all(value == expected[key] for key, value in json.loads((args.output / "metrics.json").read_text()).items())
assert (args.output / "decisions.csv").read_bytes() == (episode / "decisions.csv").read_bytes()
print(f"PASS: loaded 3 CPU policies, 360 decisions and 1800s metrics reproduced, "
      f"delay={result['cohort_delay_s']:.6f}, completed={result['completed_vehicles']}")
