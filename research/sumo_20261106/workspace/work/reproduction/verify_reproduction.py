import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import torch

workspace = Path(__file__).resolve().parent
output = workspace.parent.parent / "outputs"
base = workspace / "T_Junction_RL" / "outputs" / "v3_research"
replay = base / "colab"
archive = workspace / "T_Junction_RL_Reproduced_Results.zip"
log = workspace / "notebook_run.log"

def read_rows(path, keys):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {tuple(row[k] for k in keys): row for row in rows}
    assert len(rows) == len(indexed), f"Duplicate rows in {path}"
    return indexed

def compare_csv(before, after, keys, ignored=()):
    old = read_rows(before, keys)
    new = read_rows(after, keys)
    assert old.keys() == new.keys(), f"Different record keys in {after}"
    differences = defaultdict(int)
    max_numeric_delta = defaultdict(float)
    material = defaultdict(int)
    for key, row in old.items():
        assert row.keys() == new[key].keys()
        for column, value in row.items():
            other = new[key][column]
            if column in ignored or value == other:
                continue
            differences[column] += 1
            try:
                delta = abs(float(value) - float(other))
                if math.isfinite(delta):
                    max_numeric_delta[column] = max(max_numeric_delta[column], delta)
                if not math.isfinite(delta) or delta > 1e-9:
                    material[column] += 1
            except ValueError:
                material[column] += 1
    return {"rows": len(old), "differences": dict(differences), "max_numeric_delta": dict(max_numeric_delta),
            "differences_exceeding_absolute_1e_9_or_non_numeric": dict(material), "ignored": list(ignored)}

stage_summary = {}
for stage, expected_count in (("smoke", 4), ("quick", 24), ("full", 6)):
    originals = {p.parent.name: p for p in (base / stage).glob("*/training_status.json")}
    reproductions = {p.parent.name: p for p in (replay / stage).glob("*/training_status.json")}
    assert len(originals) == len(reproductions) == expected_count and originals.keys() == reproductions.keys(), stage
    weight_matches, score_matches, episodes = 0, 0, 0
    for name, status in reproductions.items():
        old = json.loads(originals[name].read_text(encoding="utf-8"))
        new = json.loads(status.read_text(encoding="utf-8"))
        assert new["complete"] and new["completed_episodes"] == old["completed_episodes"], name
        assert new["updates"] == old["updates"], name
        episodes += new["completed_episodes"]
        score_matches += old["best_validation_cohort_delay_s"] == new["best_validation_cohort_delay_s"]
        a, b = (torch.load(folder / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=True)["state_dict"]
                for folder in (originals[name].parent, status.parent))
        weight_matches += a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    stage_summary[stage] = {"runs": expected_count, "episodes": episodes,
                            "exact_best_validation_score_matches": score_matches,
                            "exact_checkpoint_tensor_matches": weight_matches}

case_key = ["scenario", "seed", "controller"]
evaluation = compare_csv(base / "full/final_eval/seed_level_results.csv",
                         replay / "full/final_eval/seed_level_results.csv",
                         case_key, ("wall_seconds", "episode_dir"))
assert evaluation["rows"] == 990
validation = compare_csv(base / "full/validation_eval/seed_level_results.csv",
                         replay / "full/validation_eval/seed_level_results.csv",
                         case_key, ("wall_seconds", "episode_dir"))
leaderboard = compare_csv(base / "full/validation_leaderboard.csv", replay / "full/validation_leaderboard.csv",
                          ["control_mode", "algorithm"])
means = {}
with (replay / "full/figures/scenario_delay.csv").open(encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream):
        name = row.pop("group")
        means[name] = sum(map(float, row.values())) / len(row)
assert "=== Notebook cell 13 completed ===" in log.read_text(encoding="utf-8")
tests = re.findall(r"\b(\d+) passed", log.read_text(encoding="utf-8"))
report = {
    "platform": "Windows CPU; Colab browser login unavailable",
    "local_adaptations": ["local ZIP extraction", "pip requirements and dependency check instead of bash", 
                          "8 workers", "local result ZIP instead of browser download"],
    "pytest_passes": [int(count) for count in tests],
    "training": stage_summary,
    "formal_validation": validation,
    "formal_validation_leaderboard": leaderboard,
    "heldout_evaluation": evaluation,
    "mean_delay_seconds_by_controller": means,
}
with ZipFile(archive, mode="a") as package:
    names = set(package.namelist())
    required = {"full/final_eval/seed_level_results.csv", "full/validation_leaderboard.csv",
                "full/figures/scenario_delay.csv", "full/figures/scenario_heatmap.png"}
    assert required <= names, required - names
    assert "reproduction_verification.json" not in names
    package.writestr("reproduction_verification.json", json.dumps(report, ensure_ascii=False, indent=2))
    package.write(log, "notebook_run.log")
with ZipFile(archive) as package:
    assert package.testzip() is None
destination = output / archive.name
assert not destination.exists(), destination
archive.replace(destination)
print(json.dumps(report, ensure_ascii=False, indent=2))
print(destination)


