from pathlib import Path
from hashlib import sha256
import json
import pandas as pd
import torch
import xml.etree.ElementTree as ET

run = Path("outputs/verify_20261103")
cmp = run / "comparison"
d = pd.read_csv(cmp / "seed_level_results.csv")
archive = Path("outputs/revision_comparison")
old = pd.read_csv(archive / "seed_level_results.csv")
metrics = ["cohort_delay_s", "completion_rate", "throughput_veh_h", "mean_queue_veh", "completed_vehicles"]
joint = d.merge(old, on=["scenario", "seed", "controller"], suffixes=("_new", "_archived"), validate="one_to_one")
assert len(joint) == len(d) == 168
for c in metrics:
    joint[c + "_delta"] = joint[c + "_new"] - joint[c + "_archived"]
joint[["scenario", "seed", "controller", *(c + "_delta" for c in metrics)]].to_csv(cmp / "archived_comparison_delta.csv", index=False)
paths = d.route_file.unique()
for route in paths:
    assert (cmp / route).read_text() == (archive / route).read_text()
    assert (cmp / route).with_suffix(".json").is_file()
print("Matched XML after normalizing line endings:", len(paths))
print("Max absolute V1 cohort delay difference vs archived run:", joint[joint.controller.eq("v1")].cohort_delay_s_delta.abs().max())
for seed in (42, 43, 44):
    model_run = run / "training" / f"seed{seed}"
    logs = model_run / "logs"
    decisions = pd.read_csv(logs / "training.csv")
    val = pd.read_csv(logs / "validation.csv")
    updates = pd.read_csv(logs / "updates.csv")
    ckpt = torch.load(model_run / "checkpoints/training_checkpoint.pt", map_location="cpu", weights_only=False)
    best = torch.load(model_run / "checkpoints/best_model.pt", map_location="cpu", weights_only=True)
    last = torch.load(model_run / "checkpoints/last_model.pt", map_location="cpu", weights_only=True)
    assert ckpt["next_episode"] == len(decisions) == 80
    assert ckpt["updates"] == len(updates) == 9473 and ckpt["global_step"] == 9600
    assert ckpt["config"]["seed"] == seed and ckpt["config"]["state_dim"] == 25
    assert ckpt["extra"]["net_sha256"] == sha256(Path("maps/t_junction.net.xml").read_bytes()).hexdigest()
    assert best["schema_version"] == last["schema_version"] == 2
    assert tuple(best["state_dict"]["layers.0.weight"].shape) == (128, 25)
    scores = val.groupby("episode").cohort_delay_s.mean()
    assert len(val) == 25 and best["metadata"]["episode"] == scores.idxmin()
    assert all(torch.equal(v, last["state_dict"][k]) for k, v in ckpt["online"].items())
    print("Validated training seed", seed, "best episode", scores.idxmin() + 1, "score", scores.min())

video = run / "video_seed42"
same_demand = d.query("scenario == 'heldout_reversal' and seed == 3120 and controller == 'v2_42'").iloc[0]
metrics = json.loads((video / "metrics.json").read_text())
assert metrics["cohort_delay_s"] == same_demand.cohort_delay_s
assert len((video / "trajectory.jsonl").read_text().splitlines()) == 1800
assert (video / "playback.mp4").stat().st_size > 0
print("Video rerun metric agrees with comparison", metrics["cohort_delay_s"])
