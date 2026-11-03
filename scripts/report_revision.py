"""Summarize measured results and verify the archive inputs."""
import argparse
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=Path("outputs/revision_comparison"))
    parser.add_argument("--training-root", type=Path, default=Path("outputs/revision"))
    args = parser.parse_args()
    root = args.comparison
    results = pd.read_csv(root / "seed_level_results.csv")
    assert len(results) == 168
    paired = results.groupby(["scenario", "seed"])
    assert paired.route_sha256.nunique().eq(1).all()
    assert paired.controller.nunique().eq(7).all()
    manifest = json.loads((root / "manifest.json").read_text())
    for seed in (42, 43, 44):
        model = args.training_root / f"seed{seed}/checkpoints/best_model.pt"
        assert manifest["models"][f"v2_{seed}"] == sha256(model)
    xmls = [*root.rglob("tripinfo.xml"), *args.training_root.rglob("tripinfo.xml")]
    for path in xmls:
        assert ET.parse(path).getroot().tag == "tripinfos", path

    counts = []
    for seed in (42, 43, 44):
        run = args.training_root / f"seed{seed}"
        d = pd.concat([pd.read_csv(p) for p in (run / "training_episodes").glob("*/decisions.csv")])
        status = json.loads((run / "training_status.json").read_text())
        model = json.loads((run / "checkpoints/best_model.metadata.json").read_text())
        assert status["complete"] and len(d) == 9600
        counts.append({"seed": seed, "episodes": status["completed_episodes"],
                       "updates": status["updates"], "decisions": len(d),
                       "internal_active": int(d.internal_cost.gt(0).sum()),
                       "long_wait_active": int(d.starvation_cost.gt(0).sum()),
                       "best_episode": model["episode"] + 1,
                       "validation_delay": model["validation_score"]})
    pd.DataFrame(counts).to_csv(root / "training_verification.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    labels = {"fixed_time": "Fixed", "rule_based": "Rule", "rule_observed": "Rule (same input)",
              "v1": "DQN v1", "v2_42": "DQN v2 / 42", "v2_43": "DQN v2 / 43", "v2_44": "DQN v2 / 44"}
    names = ["medium", "imbalanced", "east_bias", "south_bias", "unseen", "heldout_reversal"]
    for ax, metric, title in zip(axes, ["cohort_delay_s", "completion_rate"],
                                ["Observed delay per request (s)", "Completion rate (%)"]):
        means = results.groupby(["scenario", "controller"])[metric].mean().unstack().loc[names, list(labels)]
        if metric == "completion_rate":
            means *= 100
        means.rename(columns=labels).plot.bar(ax=ax, legend=False, width=0.85)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=28)
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(loc="lower left", fontsize=7)
    fig.suptitle("SUMO T-junction | 4 matched demand seeds per scenario, 1800 s per run")
    fig.savefig(root / "revision_summary.png", dpi=160)
    plt.close(fig)
    write_json(root / "verification.json", {"comparison_episodes": len(results),
        "matched_demand_groups": len(paired), "parsed_tripinfo_files": len(xmls),
        "collisions": int(results.collision_events.sum()), "teleports": int(results.teleport_events.sum())})


if __name__ == "__main__":
    main()
