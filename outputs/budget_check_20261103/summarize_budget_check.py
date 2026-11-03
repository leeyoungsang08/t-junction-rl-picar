"""Summarize a predeclared paired comparison with separate traffic and training seeds."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t

ROOT = Path("outputs/budget_check_20261103")
EVAL = ROOT / "evaluation"
RUN80 = Path("outputs/verify_20261103/training")
SEEDS = (42, 43, 44)
STRESS = ("unseen", "heldout_reversal")
METRICS = ["cohort_delay_s", "completion_rate", "throughput_veh_h", "mean_queue_veh",
           "uninserted_vehicles", "unfinished_vehicles", "collision_events", "teleport_events"]


def main():
    data = pd.read_csv(EVAL / "seed_level_results.csv")
    manifest = json.loads((EVAL / "manifest.json").read_text())
    assert len(data) == 320 and sorted(data.seed.unique()) == manifest["traffic_seeds"]
    assert len(list((EVAL / "episodes").glob("*/result_row.json"))) == len(data)
    assert (data.observed_seconds == manifest["horizon_seconds"]).all()
    assert (data.completed_vehicles + data.unfinished_vehicles + data.uninserted_vehicles
            == data.requested_vehicles).all()
    assert data.groupby(["scenario", "seed"]).requested_vehicles.nunique().eq(1).all()
    mean = data.groupby(["scenario", "controller"])[METRICS].mean()
    mean.to_csv(EVAL / "scenario_means.csv")
    pairs = []
    checks = []
    for scenario, group in data.groupby("scenario"):
        delay = group.pivot(index="seed", columns="controller", values="cohort_delay_s")
        completion = group.pivot(index="seed", columns="controller", values="completion_rate")
        for seed in SEEDS:
            for budget in (80, 240):
                label = f"v2_{seed}_{budget}"
                references = ["rule_based", "rule_observed"]
                if budget == 240:
                    references.append(f"v2_{seed}_80")
                for ref in references:
                    delta = delay[label] - delay[ref]
                    n = len(delta)
                    half = t.ppf(0.975, n - 1) * delta.sem()
                    pairs.append({"scenario": scenario, "training_seed": seed, "budget": budget,
                                  "reference": ref, "n": n, "wins": int(delta.lt(0).sum()),
                                  "delta_mean": delta.mean(), "ci95_low": delta.mean() - half,
                                  "ci95_high": delta.mean() + half})
                if scenario in STRESS:
                    rule_delay = min(delay.rule_based.mean(), delay.rule_observed.mean())
                    rule_completion = max(completion.rule_based.mean(), completion.rule_observed.mean())
                    checks.append({"scenario": scenario, "training_seed": seed, "budget": budget,
                                   "delay_vs_lower_rule_s": delay[label].mean() - rule_delay,
                                   "completion_vs_higher_rule_pp": 100 * (completion[label].mean() - rule_completion),
                                   "pass": delay[label].mean() < rule_delay and
                                           completion[label].mean() >= rule_completion - 0.02})
    pd.DataFrame(pairs).to_csv(EVAL / "paired_deltas.csv", index=False)
    gate = pd.DataFrame(checks)
    gate.to_csv(EVAL / "reliability_criterion.csv", index=False)

    curves = []
    models = []
    for seed in SEEDS:
        run = ROOT / "training" / f"seed{seed}"
        val = pd.read_csv(run / "logs/validation.csv")
        assert len(val) == 65
        series = val.groupby("episode").cohort_delay_s.mean()
        curves.extend({"training_seed": seed, "episode": i + 1, "mean_delay_s": score}
                      for i, score in series.items())
        for budget, root in ((80, RUN80), (240, ROOT / "training")):
            ckpts = root / f"seed{seed}" / "checkpoints"
            meta = json.loads((ckpts / "best_model.metadata.json").read_text())
            assert meta["episode"] == series[series.index < budget].idxmin()
            models.append({"training_seed": seed, "budget": budget, "best_episode": meta["episode"] + 1,
                           "best_validation_delay_s": meta["validation_score"],
                           "last_validation_delay_s": series.loc[budget - 1],
                           "updates_at_budget": budget * 120 - 127,
                           "updates_in_selected_model": meta["updates"]})
    pd.DataFrame(models).to_csv(EVAL / "model_selection.csv", index=False)
    curves = pd.DataFrame(curves)
    curves.to_csv(EVAL / "validation_curves.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for seed, series in curves.groupby("training_seed"):
        axes[0].plot(series.episode, series.mean_delay_s, ".-", label=str(seed))
    axes[0].axvline(80, linestyle="--", color="0.5")
    axes[0].set(title="Validation during continued training", xlabel="Completed training episodes",
                ylabel="Delay (s per request)")
    axes[0].legend(title="Training seed")
    for axis, scenario in zip(axes[1:], STRESS):
        means = mean.loc[scenario].cohort_delay_s
        budgets = pd.DataFrame({str(b): [means.loc[f"v2_{s}_{b}"] for s in SEEDS] for b in (80, 240)},
                               index=[str(s) for s in SEEDS])
        budgets.plot.bar(ax=axis, legend=False, rot=0)
        axis.axhline(means.rule_based, color="#8c564b", linestyle="--", label="Rule")
        axis.axhline(means.rule_observed, color="#9467bd", linestyle=":", label="Rule (same input)")
        axis.set(title=scenario, xlabel="Training seed", ylabel="Delay (s per request)")
    axes[-1].legend([*axes[-1].containers, *axes[-1].lines], ["80 episodes", "240 episodes", "Rule", "Rule (same input)"], fontsize=8)
    fig.savefig(EVAL / "budget_comparison.png", dpi=155)
    plt.close(fig)
    print("Rows", len(data), "matched groups", data.groupby(["scenario", "seed"]).ngroups)
    print("Collisions", int(data.collision_events.sum()), "teleports", int(data.teleport_events.sum()))
    print("Criterion checks\n", gate.to_string(index=False))
    print("Model selection\n", pd.DataFrame(models).to_string(index=False))
    print("Delay means\n", mean.cohort_delay_s.unstack().to_string())
    print("Completion means\n", (mean.completion_rate.unstack() * 100).to_string())


if __name__ == "__main__":
    main()

