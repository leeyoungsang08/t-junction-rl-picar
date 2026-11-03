"""Check matched evaluation receipts and audit decisions from the current run."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()
    root = args.comparison
    data = pd.read_csv(root / "seed_level_results.csv")
    plan = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    controllers = {"fixed_time", "rule_based", "rule_observed", *plan["models"]}
    assert len(data) == len(plan["scenarios"]) * len(plan["seeds"]) * len(controllers)
    assert set(data["scenario"]) == set(plan["scenarios"])
    assert set(data["seed"]) == set(plan["seeds"])
    assert not data.duplicated(["scenario", "seed", "controller"]).any()

    audits = []
    for _, batch in data.groupby(["scenario", "seed"]):
        assert set(batch.controller) == controllers
        assert batch.route_file.nunique() == batch.route_sha256.nunique() == 1
        route = root / batch.route_file.iloc[0]
        assert sha256(route) == batch.route_sha256.iloc[0]
        assert batch.requested_vehicles.nunique() == 1
        for row in batch.itertuples():
            episode = root / row.episode_dir
            assert row.observed_seconds == row.horizon_seconds == plan["horizon"]
            assert row.completed_vehicles + row.unfinished_vehicles + row.uninserted_vehicles == row.requested_vehicles
            decision = pd.read_csv(episode / "decisions.csv")
            series = pd.read_csv(episode / "timeseries.csv")
            assert len(series) == plan["horizon"]
            stage_groups = series.groupby((series.stage.ne(series.stage.shift()) |
                                           series.phase.ne(series.phase.shift())).cumsum())
            for _, stage in list(stage_groups)[:-1]:
                stage_name = stage.stage.iloc[0]
                if stage_name == "green":
                    assert 10 <= len(stage) <= 45
                    if row.controller == "fixed_time":
                        assert len(stage) == 20
                else:
                    assert len(stage) == {"yellow": 3, "all_red": 1}[stage_name]
            assert series.loc[series.stage.eq("all_red"), "tls_state"].eq("rrrrrr").all()
            mask = decision.action_mask.map(json.loads)
            assert all(m[a] for m, a in zip(mask, decision.action))
            assert decision.forced_switches.sum() == row.forced_switches
            if row.controller not in plan["models"]:
                continue
            assert row.model_sha256 == plan["models"][row.controller]
            voluntary = decision[mask.map(all)].copy()
            assert (voluntary.action == voluntary.q_switch.gt(voluntary.q_keep).astype(int)).all()
            state = voluntary.state.map(json.loads)
            voluntary["phase"] = state.map(lambda x: x["phase"])
            voluntary["green_elapsed"] = state.map(lambda x: x["green_elapsed"])
            same_clock = voluntary.groupby(["phase", "green_elapsed"]).action.nunique()
            audits.append({"scenario": row.scenario, "seed": row.seed, "controller": row.controller,
                           "both_valid": len(voluntary), "keep": int(voluntary.action.eq(0).sum()),
                           "switch": int(voluntary.action.eq(1).sum()),
                           "same_phase_elapsed_with_both_choices": int(same_clock.eq(2).sum()),
                           "forced_switches": int(row.forced_switches),
                           "seconds_with_pending": int(series.pending_total.gt(0).sum()),
                           "seconds_with_starvation_cost": int(series.starvation_cost.gt(0).sum()),
                           "seconds_with_internal_cost": int(series.internal_cost.gt(0).sum())})
    pd.DataFrame(audits).to_csv(root / "decision_audit.csv", index=False)
    measures = ["cohort_delay_s", "completion_rate", "throughput_veh_h", "mean_queue_veh", "max_queue_veh",
                "requested_vehicles", "completed_vehicles", "unfinished_vehicles", "uninserted_vehicles",
                "mean_waiting_s", "collision_events", "teleport_events"]
    stats = data.groupby(["scenario", "controller"])[measures].agg(["mean", "std"])
    stats.columns = ["_".join(c) for c in stats.columns]
    stats.to_csv(root / "scenario_metrics.csv")
    dqn = data[data.controller.str.startswith("v2_")].pivot(index=["scenario", "seed"],
                                                            columns="controller", values="cohort_delay_s")
    reference = data.pivot(index=["scenario", "seed"], columns="controller", values="cohort_delay_s")
    for label in ("v1", "rule_based", "rule_observed"):
        dqn.sub(reference[label], axis=0).to_csv(root / f"paired_delay_vs_{label}.csv")
    print(f"{len(data)} episodes, {data.groupby(['scenario', 'seed']).ngroups} matched demand groups, "
          f"{int(data.collision_events.sum())} collisions, {int(data.teleport_events.sum())} teleports")


if __name__ == "__main__":
    main()
