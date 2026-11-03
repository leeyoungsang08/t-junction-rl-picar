import numpy as np
import pandas as pd
from src.metrics import compare_results, episode_metrics
from src.utils import atomic_text


def test_paired_comparison_and_zero_denominator():
    rows = []
    for seed in range(5):
        for c, wait in [("fixed_time", 10 + seed), ("dqn", 8 + seed)]:
            rows.append(dict(scenario="low", seed=seed, controller=c, mean_waiting_s=wait,
                             cohort_delay_s=wait, throughput_veh_h=0, mean_queue_veh=1))
    summary, paired = compare_results(pd.DataFrame(rows))
    wait = paired[paired.metric == "mean_waiting_s"].iloc[0]
    assert wait.n_pairs == 5 and wait.difference_other_minus_fixed == -2
    assert wait.ci95_low == wait.ci95_high == -2
    assert pd.isna(paired[paired.metric == "throughput_veh_h"].iloc[0].improvement_pct)


def test_uninserted_vehicles_increase_cohort_delay(tmp_path):
    path = tmp_path / "trips.xml"
    atomic_text(path, '<tripinfos><tripinfo id="v0" arrival="10" depart="0" duration="10" waitingTime="0" timeLoss="0" departDelay="0"/></tripinfos>')
    manifest = {"horizon_seconds": 100, "vehicles": {"v0": {"origin": "W", "requested_depart": 0}, "v1": {"origin": "S", "requested_depart": 20}}}
    m = episode_metrics(path, manifest, [], {"v0"}, {"v0"}, 0, 0, 0, 0)
    assert m["mean_waiting_s"] == 0 and m["cohort_delay_s"] == 40
    assert m["uninserted_vehicles"] == 1 and m["completion_rate"] == 0.5


def test_uninserted_tripinfo_does_not_double_count_departure_delay(tmp_path):
    path = tmp_path / "trips.xml"
    atomic_text(path, '<tripinfos><tripinfo id="pending" arrival="-1" depart="-1" waitingTime="0" departDelay="80"/></tripinfos>')
    manifest = {"horizon_seconds": 100, "vehicles": {"pending": {"origin": "W", "requested_depart": 20}}}
    m = episode_metrics(path, manifest, [], set(), set(), 0, 0, 0, 0)
    assert m["cohort_delay_s"] == 80
