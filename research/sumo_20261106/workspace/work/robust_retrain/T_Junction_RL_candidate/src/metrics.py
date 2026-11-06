import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from src.utils import DIRECTIONS


def mean(values):
    return float(np.mean(values)) if len(values) else None


def episode_metrics(tripinfo, manifest, timeseries, departed_ids, arrived_ids, switches, forced_switches, collisions, teleports):
    horizon = manifest["horizon_seconds"]
    requests = manifest["vehicles"]
    trips = {x.get("id"): {k: float(x.get(k, 0)) for k in ("arrival", "depart", "duration", "waitingTime", "timeLoss", "departDelay")}
             for x in ET.parse(tripinfo).findall("tripinfo")}
    complete = {vid: v for vid, v in trips.items() if v["arrival"] >= 0 and vid in arrived_ids}
    uninserted = set(requests) - set(departed_ids)
    uninserted_delay = sum(max(0, horizon - requests[vid]["requested_depart"]) for vid in uninserted)
    inserted_trips = {vid: v for vid, v in trips.items() if vid in departed_ids}
    if inserted_trips.keys() != set(departed_ids):
        raise ValueError("Tripinfo is missing records for vehicles that entered")
    total_wait = sum(v["waitingTime"] for v in inserted_trips.values())
    total_depart_delay = sum(v["departDelay"] for v in inserted_trips.values())
    finished_wait = [v["waitingTime"] for v in complete.values()]
    queue = [r["queue_total"] for r in timeseries]
    result = {
        "requested_vehicles": len(requests), "departed_vehicles": len(departed_ids),
        "completed_vehicles": len(arrived_ids), "unfinished_vehicles": len(set(departed_ids) - set(arrived_ids)),
        "uninserted_vehicles": len(uninserted), "completion_rate": len(arrived_ids) / len(requests) if requests else None,
        "mean_waiting_s": mean(finished_wait),
        "max_waiting_s": max((v["waitingTime"] for v in inserted_trips.values()), default=0),
        "incomplete_vehicles": len(requests) - len(arrived_ids), "mean_travel_s": mean([v["duration"] for v in complete.values()]),
        "mean_time_loss_s": mean([v["timeLoss"] for v in complete.values()]),
        "p95_waiting_s": float(np.percentile(finished_wait, 95)) if finished_wait else None,
        "mean_queue_veh": mean(queue), "max_queue_veh": max(queue, default=0),
        "throughput_veh_h": len(arrived_ids) * 3600 / horizon,
        "mean_depart_delay_s": mean([v["departDelay"] for v in inserted_trips.values()]),
        "uninserted_delay_s": uninserted_delay,
        "cohort_delay_s": (total_wait + total_depart_delay + uninserted_delay) / len(requests) if requests else 0.0,
        "max_consecutive_wait_s": max((r["max_consecutive_wait_s"] for r in timeseries), default=0),
        "signal_switches": switches, "forced_switches": forced_switches,
        "collision_events": collisions, "teleport_events": teleports,
        "observed_seconds": horizon,
    }
    direction_delays = []
    all_cohort_delays = []
    for d in DIRECTIONS:
        cohort = [vid for vid, request in requests.items() if request["origin"] == d]
        delays = [
            max(0, horizon - requests[vid]["requested_depart"]) if vid in uninserted
            else inserted_trips.get(vid, {}).get("waitingTime", 0) + inserted_trips.get(vid, {}).get("departDelay", 0)
            for vid in cohort
        ]
        all_cohort_delays.extend(delays)
        if delays:
            direction_delays.append(float(np.mean(delays)))
        result[f"mean_waiting_{d}_s"] = mean([complete[vid]["waitingTime"] for vid in cohort if vid in complete])
        result[f"cohort_delay_{d}_s"] = mean(delays)
        result[f"p95_cohort_delay_{d}_s"] = float(np.percentile(delays, 95)) if delays else None
        result[f"uninserted_{d}"] = sum(vid in uninserted for vid in cohort)
    result["median_cohort_delay_s"] = float(np.median(all_cohort_delays)) if all_cohort_delays else None
    result["p95_cohort_delay_s"] = float(np.percentile(all_cohort_delays, 95)) if all_cohort_delays else None
    result["worst_direction_cohort_delay_s"] = max(direction_delays, default=None)
    result["direction_delay_gap_s"] = max(direction_delays) - min(direction_delays) if len(direction_delays) >= 2 else None
    return result


def compare_results(frame, baselines=("fixed_time",)):
    numeric = [c for c in frame.select_dtypes(include="number") if c not in ("seed", "training_seed", "horizon_seconds")]
    summary = frame.groupby(["scenario", "controller"])[numeric].agg(["mean", "std", "count"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    paired = []
    for baseline in baselines:
        fixed = frame[frame.controller == baseline].set_index(["scenario", "seed"])
        for controller in frame.controller.unique():
            if controller == baseline:
                continue
            other = frame[frame.controller == controller].set_index(["scenario", "seed"])
            joint = fixed.merge(other, left_index=True, right_index=True, suffixes=("_fixed", "_other"), how="inner")
            for scenario, group in joint.groupby(level="scenario"):
                for metric in ["mean_waiting_s", "cohort_delay_s", "throughput_veh_h", "mean_queue_veh",
                               "completion_rate", "p95_cohort_delay_s", "worst_direction_cohort_delay_s"]:
                    if metric + "_fixed" not in group or metric + "_other" not in group:
                        continue
                    data = group[[metric + "_fixed", metric + "_other"]].dropna()
                    diff = data.iloc[:, 1] - data.iloc[:, 0]
                    n = len(diff)
                    avg = mean(diff)
                    half = float(student_t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)) if n > 1 else None
                    base = mean(data.iloc[:, 0])
                    measured = mean(data.iloc[:, 1])
                    sign = 1 if metric in ("throughput_veh_h", "completion_rate") else -1
                    improvement = sign * (measured - base) / base * 100 if base not in (None, 0) else None
                    paired.append({"scenario": scenario, "controller": controller, "metric": metric, "n_pairs": n,
                                   "baseline": baseline, "difference_other_minus_baseline": avg,
                                   "difference_other_minus_fixed": avg if baseline == "fixed_time" else None, "ci95_low": avg - half if half is not None else None,
                                   "ci95_high": avg + half if half is not None else None, "improvement_pct": improvement})
    return summary, pd.DataFrame(paired)
