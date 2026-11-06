import csv
import hashlib
import io
import itertools
import json
import math
import re
import statistics as st
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

WORK = Path(__file__).resolve().parent
OUTPUTS = WORK.parents[1] / "outputs"
PROJECT = "T_Junction_RL/"
V3 = PROJECT + "outputs/v3_research/"
REFERENCE = OUTPUTS / "T_Junction_RL_Project_Validated_Final.zip"
REPRODUCED = OUTPUTS / "T_Junction_RL_Reproduced_Results.zip"

def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))

def group(label):
    return re.sub(r"_seed\d+$", "", label)

def checked_eval(package, prefix, plan_root):
    plan = json.loads(package.read(prefix + "evaluation_plan.json"))
    routes = json.loads(package.read(prefix + "demand_manifest.json"))
    table = csv_rows(package.read(prefix + "seed_level_results.csv"))
    if prefix.startswith(PROJECT):
        assert hashlib.sha256(package.read(PROJECT + "maps/t_junction.net.xml")).hexdigest() == plan["net_sha256"]
    for controller, digest in plan["model_sha256"].items():
        assert hashlib.sha256(package.read(plan_root + controller + "/checkpoints/best_model.pt")).hexdigest() == digest
    demands = {}
    for entry in routes:
        route = prefix + entry["route"]
        raw = package.read(route)
        manifest = json.loads(package.read(route[:-4] + ".json"))
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"] == manifest["route_sha256"]
        xml = ET.fromstring(raw)
        requests = {v.attrib["id"]: float(v.attrib["depart"]) for v in xml.findall("vehicle")}
        assert len(requests) == len(xml.findall("vehicle")) == manifest["requested_count"]
        assert requests == {k: v["requested_depart"] for k, v in manifest["vehicles"].items()}
        assert entry["seed"] == manifest["seed"] and manifest["split"] == "final"
        demands[entry["scenario"], entry["seed"]] = (requests, entry["sha256"])
    keys = {(r["scenario"], int(r["seed"]), r["controller"]) for r in table}
    expected = set(itertools.product(plan["scenarios"], range(plan["seed_start"], plan["seed_start"] + plan["seeds"]), plan["controllers"]))
    assert len(keys) == len(table) == 990 and keys == expected
    checked = []
    max_delay_delta = 0
    header_dates = []
    for row in table:
        scenario, seed, label = row["scenario"], int(row["seed"]), row["controller"]
        requests, digest = demands[scenario, seed]
        assert row["route_sha256"] == digest
        episode = prefix + row["episode_dir"] + "/"
        trip_raw = package.read(episode + "tripinfo.xml")
        header = trip_raw.split(b"-->", 1)[0]
        assert b"Eclipse SUMO libsumo 1.28.0" in header
        assert f'<seed value="{seed}"'.encode() in header
        header_dates.append(re.search(rb"generated on (\S+)", header).group(1).decode())
        trips = {r.attrib["id"]: r.attrib for r in ET.fromstring(trip_raw).findall("tripinfo")}
        assert trips.keys() <= requests.keys()
        assert all(float(t["depart"]) >= 0 for t in trips.values())
        entered, arrived = len(trips), sum(float(t["arrival"]) >= 0 for t in trips.values())
        uninserted = requests.keys() - trips.keys()
        horizon = plan["horizon_seconds"]
        uninserted_delay = math.fsum(horizon - requests[vid] for vid in uninserted)
        delay = (math.fsum(float(t["waitingTime"]) + float(t["departDelay"]) for t in trips.values()) + uninserted_delay) / len(requests)
        max_delay_delta = max(max_delay_delta, abs(float(row["cohort_delay_s"]) - delay))
        assert math.isclose(float(row["cohort_delay_s"]), delay, rel_tol=0, abs_tol=1e-9)
        assert math.isclose(float(row["completion_rate"]), arrived / len(requests), abs_tol=1e-12)
        assert math.isclose(float(row["uninserted_delay_s"]), uninserted_delay, rel_tol=0, abs_tol=1e-9)
        metrics = json.loads(package.read(episode + "metrics.json"))
        for key, value in {
            "requested_vehicles": len(requests), "departed_vehicles": entered,
            "completed_vehicles": arrived, "uninserted_vehicles": len(uninserted),
            "unfinished_vehicles": entered - arrived, "incomplete_vehicles": len(requests) - arrived,
        }.items():
            assert int(row[key]) == metrics[key] == value, (scenario, seed, label, key)
        trace = package.read(episode + "timeseries.csv").splitlines()
        records = [line for line in trace[1:] if line.strip()]
        assert len(records) == horizon
        end = dict(zip(trace[0].decode().split(","), records[-1].decode().split(",")))
        assert int(end["time"]) == horizon
        assert int(end["cumulative_departed"]) == entered
        assert int(end["cumulative_arrived"]) == arrived
        assert all(int(row[k]) == metrics[k] for k in ("collision_events", "teleport_events"))
        checked.append(dict(scenario=scenario, seed=seed, controller=label, group=group(label),
                            delay=delay, requested=len(requests), arrived=arrived,
                            teleports=int(row["teleport_events"]), collisions=int(row["collision_events"])))
    evidence = {"rows": len(checked), "unique_demand_files": len(demands), "sumo_seconds": len(checked)*plan["horizon_seconds"],
                "tripinfo_date_range": [min(header_dates),max(header_dates)], "max_delay_vs_csv": max_delay_delta}
    return checked, evidence, set(d for _, d in demands.values())

def stats(rows):
    groups = sorted({r["group"] for r in rows})
    grouped = {name: [r for r in rows if r["group"] == name] for name in groups}
    overall = {name: {
        "macro_delay": st.mean(r["delay"] for r in batch),
        "macro_completion": st.mean(r["arrived"]/r["requested"] for r in batch),
        "request_weighted_delay": math.fsum(r["delay"]*r["requested"] for r in batch)/sum(r["requested"] for r in batch),
        "teleports": sum(r["teleports"] for r in batch), "collisions": sum(r["collisions"] for r in batch),
    } for name, batch in grouped.items()}
    scenario_groups = defaultdict(list)
    for r in rows:
        scenario_groups[r["scenario"], r["seed"], r["group"]].append(r["delay"])
    seed_means = {key: st.mean(values) for key, values in scenario_groups.items()}
    scenarios = sorted({r["scenario"] for r in rows})
    paired = {}
    for scenario in scenarios:
        differences = [seed_means[scenario, seed, "direct_rainbow_lite"] - seed_means[scenario, seed, "max_pressure_proxy"]
                       for seed in range(70000, 70010)]
        delta = st.mean(differences)
        half = 2.2621571628540993 * st.stdev(differences) / math.sqrt(len(differences))
        paired[scenario] = {"direct": st.mean(seed_means[scenario, seed, "direct_rainbow_lite"] for seed in range(70000, 70010)),
                            "proxy": st.mean(seed_means[scenario, seed, "max_pressure_proxy"] for seed in range(70000, 70010)),
                            "delta": delta, "ci95": [delta-half, delta+half],
                            "n": len(differences), "direct_wins": sum(d<0 for d in differences)}
    return overall, paired

def main():
    with ZipFile(REFERENCE) as package:
        original, original_info, final_hashes = checked_eval(package, V3+"full/final_eval/", V3+"full/")
        dev = defaultdict(set)
        dev_seeds = defaultdict(set)
        for name in package.namelist():
            if not name.startswith(V3) or not name.endswith(".rou.json"):
                continue
            item = json.loads(package.read(name))
            if item["split"] in {"train", "validation"}:
                dev[item["split"]].add(item["route_sha256"])
                dev_seeds[item["split"]].add(item["seed"])
        assert not final_hashes.intersection(set.union(*dev.values()))
        assert not dev_seeds["train"].intersection(dev_seeds["validation"] | set(range(70000, 70010)))
        assert not dev_seeds["validation"].intersection(range(70000,70010))
        print("Delivery archive: 990 SUMO rows independently calculated", flush=True)
    with ZipFile(REPRODUCED) as package:
        reproduced, reproduced_info, other_hashes = checked_eval(package, "full/final_eval/", "full/")
        assert final_hashes == other_hashes
        comparison = json.loads(package.read("reproduction_verification.json"))
        assert comparison["heldout_evaluation"]["rows"] == 990
        print("Reproduction archive: 990 SUMO rows independently calculated", flush=True)
    keys = lambda r: (r["scenario"], r["seed"], r["controller"])
    before, after = sorted(original, key=keys), sorted(reproduced, key=keys)
    assert before == after
    overall, paired = stats(original)
    report = {"delivery": original_info, "reproduction": reproduced_info,
              "identical_independent_rows": len(original),
              "distinct_development_route_hashes": {k: len(v) for k,v in dev.items()},
              "overall": overall, "paired_direct_vs_proxy": paired,
              "reproduction_checkpoint_and_validation_check": comparison["training"]}
    (WORK/"archived_summary.json").write_text(json.dumps(report,indent=2), encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
