import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("T_JUNCTION_PROJECT", HERE.parent / "T_Junction_RL"))
sys.path.insert(0, str(PROJECT))
from src.utils import binary, sha256, write_json

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
HORIZON = 1200
SEEDS = range(81000, 81012)
DIRECTIONS = "WES"
ODS = [a + b for a in DIRECTIONS for b in DIRECTIONS if a != b]
# Each block: seconds [start,end), total nominal rate (veh/h), W/E/S shares.
SCENARIOS = {
    "balanced": [(0, 1080, 690, [1, 1, 1])],
    "reversal": [(0, 360, 930, [7, 2, 1]), (360, 720, 930, [2, 7, 1]),
                 (720, 1080, 930, [1, 2, 7])],
    "south_pulse": [(0, 360, 650, [4, 4, 2]), (360, 720, 1480, [1, 1, 12]),
                    (720, 1080, 650, [4, 4, 2])],
}
MODELS = tuple(f"direct_rainbow_lite_seed{n}" for n in (42, 43, 44))
BASELINES = ("fixed20", "queue_proxy", "queue_wait")
FULL = PROJECT / "outputs/v3_research/full"
SOURCE_NET = PROJECT / "maps/t_junction.net.xml"
MAPS = HERE / "maps"
DEMAND = HERE / "demand"
OUT = HERE / "runs"

def xmlfile(path, root):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

def generate_map():
    MAPS.mkdir(exist_ok=True)
    coords = {"J": (0, 0), "W": (-390, -82), "E": (202, 78), "S": (104, -118)}
    speeds = {"W": 13.89, "E": 11.11, "S": 8.33}
    width = {"W": 3.5, "E": 3.2, "S": 3.0}
    nodes, edges, connections = [ET.Element(tag) for tag in ("nodes", "edges", "connections")]
    for d, (x, y) in coords.items():
        ET.SubElement(nodes, "node", id=d, x=str(x), y=str(y), type="traffic_light" if d == "J" else "dead_end")
    for d in DIRECTIONS:
        for a, b, suffix in [(d, "J", "in"), ("J", d, "out")]:
            ET.SubElement(edges, "edge", {"id": f"{d}_{suffix}", "from": a, "to": b, "numLanes": "1",
                                         "speed": str(speeds[d]), "width": str(width[d])})
    for a, b in ODS:
        ET.SubElement(connections, "connection", {"from": f"{a}_in", "to": f"{b}_out", "fromLane": "0", "toLane": "0"})
    for suffix, root in [("nod", nodes), ("edg", edges), ("con", connections)]:
        xmlfile(MAPS / f"skew_t.{suffix}.xml", root)
    cmd = [binary("netconvert"), "-n", "skew_t.nod.xml", "-e", "skew_t.edg.xml", "-x", "skew_t.con.xml",
           "-o", "skew_t.net.xml", "--offset.disable-normalization", "true", "--no-turnarounds", "true",
           "--write-license", "false"]
    subprocess.run(cmd, cwd=MAPS, check=True)
    target = MAPS / "skew_t.net.xml"
    assert sha256(target) != sha256(SOURCE_NET)
    info = {}
    for label, net in [("source", SOURCE_NET), ("new", target)]:
        network = ET.parse(net)
        links = [(c.get("from"), c.get("to")) for c in network.findall("connection") if c.get("tl") == "J"]
        assert set(links) == {(f"{a}_in", f"{b}_out") for a, b in ODS} and len(links) == 6
        info[label] = {"sha256": sha256(net), "approaches": {
            e.get("id"): {k: e.find("lane").get(k) for k in ("length", "speed", "width", "shape")}
            for e in network.findall("edge") if e.get("id").endswith("_in")}}
    return target, info

def generate_demand(scenario, seed):
    DEMAND.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    route = DEMAND / f"{scenario}_{seed}.rou.xml"
    doc = ET.Element("routes")
    ET.SubElement(doc, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5",
                  minGap="2.5", maxSpeed="13.89", speedDev="0")
    for a, b in ODS:
        ET.SubElement(doc, "route", id=a+b, edges=f"{a}_in {b}_out")
    vehicles = {}
    for lo, hi, rate, shares in SCENARIOS[scenario]:
        shares = np.asarray(shares, dtype=float)
        time = float(lo)
        while True:
            time += rng.exponential(3600/rate)
            if time >= hi:
                break
            a = str(rng.choice(list(DIRECTIONS), p=shares/shares.sum()))
            b = str(rng.choice([d for d in DIRECTIONS if d != a]))
            depart = min(round(time, 3), hi-0.001)
            vid = f"v{len(vehicles):05d}"
            vehicles[vid] = dict(requested_depart=depart, origin=a, od=a+b)
            ET.SubElement(doc, "vehicle", id=vid, type="car", route=a+b, depart=str(depart), departLane="0", departSpeed="max")
    xmlfile(route, doc)
    write_json(route.with_suffix(".json"), dict(horizon_seconds=HORIZON, seed=seed, scenario=scenario, vehicles=vehicles,
        requested_count=len(vehicles), route_sha256=sha256(route)))
    return route

def run_task(task):
    from copy import deepcopy
    import libsumo
    from src.baselines import direct_phase_baseline
    from src.environment import TJunctionEnv
    from src.inference import Policy
    geometry, net, scenario, seed, route, env_cfg = task
    origin_start = libsumo.start
    def start(command, *a, **kw):
        command = list(command)
        command[command.index("-n")+1] = str(net)
        return origin_start([*command, "--time-to-teleport", "-1", "--collision.check-junctions", "true",
                             "--collision.action", "warn"], *a, **kw)
    rows = []
    libsumo.start = start
    try:
        for controller in (*MODELS, *BASELINES):
            cfg = deepcopy(env_cfg)
            cfg["simulation"]["horizon_seconds"] = HORIZON
            if controller == "fixed20":
                cfg["simulation"]["decision_seconds"] = 1
            policy = Policy(FULL / controller / "checkpoints/best_model.pt") if controller in MODELS else None
            directory = OUT / geometry / scenario / str(seed) / controller
            capture = (geometry, scenario, seed) == ("new", "reversal", 81000) and controller in (MODELS[0], "queue_proxy", "fixed20")
            env = TJunctionEnv(cfg, route, seed, directory, capture=capture)
            try:
                _, info = env.reset(seed=seed)
                groups = tuple(env.controller.groups)
                while True:
                    if policy:
                        p = policy.predict(info["raw_state"])
                        assert p["valid_actions"] == info["action_mask"].tolist()
                        act = p["action_id"]
                    else:
                        act = direct_phase_baseline({"fixed20":"fixed_time", "queue_proxy":"max_pressure_proxy",
                                                     "queue_wait":"rule_observed"}[controller], info, cfg)
                    _, _, _, done, info = env.step(act)
                    if done:
                        break
                native = env.tripinfo.read_text(encoding="utf-8")
                assert "Eclipse SUMO libsumo 1.28.0" in native.split("-->", 1)[0]
                assert net.name in native.split("-->", 1)[0] and '<collision.check-junctions value="true"' in native
                rows.append(dict(geometry=geometry, scenario=scenario, seed=seed, controller=controller,
                                 net_sha256=sha256(net), route_sha256=sha256(route), tls_groups="".join(groups), **info["metrics"]))
            finally:
                env.close()
    finally:
        libsumo.start = origin_start
    return rows

def main():
    new_map, maps = generate_map()
    plan = json.loads((FULL / "final_eval/evaluation_plan.json").read_text())
    assert maps["source"]["sha256"] == plan["net_sha256"]
    models = {c: sha256(FULL / c / "checkpoints/best_model.pt") for c in MODELS}
    assert models == {c: plan["model_sha256"][c] for c in MODELS}
    route_rows = {}
    tasks = []
    for scenario, seed in itertools.product(SCENARIOS, SEEDS):
        route = generate_demand(scenario, seed)
        route_rows[route.name] = sha256(route)
        for geometry, net in [("source", SOURCE_NET), ("new", new_map)]:
            tasks.append((geometry, net, scenario, seed, route, plan["environment"][MODELS[0]]))
    assert len(set(route_rows.values())) == 36
    write_json(HERE / "plan.json", dict(horizon_seconds=HORIZON, scenarios=SCENARIOS, seeds=list(SEEDS), maps=maps,
        route_hashes=route_rows, model_hashes=models, baselines=BASELINES, environment=plan["environment"][MODELS[0]],
        options=["--time-to-teleport", "-1", "--collision.check-junctions", "true", "--collision.action", "warn"]))
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for batch in pool.map(run_task, tasks):
            rows.extend(batch)
            if len(rows) % 36 == 0:
                print(f"SUMO {len(rows)}/{len(tasks)*6}", flush=True)
    with (HERE / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
