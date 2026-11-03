from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from src.utils import ROOT, atomic_text, load_yaml, sha256, write_json


def generate_traffic(scenario, seed, horizon, split="train", directory=None):
    cfg = load_yaml("scenarios.yaml")
    bounds = cfg["splits"][split]
    if not bounds["start"] <= seed < bounds["start"] + bounds["count"]:
        raise ValueError(f"Seed {seed} does not belong to split {split}")
    spec = cfg["scenarios"][scenario]
    if spec.get("test_only") and split != "test":
        raise ValueError("Unseen scenario is test-only")
    root_dir = Path(directory or ROOT / "routes/generated") / split
    path = root_dir / f"{scenario}_seed{seed}_t{horizon}.rou.xml"
    rng = np.random.default_rng(seed)
    events = []
    for segment in spec.get("segments", [[0, 1, 1]]):
        start, stop, factor = segment[:3]
        weights = np.asarray(segment[3] if len(segment) == 4 else spec["weights"], dtype=float)
        weights /= weights.sum()
        t = start * horizon
        rate = spec["rate_veh_h"] * factor / 3600
        while rate > 0:
            t += rng.exponential(1 / rate)
            if t >= stop * horizon:
                break
            events.append((round(float(t), 3), str(rng.choice(cfg["od_order"], p=weights))))
    events.sort()
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5", minGap="2.5", maxSpeed="13.89", speedDev="0")
    for od in cfg["od_order"]:
        a, b = od.split("_")
        ET.SubElement(root, "route", id=od, edges=f"{a}_in {b}_out")
    vehicles = {}
    for i, (depart, od) in enumerate(events):
        vid = f"v{i:06d}"
        ET.SubElement(root, "vehicle", id=vid, type="car", route=od, depart=str(depart), departLane="0", departSpeed="max")
        vehicles[vid] = {"requested_depart": depart, "origin": od[0], "od": od}
    ET.indent(root)
    atomic_text(path, ET.tostring(root, encoding="unicode") + "\n")
    write_json(path.with_suffix(".json"), {"scenario": scenario, "seed": seed, "split": split,
              "horizon_seconds": horizon, "route_sha256": sha256(path), "vehicles": vehicles})
    return path
