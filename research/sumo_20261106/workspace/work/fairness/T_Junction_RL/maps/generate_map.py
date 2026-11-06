import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
from src.utils import ROOT, DIRECTIONS, atomic_text, binary, load_yaml

OD = [(a, b) for a in DIRECTIONS for b in DIRECTIONS if a != b]


def xml_write(root, path):
    ET.indent(root)
    atomic_text(path, ET.tostring(root, encoding="unicode") + "\n")


def validate_map(path):
    tree = ET.parse(path)
    edges = {e.get("id"): e for e in tree.findall("edge") if e.get("function") != "internal"}
    assert set(edges) == {f"{d}_{kind}" for d in DIRECTIONS for kind in ("in", "out")}
    assert all(len(e.findall("lane")) == 1 for e in edges.values())
    junctions = {j.get("id"): j for j in tree.findall("junction")}
    assert set(junctions) >= {"J", "W", "E", "S"}
    assert float(junctions["W"].get("x")) < float(junctions["J"].get("x")) < float(junctions["E"].get("x"))
    assert float(junctions["S"].get("y")) < float(junctions["J"].get("y"))
    links = [c for c in tree.findall("connection") if c.get("tl") == "J"]
    assert {(c.get("from"), c.get("to")) for c in links} == {(f"{a}_in", f"{b}_out") for a, b in OD}
    assert len(links) == 6 and len({c.get("linkIndex") for c in links}) == 6
    return links


def generate(config=None, directory=None):
    cfg = (config or load_yaml("base.yaml"))["map"]
    if cfg["tls_id"] != "J":
        raise ValueError("This single-junction map uses tls_id J")
    directory = Path(directory or ROOT / "maps")
    directory.mkdir(parents=True, exist_ok=True)
    length = cfg["length_m"]
    nodes = ET.Element("nodes")
    for name, x, y in [("J", 0, 0), ("W", -length, 0), ("E", length, 0), ("S", 0, -length)]:
        ET.SubElement(nodes, "node", id=name, x=str(x), y=str(y), type="traffic_light" if name == "J" else "priority")
    edges = ET.Element("edges")
    for d in DIRECTIONS:
        for kind, source, dest in [("in", d, "J"), ("out", "J", d)]:
            ET.SubElement(edges, "edge", {"id": f"{d}_{kind}", "from": source, "to": dest,
                          "numLanes": "1", "speed": str(cfg["speed_mps"]), "width": str(cfg["lane_width_m"])})
    cons = ET.Element("connections")
    for a, b in OD:
        ET.SubElement(cons, "connection", {"from": f"{a}_in", "to": f"{b}_out", "fromLane": "0", "toLane": "0"})
    for root, ext in [(nodes, "nod"), (edges, "edg"), (cons, "con")]:
        xml_write(root, directory / f"t_junction.{ext}.xml")
    net = directory / "t_junction.net.xml"
    args = [binary("netconvert"), "--node-files", str(directory / "t_junction.nod.xml"),
            "--edge-files", str(directory / "t_junction.edg.xml"), "--connection-files", str(directory / "t_junction.con.xml"),
            "--no-turnarounds", "true", "--offset.disable-normalization", "true", "--output-file", str(net)]
    with TemporaryDirectory(dir=directory) as temporary:
        candidate = Path(temporary) / net.name
        args[-1] = str(candidate)
        result = subprocess.run(args, capture_output=True, text=True)
        atomic_text(directory / "netconvert.log", result.stdout + result.stderr)
        result.check_returncode()
        links = validate_map(candidate)
        tree = ET.parse(candidate)
    for logic in tree.findall("tlLogic"):
        tree.getroot().remove(logic)
    logic = ET.Element("tlLogic", id="J", type="static", programID="safe_default", offset="0")
    n = max(int(c.get("linkIndex")) for c in links) + 1
    signal_cfg = (config or load_yaml("base.yaml"))["signal"]
    for d in DIRECTIONS:
        for color, seconds in [("G", signal_cfg["fixed_green"]), ("y", signal_cfg["yellow"]), ("r", signal_cfg["all_red"])]:
            state = ["r"] * n
            for c in links:
                if c.get("from") == f"{d}_in":
                    state[int(c.get("linkIndex"))] = color
            ET.SubElement(logic, "phase", duration=str(seconds), state="".join(state))
    tree.getroot().insert(0, logic)
    xml_write(tree.getroot(), net)
    sumocfg = ET.Element("configuration")
    ET.SubElement(ET.SubElement(sumocfg, "input"), "net-file", value="t_junction.net.xml")
    xml_write(sumocfg, directory / "t_junction.sumocfg")
    validate_map(net)
    return net


if __name__ == "__main__":
    print(generate())
