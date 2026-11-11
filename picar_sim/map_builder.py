"""Build the PiCar course as a metre-scale SUMO network (Real-to-Sim).

Layout (top view, x east, y north, origin = south-west corner of the course):
  outer loop rectangle with rounded corners + a straight centre connector.
  J  = bottom T (controlled): arms W (loop, west half), E (loop, east half), N (connector)
  J2 = top T, run by a fixed rule (queue rule) through its own supervisor - PiCars cannot yield to
       cross traffic by themselves, so the real J2 also needs central GO/STOP.
Each road carries one lane per direction. Approach edges: J <- W_in/E_in/N_in, J2 <- W_out/E_out/N_out.
"""
import argparse
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import os

import yaml

PKG = Path(__file__).resolve().parent
APPROACHES = ("W", "E", "N")


def config_path():
    """course.yaml, or the file named by PICAR_COURSE_CONFIG (a calibrated copy; its SUMO net is kept apart)."""
    return Path(os.environ.get("PICAR_COURSE_CONFIG") or PKG / "config/course.yaml").expanduser()


def maps_dir():
    if not os.environ.get("PICAR_COURSE_CONFIG"):
        return PKG / "maps"
    p = config_path()
    return p.parent / (p.stem + "_maps")


def load_config(path=None):
    return yaml.safe_load(Path(path or config_path()).read_text(encoding="utf-8"))


def _arc(cx, cy, r, a0, a1, n=8):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def road_shapes(cfg):
    c = cfg["course"]
    lw = c["lane_width_m"]
    x0, x1 = lw, c["width_m"] - lw            # road centreline rectangle (road width = 2 lanes)
    y0, y1 = lw, c["depth_m"] - lw
    r, xc = c["corner_radius_m"], c["connector_x_m"]
    if not (x0 + r < xc < x1 - r) or y1 - y0 < 2 * r:
        raise ValueError("Course too small for the corner radius / connector position")
    J, J2 = (xc, y0), (xc, y1)
    west = [J, (x0 + r, y0), *_arc(x0 + r, y0 + r, r, 270, 180)[1:], (x0, y1 - r),
            *_arc(x0 + r, y1 - r, r, 180, 90)[1:], J2]
    east = [J, (x1 - r, y0), *_arc(x1 - r, y0 + r, r, 270, 360)[1:], (x1, y1 - r),
            *_arc(x1 - r, y1 - r, r, 0, 90)[1:], J2]
    return J, J2, {"W": west, "E": east, "N": [J, J2]}


def _length(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def build(cfg=None, out_dir=None):
    import sumolib
    cfg = cfg or load_config()
    c, v = cfg["course"], cfg["vehicle"]
    out = Path(out_dir or maps_dir())
    out.mkdir(parents=True, exist_ok=True)
    J, J2, shapes = road_shapes(cfg)
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="J", x=f"{J[0]:.4f}", y=f"{J[1]:.4f}", type="traffic_light", radius="0.05")
    ET.SubElement(nodes, "node", id="J2", x=f"{J2[0]:.4f}", y=f"{J2[1]:.4f}", type="traffic_light", radius="0.05")
    edges = ET.Element("edges")
    speed = str(max(1.0, 2 * v["speed_mps"]))
    for a, pts in shapes.items():
        prio = "1" if a == "N" else "2"
        fwd = " ".join(f"{x:.4f},{y:.4f}" for x, y in pts)
        rev = " ".join(f"{x:.4f},{y:.4f}" for x, y in reversed(pts))
        # *_out leaves J towards J2, *_in arrives at J from J2. Stop line pulled back by endOffset.
        ET.SubElement(edges, "edge", {"id": f"{a}_out", "from": "J", "to": "J2", "numLanes": "1", "priority": prio,
                                      "speed": speed, "width": str(c["lane_width_m"]), "shape": fwd,
                                      "endOffset": str(c["stopline_offset_m"])})
        ET.SubElement(edges, "edge", {"id": f"{a}_in", "from": "J2", "to": "J", "numLanes": "1", "priority": prio,
                                      "speed": speed, "width": str(c["lane_width_m"]), "shape": rev,
                                      "endOffset": str(c["stopline_offset_m"])})
    cons = ET.Element("connections")
    # vehicle.turn_speed_mps {left, right, straight}: speed inside the box per movement, fitted by calibrate.py so
    # that SUMO's crossing times match the cars (a PiCar stops and spins on the spot; SUMO would sweep through).
    turn_speed = v.get("turn_speed_mps") or {}
    from picar_lab.car_agent.drive import movement
    for a in APPROACHES:          # no U-turns at either T
        for b in APPROACHES:
            if a != b:
                for jn, frm, to in (("J", f"{a}_in", f"{b}_out"), ("J2", f"{a}_out", f"{b}_in")):
                    attrs = {"from": frm, "to": to}
                    if turn_speed:
                        attrs.update(fromLane="0", toLane="0")
                    sp = turn_speed.get(movement(jn, a, b))
                    if sp:
                        attrs["speed"] = f"{float(sp):.4f}"
                    ET.SubElement(cons, "connection", attrs)
    for root, ext in ((nodes, "nod"), (edges, "edg"), (cons, "con")):
        ET.indent(root)
        ET.ElementTree(root).write(out / f"picar_course.{ext}.xml", encoding="utf-8", xml_declaration=True)
    net = out / "picar_course.net.xml"
    cmd = [sumolib.checkBinary("netconvert"),
           "-n", str(out / "picar_course.nod.xml"), "-e", str(out / "picar_course.edg.xml"),
           "-x", str(out / "picar_course.con.xml"), "-o", str(net),
           "--no-turnarounds", "true", "--offset.disable-normalization", "true",
           "--default.lanewidth", str(c["lane_width_m"]), "--default.junctions.radius", "0.05",
           "--geometry.min-radius", "0.1", "--junctions.limit-turn-speed", "-1",
           "--tls.default-type", "static", "--tls.yellow.time", "1", "--tls.allred.time", "1",
           "--no-internal-links", "false", "--junctions.internal-link-detail", "8"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    (out / "netconvert.log").write_text(res.stdout + res.stderr, encoding="utf-8")
    if res.returncode:
        raise RuntimeError(res.stderr)
    info = validate(net, cfg)
    (out / "map_summary.yaml").write_text(yaml.safe_dump(info, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return net, info


def validate(net_path, cfg):
    import sumolib.net
    net = sumolib.net.readNet(str(net_path), withInternal=True)
    info = {"net": str(Path(net_path).name), "lanes": {}, "junction_internal_m": {}}
    for a in APPROACHES:
        for kind in ("in", "out"):
            e = net.getEdge(f"{a}_{kind}")
            info["lanes"][f"{a}_{kind}"] = round(e.getLane(0).getLength(), 3)
    for jn, (src, dst) in {"J": ("in", "out"), "J2": ("out", "in")}.items():
        tls = net.getTLS(jn)
        movements = {(c[0].getEdge().getID(), c[1].getEdge().getID()) for conns in tls.getLinks().values() for c in conns}
        expected = {(f"{a}_{src}", f"{b}_{dst}") for a in APPROACHES for b in APPROACHES if a != b}
        if movements != expected:
            raise ValueError(f"TLS {jn} must control exactly 6 movements, got {sorted(movements)}")
    for a in APPROACHES:
        for b in APPROACHES:
            if a == b:
                continue
            conn = net.getEdge(f"{a}_in").getConnections(net.getEdge(f"{b}_out"))[0]
            via = conn.getViaLaneID()
            length = 0.0
            while via:
                lane = net.getLane(via)
                length += lane.getLength()
                nxt = [c for c in lane.getOutgoing()]
                via = nxt[0].getViaLaneID() if nxt and nxt[0].getViaLaneID() else ""
            info["junction_internal_m"][f"{a}->{b}"] = round(length, 3)
    lw = cfg["course"]["lane_width_m"]
    info["loop_centreline_m"] = round(_length(road_shapes(cfg)[2]["W"]) * 2, 3)
    info["inner_lane_corner_radius_m"] = round(cfg["course"]["corner_radius_m"] - lw / 2, 3)
    info["outer_lane_corner_radius_m"] = round(cfg["course"]["corner_radius_m"] + lw / 2, 3)
    veh = cfg["vehicle"]
    info["segment_capacity_cars"] = segment_capacity(info["lanes"], veh, cfg["course"])
    return info


def segment_capacity(lanes, veh, course):
    """Cars that fit on a segment such that the LAST one admitted can still clear the upstream box
    (rear past the exit beam) while the first one waits at the downstream stop line."""
    usable = {e: L - course["exit_line_m"] - course["arrive_window_m"] for e, L in lanes.items()}
    return {e: max(1, int((u + veh["min_gap_m"]) // (veh["length_m"] + veh["min_gap_m"]))) for e, u in usable.items()}


def plot(net_path, png, cfg=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sumolib.net
    cfg = cfg or load_config()
    net = sumolib.net.readNet(str(net_path), withInternal=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    c = cfg["course"]
    ax.set_aspect("equal")
    ax.set_xlim(-.2, c["width_m"] + .2)
    ax.set_ylim(-.2, c["depth_m"] + .2)
    fig.canvas.draw()
    pts_per_m = ax.get_window_extent().width * 72 / fig.dpi / (c["width_m"] + .4)
    ax.add_patch(plt.Rectangle((0, 0), c["width_m"], c["depth_m"], fill=False, ls="--", color="0.6"))
    for e in net.getEdges(withInternal=True):
        for lane in e.getLanes():
            xs, ys = zip(*lane.getShape())
            internal = e.getFunction() == "internal"
            ax.plot(xs, ys, color="#c9a227" if internal else "#4a5560", lw=1 if internal else lane.getWidth() * pts_per_m * .92,
                    alpha=.35 if not internal else .9,
                    solid_capstyle="butt")
            if not internal:
                ax.annotate("", xy=lane.getShape()[-1], xytext=lane.getShape()[-2],
                            arrowprops=dict(arrowstyle="->", color="white", lw=1.2))
    for a, kind in [(a, k) for a in APPROACHES for k in ("in", "out")]:      # stop lines and approach markers
        lane = net.getEdge(f"{a}_{kind}").getLane(0)
        L = lane.getLength()
        for pos, col in ((L, "#c0392b"), (max(0.0, L - c["approach_zone_m"]), "#2e86c1")):
            x, y = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), pos)
            ax.plot(x, y, "s", color=col, ms=7)
        x, y = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), L)
        if kind == "in":
            ax.text(x, y + .12, f"{a} stop", color="#c0392b", fontsize=8, ha="center")
    for n, label in (("J", "J (RL controlled)"), ("J2", "J2 (fixed queue rule)")):
        x, y = net.getNode(n).getCoord()
        ax.text(x + .45, y + (-.12 if n == "J" else .05), label, fontsize=9, weight="bold")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("PiCar course in SUMO (metres). red=stop line, blue=APPROACH marker; ASSUMED dims", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--out")
    args = p.parse_args()
    cfg = load_config(args.config)
    net, info = build(cfg, args.out)
    plot(net, Path(net).with_name("picar_course_topview.png"), cfg)
    print(yaml.safe_dump(info, sort_keys=False))
