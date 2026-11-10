"""Generate the Gazebo Fortress world for the PiCar course from picar_sim/config/course.yaml.

Outputs (in --out):
  course_texture.png   top-view floor: white, black lane tapes, thin approach markers, thick stop lines
                       (no tape inside the junction boxes: cars cross them with JunctionManeuver)
  picar_course.sdf     world with the floor and N PiCar models (diff drive + downward line camera + pose publisher)
  maneuvers.yaml       crossing plan for every (junction, arm_in, arm_out)
  course_meta.yaml     cars (start edge, route), exit beams, stop lines, sensor geometry, car size

python3 -m picar_gazebo.world_gen --cars 3 --seed 1 --out ~/.cache/picar_gazebo/run1
"""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from picar_gazebo.core import (APPROACH_TAPE_M, APPROACHES, CROSS_TAPE_LEN, LINE_TAPE_M, STOP_TAPE_M, Course,
                               sensor_columns)

CAR = {"length": 0.24, "width": 0.17, "wheel_radius": 0.033, "wheel_sep": 0.17, "mass": 0.8,
       "sensor_offset": 0.125, "cam_height": 0.03, "cam_hfov": 1.85, "cam_w": 32, "cam_h": 8,
       "sensor_y": [0.0225, 0.0075, -0.0075, -0.0225]}
COLORS = ["0.85 0.2 0.2 1", "0.2 0.45 0.85 1", "0.2 0.7 0.3 1", "0.9 0.65 0.1 1", "0.6 0.3 0.8 1", "0.1 0.7 0.7 1"]


def draw_texture(course, path, ppm=400):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    W, H = course.width, course.depth
    fig = plt.figure(figsize=(W * ppm / 100, H * ppm / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor("white")
    m2pt = ppm * 72 / 100
    for pts in course.lanes.values():
        seg, segs = [], []
        for p in pts:                                    # break the tape inside junction boxes
            if course.in_box(p, margin=-1e-6):
                if len(seg) > 1:
                    segs.append(seg)
                seg = []
            else:
                seg.append(p)
        if len(seg) > 1:
            segs.append(seg)
        for s in segs:
            s = np.array(s)
            ax.plot(s[:, 0], s[:, 1], color="black", lw=LINE_TAPE_M * m2pt, solid_capstyle="butt")
    for jn in ("J", "J2"):
        for a in APPROACHES:
            far, near, u = course.stopline(jn, a)
            ax.add_patch(_tape(near, u, STOP_TAPE_M, CROSS_TAPE_LEN, Polygon))
            mark, u = course.approach_marker(jn, a)
            ax.add_patch(_tape(mark - u * APPROACH_TAPE_M / 2, u, APPROACH_TAPE_M, CROSS_TAPE_LEN, Polygon))
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    return int(round(W * ppm)), int(round(H * ppm))


def _tape(start, u, thick, length, Polygon):
    n = np.array([-u[1], u[0]])
    a, b = start + n * length / 2, start - n * length / 2
    return Polygon([a, b, b + u * thick, a + u * thick], closed=True, color="black")


def make_routes(n_cars, seed, course, length=24):
    """Start segments spread over the course; cyclic random routes without U-turns (same rule as SUMO)."""
    rng = np.random.default_rng(seed)
    starts = []
    candidates = [(e, f) for e in course.lanes if not e.startswith("N") for f in np.round(np.arange(0.1, 0.91, 0.05), 2)]
    tapes = []
    for jn in ("J", "J2"):
        for a in APPROACHES:
            tapes += [course.stopline(jn, a)[0], course.approach_marker(jn, a)[0]]
    rng.shuffle(candidates)
    used = set()
    for e, f in candidates:
        pts = course.lanes[e]
        L = np.cumsum([0] + [np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])])
        if L[-1] < 1.2 or (e, f) in used:
            continue                                   # connector lanes are too short to start on
        s = f * L[-1]
        i = int(np.searchsorted(L, s) - 1)
        p = pts[i] + (pts[i + 1] - pts[i]) * (s - L[i]) / max(L[i + 1] - L[i], 1e-9)
        if course.in_box(p, margin=0.25) or min(np.linalg.norm(p - q) for q in tapes) < 0.35 \
                or any(np.linalg.norm(p - q) < 0.45 for _, q, _ in starts):
            continue                                   # do not start on/near a marker or stop line
        yaw = math.atan2(*(pts[i + 1] - pts[i])[::-1])
        j0, j1 = int(np.searchsorted(L, s - 0.2)), min(len(pts) - 2, int(np.searchsorted(L, s + 0.2)))
        h0, h1 = pts[j0 + 1] - pts[j0], pts[j1 + 1] - pts[j1]
        if abs(math.atan2(h0[0] * h1[1] - h0[1] * h1[0], h0 @ h1)) > math.radians(3):
            continue                                   # start on a straight so the sensor bar sees the tape
        starts.append((e, p, yaw))
        used.add((e, f))
        if len(starts) == n_cars:
            break
    if len(starts) < n_cars:
        raise ValueError("not enough start slots for that many cars")
    cars = []
    for k, (e, p, yaw) in enumerate(starts):
        route, cur = [e], e
        for i in range(length - 1):
            arm, kind = cur.split("_")
            banned = {arm}
            if i == length - 2:                        # route is cyclic: no U-turn from the last segment back to the first
                banned.add(e.split("_")[0])
            b = rng.choice([x for x in APPROACHES if x not in banned])
            cur = f"{b}_out" if kind == "in" else f"{b}_in"
            route.append(cur)
        cars.append({"name": f"car{k}", "start_edge": e, "x": float(p[0]), "y": float(p[1]), "yaw": float(yaw),
                     "route": route})
    return cars


def _sub(parent, tag, text=None, **attrs):
    el = ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})
    if text is not None:
        el.text = str(text)
    return el


def _inertial(link, mass, ixx, iyy, izz):
    inertial = _sub(link, "inertial")
    _sub(inertial, "mass", mass)
    inertia = _sub(inertial, "inertia")
    for k, v in (("ixx", ixx), ("iyy", iyy), ("izz", izz), ("ixy", 0), ("ixz", 0), ("iyz", 0)):
        _sub(inertia, k, v)


def car_model(world, car, color):
    c = CAR
    name = car["name"]
    m = _sub(world, "model", name=name)
    _sub(m, "pose", f"{car['x']:.4f} {car['y']:.4f} 0 0 0 {car['yaw']:.4f}")
    ch = _sub(m, "link", name="chassis")
    _sub(ch, "pose", "0 0 0.045 0 0 0")
    L, Wd, Hh = c["length"], c["width"], 0.05
    _inertial(ch, c["mass"], c["mass"] * (Wd ** 2 + Hh ** 2) / 12, c["mass"] * (L ** 2 + Hh ** 2) / 12,
              c["mass"] * (L ** 2 + Wd ** 2) / 12)
    for kind in ("collision", "visual"):
        el = _sub(ch, kind, name=f"body_{kind}")
        _sub(_sub(_sub(el, "geometry"), "box"), "size", f"{L} {Wd} {Hh}")
        if kind == "visual":
            mat = _sub(el, "material")
            _sub(mat, "ambient", color)
            _sub(mat, "diffuse", color)
    cam = _sub(ch, "sensor", name="line_cam", type="camera")
    _sub(cam, "pose", f"{c['sensor_offset']} 0 {c['cam_height'] - 0.045} 0 1.5708 0")
    _sub(cam, "always_on", 1)
    _sub(cam, "update_rate", 30)
    _sub(cam, "topic", f"/model/{name}/line_cam")
    camera = _sub(cam, "camera")
    _sub(camera, "horizontal_fov", c["cam_hfov"])
    img = _sub(camera, "image")
    _sub(img, "width", c["cam_w"])
    _sub(img, "height", c["cam_h"])
    _sub(img, "format", "R8G8B8")
    clip = _sub(camera, "clip")
    _sub(clip, "near", 0.005)
    _sub(clip, "far", 1.0)
    r = c["wheel_radius"]
    for side, y in (("left", c["wheel_sep"] / 2), ("right", -c["wheel_sep"] / 2)):
        w = _sub(m, "link", name=f"{side}_wheel")
        _sub(w, "pose", f"0 {y} {r} -1.5708 0 0")
        _inertial(w, 0.05, 2e-5, 2e-5, 3e-5)
        for kind in ("collision", "visual"):
            el = _sub(w, kind, name=f"{side}_{kind}")
            cyl = _sub(_sub(el, "geometry"), "cylinder")
            _sub(cyl, "radius", r)
            _sub(cyl, "length", 0.02)
        j = _sub(m, "joint", name=f"{side}_wheel_joint", type="revolute")
        _sub(j, "parent", "chassis")
        _sub(j, "child", f"{side}_wheel")
        ax = _sub(j, "axis")
        _sub(ax, "xyz", "0 0 1")
        lim = _sub(ax, "limit")
        _sub(lim, "lower", -1e16)
        _sub(lim, "upper", 1e16)
    for nm, x in (("front_caster", 0.09), ("rear_caster", -0.09)):
        cs = _sub(m, "link", name=nm)
        _sub(cs, "pose", f"{x} 0 0.012 0 0 0")
        _inertial(cs, 0.01, 1e-6, 1e-6, 1e-6)
        col = _sub(cs, "collision", name=f"{nm}_collision")
        _sub(_sub(_sub(col, "geometry"), "sphere"), "radius", 0.012)
        ode = _sub(_sub(_sub(col, "surface"), "friction"), "ode")
        _sub(ode, "mu", 0.0)
        _sub(ode, "mu2", 0.0)
        j = _sub(m, "joint", name=f"{nm}_joint", type="ball")
        _sub(j, "parent", "chassis")
        _sub(j, "child", nm)
    dd = _sub(m, "plugin", filename="ignition-gazebo-diff-drive-system", name="ignition::gazebo::systems::DiffDrive")
    for k, v in (("left_joint", "left_wheel_joint"), ("right_joint", "right_wheel_joint"),
                 ("wheel_separation", c["wheel_sep"]), ("wheel_radius", r), ("topic", f"/model/{name}/cmd_vel"),
                 ("odom_publish_frequency", 20), ("max_linear_acceleration", 1.0), ("max_angular_acceleration", 8.0)):
        _sub(dd, k, v)
    pp = _sub(m, "plugin", filename="ignition-gazebo-pose-publisher-system",
              name="ignition::gazebo::systems::PosePublisher")
    for k, v in (("publish_link_pose", "false"), ("publish_model_pose", "true"), ("publish_nested_model_pose", "false"),
                 ("use_pose_vector_msg", "false"), ("update_frequency", 30)):
        _sub(pp, k, v)


def build_world(course, cars, texture_path):
    sdf = ET.Element("sdf", version="1.8")
    world = _sub(sdf, "world", name="picar_course")
    phys = _sub(world, "physics", name="1ms", type="ignored")
    _sub(phys, "max_step_size", 0.001)
    _sub(phys, "real_time_factor", 1.0)
    for fn, nm in (("ignition-gazebo-physics-system", "ignition::gazebo::systems::Physics"),
                   ("ignition-gazebo-user-commands-system", "ignition::gazebo::systems::UserCommands"),
                   ("ignition-gazebo-scene-broadcaster-system", "ignition::gazebo::systems::SceneBroadcaster")):
        _sub(world, "plugin", filename=fn, name=nm)
    sens = _sub(world, "plugin", filename="ignition-gazebo-sensors-system", name="ignition::gazebo::systems::Sensors")
    _sub(sens, "render_engine", "ogre2")
    light = _sub(world, "light", type="directional", name="sun")
    _sub(light, "cast_shadows", "false")
    _sub(light, "pose", "0 0 10 0 0 0")
    _sub(light, "diffuse", "0.9 0.9 0.9 1")
    _sub(light, "specular", "0.1 0.1 0.1 1")
    _sub(light, "direction", "-0.2 0.1 -0.9")
    scene = _sub(world, "scene")
    _sub(scene, "ambient", "0.8 0.8 0.8 1")
    _sub(scene, "background", "0.7 0.7 0.75 1")
    g = _sub(world, "model", name="floor")
    _sub(g, "static", "true")
    link = _sub(g, "link", name="floor_link")
    col = _sub(link, "collision", name="floor_collision")
    pl = _sub(_sub(col, "geometry"), "plane")
    _sub(pl, "normal", "0 0 1")
    _sub(pl, "size", "20 20")
    vis = _sub(link, "visual", name="course_visual")
    _sub(vis, "pose", f"{course.width / 2} {course.depth / 2} 0.0005 0 0 0")
    pl = _sub(_sub(vis, "geometry"), "plane")
    _sub(pl, "normal", "0 0 1")
    _sub(pl, "size", f"{course.width} {course.depth}")
    mat = _sub(vis, "material")
    for k in ("ambient", "diffuse"):
        _sub(mat, k, "1 1 1 1")
    _sub(mat, "specular", "0 0 0 1")
    metal = _sub(_sub(mat, "pbr"), "metal")
    _sub(metal, "albedo_map", str(Path(texture_path).resolve()))
    _sub(metal, "roughness", 1.0)
    _sub(metal, "metalness", 0.0)
    for k, car in enumerate(cars):
        car_model(world, car, COLORS[k % len(COLORS)])
    ET.indent(sdf)
    return ET.tostring(sdf, encoding="unicode", xml_declaration=True)


def generate(out, cars=3, seed=1, cfg=None):
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    course = Course(cfg)
    texture = out / "course_texture.png"
    px = draw_texture(course, texture)
    car_list = make_routes(cars, seed, course)
    (out / "picar_course.sdf").write_text(build_world(course, car_list, texture), encoding="utf-8")
    stop_min = 0.04
    plans = {f"{jn}:{a}:{b}": course.maneuver(jn, a, b, CAR["sensor_offset"], stop_min)
             for jn in ("J", "J2") for a in APPROACHES for b in APPROACHES if a != b}
    (out / "maneuvers.yaml").write_text(yaml.safe_dump({"sensor_offset_m": CAR["sensor_offset"],
                                                        "stop_line_min_m": stop_min, "maneuvers": plans},
                                                       sort_keys=False), encoding="utf-8")
    beams, stops = [], []
    for jn in ("J", "J2"):
        for a in APPROACHES:
            p, u, edge = course.exit_line(jn, a)
            beams.append({"junction": jn, "edge": edge, "point": [float(v) for v in p], "normal": [float(v) for v in u]})
            far, near, u2 = course.stopline(jn, a)
            stops.append({"junction": jn, "arm": a, "far": [float(v) for v in far], "normal": [float(v) for v in u2]})
    cols = sensor_columns(CAR["cam_w"], CAR["cam_hfov"], CAR["cam_height"], CAR["sensor_y"])
    meta = {"cars": car_list, "beams": beams, "stoplines": stops, "car": CAR, "sensor_columns": cols,
            "texture_px": list(px), "course": {"width": course.width, "depth": course.depth, "lane_width": course.lw}}
    (out / "course_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return out


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="~/.cache/picar_gazebo/latest")
    a = ap.parse_args()
    print(generate(a.out, a.cars, a.seed))


if __name__ == "__main__":
    main_cli()
