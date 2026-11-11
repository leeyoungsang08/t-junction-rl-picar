"""2D kinematic pre-check of the Gazebo setup (no ROS, no Gazebo needed).

Each car: unicycle model with acceleration limits and motor imperfections, a front camera rendered from the
generated course_texture.png (the floor Gazebo renders) through the same CameraModel, IR sensors read from the
texture, virtual ultrasonic, and the real CameraBrain + CarAgent (--driver ir: the legacy centre-tape CarBrain).
Cars talk to the real CentralServer over UDP; exit beams come from the same exit lines as in Gazebo.

python3 -m picar_gazebo.kinematic_check --cars 3 --seconds 180 --out /tmp/kin --video
"""
import argparse
import json
import math
import socket
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from picar_gazebo.core import Course, crossed, footprint, forward_range, lane_signed, overlap
from picar_gazebo.world_gen import generate
from picar_lab.car_agent.agent import CarAgent
from picar_lab.car_agent.brain import CarBrain
from picar_lab.car_agent.camera_brain import CameraBrain
from picar_lab.car_agent.camera_calib import camera_from_calibration
from picar_lab.car_agent.lane_camera import CameraModel
from picar_gazebo.cam_render import CamRenderer
from picar_lab.car_agent.protocol import VERSION, decode, encode
from picar_sim.central_server import CentralServer


# brightness of the car bodies in the camera (world_gen.COLORS as gray under Gazebo's light): some are darker
# than the detector's threshold, some are not
CAR_GRAY = [0.38, 0.45, 0.39, 0.50, 0.51, 0.45]

# "unknown real car": the controller only learns these through calibrate()
MOTOR = {"latency_s": 0.05, "deadband": 0.03, "gain_l": 1.0, "gain_r": 0.97, "acc": 1.0, "alpha": 8.0}


class KinCar:
    def __init__(self, spec, plans, car, speed, motor, calib, driver="camera", approach_after=None, lane_width=0.21,
                 rig=None, turn_rate_dps=90.0, stop_after=None):
        self.name, self.x, self.y, self.yaw = spec["name"], spec["x"], spec["y"], spec["yaw"]
        self.v = self.w = 0.0
        self.agent = CarAgent(self.name, spec["route"])
        self.driver = driver
        self.rig = rig                      # virtual_fleet.VirtualCar: this car's hidden hardware (None: ideal)
        self.t_enter = self.t_go = None     # field log: crossing start / waiting for the wheels to move
        self.cruise = [0.0, 0.0]            # field log: sum of speed, time at full commanded speed
        if driver == "camera":
            cam = camera_from_calibration(calib, car.get("front_cam")) if "cam_yaw_deg" in calib else car.get("front_cam")
            self.brain = CameraBrain(self.agent, plans, approach_after or {}, speed_mps=speed, lane_width=lane_width,
                                     camera=cam, approach_in_m=spec.get("approach_in_m"), stop_after=stop_after,
                                     **{"turn_rate_dps": turn_rate_dps, **calib})
        else:
            self.brain = CarBrain(self.agent, plans, speed_mps=speed, stop_min_m=0.04, **calib)
        self.car, self.motor = car, motor
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.setblocking(False)
        self.cmd_queue = []
        self.prev_rear = None

    def sensor_points(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        off = self.car["sensor_offset"]
        return [(self.x + c * off - s * y, self.y + s * off + c * y) for y in self.car["sensor_y"]]

    def pose(self):
        return (self.x, self.y, self.yaw)


def physics_step(k, t, dt, motor, car):
    """Unicycle with command latency, wheel deadband, left/right gain mismatch and accel limits."""
    while k.cmd_queue and k.cmd_queue[0][0] <= t + 1e-9:
        _, cv, cw = k.cmd_queue.pop(0)
        vl, vr = cv - cw * car["wheel_sep"] / 2, cv + cw * car["wheel_sep"] / 2
        vl = 0.0 if abs(vl) < motor["deadband"] else vl * motor["gain_l"]
        vr = 0.0 if abs(vr) < motor["deadband"] else vr * motor["gain_r"]
        k.target = ((vl + vr) / 2, (vr - vl) / car["wheel_sep"])
    tv, tw = getattr(k, "target", (0.0, 0.0))
    k.v += float(np.clip(tv - k.v, -motor["acc"] * dt, motor["acc"] * dt))
    k.w += float(np.clip(tw - k.w, -motor["alpha"] * dt, motor["alpha"] * dt))
    k.x += k.v * math.cos(k.yaw) * dt
    k.y += k.v * math.sin(k.yaw) * dt
    k.yaw += k.w * dt


def calibrate(motor, car):
    """Runs picar_lab.car_agent.calibration.TESTS on the plant and measures like a person with a tape
    measure and protractor would on the real car (same code path as Gazebo's calibrate_node)."""
    from types import SimpleNamespace
    from picar_lab.car_agent.calibration import TESTS, from_measurements
    meas = {"spin": []}
    for name in ("straight", "spin", "brake_slow", "spin", "brake_fast", "spin"):
        (cv, cw), dur = TESTS[name]
        k = SimpleNamespace(x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0, cmd_queue=[])
        t, stop = 0.0, None
        while t < dur + 2.0:
            if abs(t / 0.05 - round(t / 0.05)) < 1e-6:            # 20 Hz controller
                on = t < dur - 1e-9
                if not on and stop is None:
                    stop = (k.x, k.y)
                k.cmd_queue.append((t + motor["latency_s"], *((cv, cw) if on else (0.0, 0.0))))
            physics_step(k, t, 0.01, motor, car)
            t = round(t + 0.01, 6)
        if name == "straight":
            meas["straight_m"], meas["drift_deg"] = math.hypot(k.x, k.y), math.degrees(k.yaw)
        elif name == "spin":
            meas["spin"].append(math.degrees(k.yaw))
        else:
            meas[f"roll_{name.split('_')[1]}_m"] = math.hypot(k.x - stop[0], k.y - stop[1])
    cal = from_measurements(meas["straight_m"], meas["drift_deg"], meas["roll_slow_m"], meas["roll_fast_m"],
                            meas["spin"])
    cal.pop("measured")
    return cal


def run(cars=3, seconds=180, seed=1, out="/tmp/kin", policy=None, mode="shadow", video=False, port=0,
        calibrated=True, motor=None, ir_flip=0.0, driver="camera", truths=None, calibs=None, real_cfg=None,
        map_cfg=None, turn_rate_dps=90.0, speed=0.25, cars_visible=True):
    """truths: per-car hidden hardware (virtual_fleet); calibs: per-car calibration dicts (what was measured).
    real_cfg: the course as it really is (floor, beams, checks); map_cfg: the course as the cars and the plan
    table believe it is (maneuvers). Both default to picar_sim/config/course.yaml."""
    out = Path(out)
    generate(out, cars, seed, real_cfg)
    meta = yaml.safe_load((out / "course_meta.yaml").read_text())
    if map_cfg is not None:                                # cars plan with the map, not with the real floor
        generate(out / "map", cars, seed, map_cfg)
        man = yaml.safe_load((out / "map" / "maneuvers.yaml").read_text())
    else:
        man = yaml.safe_load((out / "maneuvers.yaml").read_text())
    plans, approach_after = man["maneuvers"], man.get("approach_after_exit_m", {})
    tex = np.asarray(Image.open(out / "course_texture.png").convert("L"), dtype=float) / 255.0
    H, W = tex.shape
    course = Course(real_cfg)
    map_lw = Course(map_cfg).lw if map_cfg is not None else course.lw
    tx0, tx1, ty0, ty1 = course.tex_bounds()
    ppm = W / (tx1 - tx0)
    car = meta["car"]
    motor = motor or dict(MOTOR)
    calib = calibrate(motor, car) if calibrated else {"trim_per_m": 0.0, "turn_scale": 1.0, "accel_mps2": 1e9, "latency_s": 0.0}
    server = CentralServer("127.0.0.1", port, policy, mode, cfg=map_cfg)
    beam_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fleet, renderers = [], {}
    for i, c in enumerate(meta["cars"]):
        rig = cal_i = None
        if truths is not None:
            from picar_gazebo.virtual_fleet import VirtualCar, true_camera
            rig = VirtualCar(truths[i], car, seed * 100 + i)
            renderers[c["name"]] = CamRenderer(tex, course.tex_bounds(), CameraModel(true_camera(truths[i], car.get("front_cam"))))
        cal_i = dict(calibs[i]) if calibs is not None else calib
        mot_i = {**motor, "latency_s": rig.t["latency_s"]} if rig else motor
        fleet.append(KinCar(c, plans, car, speed, mot_i, cal_i, driver, approach_after, map_lw, rig, turn_rate_dps,
                            man.get("stop_after_exit_m")))
    renderer = CamRenderer(tex, course.tex_bounds(), CameraModel(car.get("front_cam"))) if driver == "camera" else None
    lane_tol = meta.get("lane_tolerance_m", 0.08)
    lateral = {}                                           # signed lane offset samples per car (straight / curve)
    field = []                                             # what a person with a stopwatch could log on the real course
    for k in fleet:
        k.agent.register(0.0)
    dt_phys, dt_ctrl = 0.01, 0.05
    t, next_ctrl, frames = 0.0, 0.0, []
    m = {"collisions": 0, "lane_departures": 0, "illegal_crossings": 0, "crossings": 0, "faults": 0,
         "exit_events": 0, "stop_errors_m": [], "max_all_stopped_s": 0.0}
    idle_since = None
    contact, departed, stopped, close = set(), set(), {}, set()
    seq = 0
    noise = np.random.default_rng(seed + 1000)

    def dark(px, py):
        u, v = int((px - tx0) * ppm), int((ty1 - py) * ppm)
        return 1 if 0 <= u < W and 0 <= v < H and tex[v, u] < 0.5 else 0

    while t < seconds - 1e-9:
        if t >= next_ctrl - 1e-9:
            next_ctrl += dt_ctrl
            for k in fleet:
                while True:
                    try:
                        raw, _ = k.sock.recvfrom(2048)
                    except BlockingIOError:
                        break
                    if k.rig:
                        k.rig.radio(k.rig.rx, raw, t)
                        continue
                    msg = decode(raw)
                    if msg:
                        k.agent.on_packet(msg, t)
                for raw in (k.rig.due(k.rig.rx, t) if k.rig else []):
                    msg = decode(raw)
                    if msg:
                        k.agent.on_packet(msg, t)
                values = [dark(*p) for p in k.sensor_points()]
                if ir_flip > 0:
                    values = [1 - x if noise.random() < ir_flip else x for x in values]
                rng = forward_range(k.pose(), [o.pose() for o in fleet if o is not k], car["length"], car["width"])
                n_ev = len(k.brain.events)
                if k.rig:
                    values, rng = k.rig.ir(values), k.rig.sonar(rng)
                    seen = [(o.x, o.y, o.yaw, CAR_GRAY[j % len(CAR_GRAY)], car["length"], car["width"])
                            for j, o in enumerate(fleet) if o is not k] if cars_visible else ()
                    img = k.rig.camera(renderers[k.name].render(k.x, k.y, k.yaw, seen), t)
                    v, w = k.brain.step(img, values, t, range_m=rng)
                elif renderer is not None:
                    v, w = k.brain.step(renderer.render(k.x, k.y, k.yaw), values, t, range_m=rng)
                else:
                    v, w = k.brain.step(values, t, range_m=rng)
                for _, text in k.brain.events[n_ev:]:
                    if text.startswith("approach"):
                        k.t_appr = t
                    if text.startswith("enter"):
                        if text.split()[1] == "J" and getattr(k, "t_appr", None) is not None:
                            field.append({"car": k.name, "metric": "j_request_to_entry_s", "value": round(t - k.t_appr, 2)})
                        k.t_appr = None
                        m["crossings"] += 1
                        m["illegal_crossings"] += "legal=False" in text
                        k.t_enter, k.t_go, k.move = t, t, text.split()[1] + ":" + text.split()[2]
                    if text.startswith("left box"):
                        sgn, _st, herr = lane_signed(course, np.array([k.x, k.y]), k.yaw)
                        m.setdefault("exit_log", []).append({"car": k.name, "t": round(t, 1), "move": getattr(k, "move", ""),
                                                             "offset_mm": round(sgn * 1000), "heading_deg": round(herr, 1)})
                    if text.startswith("FAULT"):
                        m["faults"] += 1
                        m.setdefault("fault_log", []).append({"t": round(t, 1), "car": k.name, "what": text,
                                                              "pose": [round(q, 3) for q in k.pose()]})
                if v >= 0.95 * speed and abs(w) < 0.3 and k.brain.mode == "follow":
                    k.cruise[0] += abs(k.v) * dt_ctrl
                    k.cruise[1] += dt_ctrl
                k.cmd_queue.append((t + k.motor["latency_s"], v, w))
                for msg in k.agent.tick(t):
                    if k.rig:
                        k.rig.radio(k.rig.tx, encode(msg), t)
                    else:
                        k.sock.sendto(encode(msg), server.address)
                for raw in (k.rig.due(k.rig.tx, t) if k.rig else []):
                    k.sock.sendto(raw, server.address)
            for jn in ("J", "J2"):
                seq += 1
                beam_sock.sendto(encode({"v": VERSION, "type": "beam_heartbeat", "node": f"kin-{jn}", "seq": seq,
                                         "junction": jn, "ok": True}), server.address)
            server.poll(t)
            if video and int(round(t / dt_ctrl)) % 4 == 0:
                frames.append((t, [(k.x, k.y, k.yaw, k.brain.mode) for k in fleet]))
        for k in fleet:                                  # physics
            if k.rig:
                k.rig.physics(k, t, dt_phys)
            else:
                physics_step(k, t, dt_phys, k.motor, car)
            if k.t_go is not None and abs(k.v) > 0.02:
                field.append({"car": k.name, "metric": "startup_delay_s", "value": round(t - k.t_go, 3)})
                k.t_go = None
            rear = np.array([k.x - math.cos(k.yaw) * car["length"] / 2, k.y - math.sin(k.yaw) * car["length"] / 2])
            for b in meta["beams"]:
                if crossed(k.prev_rear, rear, np.array(b["point"]), np.array(b["normal"]), course.lw / 2):
                    seq += 1
                    beam_sock.sendto(encode({"v": VERSION, "type": "exit_event", "node": "kin", "seq": seq,
                                             "junction": b["junction"], "edge": b["edge"]}), server.address)
                    m["exit_events"] += 1
                    if k.t_enter is not None:
                        field.append({"car": k.name, "metric": "crossing_time_s", "value": round(t - k.t_enter, 2),
                                      "move": k.move})
                        k.t_enter = None
            k.prev_rear = rear
            centre = np.array([k.x, k.y])
            if int(round(t / dt_phys)) % 10 == 0:                     # lane check at 10 Hz
                if k.brain.mode == "follow" and abs(k.v) > 0.1 and not course.in_box(centre, 0.15):
                    sgn, st, _h = lane_signed(course, centre, k.yaw)
                    lateral.setdefault(k.name, {True: [], False: []})[st].append(sgn)
                off = (not course.in_box(centre, 0.05)) and course.lane_distance(centre) > lane_tol
                if off and k.name not in departed:
                    m["lane_departures"] += 1
                    m.setdefault("departure_log", []).append(
                        {"t": round(t, 1), "car": k.name, "pose": [round(q, 3) for q in k.pose()],
                         "dist": round(course.lane_distance(centre), 3), "mode": k.brain.mode,
                         "edge": k.agent.edge, "since_box": None if getattr(k.brain, "s_left_box", None) is None
                         else round(k.brain.odo.s - k.brain.s_left_box, 2)})
                departed.discard(k.name) if not off else departed.add(k.name)
            if abs(k.v) < 0.005 and k.brain.mode == "hold":
                stopped.setdefault(k.name, t)
                if t - stopped[k.name] > 0.5 and stopped[k.name] >= 0:
                    front = centre + np.array([math.cos(k.yaw), math.sin(k.yaw)]) * car["length"] / 2
                    for s in meta["stoplines"]:
                        n = np.array(s["normal"])
                        d = float(np.dot(np.array(s["far"]) - front, n))
                        if -0.15 < d < 0.4 and abs(float(np.dot(front - np.array(s["far"]), [n[1], -n[0]]))) < course.lw / 2:
                            m["stop_errors_m"].append(round(d, 4))
                            field.append({"car": k.name, "metric": "stop_error_m", "value": round(d, 4)})
                            break
                    stopped[k.name] = -1.0                      # logged for this stop
            elif abs(k.v) >= 0.005:
                stopped.pop(k.name, None)
        if all(abs(k.v) < 0.005 and abs(k.w) < 0.02 for k in fleet):          # nobody moves: deadlock clock
            idle_since = t if idle_since is None else idle_since
            m["max_all_stopped_s"] = max(m["max_all_stopped_s"], round(t - idle_since, 1))
        else:
            idle_since = None
        for i, a in enumerate(fleet if int(round(t / dt_phys)) % 5 == 0 else []):   # contacts at 20 Hz
            for b in fleet[i + 1:]:
                hit = overlap(footprint(*a.pose(), car["length"], car["width"]), footprint(*b.pose(), car["length"], car["width"]))
                key = (a.name, b.name)
                if hit and key not in contact:
                    m["collisions"] += 1
                    m.setdefault("collision_log", []).append(
                        {"t": round(t, 2), "cars": [a.name, b.name],
                         "poses": [[round(x, 3) for x in a.pose()], [round(x, 3) for x in b.pose()]],
                         "modes": [a.brain.mode, b.brain.mode],
                         "phases": [a.brain.maneuver.phase if a.brain.maneuver else "",
                                    b.brain.maneuver.phase if b.brain.maneuver else ""]})
                    contact.add(key)
                elif not hit:
                    contact.discard(key)
                # near miss: less than 1.5 cm between the bodies. In Gazebo (and on the floor) a touch is not a
                # number: the two cars lock and push each other round (fleet run 018), so the margin matters
                pad = 0.015
                near = overlap(footprint(*a.pose(), car["length"] + pad, car["width"] + pad),
                               footprint(*b.pose(), car["length"] + pad, car["width"] + pad))
                if near and key not in close:
                    close.add(key)
                    jd = min(float(np.hypot(a.x - cx, a.y - cy)) for cx, cy in course.centres.values())
                    m["near_misses"] = m.get("near_misses", 0) + 1
                    m.setdefault("near_log", []).append({"t": round(t, 2), "cars": [a.name, b.name],
                                                         "where": "junction" if jd < 0.6 else "road",
                                                         "pose": [round(x, 3) for x in a.pose()]})
                elif not near:
                    close.discard(key)
        t = round(t + dt_phys, 6)
    errs = m["stop_errors_m"]
    for k in fleet:
        if k.cruise[1] > 5.0:
            field.append({"car": k.name, "metric": "speed_mps", "value": round(k.cruise[0] / k.cruise[1], 4)})
    m["field"] = field
    m["lateral_mm"] = {c: {("straight" if st else "curve"): [round(float(np.mean(v)) * 1000, 1), round(float(np.std(v)) * 1000, 1),
                                                              round(float(np.max(np.abs(v))) * 1000, 1)]
                           for st, v in d.items() if len(v) > 20} for c, d in lateral.items()}   # mean, sd, max |.|
    m["throughput_per_min"] = round(m["crossings"] / seconds * 60, 2)
    m.update(seconds=seconds, cars=cars, calibration=calib if calibs is None else calibs, server=server.stats, anomalies=sorted({a["kind"] for a in server.anomalies}),
             stop_error_mean_m=round(float(np.mean(errs)), 4) if errs else None,
             stop_error_max_abs_m=round(float(np.max(np.abs(errs))), 4) if errs else None,
             modes={k.name: k.brain.mode for k in fleet},
             events={k.name: [f"{tt:.1f} {e}" for tt, e in k.brain.events[-6:]] for k in fleet})
    (out / "kinematic_metrics.json").write_text(json.dumps(m, indent=1))
    server.close()
    if video:
        animate(out, frames, course, car)
    return m


def animate(out, frames, course, car):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter
    from matplotlib.patches import Polygon
    img = Image.open(out / "course_texture.png")
    fig, ax = plt.subplots(figsize=(8.75, 5.5))
    ax.imshow(img, extent=course.tex_bounds())
    ax.set_xlim(0, course.width)
    ax.set_ylim(0, course.depth)
    ax.set_aspect("equal")
    ax.axis("off")
    colors = ["#d33", "#36c", "#3a3", "#e90", "#93c", "#1aa"]
    patches = [ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, color=colors[i % 6], alpha=.85))
               for i in range(len(frames[0][1]))]
    title = ax.set_title("")

    def draw(i):
        t, states = frames[i]
        for p, (x, y, yaw, mode) in zip(patches, states):
            p.set_xy(footprint(x, y, yaw, car["length"], car["width"]))
        title.set_text(f"t = {t:5.1f} s   " + "  ".join(f"car{j}:{s[3]}" for j, s in enumerate(states)))
        return patches + [title]

    try:
        writer = FFMpegWriter(fps=10)
        path = out / "kinematic_check.mp4"
        import matplotlib.animation as anim
        anim.FuncAnimation(fig, draw, frames=len(frames), blit=False).save(path, writer=writer, dpi=90)
    except (FileNotFoundError, RuntimeError):
        path = out / "kinematic_check.gif"
        import matplotlib.animation as anim
        anim.FuncAnimation(fig, draw, frames=len(frames), blit=False).save(path, writer=PillowWriter(fps=10), dpi=70)
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="/tmp/kin")
    ap.add_argument("--policy")
    ap.add_argument("--mode", default="shadow")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--no-calibration", action="store_true", help="drive with trim 0 / turn_scale 1")
    ap.add_argument("--driver", choices=["camera", "ir"], default="camera")
    a = ap.parse_args()
    print(json.dumps(run(a.cars, a.seconds, a.seed, a.out, a.policy, a.mode, a.video,
                         calibrated=not a.no_calibration, driver=a.driver), indent=1))
