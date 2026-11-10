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

from picar_gazebo.core import Course, crossed, footprint, forward_range, overlap
from picar_gazebo.world_gen import generate
from picar_lab.car_agent.agent import CarAgent
from picar_lab.car_agent.brain import CarBrain
from picar_lab.car_agent.camera_brain import CameraBrain
from picar_lab.car_agent.lane_camera import CameraModel
from picar_gazebo.cam_render import CamRenderer
from picar_lab.car_agent.protocol import VERSION, decode, encode
from picar_sim.central_server import CentralServer


# "unknown real car": the controller only learns these through calibrate()
MOTOR = {"latency_s": 0.05, "deadband": 0.03, "gain_l": 1.0, "gain_r": 0.97, "acc": 1.0, "alpha": 8.0}


class KinCar:
    def __init__(self, spec, plans, car, speed, motor, calib, driver="camera", approach_after=None, lane_width=0.21):
        self.name, self.x, self.y, self.yaw = spec["name"], spec["x"], spec["y"], spec["yaw"]
        self.v = self.w = 0.0
        self.agent = CarAgent(self.name, spec["route"])
        self.driver = driver
        if driver == "camera":
            self.brain = CameraBrain(self.agent, plans, approach_after or {}, speed_mps=speed, lane_width=lane_width,
                                     camera=car.get("front_cam"), approach_in_m=spec.get("approach_in_m"), **calib)
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
        calibrated=True, motor=None, ir_flip=0.0, driver="camera"):
    out = Path(out)
    generate(out, cars, seed)
    meta = yaml.safe_load((out / "course_meta.yaml").read_text())
    man = yaml.safe_load((out / "maneuvers.yaml").read_text())
    plans, approach_after = man["maneuvers"], man.get("approach_after_exit_m", {})
    tex = np.asarray(Image.open(out / "course_texture.png").convert("L"), dtype=float) / 255.0
    H, W = tex.shape
    course = Course()
    tx0, tx1, ty0, ty1 = course.tex_bounds()
    ppm = W / (tx1 - tx0)
    car = meta["car"]
    motor = motor or dict(MOTOR)
    calib = calibrate(motor, car) if calibrated else {"trim_per_m": 0.0, "turn_scale": 1.0, "accel_mps2": 1e9, "latency_s": 0.0}
    server = CentralServer("127.0.0.1", port, policy, mode)
    beam_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fleet = [KinCar(c, plans, car, 0.25, motor, calib, driver, approach_after, course.lw) for c in meta["cars"]]
    renderer = CamRenderer(tex, course.tex_bounds(), CameraModel(car.get("front_cam"))) if driver == "camera" else None
    lane_tol = meta.get("lane_tolerance_m", 0.08)
    for k in fleet:
        k.agent.register(0.0)
    dt_phys, dt_ctrl = 0.01, 0.05
    t, next_ctrl, frames = 0.0, 0.0, []
    m = {"collisions": 0, "lane_departures": 0, "illegal_crossings": 0, "crossings": 0, "faults": 0,
         "exit_events": 0, "stop_errors_m": []}
    contact, departed, stopped = set(), set(), {}
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
                    msg = decode(raw)
                    if msg:
                        k.agent.on_packet(msg, t)
                values = [dark(*p) for p in k.sensor_points()]
                if ir_flip > 0:
                    values = [1 - x if noise.random() < ir_flip else x for x in values]
                rng = forward_range(k.pose(), [o.pose() for o in fleet if o is not k], car["length"], car["width"])
                n_ev = len(k.brain.events)
                if renderer is not None:
                    v, w = k.brain.step(renderer.render(k.x, k.y, k.yaw), values, t, range_m=rng)
                else:
                    v, w = k.brain.step(values, t, range_m=rng)
                for _, text in k.brain.events[n_ev:]:
                    if text.startswith("enter"):
                        m["crossings"] += 1
                        m["illegal_crossings"] += "legal=False" in text
                    if text.startswith("FAULT"):
                        m["faults"] += 1
                k.cmd_queue.append((t + motor["latency_s"], v, w))
                for msg in k.agent.tick(t):
                    k.sock.sendto(encode(msg), server.address)
            for jn in ("J", "J2"):
                seq += 1
                beam_sock.sendto(encode({"v": VERSION, "type": "beam_heartbeat", "node": f"kin-{jn}", "seq": seq,
                                         "junction": jn, "ok": True}), server.address)
            server.poll(t)
            if video and int(round(t / dt_ctrl)) % 4 == 0:
                frames.append((t, [(k.x, k.y, k.yaw, k.brain.mode) for k in fleet]))
        for k in fleet:                                  # physics
            physics_step(k, t, dt_phys, motor, car)
            rear = np.array([k.x - math.cos(k.yaw) * car["length"] / 2, k.y - math.sin(k.yaw) * car["length"] / 2])
            for b in meta["beams"]:
                if crossed(k.prev_rear, rear, np.array(b["point"]), np.array(b["normal"]), course.lw / 2):
                    seq += 1
                    beam_sock.sendto(encode({"v": VERSION, "type": "exit_event", "node": "kin", "seq": seq,
                                             "junction": b["junction"], "edge": b["edge"]}), server.address)
                    m["exit_events"] += 1
            k.prev_rear = rear
            centre = np.array([k.x, k.y])
            if int(round(t / dt_phys)) % 10 == 0:                     # lane check at 10 Hz
                off = (not course.in_box(centre, 0.05)) and course.lane_distance(centre) > lane_tol
                if off and k.name not in departed:
                    m["lane_departures"] += 1
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
                            break
                    stopped[k.name] = -1.0                      # logged for this stop
            elif abs(k.v) >= 0.005:
                stopped.pop(k.name, None)
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
        t = round(t + dt_phys, 6)
    errs = m["stop_errors_m"]
    m.update(seconds=seconds, cars=cars, calibration=calib, server=server.stats, anomalies=sorted({a["kind"] for a in server.anomalies}),
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
