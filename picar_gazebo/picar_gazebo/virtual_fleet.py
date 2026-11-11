"""Virtual real fleet: stand-in for the six physical PiCars until they can be measured.

Every car gets its own hidden "true" hardware (motors, camera mount, sensors, radio), sampled from ranges that are
typical for this class of car (cheap DC gear motors, Pi camera on a printed bracket, IR line bar, HC-SR04, Wi-Fi UDP).
The driving code never sees these numbers. It only gets what a person could measure on the real car:

  measure_motors()  straight run, stop roll-out at two speeds, in-place spins, read with tape measure + protractor
                    (reading errors included) -> trim_per_m, latency_s, accel_mps2, turn_scale
  measure_camera()  one photo with the car parked in a straight lane (placement error included)
                    -> cam_pitch_deg, cam_yaw_deg, cam_y_m

So the whole Real-to-Sim loop (measure -> per-car calibration -> drive -> compare) can be rehearsed in simulation,
and on the real cars the same procedure and file formats apply. Used by picar_gazebo.kinematic_check (2D) and by
the Gazebo launch (motor_model / camera mount per car).

All ranges are ASSUMPTIONS (docs/PICAR_VIRTUAL_FLEET_KO.md lists the reasoning); replace PRIORS with measured
spreads once the cars are available.
"""
import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import yaml

from picar_lab.car_agent.calibration import FAST, SLOW, STRAIGHT_V, from_measurements
from picar_lab.car_agent.camera_calib import calibrate_mount, to_calibration
from picar_lab.car_agent.lane_camera import CAMERA, CameraModel

# name: (kind, a, b, lo, hi)   kind n = normal(mean a, sd b) clipped to [lo, hi]; u = uniform(a, b)
PRIORS = {
    # motors / drivetrain
    "gain_l": ("n", 1.00, 0.04, 0.88, 1.10),        # same PWM -> speed differs +-4 % (1 sd) between cheap gear motors
    "gain_r": ("n", 1.00, 0.04, 0.88, 1.10),
    "speed_scale": ("u", 0.92, 1.02),               # battery charge state at the start of a run
    "sag_per_min": ("u", 0.000, 0.004),             # speed lost per minute as the battery drains
    "deadband_mps": ("u", 0.02, 0.05),              # wheel speed below which the motor does not turn
    "latency_s": ("u", 0.05, 0.12),                 # command -> wheels (ROS + PWM + motor time constant)
    "acc": ("u", 0.7, 1.3),                         # m/s^2
    "alpha": ("u", 5.0, 10.0),                      # rad/s^2
    "spin_eff": ("u", 0.82, 0.97),                  # in-place spin: achieved / commanded yaw rate (tyre scrub)
    "spin_sd": ("u", 0.010, 0.030),                 # spin-to-spin variation of that ratio
    "spin_stick_prob": ("u", 0.00, 0.08),           # a wheel hesitates at the start of a spin (-8 % that time)
    "ripple": ("u", 0.01, 0.03),                    # slow random speed fluctuation (floor, gears)
    # front camera mount and image
    "cam_pitch_err_deg": ("n", 0.0, 1.5, -3.0, 3.0),
    "cam_yaw_err_deg": ("n", 0.0, 1.5, -3.0, 3.0),
    "cam_y_err_m": ("n", 0.0, 0.004, -0.008, 0.008),
    "cam_z_err_m": ("n", 0.0, 0.003, -0.006, 0.006),
    "cam_noise": ("u", 0.01, 0.04),                 # pixel noise sd (image range 0..1)
    "cam_gain": ("u", 0.80, 1.15),                  # exposure / lighting
    "cam_delay_s": ("u", 0.03, 0.09),               # capture -> lane result
    "cam_drop_prob": ("u", 0.00, 0.05),             # lost frames
    # other sensors
    "ir_flip_prob": ("u", 0.000, 0.010),            # per channel per reading
    "sonar_sd_m": ("u", 0.005, 0.015),
    "sonar_drop_prob": ("u", 0.00, 0.05),
    # radio
    "udp_loss": ("u", 0.00, 0.05),
    "udp_delay_s": ("u", 0.005, 0.040),
}
NOMINAL = {"gain_l": 1.0, "gain_r": 1.0, "speed_scale": 1.0, "sag_per_min": 0.0, "deadband_mps": 0.03,
           "latency_s": 0.05, "acc": 1.0, "alpha": 8.0, "spin_eff": 1.0, "spin_sd": 0.0, "spin_stick_prob": 0.0,
           "ripple": 0.0, "cam_pitch_err_deg": 0.0, "cam_yaw_err_deg": 0.0, "cam_y_err_m": 0.0, "cam_z_err_m": 0.0,
           "cam_noise": 0.0, "cam_gain": 1.0, "cam_delay_s": 0.0, "cam_drop_prob": 0.0, "ir_flip_prob": 0.0,
           "sonar_sd_m": 0.0, "sonar_drop_prob": 0.0, "udp_loss": 0.0, "udp_delay_s": 0.0}
# reading errors of a careful person (1 sd): tape measure, protractor / phone compass, parking the car by eye
MEASURE = {"tape_m": 0.002, "angle_deg": 1.5, "park_lateral_m": 0.003, "park_yaw_deg": 0.7, "repeats": 3}


def _draw(spec, rng):
    if spec[0] == "u":
        return float(rng.uniform(spec[1], spec[2]))
    return float(np.clip(rng.normal(spec[1], spec[2]), spec[3], spec[4]))


def sample_fleet(n=6, seed=7, scale=1.0, worst=None):
    """n car truths. scale widens (>1) or narrows (<1) every spread around its centre; worst={key: value} pins
    parameters (stress tests)."""
    rng = np.random.default_rng(seed)
    fleet = []
    for i in range(n):
        t = {"name": f"car{i}"}
        for k, spec in PRIORS.items():
            v = _draw(spec, rng)
            if scale != 1.0:
                mid = spec[1] if spec[0] == "n" else (spec[1] + spec[2]) / 2
                v = mid + (v - mid) * scale
            t[k] = round(v, 5)
        t.update(worst or {})
        fleet.append(t)
    return fleet


def nominal_fleet(n=6):
    return [{"name": f"car{i}", **NOMINAL} for i in range(n)]


def save_fleet(path, fleet, note=""):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"note": note or "virtual real fleet: hidden truth, ASSUMED ranges (virtual_fleet.PRIORS)",
                        "cars": fleet}, f, sort_keys=False, allow_unicode=True)


def load_fleet(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["cars"]


def true_camera(truth, cam=None):
    c = {**CAMERA, **(cam or {})}
    c["pitch_deg"] = c["pitch_deg"] + truth["cam_pitch_err_deg"]
    c["yaw_deg"] = truth["cam_yaw_err_deg"]
    c["y"] = truth["cam_y_err_m"]
    c["z"] = c["z"] + truth["cam_z_err_m"]
    return c


class VirtualCar:
    """Hidden hardware of one car: plant physics and what its sensors and radio really deliver."""

    def __init__(self, truth, geom, seed=0):
        self.t, self.geom = {**NOMINAL, **truth}, geom
        self.rng = np.random.default_rng(seed)
        self.spinning, self.spin_e, self.rip = False, 1.0, 0.0
        self.frames = deque()
        self.rx, self.tx = [], []

    # ---------------------------------------------------------------- plant
    def physics(self, k, t, dt):
        tr, sep = self.t, self.geom["wheel_sep"]
        while k.cmd_queue and k.cmd_queue[0][0] <= t + 1e-9:
            _, cv, cw = k.cmd_queue.pop(0)
            spin = abs(cw) > 0.3 and abs(cv) < 0.02
            if spin and not self.spinning:                    # every spin scrubs a little differently
                e = self.rng.normal(tr["spin_eff"], tr["spin_sd"]) if tr["spin_sd"] > 0 else tr["spin_eff"]
                if self.rng.random() < tr["spin_stick_prob"]:
                    e *= 0.92
                self.spin_e = float(np.clip(e, 0.5, 1.05))
            self.spinning = spin
            scale = max(0.5, tr["speed_scale"] - tr["sag_per_min"] * t / 60.0)
            half = cw * sep / 2 * (self.spin_e if spin else 1.0)
            vl, vr = cv - half, cv + half
            vl = 0.0 if abs(vl) < tr["deadband_mps"] else vl * tr["gain_l"] * scale
            vr = 0.0 if abs(vr) < tr["deadband_mps"] else vr * tr["gain_r"] * scale
            k.target = ((vl + vr) / 2, (vr - vl) / sep)
        tv, tw = getattr(k, "target", (0.0, 0.0))
        if tr["ripple"] > 0:                                   # slow fluctuation (correlation time ~0.5 s)
            self.rip += (-self.rip / 0.5) * dt + tr["ripple"] * math.sqrt(2 * dt / 0.5) * self.rng.normal()
            tv *= 1.0 + self.rip
        k.v += float(np.clip(tv - k.v, -tr["acc"] * dt, tr["acc"] * dt))
        k.w += float(np.clip(tw - k.w, -tr["alpha"] * dt, tr["alpha"] * dt))
        k.x += k.v * math.cos(k.yaw) * dt
        k.y += k.v * math.sin(k.yaw) * dt
        k.yaw += k.w * dt

    # ---------------------------------------------------------------- sensors
    def camera(self, img, t):
        """img: ideal render for the true mount. Returns the frame the lane code gets now (None: no new frame)."""
        tr = self.t
        if tr["cam_noise"] > 0 or tr["cam_gain"] != 1.0:
            img = np.clip(img * tr["cam_gain"] + self.rng.normal(0, tr["cam_noise"], img.shape), 0.0, 1.0)
        self.frames.append((t, img))
        out = None
        while self.frames and self.frames[0][0] <= t - tr["cam_delay_s"] + 1e-9:
            out = self.frames.popleft()[1]
        if out is not None and self.rng.random() < tr["cam_drop_prob"]:
            return None
        return out

    def ir(self, values):
        p = self.t["ir_flip_prob"]
        return [1 - v if p > 0 and self.rng.random() < p else v for v in values]

    def sonar(self, rng_m):
        tr = self.t
        if self.rng.random() < tr["sonar_drop_prob"]:
            return None                                        # no echo this cycle: brain keeps its last decision
        return max(0.02, rng_m + self.rng.normal(0, tr["sonar_sd_m"])) if tr["sonar_sd_m"] > 0 else rng_m

    # ---------------------------------------------------------------- radio (both directions)
    def radio(self, queue, data, t):
        if self.rng.random() >= self.t["udp_loss"]:
            queue.append((t + self.t["udp_delay_s"] * self.rng.uniform(0.5, 1.5), data))

    @staticmethod
    def due(queue, t):
        out = [d for tt, d in queue if tt <= t + 1e-9]
        queue[:] = [(tt, d) for tt, d in queue if tt > t + 1e-9]
        return out


# -------------------------------------------------------------------- measurements a person can make
def _drive(vc, cv, cw, dur, settle=2.0):
    """Open-loop command for dur seconds, then stop. Returns final pose and the pose when stop was issued."""
    k = SimpleNamespace(x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0, cmd_queue=[])
    t, stop = 0.0, None
    lat = vc.t["latency_s"]
    while t < dur + settle:
        if abs(t / 0.05 - round(t / 0.05)) < 1e-6:
            on = t < dur - 1e-9
            if not on and stop is None:
                stop = (k.x, k.y)
            k.cmd_queue.append((t + lat, *((cv, cw) if on else (0.0, 0.0))))
        vc.physics(k, t, 0.01)
        t = round(t + 0.01, 6)
    return k, stop


def measure_motors(truth, geom, turn_rate_dps=45.0, seed=0, measure=None):
    """The calibration tests of picar_lab.car_agent.calibration, read off with tape measure and protractor.
    Returns (calibration dict, report)."""
    ms = {**MEASURE, **(measure or {})}
    rng = np.random.default_rng(seed + 500)
    vc = VirtualCar(truth, geom, seed + 600)
    n = int(ms["repeats"])
    tape = lambda v: v + rng.normal(0, ms["tape_m"])
    ang = lambda v: v + rng.normal(0, ms["angle_deg"])
    straight, drift, slow, fast, spins = [], [], [], [], []
    for _ in range(n):
        k, _s = _drive(vc, STRAIGHT_V, 0.0, 1.0 / STRAIGHT_V)
        straight.append(tape(math.hypot(k.x, k.y)))
        drift.append(ang(math.degrees(k.yaw)))
        k, st = _drive(vc, SLOW, 0.0, 3.0)
        slow.append(max(0.0, tape(math.hypot(k.x - st[0], k.y - st[1]))))
        k, st = _drive(vc, FAST, 0.0, 3.0)
        fast.append(max(0.0, tape(math.hypot(k.x - st[0], k.y - st[1]))))
        for sign in (1.0, -1.0):                             # counter-clockwise and clockwise
            k, _s = _drive(vc, 0.0, sign * math.radians(turn_rate_dps), 90.0 / turn_rate_dps)
            spins.append(abs(ang(math.degrees(k.yaw))))
    cal = from_measurements(float(np.mean(straight)), float(np.mean(drift)), float(np.mean(slow)),
                            float(np.mean(fast)), [float(s) for s in spins])
    cal.pop("measured")
    cal["turn_rate_dps"] = float(turn_rate_dps)
    report = {"straight_m": [round(v, 3) for v in straight], "drift_deg": [round(v, 1) for v in drift],
              "roll_slow_m": [round(v, 3) for v in slow], "roll_fast_m": [round(v, 3) for v in fast],
              "spin_deg": [round(v, 1) for v in spins], "spin_sd_deg": round(float(np.std(spins)), 2),
              "cruise_mps": round(float(np.mean(straight)) / (1.0 / STRAIGHT_V) * 0.25 / STRAIGHT_V, 3)}
    return cal, report


def measure_camera(truth, geom, renderer_factory, pose, lane_width, seed=0, measure=None):
    """One frame with the car parked (by eye) at `pose` = middle of a straight lane -> mount calibration."""
    ms = {**MEASURE, **(measure or {})}
    rng = np.random.default_rng(seed + 700)
    vc = VirtualCar({**truth, "cam_delay_s": 0.0, "cam_drop_prob": 0.0}, geom, seed + 800)
    x0, y0, yaw = pose
    rend = renderer_factory(true_camera(truth, geom.get("front_cam")))
    mounts, errs = [], []
    for _ in range(int(ms["repeats"])):                      # park again for every photo; the results are averaged
        off = rng.normal(0, ms["park_lateral_m"])
        yaw_p = yaw + math.radians(rng.normal(0, ms["park_yaw_deg"]))
        x, y = x0 - math.sin(yaw) * off, y0 + math.cos(yaw) * off
        mount, err = calibrate_mount(vc.camera(rend.render(x, y, yaw_p), 0.0), lane_width, geom.get("front_cam"))
        mounts.append(mount)
        errs.append(err)
    mean = {k: float(np.mean([m[k] for m in mounts])) for k in mounts[0]}
    return to_calibration(mean), {"rms_mm": round(max(errs) * 1000, 1), "photos": len(mounts)}


def renderer_factory_for(tex, bounds):
    from picar_gazebo.cam_render import CamRenderer
    return lambda cam: CamRenderer(tex, bounds, CameraModel(cam))
