"""Per-car driving calibration (car_calibration.yaml), shared by Gazebo and the real PiCar.

Junction maneuvers are open loop (no tape inside the box), so four numbers per car decide whether the car
finds the lane line again after a turn. Measured once per car (and again after changing battery/tyres):

  speed_scale : straight run commanded for 1 m -> metres really driven (battery, motor strength)
  trim_per_m  : same run with trim 0 -> yaw drift per metre (left/right motor mismatch)
  latency_s,  : roll-out after a stop command at two speeds -> command latency and braking decel
  accel_mps2
  turn_scale  : in-place spin commanded for 90 deg -> measured angle / 90

Gazebo: `ros2 launch picar_gazebo calibrate.launch.py` measures them from the simulated pose.
Real car: drive the same tests (picar_lab/car_agent/calib_drive_node.py or by hand), measure with a tape
measure and a protractor, then

  python3 -m picar_lab.car_agent.calibration --out car_calibration.yaml \
      --straight-m 1.00 --drift-deg -8 --roll-slow-m 0.020 --roll-fast-m 0.045 --spin-deg 84,86,85

drift-deg: heading change after the straight run, + = turned left (CCW).
"""
import argparse
import math
from pathlib import Path

import yaml

from picar_lab.car_agent.drive import calibrate_brake2, calibrate_trim, calibrate_turn_scale

DEFAULT = {"trim_per_m": 0.0, "turn_scale": 1.0, "latency_s": 0.05, "accel_mps2": 1.0, "speed_scale": 1.0}
SLOW, FAST = 0.15, 0.25          # cruise speeds of the two brake tests (m/s)
STRAIGHT_V, TURN_RATE_DPS = 0.20, 90.0

# open-loop test commands: name -> ((v, w), seconds). The same sequence runs in Gazebo and on the car.
TESTS = {
    "straight": ((STRAIGHT_V, 0.0), 1.0 / STRAIGHT_V),                     # 1 m
    "brake_slow": ((SLOW, 0.0), 3.0),
    "brake_fast": ((FAST, 0.0), 3.0),
    "spin": ((0.0, math.radians(TURN_RATE_DPS)), 90.0 / TURN_RATE_DPS),    # commanded 90 deg CCW
}


def from_measurements(straight_m, drift_deg, roll_slow_m, roll_fast_m, spin_deg, slow=SLOW, fast=FAST):
    spins = [abs(float(s)) for s in (spin_deg if isinstance(spin_deg, (list, tuple)) else [spin_deg])]
    lat, acc = calibrate_brake2(roll_slow_m, slow, roll_fast_m, fast)
    lat, acc = min(max(lat, 0.0), 0.3), max(acc, 0.2)
    t_run = TESTS["straight"][1]                       # commanded STRAIGHT_V for t_run s, then stop
    # expected distance of a perfect car with this latency/acceleration: ramp up, cruise, ramp down
    ideal = STRAIGHT_V * t_run                         # (accel loss and stop roll-out cancel for a symmetric ramp)
    return {"speed_scale": round(min(1.3, max(0.6, straight_m / ideal)), 4),
            "trim_per_m": round(calibrate_trim(math.radians(drift_deg), straight_m), 4),
            "turn_scale": round(calibrate_turn_scale(sum(spins) / len(spins), 90.0), 4),
            "latency_s": round(lat, 3), "accel_mps2": round(acc, 3),
            "measured": {"straight_m": straight_m, "drift_deg": drift_deg, "roll_slow_m": roll_slow_m,
                         "roll_fast_m": roll_fast_m, "spin_deg": spins, "slow_mps": slow, "fast_mps": fast}}


def load(path):
    """Returns the four CarBrain keyword arguments (defaults when the file is missing or empty)."""
    if not path or not Path(path).expanduser().exists():
        return dict(DEFAULT)
    data = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    return {k: float(data.get(k, v)) for k, v in DEFAULT.items()}


def load_camera(path, cam=None):
    """Camera dict for CameraBrain(camera=...) with the mount from camera_calib.py, when the file has it."""
    from picar_lab.car_agent.camera_calib import camera_from_calibration
    data = {}
    if path and Path(path).expanduser().exists():
        data = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    return camera_from_calibration(data, cam) if "cam_yaw_deg" in data else dict(cam or {})


def save(path, cal, source):
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump({**cal, "source": source}, sort_keys=False, allow_unicode=True))
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="car_calibration.yaml")
    ap.add_argument("--straight-m", type=float, required=True)
    ap.add_argument("--drift-deg", type=float, required=True)
    ap.add_argument("--roll-slow-m", type=float, required=True, help=f"roll-out after stop at {SLOW} m/s")
    ap.add_argument("--roll-fast-m", type=float, required=True, help=f"roll-out after stop at {FAST} m/s")
    ap.add_argument("--spin-deg", required=True, help="measured angle(s) of a commanded 90 deg spin, comma separated")
    a = ap.parse_args()
    cal = from_measurements(a.straight_m, a.drift_deg, a.roll_slow_m, a.roll_fast_m,
                            [float(x) for x in a.spin_deg.split(",")])
    print(yaml.safe_dump(cal, sort_keys=False))
    print("written:", save(a.out, cal, "manual measurements"))


if __name__ == "__main__":
    main()
