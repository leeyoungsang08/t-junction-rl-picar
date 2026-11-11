"""Fit SUMO parameters from PiCar low-speed logs and check sim-vs-real on held-out runs.

Input CSV (one row per measurement; template: picar_sim/config/field_measurements.template.csv):
  run_id, metric, value
metrics: speed_mps, accel_mps2, decel_mps2, startup_delay_s, stop_error_m, crossing_time_s,
         crossing_straight_s, crossing_left_s, crossing_right_s (stop line -> exit beam, per movement),
         command_latency_s, ir_detect_prob, status_loss_prob, course_width_m, course_depth_m,
         stopline_offset_m, corner_radius_m
Runs are split 60/40 by run_id (never by sample) into calibration / validation sets.

python -m picar_sim.calibrate --csv field.csv --out picar_sim/config/course.calibrated.yaml
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from picar_sim.map_builder import APPROACHES, build, load_config

NOMINAL = {"speed_mps": ("vehicle", "speed_mps"), "accel_mps2": ("vehicle", "accel_mps2"),
           "decel_mps2": ("vehicle", "decel_mps2"), "startup_delay_s": ("vehicle", "startup_delay_s"),
           "course_width_m": ("course", "width_m"), "course_depth_m": ("course", "depth_m"),
           "stopline_offset_m": ("course", "stopline_offset_m"), "corner_radius_m": ("course", "corner_radius_m")}
RANGES = ["speed_mps", "accel_mps2", "decel_mps2", "startup_delay_s", "command_latency_s",
          "ir_detect_prob", "status_loss_prob"]


def split_runs(df, frac=0.6):
    h = df.run_id.astype(str).map(lambda r: int(hashlib.sha1(r.encode()).hexdigest(), 16) % 1000 / 1000)
    return df[h < frac], df[h >= frac]


MOVES = {"straight": ("W", "E_out"), "left": ("W", "N_out"), "right": ("N", "W_out")}   # at J


def sim_crossing_time(cfg, approach="W", trials=5, out=None):
    """Single car released from the stop line: time from GO to EXIT of the conflict box."""
    from picar_sim.env import PicarEnv
    import copy
    cfg = copy.deepcopy(cfg)
    cfg["signal"]["unprotected_right"] = False     # the measured car must wait at the line for its GO
    env = PicarEnv(cfg)
    env.auto_route = False
    times = []
    for s in range(trials):
        env.reset(seed=9000 + s, cars=1)
        car = "car0"
        k = env.k
        cur = k.vehicle.getRoadID(car)
        out = out or ("E_out" if approach != "E" else "W_out")
        path = [cur] if cur == f"{approach}_in" else list(k.simulation.findRoute(cur, f"{approach}_in").edges)
        k.vehicle.setRoute(car, path + [out])
        go, was_in = None, False
        for _ in range(int(120 / env.dt)):
            if car not in k.vehicle.getIDList():
                break
            jn, st, arm = env.truth.get(car, (None, "", None))
            if jn == "J2" and arm:                 # on the way there: the top T shows green to wherever the car is
                env.sup2.phase = APPROACHES.index(arm)
            if jn == "J" and st == "ARRIVE" and arm == approach and go is None:
                go = env.t
                env.sup.phase = APPROACHES.index(approach)
            if go is None:      # drive the car round to the measured approach; other arms pass freely
                permits = {a: ("STOP" if a == approach else "GO") for a in "WEN"}
            else:
                permits = {a: ("GO" if a == approach else "STOP") for a in "WEN"}
            env._substep({"J": permits, "J2": {a: "GO" for a in "WEN"}})   # single car: J2 open
            jn, st, _ = env.truth.get(car, (None, "", None))
            if go is not None and jn == "J" and st == "IN":
                was_in = True
            elif was_in:                       # rear passed the exit beam
                times.append(env.t - go)
                break
        env.close()
    return float(np.mean(times)) if times else float("nan")


def calibrate(csv, out):
    df = pd.read_csv(csv)
    cal, val = split_runs(df)
    cfg = load_config()
    report = {"n_runs_calibration": int(cal.run_id.nunique()), "n_runs_validation": int(val.run_id.nunique())}
    for metric, (sec, key) in NOMINAL.items():
        v = cal[cal.metric == metric].value
        if len(v):
            cfg[sec][key] = float(v.median())
    stop = cal[cal.metric == "stop_error_m"].value
    if len(stop):     # stop line must sit beyond the worst-case stop overshoot
        cfg["course"]["stopline_offset_m"] = max(cfg["course"]["stopline_offset_m"], float(np.percentile(stop, 95)) + 0.03)
    for metric in RANGES:
        v = cal[cal.metric == metric].value
        if len(v) >= 5:
            cfg["randomization"][metric] = [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
            report[f"range_{metric}"] = cfg["randomization"][metric]
    # Crossing time per movement (metrics crossing_straight_s / crossing_left_s / crossing_right_s): fit the speed
    # inside the box so that SUMO needs as long as the cars do, then check on the held-out runs.
    out = Path(out)
    os.environ["PICAR_COURSE_CONFIG"] = str(out)           # the calibrated config gets its own SUMO net (<stem>_maps)
    cfg["vehicle"].setdefault("turn_speed_mps", {})

    def rebuild():
        out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        build(cfg)

    rebuild()
    for move, (arm, edge_out) in MOVES.items():
        target = cal[cal.metric == f"crossing_{move}_s"].value
        if not len(target):
            continue
        target = float(target.mean())
        lo, hi = 0.03, 2 * cfg["vehicle"]["speed_mps"]     # above ~2x the car speed SUMO slows turns down again
        for _ in range(10):                                # bisection on the box speed (crossing time falls with it)
            mid = (lo * hi) ** 0.5
            cfg["vehicle"]["turn_speed_mps"][move] = round(mid, 4)
            rebuild()
            t = sim_crossing_time(cfg, arm, trials=2, out=edge_out)
            lo, hi = (lo, mid) if t < target else (mid, hi)
        cfg["vehicle"]["turn_speed_mps"][move] = round((lo * hi) ** 0.5, 4)
        report[f"box_speed_{move}_mps"] = cfg["vehicle"]["turn_speed_mps"][move]
    rebuild()
    for move, (arm, edge_out) in MOVES.items():
        real = val[val.metric == f"crossing_{move}_s"].value
        if len(real):
            sim = float(sim_crossing_time(cfg, arm, trials=3, out=edge_out))
            rel = abs(sim - float(real.mean())) / float(real.mean())
            report[f"crossing_{move}"] = {"real_mean_s": round(float(real.mean()), 2), "sim_mean_s": round(sim, 2),
                                          "rel_error": round(rel, 3), "pass_10pct": bool(rel <= 0.10)}
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="picar_sim/config/course.calibrated.yaml")
    a = p.parse_args()
    print(yaml.safe_dump(calibrate(a.csv, a.out), sort_keys=False))
