"""Fit SUMO parameters from PiCar low-speed logs and check sim-vs-real on held-out runs.

Input CSV (one row per measurement; template: picar_sim/config/field_measurements.template.csv):
  run_id, metric, value
metrics: speed_mps, accel_mps2, decel_mps2, startup_delay_s, stop_error_m, crossing_time_s,
         command_latency_s, ir_detect_prob, status_loss_prob, course_width_m, course_depth_m,
         stopline_offset_m, corner_radius_m
Runs are split 60/40 by run_id (never by sample) into calibration / validation sets.

python -m picar_sim.calibrate --csv field.csv --out picar_sim/config/course.calibrated.yaml
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from picar_sim.map_builder import build, load_config

NOMINAL = {"speed_mps": ("vehicle", "speed_mps"), "accel_mps2": ("vehicle", "accel_mps2"),
           "decel_mps2": ("vehicle", "decel_mps2"), "startup_delay_s": ("vehicle", "startup_delay_s"),
           "course_width_m": ("course", "width_m"), "course_depth_m": ("course", "depth_m"),
           "stopline_offset_m": ("course", "stopline_offset_m"), "corner_radius_m": ("course", "corner_radius_m")}
RANGES = ["speed_mps", "accel_mps2", "decel_mps2", "startup_delay_s", "command_latency_s",
          "ir_detect_prob", "status_loss_prob"]


def split_runs(df, frac=0.6):
    h = df.run_id.astype(str).map(lambda r: int(hashlib.sha1(r.encode()).hexdigest(), 16) % 1000 / 1000)
    return df[h < frac], df[h >= frac]


def sim_crossing_time(cfg, approach="W", trials=5):
    """Single car released from the stop line: time from GO to EXIT of the conflict box."""
    from picar_sim.env import PicarEnv
    env = PicarEnv(cfg)
    env.auto_route = False
    times = []
    for s in range(trials):
        env.reset(seed=9000 + s, cars=1)
        car = "car0"
        k = env.k
        cur = k.vehicle.getRoadID(car)
        out = "E_out" if approach != "E" else "W_out"
        path = [cur] if cur == f"{approach}_in" else list(k.simulation.findRoute(cur, f"{approach}_in").edges)
        k.vehicle.setRoute(car, path + [out])
        go, was_in = None, False
        for _ in range(int(120 / env.dt)):
            if car not in k.vehicle.getIDList():
                break
            jn, st, arm = env.truth.get(car, (None, "", None))
            if jn == "J" and st == "ARRIVE" and arm == approach and go is None:
                go = env.t
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
    Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    build(cfg)
    real = val[val.metric == "crossing_time_s"].value
    if len(real):
        sim = sim_crossing_time(cfg)
        report["crossing_time_real_mean_s"] = float(real.mean())
        report["crossing_time_sim_mean_s"] = sim
        report["crossing_time_rel_error"] = abs(sim - real.mean()) / real.mean()
        report["crossing_time_pass_10pct"] = bool(report["crossing_time_rel_error"] <= 0.10)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="picar_sim/config/course.calibrated.yaml")
    a = p.parse_args()
    print(yaml.safe_dump(calibrate(a.csv, a.out), sort_keys=False))
