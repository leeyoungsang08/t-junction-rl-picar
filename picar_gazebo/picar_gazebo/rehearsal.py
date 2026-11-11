"""Real-to-Sim rehearsal with the virtual real fleet (no ROS, no Gazebo; the same steps apply to the real cars).

  python3 -m picar_gazebo.rehearsal fleet   --cars 6 --seed 7 --out runs/rehearsal        # hidden truth
  python3 -m picar_gazebo.rehearsal measure --out runs/rehearsal                          # per-car calibration
  python3 -m picar_gazebo.rehearsal drive   --out runs/rehearsal --calib full --seconds 180 --seed 1
  python3 -m picar_gazebo.rehearsal ladder  --out runs/rehearsal                          # none -> shared -> full

calib levels: none (drive uncalibrated), shared (car0's numbers for everyone), motors (per car, motors only),
full (per car, motors + camera mount).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from picar_gazebo import virtual_fleet as vf
from picar_gazebo.core import Course
from picar_gazebo.kinematic_check import run
from picar_gazebo.world_gen import CAR, generate

def straight_spot(course, ahead=0.9, behind=0.2):
    """(x, y, yaw) in the middle of a lane with `ahead` metres of straight, line-only floor in front of the car
    (camera calibration needs both lane lines straight through the whole view, no stop line, no junction box)."""
    best = None
    for e, pts in course.lanes.items():
        L = course.arc[e]
        for s in np.arange(behind, L[-1] - ahead, 0.05):
            i0, i1, i = (int(np.searchsorted(L, v)) for v in (s - behind, s + ahead, s))
            seg = pts[i0:i1 + 1]
            u = (seg[-1] - seg[0]) / np.linalg.norm(seg[-1] - seg[0])
            dev = np.abs((seg - seg[0]) @ np.array([-u[1], u[0]])).max()
            if dev > 0.002 or any(course.in_box(q, margin=0.30) for q in seg[::10]):
                continue
            room = min(s - behind, L[-1] - ahead - s)             # prefer the middle of the longest straight
            if best is None or room > best[0]:
                best = (room, float(pts[i][0]), float(pts[i][1]), float(np.arctan2(u[1], u[0])))
    if best is None:
        raise ValueError("no straight piece of lane long enough for the camera calibration")
    return best[1:]


UNCAL = {"trim_per_m": 0.0, "turn_scale": 1.0, "accel_mps2": 1.0, "latency_s": 0.05, "speed_scale": 1.0}


def cmd_fleet(out, cars, seed, scale=1.0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fleet = vf.sample_fleet(cars, seed, scale)
    vf.save_fleet(out / "fleet_truth.yaml", fleet, f"virtual real fleet seed {seed} scale {scale} (ASSUMED ranges)")
    return fleet


def cmd_measure(out, turn_rate=45.0, real_cfg=None, seed=0, measure=None):
    """Per-car calibration files out/calib/<car>.yaml, as they would be written after measuring the real cars."""
    out = Path(out)
    fleet = vf.load_fleet(out / "fleet_truth.yaml")
    park = out / "_park"
    generate(park, 1, 1, real_cfg)
    tex = np.asarray(Image.open(park / "course_texture.png").convert("L"), dtype=float) / 255.0
    course = Course(real_cfg)
    factory = vf.renderer_factory_for(tex, course.tex_bounds())
    spot = straight_spot(course)                              # a straight piece of lane, away from stop lines
    (out / "calib").mkdir(exist_ok=True)
    report = {}
    for i, truth in enumerate(fleet):
        cal, rep = vf.measure_motors(truth, CAR, turn_rate, seed * 100 + i, measure)
        cam, rep_c = vf.measure_camera(truth, CAR, factory, spot, course.lw,
                                       seed * 100 + i, measure)
        cal.update(cam)
        (out / "calib" / f"{truth['name']}.yaml").write_text(
            yaml.safe_dump({**cal, "source": "virtual fleet rehearsal (tape + protractor + one photo)"}, sort_keys=False))
        report[truth["name"]] = {"calibration": cal, "motor_tests": rep, "camera_test": rep_c,
                                 "truth_camera": {k: truth[k] for k in ("cam_pitch_err_deg", "cam_yaw_err_deg", "cam_y_err_m")}}
    (out / "measure_report.json").write_text(json.dumps(report, indent=1))
    return report


def load_calibs(out, level, n):
    out = Path(out)
    cals = [yaml.safe_load((out / "calib" / f"car{i}.yaml").read_text()) for i in range(n)]
    for c in cals:
        c.pop("source", None)
        c.pop("turn_rate_dps", None)
    cam_keys = ("cam_pitch_deg", "cam_yaw_deg", "cam_y_m")
    if level == "none":
        return [dict(UNCAL) for _ in cals]
    if level == "shared":
        return [{k: v for k, v in cals[0].items() if k not in cam_keys} for _ in cals]
    if level == "motors":
        return [{k: v for k, v in c.items() if k not in cam_keys} for c in cals]
    return cals


def summarize(m):
    errs = np.abs(m["stop_errors_m"]) if m["stop_errors_m"] else np.array([0.0])
    cross = [f["value"] for f in m["field"] if f["metric"] == "crossing_time_s"]
    jw = [f["value"] for f in m["field"] if f["metric"] == "j_request_to_entry_s"]
    near = [n["where"] for n in m.get("near_log", [])]
    return {"collisions": m["collisions"], "near_road": near.count("road"), "near_junction": near.count("junction"),
            "lane_departures": m["lane_departures"], "faults": m["faults"],
            "illegal": m["illegal_crossings"], "deadlock_s": m["max_all_stopped_s"],
            "stop_p95_m": round(float(np.percentile(errs, 95)), 3), "crossings": m["crossings"],
            "throughput_per_min": m["throughput_per_min"],
            "crossing_time_s": round(float(np.mean(cross)), 2) if cross else None,
            "j_entries": len(jw), "j_request_to_entry_s": round(float(np.mean(jw)), 2) if jw else None,
            "j_request_to_entry_max_s": round(float(np.max(jw)), 1) if jw else None}


def cmd_drive(out, level="full", seconds=180, seed=1, cars=None, policy=None, mode="shadow", turn_rate=45.0,
              real_cfg=None, map_cfg=None, tag=None, truths=None, video=False):
    out = Path(out)
    fleet = truths if truths is not None else vf.load_fleet(out / "fleet_truth.yaml")
    n = cars or len(fleet)
    calibs = load_calibs(out, level, len(fleet))[:n] if level != "ideal" else [dict(UNCAL, accel_mps2=1.0) for _ in range(n)]
    run_dir = out / "drive" / (tag or f"{level}_s{seed}")
    m = run(cars=n, seconds=seconds, seed=seed, out=run_dir, policy=policy, mode=mode, truths=fleet[:n], calibs=calibs,
            real_cfg=real_cfg, map_cfg=map_cfg, turn_rate_dps=turn_rate, video=video)
    s = summarize(m)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=1))
    return s, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fleet", "measure", "drive", "ladder"])
    ap.add_argument("--out", default="runs/rehearsal")
    ap.add_argument("--cars", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--calib", default="full", choices=["none", "shared", "motors", "full"])
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--policy")
    ap.add_argument("--mode", default="shadow")
    ap.add_argument("--turn-rate", type=float, default=45.0)
    ap.add_argument("--tag")
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    if a.cmd == "fleet":
        print(yaml.safe_dump(cmd_fleet(a.out, a.cars, a.seed, a.scale), sort_keys=False))
    elif a.cmd == "measure":
        print(json.dumps(cmd_measure(a.out, a.turn_rate), indent=1))
    elif a.cmd == "drive":
        print(json.dumps(cmd_drive(a.out, a.calib, a.seconds, a.seed, a.cars, a.policy, a.mode, a.turn_rate,
                                   tag=a.tag, video=a.video)[0], indent=1))
    else:
        for level in ("none", "shared", "motors", "full"):
            print(level, json.dumps(cmd_drive(a.out, level, a.seconds, 1, a.cars, turn_rate=a.turn_rate)[0]))


if __name__ == "__main__":
    main()
