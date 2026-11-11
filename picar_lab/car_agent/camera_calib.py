"""Front-camera mount calibration from ONE photo (numpy only; runs on the Pi).

The lane follower turns pixels into metres with the camera mount (pitch, yaw, sideways offset). A camera screwed on
2 deg crooked moves the lane centre the car aims at by about 1 cm, and the lane only leaves 2 cm each side.

Procedure (real car and simulation are the same):
  1. Park the car in the middle of a straight lane, pointing along it (tape measure: same gap to both lines).
  2. Save one camera frame and run
       python3 -m picar_lab.car_agent.camera_calib --image frame.png --lane-width 0.21 --out car_calibration.yaml
  3. The result (cam_pitch_deg, cam_yaw_deg, cam_y_m) is merged into the car's calibration file; the drive node
     builds its CameraModel from it.

Method: search the mount for which the bird's-eye view shows the two lane lines where they really are:
right line at -lane/2, left line at +lane/2, both parallel to the car. Height and forward offset are taken from
a ruler (they are confounded with pitch).
"""
import argparse
from pathlib import Path

import numpy as np

from picar_lab.car_agent.lane_camera import CAMERA, CameraModel, LaneDetector, at

KEYS = ("pitch_deg", "yaw_deg", "y")


def _cost(gray, cam, lane_width, fs):
    det = LaneDetector(CameraModel(cam), lane_width)
    lanes = det.detect(gray)
    r, lf = lanes["right"], lanes["left"]
    if r is None or lf is None:
        return 1e3
    f = fs[(fs >= max(r["f0"], lf["f0"])) & (fs <= min(r["f1"], lf["f1"]))]
    if len(f) < 3:
        return 1e3
    er = at(r["fit"], f) + lane_width / 2
    el = at(lf["fit"], f) - lane_width / 2
    return float(np.sqrt(np.mean(er ** 2 + el ** 2)))


def calibrate_mount(gray, lane_width=0.21, cam=None, span=(6.0, 6.0, 0.03), rounds=4):
    """Returns ({"pitch_deg", "yaw_deg", "y"}, rms error in metres). Coarse-to-fine grid search around `cam`."""
    base = {**CAMERA, **(cam or {})}
    base.setdefault("yaw_deg", 0.0)
    base.setdefault("y", 0.0)
    gray = np.asarray(gray, float)
    fs = np.linspace(0.18, 0.50, 17)
    best = {k: float(base[k]) for k in KEYS}
    best_cost = _cost(gray, {**base, **best}, lane_width, fs)
    half = list(span)
    for _ in range(rounds):
        grids = [np.linspace(best[k] - h, best[k] + h, 7) for k, h in zip(KEYS, half)]
        for p in grids[0]:
            for yw in grids[1]:
                for y in grids[2]:
                    c = _cost(gray, {**base, "pitch_deg": p, "yaw_deg": yw, "y": y}, lane_width, fs)
                    if c < best_cost - 1e-9:
                        best_cost, best = c, {"pitch_deg": float(p), "yaw_deg": float(yw), "y": float(y)}
        half = [h / 3.0 for h in half]
    return best, best_cost


def to_calibration(mount):
    return {"cam_pitch_deg": round(mount["pitch_deg"], 2), "cam_yaw_deg": round(mount["yaw_deg"], 2),
            "cam_y_m": round(mount["y"], 4)}


def camera_from_calibration(cal, cam=None):
    """Camera dict for CameraModel / CameraBrain(camera=...) from a calibration dict (missing keys: nominal)."""
    c = {**CAMERA, **(cam or {})}
    if cal.get("cam_pitch_deg") is not None:
        c["pitch_deg"] = float(cal["cam_pitch_deg"])
    c["yaw_deg"] = float(cal.get("cam_yaw_deg", 0.0) or 0.0)
    c["y"] = float(cal.get("cam_y_m", 0.0) or 0.0)
    return c


def main():
    import yaml
    from PIL import Image
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--lane-width", type=float, default=0.21)
    ap.add_argument("--out", help="car calibration yaml to merge the result into")
    a = ap.parse_args()
    img = Image.open(a.image).convert("L").resize((CAMERA["w"], CAMERA["h"]))
    mount, err = calibrate_mount(np.asarray(img, float) / 255.0, a.lane_width)
    cal = to_calibration(mount)
    print(cal, f"rms {err * 1000:.1f} mm")
    if err > 0.01:
        print("WARNING: rms > 10 mm - are both lane lines visible and the car centred and straight?")
    if a.out:
        p = Path(a.out).expanduser()
        data = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}
        data.update(cal)
        p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        print("merged into", p)


if __name__ == "__main__":
    main()
