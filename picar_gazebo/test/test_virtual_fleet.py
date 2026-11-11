"""Virtual real fleet + Real-to-Sim rehearsal (no ROS, no Gazebo)."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "picar_gazebo")]
os.environ.setdefault("PICAR_REPO", str(REPO))

from picar_gazebo import rehearsal as rh                      # noqa: E402
from picar_gazebo import virtual_fleet as vf                  # noqa: E402
from picar_gazebo.core import Course                          # noqa: E402
from picar_gazebo.world_gen import CAR, generate              # noqa: E402
from picar_lab.car_agent.camera_brain import CameraBrain      # noqa: E402
from picar_lab.car_agent.agent import CarAgent                # noqa: E402
from picar_lab.car_agent.camera_calib import calibrate_mount  # noqa: E402
from picar_lab.car_agent.lane_camera import LaneFollower      # noqa: E402


def test_fleet_is_reproducible_and_inside_the_assumed_ranges():
    a, b = vf.sample_fleet(6, 7), vf.sample_fleet(6, 7)
    assert a == b and a != vf.sample_fleet(6, 8)
    for car in a:
        for key, spec in vf.PRIORS.items():
            lo, hi = (spec[1], spec[2]) if spec[0] == "u" else (spec[3], spec[4])
            assert lo - 1e-9 <= car[key] <= hi + 1e-9, key
    assert len({c["gain_r"] for c in a}) == 6                 # six different cars


def test_motor_measurement_recovers_the_hidden_drivetrain():
    truth = {**vf.NOMINAL, "name": "car0", "gain_l": 1.0, "gain_r": 0.90, "speed_scale": 0.95, "spin_eff": 0.85,
             "latency_s": 0.10}
    cal, rep = vf.measure_motors(truth, CAR, 45.0, seed=1)
    assert cal["speed_scale"] == pytest.approx(0.95 * 0.95, abs=0.03)          # (1.0 + 0.9) / 2 * 0.95
    assert cal["trim_per_m"] > 0.3                                             # weaker right motor: steer left
    assert cal["turn_scale"] == pytest.approx(0.85 * 0.95 * 0.95, abs=0.05)
    assert 0.05 <= cal["latency_s"] <= 0.16


def test_camera_calibration_finds_the_mount(tmp_path):
    from PIL import Image
    generate(tmp_path, 1, 1)
    course = Course()
    tex = np.asarray(Image.open(tmp_path / "course_texture.png").convert("L"), dtype=float) / 255.0
    factory = vf.renderer_factory_for(tex, course.tex_bounds())
    truth = {**vf.NOMINAL, "cam_yaw_err_deg": 2.0, "cam_y_err_m": -0.005, "cam_pitch_err_deg": 1.0}
    x, y, yaw = rh.straight_spot(course)
    img = factory(vf.true_camera(truth)).render(x, y, yaw)
    mount, err = calibrate_mount(img, course.lw)
    assert err < 0.004
    assert mount["yaw_deg"] == pytest.approx(2.0, abs=0.5)
    assert mount["y"] == pytest.approx(-0.005, abs=0.003)
    assert mount["pitch_deg"] == pytest.approx(36.0, abs=0.7)


def _brain(**kw):
    return CameraBrain(CarAgent("car0", ["W_in", "E_out"]), {}, {}, **kw)


def test_stop_line_needs_four_channels_or_two_readings():
    b = _brain()
    assert not b._stop_line([1, 1, 1, 0])          # on a lane line with one noise flip: not a stop line
    assert b._stop_line([1, 1, 1, 0])              # the same again: confirmed
    b = _brain()
    assert b._stop_line([1, 1, 1, 1])


def test_stop_line_too_soon_after_a_box_is_ignored():
    b = _brain(stop_after={"W_in": 4.0})
    b.s_left_box, b.stop_after_m = 0.0, 4.0
    b.odo.s = 1.0
    assert not b._stop_line([1, 1, 1, 1])
    b.odo.s = 3.0
    assert b._stop_line([1, 1, 1, 1])


def test_speed_scale_raises_the_command_not_the_odometer():
    b = _brain(speed_scale=0.8)
    assert b.odo.cmd(0.2) == pytest.approx(0.25)
    b.odo.command(0.2, 0.0)
    b.odo.update(2.0)
    assert b.odo.s == pytest.approx(0.2 * 2.0, abs=0.05)


def test_lane_integral_does_not_wind_up_while_standing():
    f = LaneFollower()
    lanes = {"right": {"fit": (-0.085, 0.0, 0.0), "f0": 0.15, "f1": 0.58, "l0": -0.085, "side": -0.085}, "left": None}
    for i in range(100):
        f.update(lanes, i * 0.05, moving=False)
    assert f.i_term == 0.0
    for i in range(100, 140):
        f.update(lanes, i * 0.05, moving=True)
    assert 0.0 < f.i_term <= f.i_max


def test_six_different_cars_drive_clean_with_per_car_calibration(tmp_path):
    """The headline check, short version: 6 cars with different hidden hardware, each with its own measured
    calibration, 60 s: no fault, no illegal entry, nobody stuck."""
    rh.cmd_fleet(tmp_path, 6, 7)
    rh.cmd_measure(tmp_path)
    s, m = rh.cmd_drive(tmp_path, "full", 60, 1)
    assert s["faults"] == 0 and s["illegal"] == 0 and s["deadlock_s"] < 10 and s["lane_departures"] == 0
    assert s["crossings"] >= 10 and s["stop_p95_m"] < 0.05


def test_gazebo_world_carries_each_cars_camera_mount(tmp_path):
    fleet = [{**vf.NOMINAL, **t} for t in vf.sample_fleet(3, 7)]
    generate(tmp_path, 3, 1, None, fleet)
    sdf = (tmp_path / "picar_course.sdf").read_text()
    for t in fleet:
        assert f"{math.radians(t['cam_yaw_err_deg']):.5f}</pose>" in sdf
