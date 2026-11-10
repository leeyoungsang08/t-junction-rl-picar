"""ROS-free checks of the Gazebo course generator and the shared driving logic (run with plain pytest)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "picar_gazebo")]

from picar_gazebo.core import (Course, LINE_TAPE_M, footprint, ir_from_gray, overlap,  # noqa: E402
                               sensor_columns)
from picar_lab.car_agent.calibration import TESTS, from_measurements  # noqa: E402
from picar_lab.car_agent.drive import (JunctionManeuver, LineFollower, Odometer, calibrate_brake2,  # noqa: E402
                                       movement, wheel_pwm)


@pytest.fixture(scope="module")
def course():
    return Course()


def test_stoplines_sit_outside_the_box(course):
    off = course.cfg["course"]["stopline_offset_m"]
    for jn in ("J", "J2"):
        for arm in ("W", "E", "N"):
            far, near, u = course.stopline(jn, arm)
            assert not course.in_box(far) and not course.in_box(near)
            assert course.in_box(far + u * (off + 0.01))          # the box starts right after the offset


def test_maneuver_plans_are_consistent(course):
    for jn in ("J", "J2"):
        for a in ("W", "E", "N"):
            for b in ("W", "E", "N"):
                if a == b:
                    continue
                p = course.maneuver(jn, a, b, 0.125, 0.04)
                assert p["turn"] == movement(jn, a, b)
                assert 0 <= p["d2_min"] < p["d2_max"]
                if p["turn"] == "straight":
                    assert p["d1"] == 0
                else:
                    assert 0.2 < p["d1"] < 1.0 and abs(p["angle_deg"]) == 90


def test_line_tape_wider_than_sensor_spacing():
    # 4 sensors 15 mm apart: an 18 mm tape left only a 1.5 mm two-sensor zone (zigzag); 25 mm is the default
    assert LINE_TAPE_M >= 0.025


def test_ir_from_gray_reads_columns():
    gray = np.full((8, 32), 0.9)
    cols = [5, 12, 19, 26]
    gray[:, 11:14] = 0.1
    assert ir_from_gray(gray, cols, threshold=0.35) == [0, 1, 0, 0]
    assert len(sensor_columns(32, 1.85, 0.03, [0.0225, 0.0075, -0.0075, -0.0225])) == 4


def test_overlap_of_footprints():
    a = footprint(0, 0, 0, 0.24, 0.17)
    assert overlap(a, footprint(0.2, 0, 0, 0.24, 0.17))
    assert not overlap(a, footprint(0.3, 0, 0.3, 0.24, 0.17))


def test_follower_steers_towards_the_line():
    f = LineFollower(dark=1)
    v, w, ok = f.update([0, 0, 1, 1], 0.0)           # line right of centre -> turn right (w < 0)
    assert ok and w < 0 and v > 0
    v, w, ok = f.update([1, 1, 1, 1], 0.05)          # cross tape -> straight
    assert w == 0 and ok


def test_odometer_and_brake_calibration_agree():
    # plant: latency 0.05 s, decel 1.0 m/s^2 -> roll = v*L + v^2/2a
    roll = lambda v: v * 0.05 + v * v / 2.0
    L, a = calibrate_brake2(roll(0.15), 0.15, roll(0.25), 0.25)
    assert L == pytest.approx(0.05, abs=1e-6) and a == pytest.approx(1.0, rel=1e-6)
    o = Odometer(L, a)
    o.command(0.25, 0.0)
    o.update(2.0)
    assert o.stop_distance() == pytest.approx(roll(0.25), rel=1e-3)


def test_maneuver_turns_then_finds_the_line():
    plan = {"turn": "left", "d1": 0.30, "angle_deg": 90, "d2_min": 0.05, "d2_max": 0.4}
    m = JunctionManeuver(plan, speed_mps=0.2, turn_rate_dps=90)
    t, phases = 0.0, []
    while not (m.done or m.failed) and t < 20:
        line = [0, 1, 1, 0] if m.phase == "d2" and m.dist > 0.1 else [0, 0, 0, 0]
        v, w = m.update(line, t)
        phases.append(m.phase)
        t += 0.05
    assert m.done and "turn" in phases and "d2" in phases


def test_maneuver_ignores_cross_tape_and_searches():
    plan = {"turn": "straight", "d1": 0.0, "angle_deg": 0, "d2_min": 0.0, "d2_max": 0.2}
    m = JunctionManeuver(plan, speed_mps=0.2)
    t = 0.0
    while m.phase in ("d1", "d2") and t < 5:
        m.update([1, 1, 1, 1], t)                    # a marker is not the lane line
        t += 0.05
    assert m.phase == "search" and not m.done
    while not (m.done or m.failed) and t < 30:
        m.update([0, 0, 0, 0], t)
        t += 0.05
    assert m.failed                                   # swept +-60 deg and back, gave up -> fault (car stops)


def test_calibration_tests_and_formula():
    assert set(TESTS) == {"straight", "brake_slow", "brake_fast", "spin"}
    cal = from_measurements(1.0, -10.0, 0.01875, 0.04375, [85, 86, 84])
    assert cal["trim_per_m"] == pytest.approx(math.radians(10), abs=1e-3)
    assert cal["turn_scale"] == pytest.approx(85 / 90, abs=1e-3)
    assert 0.03 < cal["latency_s"] < 0.07 and 0.7 < cal["accel_mps2"] < 1.4


def test_wheel_pwm_signs():
    l, r = wheel_pwm(0.2, 0.0)
    assert l == r > 0
    l, r = wheel_pwm(0.0, 1.0)
    assert l < 0 < r


def test_world_generation(tmp_path):
    from picar_gazebo.world_gen import generate
    generate(tmp_path, 3, 1)
    meta = yaml.safe_load((tmp_path / "course_meta.yaml").read_text())
    assert len(meta["cars"]) == 3 and (tmp_path / "picar_course.sdf").stat().st_size > 1000
    sdf = (tmp_path / "picar_course.sdf").read_text()
    assert "ignition-gazebo-diff-drive-system" in sdf and "line_cam" in sdf
    plans = yaml.safe_load((tmp_path / "maneuvers.yaml").read_text())["maneuvers"]
    assert len(plans) == 12


def test_kinematic_precheck_multi_car(tmp_path):
    from picar_gazebo.kinematic_check import run
    m = run(cars=3, seconds=60, seed=1, out=tmp_path)
    assert m["collisions"] == 0 and m["faults"] == 0 and m["illegal_crossings"] == 0
    assert m["crossings"] >= 6
