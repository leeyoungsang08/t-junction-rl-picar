"""Front-camera lane following, painted course lines and the unprotected right-turn rule (no ROS needed)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "picar_gazebo")]

from picar_gazebo.cam_render import CamRenderer  # noqa: E402
from picar_gazebo.core import Course  # noqa: E402
from picar_lab.car_agent.lane_camera import LaneDetector, LaneFollower  # noqa: E402
from picar_sim.right_turn import conflicts, is_right, may_enter  # noqa: E402
from picar_sim.supervisor import SupervisorStatus  # noqa: E402


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    from PIL import Image
    from picar_gazebo.world_gen import generate
    out = tmp_path_factory.mktemp("world")
    generate(out, 2, 1)
    tex = np.asarray(Image.open(out / "course_texture.png").convert("L"), float) / 255.0
    course = Course()
    return course, CamRenderer(tex, course.tex_bounds())


def _pose(course, edge, s, offset=0.0):
    pts, arc = course.lanes[edge], course.arc[edge]
    i = int(np.searchsorted(arc, s))
    d = pts[min(i + 3, len(pts) - 1)] - pts[max(i - 3, 0)]
    yaw = math.atan2(d[1], d[0])
    q = pts[i] + np.array([-math.sin(yaw), math.cos(yaw)]) * offset
    return q[0], q[1], yaw


def test_painted_lines_follow_the_real_course():
    course = Course()
    lines = course.road_lines()
    assert len(lines) == 9                      # 3 roads x (edge, divider, edge), each one run between the boxes
    for jn, c in course.centres.items():        # nothing painted inside a box except the through road's far edge
        s = 1.0 if jn == "J" else -1.0
        for ln in lines:
            d = ln - c
            inside = (np.abs(d[:, 0]) < course.lw - 0.01) & (np.abs(d[:, 1]) < course.lw - 0.01)
            assert not inside.any()


@pytest.mark.parametrize("edge,s", [("W_out", 2.0), ("E_in", 0.5), ("N_out", 0.6)])
@pytest.mark.parametrize("offset", [-0.03, 0.0, 0.03])
def test_lane_centre_from_camera_on_straights(scene, edge, s, offset):
    course, render = scene
    det, fol = LaneDetector(lane_width=course.lw), LaneFollower(lane_width=course.lw)
    y = fol.target(det.detect(render.render(*_pose(course, edge, s, offset))))
    assert y is not None and abs(y + offset) < 0.02      # car shifted left by `offset` -> centre is to its right


def test_stop_line_is_not_taken_for_a_lane_line(scene):
    course, render = scene
    far, near, u = course.stopline("J", "W")
    edge = course.approach("J", "W")[2]
    s = course.arc_of(edge, near) - 0.30                  # stop line 0.3 m ahead: inside the camera view
    det = LaneDetector(lane_width=course.lw)
    lanes = det.detect(render.render(*_pose(course, edge, s)))
    for side in ("right", "left"):
        if lanes[side]:
            # still a line along the lane (a cross tape would be ~90 deg); the divider bends into the L-shaped
            # stop-line corner, so allow up to ~45 deg at the near end (renderers differ slightly between machines)
            assert abs(lanes[side]["fit"][1]) < 1.0


def _status(phase, stage="green", fault=None):
    return SupervisorStatus(phase, stage, 5.0, 0.0, fault, {})


def test_right_turn_rule():
    assert is_right("J", "E", "N") and is_right("J", "N", "W") and not is_right("J", "W", "N")
    assert not conflicts("J", ("W", "E"), ("W", "N")) and conflicts("J", ("E", "N"), ("W", "E"))
    st = _status(0)                                       # W green
    assert may_enter("J", ("E", "N"), st, True, [], [], "ARRIVE")          # idle junction: stop, then turn right
    assert not may_enter("J", ("E", "N"), st, True, [], [], "APPROACH")    # must stop at the line first
    assert not may_enter("J", ("E", "N"), st, False, [], [], "ARRIVE")     # rule switched off
    assert not may_enter("J", ("E", "W"), st, True, [], [], "ARRIVE")      # only right turns
    assert not may_enter("J", ("E", "N"), st, True, [], [("W", "E")], "ARRIVE")   # green side is coming
    assert not may_enter("J", ("E", "N"), st, True, [("W", "E")], [], "ARRIVE")   # green car inside
    assert not may_enter("J", ("E", "N"), _status(0, "yellow"), True, [], [], "ARRIVE")
    assert not may_enter("J", ("W", "E"), st, True, [("E", "N")], [], "APPROACH")  # green waits for the turner
    assert may_enter("J", ("W", "E"), st, True, [("W", "N")], [], "APPROACH")      # same approach follows
