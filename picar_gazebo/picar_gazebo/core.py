"""Course geometry for Gazebo, derived from the same course.yaml and road shapes as the SUMO map.

No ROS / Gazebo imports here, so it can be unit-tested anywhere.
Conventions: x east, y north, metres; right-hand traffic; one lane per direction, lane width lw;
junction box = 2*lw square around each T centre (as in the SUMO map); stop line far edge = box edge - stopline_offset.
Painted lines follow the real course (photo): every road has an outer edge, a centre divider and an inner edge;
the divider ends at the stop lines, and inside a box only the through road's far edge continues.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("PICAR_REPO", Path(__file__).resolve().parents[2]))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from picar_sim.map_builder import APPROACHES, load_config, road_shapes  # noqa: E402

STOP_TAPE_M = 0.03        # thickness of the stop-line tape (photo: same tape as the lane lines)
APPROACH_TAPE_M = 0.015   # legacy approach marker (the real course has none: APPROACH comes from odometry)
LINE_TAPE_M = 0.025       # lane line tape width (>= sensor spacing + margin; 18 mm zigzags)
CROSS_TAPE_LEN = 0.30     # length of perpendicular tapes across the lane
TEX_MARGIN = 0.15         # white floor around the outer line in the texture / Gazebo floor (m)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _right(u):
    return np.array([u[1], -u[0]])


def _densify(pts, step=0.005):
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def _offset_polyline(pts, d, window=8):
    """Shift a polyline d metres to the right of its direction of travel (direction smoothed over +-window
    points so the chorded corner arcs do not produce ripples in the offset lane)."""
    pts = np.asarray(pts, float)
    out = []
    for i in range(len(pts)):
        a = pts[max(0, i - window)]
        b = pts[min(len(pts) - 1, i + window)]
        out.append(pts[i] + d * _right(_unit(b - a)))
    return np.array(out)


class Course:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        c = self.cfg["course"]
        self.lw = c["lane_width_m"]
        self.width, self.depth = c["width_m"], c["depth_m"]
        J, J2, shapes = road_shapes(self.cfg)
        self.centres = {"J": np.array(J), "J2": np.array(J2)}
        self.roads = {a: np.array(p) for a, p in shapes.items()}        # J -> J2 direction
        # lane centrelines (travel direction): a_out = J -> J2, a_in = J2 -> J
        self.lanes = {}
        for a, pts in self.roads.items():
            dense = _densify(pts)
            self.lanes[f"{a}_out"] = _offset_polyline(dense, self.lw / 2)
            self.lanes[f"{a}_in"] = _offset_polyline(dense[::-1], self.lw / 2)
        self.arc = {e: np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
                    for e, p in self.lanes.items()}

    # ---------------------------------------------------------------- junction-local geometry
    def approach(self, junction, arm):
        """Incoming lane of `arm` at `junction`: (unit direction towards the box, a point on the lane line)."""
        edge = f"{arm}_in" if junction == "J" else f"{arm}_out"
        pts = self.lanes[edge]
        return _unit(pts[-1] - pts[-2]), pts[-1], edge

    def departure(self, junction, arm):
        edge = f"{arm}_out" if junction == "J" else f"{arm}_in"
        pts = self.lanes[edge]
        return _unit(pts[1] - pts[0]), pts[0], edge

    def box_edge_along(self, junction, point, u):
        """Distance from `point` (on a lane line) along u to the junction box boundary (|.|_inf = lw)."""
        c = self.centres[junction]
        best = math.inf
        for k in range(2):
            if abs(u[k]) < 1e-9:
                continue
            for side in (-1, 1):
                t = (c[k] + side * self.lw - point[k]) / u[k]
                q = point + t * u
                if t > -1e-9 and abs(q[1 - k] - c[1 - k]) <= self.lw + 1e-9:
                    best = min(best, t)
        return best

    def stopline(self, junction, arm):
        """(far edge point, near edge point, direction) of the stop-line tape of this approach."""
        u, p, _ = self.approach(junction, arm)
        c = self.centres[junction]
        # project the lane onto the box boundary: entry point = where the lane meets the box
        back = p - u * 2.0
        entry = back + u * self.box_edge_along(junction, back, u)
        far = entry - u * self.cfg["course"]["stopline_offset_m"]
        return far, far - u * STOP_TAPE_M, u

    def _at(self, edge, s):
        pts, arc = self.lanes[edge], self.arc[edge]
        i = int(np.clip(np.searchsorted(arc, s) - 1, 0, len(arc) - 2))
        f = (s - arc[i]) / max(arc[i + 1] - arc[i], 1e-12)
        return pts[i] + (pts[i + 1] - pts[i]) * f, _unit(pts[i + 1] - pts[i])

    def approach_marker(self, junction, arm):
        """Thin tape approach_zone_m before the stop line, measured ALONG the lane (corners included).
        On a lane shorter than that (the connector) it sits just after the upstream exit beam."""
        far, near, u = self.stopline(junction, arm)
        _, _, edge = self.approach(junction, arm)
        pts, arc = self.lanes[edge], self.arc[edge]
        s_stop = arc[int(np.argmin(np.linalg.norm(pts - near, axis=1)))]
        first_out = next(i for i, q in enumerate(pts) if not self.in_box(q))
        s_min = arc[first_out] + self.cfg["course"]["exit_line_m"] + 0.08
        s = max(s_stop - self.cfg["course"]["approach_zone_m"], s_min)
        return self._at(edge, s)

    def exit_line(self, junction, arm_out):
        """Point and direction of the exit beam on the departing lane (box edge + exit_line_m)."""
        u, p, edge = self.departure(junction, arm_out)
        fwd = p + u * 2.0
        exit_pt = fwd - u * self.box_edge_along(junction, fwd, -u)
        return exit_pt + u * self.cfg["course"]["exit_line_m"], u, edge

    def maneuver(self, junction, arm_in, arm_out, sensor_offset_m, stop_min_m):
        """Plan for crossing the box, measured from the point where the stop line is detected
        (sensor bar stop_min_m past the near edge of the stop tape)."""
        from picar_lab.car_agent.drive import movement
        turn = movement(junction, arm_in, arm_out)
        u_in, p_in, _ = self.approach(junction, arm_in)
        u_out, p_out, _ = self.departure(junction, arm_out)
        far, near, _ = self.stopline(junction, arm_in)
        detect = near + u_in * stop_min_m                       # sensor position at detection
        centre = detect - u_in * sensor_offset_m                # car centre at detection
        exit_pt = self.exit_line(junction, arm_out)[0] - u_out * self.cfg["course"]["exit_line_m"]
        if turn == "straight":
            d_box = float(np.dot(exit_pt - detect, u_in))           # sensor travel until tape resumes
            return {"turn": "straight", "d1": 0.0, "angle_deg": 0.0,
                    "d2_min": round(max(0.0, d_box - 0.02), 3), "d2_max": round(d_box + 0.20, 3)}
        # turn point = intersection of the incoming and outgoing lane lines
        A = np.column_stack([u_in, -u_out])
        t, _ = np.linalg.solve(A, p_out - p_in)
        tp = p_in + t * u_in
        d1 = float(np.dot(tp - centre, u_in))
        d_after = float(np.dot(exit_pt - tp, u_out)) - sensor_offset_m
        angle = 90.0 if turn == "left" else -90.0
        return {"turn": turn, "d1": round(d1, 3), "angle_deg": angle,
                "d2_min": round(max(0.0, d_after - 0.02), 3), "d2_max": round(max(0.05, d_after) + 0.20, 3)}

    def _painted(self, p, arm, divider, so):
        for jn, c in self.centres.items():
            s = 1.0 if jn == "J" else -1.0                        # connector leaves J towards +y, J2 towards -y
            dx, dy = p[0] - c[0], (p[1] - c[1]) * s
            if abs(dx) <= self.lw + 1e-6 and -self.lw + 1e-4 < dy < self.lw + 1e-4:
                return False                                      # inside the box (far edge of the through road stays)
            if divider and arm == "N" and abs(dx) < 1e-3 and dy < self.lw + so:
                return False                                      # connector divider ends at its stop lines
            if divider and arm != "N" and abs(dy) < 1e-3 and abs(dx) < self.lw + so:
                return False                                      # loop divider ends at its stop lines
        return True

    def road_lines(self):
        """Painted lane lines as polylines (outer edge, divider, inner edge of every road), split at the gaps."""
        so = self.cfg["course"]["stopline_offset_m"]
        lines = []
        for arm, pts in self.roads.items():
            dense = _densify(pts)
            for d in (-self.lw, 0.0, self.lw):
                line = _offset_polyline(dense, d) if d else dense
                keep = [self._painted(q, arm, d == 0.0, so) for q in line]
                run = []
                for q, k in zip(line, keep):
                    if k:
                        run.append(q)
                    elif len(run) > 1:
                        lines.append(np.array(run))
                        run = []
                    else:
                        run = []
                if len(run) > 1:
                    lines.append(np.array(run))
        return lines

    def arc_of(self, edge, point):
        pts = self.lanes[edge]
        return float(self.arc[edge][int(np.argmin(np.linalg.norm(pts - np.asarray(point), axis=1)))])

    def approach_point(self, edge):
        """Where a car on `edge` sends APPROACH: approach_zone_m before the stop line along the lane, but not
        before the upstream exit beam (no tape on the real course: the car measures it by odometry)."""
        arm, kind = edge.split("_")
        down, up = ("J", "J2") if kind == "in" else ("J2", "J")
        far, near, _ = self.stopline(down, arm)
        s_stop = self.arc_of(edge, near)
        s_exit = self.arc_of(edge, self.exit_line(up, arm)[0])
        return max(s_stop - self.cfg["course"]["approach_zone_m"], s_exit + 0.05), s_exit

    def approach_after_exit(self, edge):
        s_appr, s_exit = self.approach_point(edge)
        return round(s_appr - s_exit, 3)

    def lane_tolerance(self, car_width):
        """Lane departure: car centre so far off the lane centre that the body covers a whole lane line."""
        return round((self.lw - car_width) / 2 + LINE_TAPE_M, 4)

    def tex_bounds(self):
        """(x0, x1, y0, y1) of the floor texture: the course plus TEX_MARGIN of white floor on every side."""
        return -TEX_MARGIN, self.width + TEX_MARGIN, -TEX_MARGIN, self.depth + TEX_MARGIN

    def in_box(self, point, margin=0.0):
        return any(np.max(np.abs(np.asarray(point) - c)) <= self.lw + margin for c in self.centres.values())

    def lane_distance(self, point):
        """Distance to the nearest lane centreline (vectorised over all lane segments)."""
        if not hasattr(self, "_seg_a"):
            self._seg_a = np.vstack([p[:-1] for p in self.lanes.values()])
            self._seg_ab = np.vstack([np.diff(p, axis=0) for p in self.lanes.values()])
            self._seg_ll = np.maximum((self._seg_ab ** 2).sum(1), 1e-12)
        ap = np.asarray(point, float) - self._seg_a
        t = np.clip((ap * self._seg_ab).sum(1) / self._seg_ll, 0, 1)
        return float(np.sqrt(((ap - t[:, None] * self._seg_ab) ** 2).sum(1)).min())


def footprint(x, y, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2, width / 2
    return np.array([[x + c * dx - s * dy, y + s * dx + c * dy] for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])


def overlap(a, b):
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def crossed(prev, cur, line_pt, u, half_width):
    """True when a point moved from before to after a line (normal u) within half_width of line_pt."""
    if prev is None:
        return False
    s0, s1 = float(np.dot(np.asarray(prev) - line_pt, u)), float(np.dot(np.asarray(cur) - line_pt, u))
    lateral = abs(float(np.dot(np.asarray(cur) - line_pt, _right(u))))
    return s0 < 0 <= s1 and lateral <= half_width


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def ir_from_gray(gray, sensor_cols, threshold, dark_value=1):
    """gray: HxW array in [0,1]. Average a 3-pixel column band per sensor; dark -> dark_value."""
    out = []
    for c in sensor_cols:
        band = gray[:, max(0, c - 1):c + 2]
        dark = float(band.mean()) < threshold
        out.append(dark_value if dark else 1 - dark_value)
    return out


def sensor_columns(width_px, hfov_rad, height_m, sensor_y_m):
    """Image column of each sensor (left->right) for a downward camera; +y is the car's left."""
    half = height_m * math.tan(hfov_rad / 2)
    return [int(round((0.5 - y / (2 * half)) * (width_px - 1))) for y in sensor_y_m]


def forward_range(pose, others, length, width, max_range=1.0, half_angle_deg=15.0):
    """Ultrasonic-like distance from the front bumper to the nearest point of another car inside a cone."""
    x, y, yaw = pose
    h = np.array([math.cos(yaw), math.sin(yaw)])
    front = np.array([x, y]) + h * length / 2
    best = max_range
    cos_lim = math.cos(math.radians(half_angle_deg))
    for o in others:
        fp = footprint(*o, length, width)
        pts = np.vstack([fp] + [fp[i] + (fp[(i + 1) % 4] - fp[i]) * k / 6 for i in range(4) for k in range(1, 6)])
        for q in pts:
            d = q - front
            dist = float(np.linalg.norm(d))
            if 1e-6 < dist < best and float(np.dot(d, h)) / dist >= cos_lim:
                best = dist
    return best
