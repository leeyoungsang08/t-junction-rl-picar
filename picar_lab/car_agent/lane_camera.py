"""Front-camera lane following, as on the real PiCar (UROP2 notes): gray -> binary -> bird's-eye view ->
histogram -> sliding windows -> centre offset -> steering. numpy only (OpenCV optional on the Pi).

Car frame: x forward from the car centre, y to the left, metres. The course has three lines per two-lane road
(outer edge, centre divider, inner edge), so in a lane the car sees its right line (edge or divider) and its
left line. The right line is followed (the left one is used when the right one is missing); stop lines and other
cross tapes are removed from the bird's-eye view before the histogram.

The same CameraModel produces: the bird's-eye remap used by LaneDetector, the 2D pre-check renderer
(picar_gazebo.kinematic_check) and the Gazebo camera sensor (picar_gazebo.world_gen). On the real car the mount (pitch, yaw, sideways offset) comes from
camera_calib.py: one photo with the car parked in the middle of a straight lane.
"""
import math

import numpy as np

CAMERA = {"x": 0.06, "z": 0.15, "pitch_deg": 35.0, "hfov_deg": 62.2, "w": 160, "h": 120}   # Pi camera v2 (62 deg), scaled; mount ASSUMED
BEV = {"f_min": 0.15, "f_max": 0.58, "l_half": 0.30, "res": 0.005}                        # car frame, metres


class CameraModel:
    def __init__(self, cam=None):
        c = {**CAMERA, **(cam or {})}
        self.c = c
        p = math.radians(c["pitch_deg"])
        yw = math.radians(c.get("yaw_deg", 0.0))            # mount error: + = camera looks to the left
        cy, sy = math.cos(yw), math.sin(yw)
        self.pos = np.array([c["x"], c.get("y", 0.0), c["z"]])   # y: camera left of the car centreline
        self.fwd = np.array([math.cos(p) * cy, math.cos(p) * sy, -math.sin(p)])
        self.right = np.array([sy, -cy, 0.0])
        self.down = np.array([-math.sin(p) * cy, -math.sin(p) * sy, -math.cos(p)])
        self.w, self.h = int(c["w"]), int(c["h"])
        self.fx = (self.w / 2) / math.tan(math.radians(c["hfov_deg"]) / 2)
        self.cx, self.cy = (self.w - 1) / 2, (self.h - 1) / 2

    def project(self, X, Y):
        """Ground points (car frame) -> pixel (u, v) floats and a validity mask."""
        d = np.stack([np.asarray(X, float) - self.pos[0], np.asarray(Y, float) - self.pos[1],
                      np.full(np.shape(X), -self.pos[2])], -1)
        zc = d @ self.fwd
        u = self.cx + self.fx * (d @ self.right) / np.maximum(zc, 1e-9)
        v = self.cy + self.fx * (d @ self.down) / np.maximum(zc, 1e-9)
        ok = (zc > 1e-6) & (u >= 0) & (u <= self.w - 1) & (v >= 0) & (v <= self.h - 1)
        return u, v, ok

    def ground_of_pixels(self):
        """For every pixel: ground point (X, Y) in the car frame, NaN above the horizon (renderer)."""
        u, v = np.meshgrid(np.arange(self.w), np.arange(self.h))
        ray = (self.fwd[None, None] + ((u - self.cx) / self.fx)[..., None] * self.right
               + ((v - self.cy) / self.fx)[..., None] * self.down)
        t = np.where(ray[..., 2] < -1e-6, -self.pos[2] / np.minimum(ray[..., 2], -1e-6), np.nan)
        return self.pos[0] + t * ray[..., 0], self.pos[1] + t * ray[..., 1]


class BirdsEye:
    """Nearest-pixel remap image -> bird's-eye grid; rows far -> near, columns left -> right."""

    def __init__(self, model, bev=None):
        b = {**BEV, **(bev or {})}
        self.b = b
        self.f = np.arange(b["f_max"], b["f_min"] - 1e-9, -b["res"])          # forward distance per row
        self.l = np.arange(b["l_half"], -b["l_half"] - 1e-9, -b["res"])       # lateral (left +) per column
        F, L = np.meshgrid(self.f, self.l, indexing="ij")
        u, v, ok = model.project(F, L)
        self.iu, self.iv = np.clip(np.rint(u), 0, model.w - 1).astype(int), np.clip(np.rint(v), 0, model.h - 1).astype(int)
        self.ok = ok

    def warp(self, gray):
        out = gray[self.iv, self.iu]
        return np.where(self.ok, out, 1.0)


class LaneDetector:
    """Returns {"right": fit|None, "left": fit|None}; a fit is (a, b, c) of l(f) = a + b f + c f^2 in the car frame."""

    def __init__(self, model=None, lane_width=0.21, bev=None, window_m=0.05, n_windows=8, cross_frac=0.30):
        self.model = model or CameraModel()
        self.bev = BirdsEye(self.model, bev)
        self.lw, self.win, self.nwin, self.cross_frac = lane_width, window_m, n_windows, cross_frac
        self.prev = {"right": None, "left": None}
        self.last_bev = None
        self.max_run_px = int(round(0.055 / self.bev.b["res"]))    # widest dark run in a row that can still be tape
        self.grow_px = int(round(0.04 / self.bev.b["res"]))
        self.left_obstacle = False
        self.left_limit = lane_width / 2 + 0.05                    # divider + 5 cm (car up to 5 cm right of centre)

    def binary(self, gray):
        g = np.asarray(gray, float)
        if g.max() > 1.5:
            g = g / 255.0
        bev = self.bev.warp(g)
        b = np.pad(bev, 1, mode="edge")                                  # 3x3 mean: pixel noise must not turn a
        bev = sum(b[i:i + bev.shape[0], j:j + bev.shape[1]] for i in range(3) for j in range(3)) / 9.0   # grey car body into dark specks
        ref = np.median(bev[self.bev.ok]) if self.bev.ok.any() else 1.0
        dark = (bev < 0.55 * ref) & self.bev.ok
        frac = dark.sum(1) / np.maximum(self.bev.ok.sum(1), 1)
        cross = frac > self.cross_frac                                   # stop lines / cross tapes: whole rows
        cross = cross | np.r_[cross[1:], False] | np.r_[False, cross[:-1]]
        dark[cross] = False
        # Anything wider than tape in a row is not a line: another car (it is in the picture on the real course and
        # in Gazebo; a car waiting in the next lane merged with the divider and the lane was lost or mis-read).
        # Only the wide part of each row is dropped, so a line next to a car survives above and below it.
        pad = np.zeros((dark.shape[0], 1), bool)
        edge = np.diff(np.hstack([pad, dark, pad]).astype(np.int8), axis=1)
        r0, c0 = np.nonzero(edge == 1)
        _r1, c1 = np.nonzero(edge == -1)
        wide = np.zeros_like(dark)
        for r, a, b in zip(r0, c0, c1):
            if b - a > self.max_run_px:
                wide[r, a:b] = True
        # something wide to the left of the car's path = a car in the other lane (or about to be passed)
        self.left_obstacle = bool(wide[:, self.bev.l > 0.03].any())
        if wide.any():                                   # also the rim of the wide shape (corners of a turned car
            k = self.grow_px                             # are thin in a row and would pass as short lines)
            grown = wide.copy()
            for d in range(1, k + 1):
                grown[d:] |= wide[:-d]
                grown[:-d] |= wide[d:]
            wide = grown.copy()
            for d in range(1, k + 1):
                wide[:, d:] |= grown[:, :-d]
                wide[:, :-d] |= grown[:, d:]
            dark &= ~wide
        # Nothing left of the divider matters for staying in this lane, and that is where the oncoming cars are.
        dark[:, self.bev.l > self.left_limit] = False
        self.last_bev = dark
        return dark

    def _components(self, dark, min_px=20):
        """Connected dark blobs of the bird's-eye view (each painted line is one blob once cross tapes are gone)."""
        try:
            from scipy.ndimage import label
            lab, n = label(dark, structure=np.ones((3, 3)))
            return [np.nonzero(lab == k) for k in range(1, n + 1) if (lab == k).sum() >= min_px]
        except ImportError:                                             # small BFS fallback (Pi without scipy)
            seen = np.zeros_like(dark, bool)
            out = []
            H, W = dark.shape
            for r0, c0 in zip(*np.nonzero(dark)):
                if seen[r0, c0]:
                    continue
                stack, rr, cc = [(r0, c0)], [], []
                seen[r0, c0] = True
                while stack:
                    r, c = stack.pop()
                    rr.append(r)
                    cc.append(c)
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            a, b = r + dr, c + dc
                            if 0 <= a < H and 0 <= b < W and dark[a, b] and not seen[a, b]:
                                seen[a, b] = True
                                stack.append((a, b))
                if len(rr) >= min_px:
                    out.append((np.array(rr), np.array(cc)))
            return out

    def detect(self, gray):
        """{"right": line|None, "left": line|None}; line = {"fit": (a, b, c), "f0": nearest f, "f1": farthest f,
        "l0": lateral position of its nearest visible part}."""
        dark = self.binary(gray)
        lines = []
        for rr, cc in self._components(dark):
            f, l = self.bev.f[rr], self.bev.l[cc]
            f0, f1 = float(f.min()), float(f.max())
            if f1 - f0 < 0.08:
                continue                                                # a blob, not a line along the lane
            if len(rr) / len(np.unique(rr)) * self.bev.b["res"] > 0.08:
                continue                                                # too wide for tape (a car ahead, a shadow)
            deg = 2 if f1 - f0 >= 0.12 else 1
            coef = np.polyfit(f, l, deg)[::-1]
            near = f < f0 + 0.02
            fit = tuple(np.r_[coef, [0.0] * (3 - len(coef))])
            l0 = float(l[near].mean())
            # which side of the car the line is on: where it would pass beside the car, not where its visible part
            # starts (a car standing 10 deg skewed sees its right line start left of centre, 0.35 m ahead)
            side = l0 - float(np.clip(slope(fit, f0), -1.0, 1.0)) * (f0 - self.bev.b["f_min"])
            lines.append({"fit": fit, "f0": f0, "f1": f1, "l0": l0, "side": side})
        out = {}
        for side, sign in (("right", -1), ("left", 1)):
            prev = self.prev[side]
            want = prev["l0"] if prev else sign * self.lw / 2
            # a line belongs to this side if it lies on that side of the car, or if it continues the line tracked
            # there in the last frame (in a tight curve the near end of the outer line swings past the centre)
            # the right line (road edge) is accepted across the whole view: losing it while the far edge is still
            # visible made a car that left a box 5 cm to the left steer into the opposite lane (rehearsal). The left
            # line must be where a divider can be (0.02..0.19 m to the left); farther left it is the other lane's edge.
            c = [ln for ln in lines if ln["f0"] < 0.50
                 and (-0.29 < ln["side"] < -0.005 if sign < 0 else 0.02 < ln["side"] < 0.19)
                 or (prev is not None and abs(ln["l0"] - prev["l0"]) < 0.04 and abs(ln["l0"]) < 0.22 and ln["f0"] < 0.40)]
            out[side] = min(c, key=lambda ln: abs(ln["l0"] - want) + 0.3 * (ln["f0"] - self.bev.b["f_min"])) if c else None
        if out["right"] is not None and out["right"] is out["left"]:      # one line claimed by both sides
            r_prev, l_prev = self.prev["right"], self.prev["left"]
            dr = abs(out["right"]["l0"] - (r_prev["l0"] if r_prev else -self.lw / 2))
            dl = abs(out["left"]["l0"] - (l_prev["l0"] if l_prev else self.lw / 2))
            out["right" if dr > dl else "left"] = None
        r, lft = out["right"], out["left"]
        if r and lft:                                                    # two lines must be about a lane apart
            sep = lft["l0"] - r["l0"]
            if not 0.5 * self.lw < sep < 1.6 * self.lw:
                out["left" if abs(lft["l0"] - self.lw / 2) > abs(r["l0"] + self.lw / 2) else "right"] = None
        self.prev = out
        return out


def at(fit, f):
    return fit[0] + fit[1] * f + fit[2] * f * f


def slope(fit, f):
    return fit[1] + 2 * fit[2] * f


class LaneFollower:
    """Pure pursuit on the lane centre at a look-ahead distance, centre from the right line (+ lane/2) or the
    left line (- lane/2); both lines are averaged when both are seen."""

    def __init__(self, speed_mps=0.25, lane_width=0.21, lookahead_m=0.21, lookahead_min_m=0.18, kd=0.25, w_max=3.0,
                 lost_timeout_s=0.5, prefer="right", ki=1.5, i_max=0.20, keep_right_m=0.005, left_curve_m=0.005, right_curve_m=-0.005):
        self.v, self.lw, self.ld, self.kd, self.w_max = speed_mps, lane_width, lookahead_m, kd, w_max
        # integral on the near lateral error: pure pursuit alone leaves a constant offset when one motor is weaker
        # (rehearsal: 9 % weaker right motor -> car 28 mm right of the lane centre; the lane has 20 mm to spare)
        self.ki, self.i_max, self.i_term = ki, i_max, 0.0
        # Opposite lanes pass with 4 cm between the cars, and in a corner the corners of the inner car leave 1.6 cm
        # to the outer one. A 0.30 m look-ahead starts turning before the curve and cut left curves 16-18 mm on
        # the inside: in Gazebo the two cars touched there, locked and pushed each other round (fleet run 018).
        # Hence the short look-ahead, 5 mm right of centre everywhere, 10 mm in left curves, centre in right
        # curves (the inner lane only shows the divider, moving away from it loses the lane).
        self.keep_right, self.left_curve, self.right_curve = keep_right_m, left_curve_m, right_curve_m
        self.pass_right, self.pass_shift = 0.015, 0.0     # shift while a car is seen in the other lane
        self.shift = 0.0
        self.v_curve_min, self.v_curve_gain = 0.55, 0.35      # speed factor in curves: floor and slope
        self.ld_min = lookahead_min_m
        self.lost_timeout, self.prefer = lost_timeout_s, prefer
        self.last_ok, self.prev_y, self.prev_t = None, None, None
        self.last_k = 0.0                                     # curvature of the last good frame (1/m, + = left)
        self.prev_i_t = None

    def _centre_points(self, line, sign):
        """Points of the lane centre: the line shifted half a lane along its normal (towards the lane)."""
        fs = np.linspace(line["f0"], line["f1"], 40)
        ls = at(line["fit"], fs)
        sl = slope(line["fit"], fs)
        n = np.hypot(1.0, sl)
        return fs - sign * (-sl / n) * self.lw / 2, ls - sign * (1.0 / n) * self.lw / 2

    def goal(self, lanes, ld=None):
        """(forward, lateral) of the lane-centre point at distance ld from the car centre, or None."""
        ld = ld or self.ld
        goals = []
        for side, sign in (("right", -1), ("left", 1)):
            line = lanes.get(side)
            if not line:
                continue
            F, L = self._centre_points(line, sign)
            d = np.hypot(F, L)
            if d.max() < ld:                     # line leaves the view before ld (tight curve): take its far end,
                k = int(np.argmax(d))            # never a straight extension (that under-steers the corner)
            else:
                k = int(np.argmin(np.abs(d - ld)))
            goals.append((side, np.array([F[k], L[k]])))
        if not goals:
            return None
        if len(goals) == 2:
            w = {"right": 0.6, "left": 0.4} if self.prefer == "right" else {"right": 0.5, "left": 0.5}
            return sum(w[sd] * g for sd, g in goals)
        return goals[0][1]

    def target(self, lanes):
        g = self.goal(lanes)
        return None if g is None else float(g[1])

    def update(self, lanes, now, moving=True, oncoming=False):
        """moving=False (waiting at a stop line, blocked): no integration, or the term winds up while standing.
        oncoming=True (a car seen in the other lane): keep to the right of the lane while passing it."""
        g = self.goal(lanes)
        self.pass_shift += (((self.pass_right if oncoming else 0.0) - self.pass_shift) * 0.25)   # eased in/out
        if g is None:
            ok = self.last_ok is not None and now - self.last_ok < self.lost_timeout
            # keep the last curvature for these few frames: straight on in a 25 cm corner leaves the lane in 0.2 s
            v = self.v * 0.5 if ok else 0.0
            return v, v * self.last_k, ok
        self.last_ok = now
        k0 = 2 * g[1] / max(g @ g, 1e-4)
        want = self.keep_right + (self.left_curve if k0 * self.ld > 0.25 else 0.0) \
            + (self.right_curve if k0 * self.ld < -0.25 else 0.0)
        self.shift += (want - self.shift) * 0.1                                  # eased: no step in the goal
        g = g - np.array([0.0, self.shift + self.pass_shift])
        k = 2 * g[1] / max(g @ g, 1e-4)              # pure pursuit through the goal point
        if abs(k) * self.ld > 0.6:                   # tight corner (radius < ~0.5 m): a shorter look-ahead, but only
            g2 = self.goal(lanes, self.ld_min)       # if it agrees (near the car the centre estimate is the weakest)
            k2 = 2 * g2[1] / max(g2 @ g2, 1e-4)
            if k2 * k > 0:
                g, k = g2, k2
        self.last_k = float(np.clip(k, -4.5, 4.5))
        y = float(g[1])
        dy = 0.0 if self.prev_t is None or now <= self.prev_t else (y - self.prev_y) / (now - self.prev_t)
        dy = float(np.clip(dy, -0.5, 0.5))           # goal jumps (line switch) must not kick the steering
        self.prev_y, self.prev_t = y, now
        v = self.v * max(self.v_curve_min, 1.0 - self.v_curve_gain * abs(k) * self.ld)
        dt = 0.0 if self.prev_i_t is None else min(0.2, max(0.0, now - self.prev_i_t))
        self.prev_i_t = now
        near = self.goal(lanes, self.ld_min)
        if moving and near is not None and abs(k) * self.ld < 0.3:        # on straights only: in corners the goal is not the error
            # the error is measured from where the car is meant to drive (keep_right), or the integrator pulls the
            # car back to the lane centre against the shift and winds up
            e = float(near[1]) - self.shift - self.pass_shift
            self.i_term = float(np.clip(self.i_term + self.ki * e * dt, -self.i_max, self.i_max))
        w = float(np.clip(v * k + self.kd * dy + self.i_term, -self.w_max, self.w_max))
        return v, w, True

    def lane_pose(self, lanes):
        """(lateral offset of the lane centre ahead [m, + = left], lane direction relative to the car [deg, + = the
        lane runs to the left]) or None when no lane line is seen."""
        g = self.goal(lanes)
        line = lanes.get("right") or lanes.get("left")
        if g is None or line is None or line["f1"] - line["f0"] < 0.15:      # a stub of a line has no direction
            return None
        return float(g[1]), math.degrees(math.atan(slope(line["fit"], (line["f0"] + line["f1"]) / 2)))

    def aligned(self, lanes, max_offset=0.06, max_heading_deg=30.0):
        """Lane seen and the car roughly on it (used to end a junction maneuver)."""
        g = self.goal(lanes)
        line = lanes.get("right") or lanes.get("left")
        if g is None or line is None:
            return False
        heading = math.degrees(math.atan(slope(line["fit"], line["f0"])))
        return abs(g[1]) < max_offset + 0.04 and abs(heading) < max_heading_deg
