"""Front-camera lane following, as on the real PiCar (UROP2 notes): gray -> binary -> bird's-eye view ->
histogram -> sliding windows -> centre offset -> steering. numpy only (OpenCV optional on the Pi).

Car frame: x forward from the car centre, y to the left, metres. The course has three lines per two-lane road
(outer edge, centre divider, inner edge), so in a lane the car sees its right line (edge or divider) and its
left line. The right line is followed (the left one is used when the right one is missing); stop lines and other
cross tapes are removed from the bird's-eye view before the histogram.

The same CameraModel produces: the bird's-eye remap used by LaneDetector, the 2D pre-check renderer
(picar_gazebo.kinematic_check) and the Gazebo camera sensor (picar_gazebo.world_gen). On the real car replace it
by the measured mount (height, pitch, forward offset) or by the four-point homography used before.
"""
import math

import numpy as np

CAMERA = {"x": 0.06, "z": 0.15, "pitch_deg": 35.0, "hfov_deg": 62.2, "w": 160, "h": 120}   # Pi camera v2 (62 deg), scaled; mount ASSUMED
BEV = {"f_min": 0.15, "f_max": 0.58, "l_half": 0.25, "res": 0.005}                        # car frame, metres


class CameraModel:
    def __init__(self, cam=None):
        c = {**CAMERA, **(cam or {})}
        self.c = c
        p = math.radians(c["pitch_deg"])
        self.pos = np.array([c["x"], 0.0, c["z"]])
        self.fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
        self.right = np.array([0.0, -1.0, 0.0])
        self.down = np.array([-math.sin(p), 0.0, -math.cos(p)])
        self.w, self.h = int(c["w"]), int(c["h"])
        self.fx = (self.w / 2) / math.tan(math.radians(c["hfov_deg"]) / 2)
        self.cx, self.cy = (self.w - 1) / 2, (self.h - 1) / 2

    def project(self, X, Y):
        """Ground points (car frame) -> pixel (u, v) floats and a validity mask."""
        d = np.stack([np.asarray(X, float) - self.pos[0], np.asarray(Y, float), np.full(np.shape(X), -self.pos[2])], -1)
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
        return self.pos[0] + t * ray[..., 0], t * ray[..., 1]


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

    def binary(self, gray):
        g = np.asarray(gray, float)
        if g.max() > 1.5:
            g = g / 255.0
        bev = self.bev.warp(g)
        ref = np.median(bev[self.bev.ok]) if self.bev.ok.any() else 1.0
        dark = (bev < 0.55 * ref) & self.bev.ok
        frac = dark.sum(1) / np.maximum(self.bev.ok.sum(1), 1)
        cross = frac > self.cross_frac                                   # stop lines / cross tapes: whole rows
        cross = cross | np.r_[cross[1:], False] | np.r_[False, cross[:-1]]
        dark[cross] = False
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
            if f1 - f0 < 0.05:
                continue                                                # a blob, not a line along the lane
            if len(rr) / len(np.unique(rr)) * self.bev.b["res"] > 0.08:
                continue                                                # too wide for tape (a car ahead, a shadow)
            deg = 2 if f1 - f0 >= 0.12 else 1
            coef = np.polyfit(f, l, deg)[::-1]
            near = f < f0 + 0.02
            lines.append({"fit": tuple(np.r_[coef, [0.0] * (3 - len(coef))]), "f0": f0, "f1": f1,
                          "l0": float(l[near].mean())})
        out = {}
        for side, sign in (("right", -1), ("left", 1)):
            prev = self.prev[side]
            want = prev["l0"] if prev else sign * self.lw / 2
            # a line belongs to this side if it lies on that side of the car, or if it continues the line tracked
            # there in the last frame (in a tight curve the near end of the outer line swings past the centre)
            c = [ln for ln in lines if abs(ln["l0"]) < 0.22 and ln["f0"] < 0.40
                 and (sign * ln["l0"] > 0.005 or (prev is not None and abs(ln["l0"] - prev["l0"]) < 0.04))]
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

    def __init__(self, speed_mps=0.25, lane_width=0.21, lookahead_m=0.30, lookahead_min_m=0.21, kd=0.25, w_max=3.0,
                 lost_timeout_s=0.5, prefer="right"):
        self.v, self.lw, self.ld, self.kd, self.w_max = speed_mps, lane_width, lookahead_m, kd, w_max
        self.ld_min = lookahead_min_m
        self.lost_timeout, self.prefer = lost_timeout_s, prefer
        self.last_ok, self.prev_y, self.prev_t = None, None, None

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

    def update(self, lanes, now):
        g = self.goal(lanes)
        if g is None:
            ok = self.last_ok is not None and now - self.last_ok < self.lost_timeout
            return (self.v * 0.5 if ok else 0.0), 0.0, ok
        self.last_ok = now
        k = 2 * g[1] / max(g @ g, 1e-4)              # pure pursuit through the goal point
        if abs(k) * self.ld > 0.6:                   # tight corner (radius < ~0.5 m): a shorter look-ahead, but only
            g2 = self.goal(lanes, self.ld_min)       # if it agrees (near the car the centre estimate is the weakest)
            k2 = 2 * g2[1] / max(g2 @ g2, 1e-4)
            if k2 * k > 0:
                g, k = g2, k2
        y = float(g[1])
        dy = 0.0 if self.prev_t is None or now <= self.prev_t else (y - self.prev_y) / (now - self.prev_t)
        dy = float(np.clip(dy, -0.5, 0.5))           # goal jumps (line switch) must not kick the steering
        self.prev_y, self.prev_t = y, now
        v = self.v * max(0.55, 1.0 - 0.35 * abs(k) * self.ld)
        w = float(np.clip(v * k + self.kd * dy, -self.w_max, self.w_max))
        return v, w, True

    def aligned(self, lanes, max_offset=0.06, max_heading_deg=30.0):
        """Lane seen and the car roughly on it (used to end a junction maneuver)."""
        g = self.goal(lanes)
        line = lanes.get("right") or lanes.get("left")
        if g is None or line is None:
            return False
        heading = math.degrees(math.atan(slope(line["fit"], line["f0"])))
        return abs(g[1]) < max_offset + 0.04 and abs(heading) < max_heading_deg
