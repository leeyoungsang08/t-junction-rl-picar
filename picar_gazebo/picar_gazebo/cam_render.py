"""Render the front camera from the course texture (2D pre-check; Gazebo renders the same floor with its camera)."""
import math

import numpy as np

from picar_lab.car_agent.lane_camera import CameraModel


CAR_HEIGHT = 0.07        # Gazebo model: 5 cm body on 3.3 cm wheels
SIDE_SHADE = 0.7         # side faces are darker than the top under the overhead light


class CamRenderer:
    def __init__(self, tex, bounds, model=None, floor=0.78, sky=0.7):
        self.tex = tex                                   # gray texture, rows top->bottom, values 0..1
        self.x0, self.x1, self.y0, self.y1 = bounds
        self.H, self.W = tex.shape
        self.ppm = self.W / (self.x1 - self.x0)
        self.model = model or CameraModel()
        X, Y = self.model.ground_of_pixels()
        self.valid = np.isfinite(X)
        self.X, self.Y = np.where(self.valid, X, 0.0), np.where(self.valid, Y, 0.0)
        self.floor, self.sky = floor, sky

    def render(self, x, y, yaw, others=()):
        """others: (x, y, yaw, gray, length, width[, height]) of the other cars, drawn as boxes with height
        (Gazebo's camera sees them too; painting only the footprint on the floor hid the smear behind the body
        and two cars meeting in a corner then circled each other in Gazebo but not here)."""
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + c * self.X - s * self.Y
        wy = y + s * self.X + c * self.Y
        u = np.rint((wx - self.x0) * self.ppm).astype(int)
        v = np.rint((self.y1 - wy) * self.ppm).astype(int)
        inside = self.valid & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        img = np.full(self.X.shape, self.floor)
        img[inside] = self.tex[v[inside], u[inside]] * 0.85 + 0.05        # rendered white ~0.9, black ~0.05
        px, py, pz = self.model.pos
        ex, ey = x + c * px - s * py, y + s * px + c * py            # camera position (world)
        for ox, oy, oyaw, gray, length, width, *rest in others:
            if (ox - x) ** 2 + (oy - y) ** 2 > 1.2 ** 2:
                continue
            height = rest[0] if rest else CAR_HEIGHT
            co, so = math.cos(oyaw), math.sin(oyaw)
            # segment eye -> ground point in the other car's frame, clipped against its box (slab test):
            # a 7 cm high body seen from 15 cm covers the floor behind it up to about twice its distance
            a = (co * (ex - ox) + so * (ey - oy), -so * (ex - ox) + co * (ey - oy), pz)
            b = (co * (wx - ox) + so * (wy - oy), -so * (wx - ox) + co * (wy - oy), 0.0)
            t0, t1 = np.zeros(self.X.shape), np.ones(self.X.shape)
            face = np.zeros(self.X.shape, int)
            for k, (lo, hi) in enumerate(((-length / 2, length / 2), (-width / 2, width / 2), (0.0, height))):
                d = b[k] - a[k]
                d = np.where(np.abs(d) < 1e-9, 1e-9, d)
                ta, tb = (lo - a[k]) / d, (hi - a[k]) / d
                near = np.minimum(ta, tb)
                face = np.where(near > t0, k, face)
                t0, t1 = np.maximum(t0, near), np.minimum(t1, np.maximum(ta, tb))
            body = self.valid & (t0 < t1)
            img[body] = np.where(face[body] == 2, gray, gray * SIDE_SHADE)
        img[~self.valid] = self.sky
        return img
