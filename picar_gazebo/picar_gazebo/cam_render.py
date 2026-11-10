"""Render the front camera from the course texture (2D pre-check; Gazebo renders the same floor with its camera)."""
import math

import numpy as np

from picar_lab.car_agent.lane_camera import CameraModel


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

    def render(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + c * self.X - s * self.Y
        wy = y + s * self.X + c * self.Y
        u = np.rint((wx - self.x0) * self.ppm).astype(int)
        v = np.rint((self.y1 - wy) * self.ppm).astype(int)
        inside = self.valid & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        img = np.full(self.X.shape, self.floor)
        img[inside] = self.tex[v[inside], u[inside]] * 0.85 + 0.05        # rendered white ~0.9, black ~0.05
        img[~self.valid] = self.sky
        return img
