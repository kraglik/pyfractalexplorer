import math

import numpy as np


class Quaternion:
    def __init__(self, x, y, z, w):
        self.data = np.array([x, y, z, w], dtype=np.float32)

    def __mul__(self, other):
        l = self.data[:3]
        r = other.data[:3]

        xyz = l * other.w + r * self.w + np.cross(l, r)

        return Quaternion(
            *list(xyz),
            self.w * other.w - np.dot(l, r)
        )

    @staticmethod
    def from_axis_angle(axis: np.array, w):
        x, y, z = list(axis)

        sin_h_w = math.sin(w * 0.5)

        x *= sin_h_w
        y *= sin_h_w
        z *= sin_h_w
        w = math.cos(w * 0.5)

        return Quaternion(x, y, z, w).normalized()

    def normalized(self):
        return Quaternion(*list(self.data / np.linalg.norm(self.data)))

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    @property
    def z(self):
        return self.data[2]

    @property
    def w(self):
        return self.data[3]
