import numpy as np
import pyopencl as cl
import pyopencl.cltypes


class Camera:

    camera_dtype = np.dtype([
        ("pos", cl.cltypes.float3),
        ("dir", cl.cltypes.float3),
        ("up", cl.cltypes.float3),
        ("right", cl.cltypes.float3),
        ("view_plane_distance", cl.cltypes.float),
        ("ratio", cl.cltypes.float),
        ("shift_multiplier", cl.cltypes.float),
        ("height", cl.cltypes.int),
        ("width", cl.cltypes.int)
    ])

    def __init__(self,
                 position=np.array([10, 0, 0], dtype=cl.cltypes.float),
                 direction=np.array([-1, 0, 0], dtype=cl.cltypes.float),
                 up=np.array([0, 1, 0], dtype=cl.cltypes.float),
                 target=None,
                 view_plane_distance=1.0,
                 ratio=1.0,
                 shift_multiplier=0.01,
                 height=500,
                 width=500):
        self.position = position
        self.direction = direction if target is None else (target - position)
        self.direction /= np.linalg.norm(self.direction)

        self.right = np.cross(self.direction, up)
        self.right /= np.linalg.norm(self.right)

        self.up = up / np.linalg.norm(up)
        self.view_plane_distance = view_plane_distance
        self.ratio = ratio
        self.shift_multiplier = shift_multiplier
        self.height = height
        self.width = width

    def zoom(self, value):
        self.view_plane_distance *= value

    @property
    def raw(self):
        raw_camera = np.array([
            tuple(self.position) + (),
            tuple(self.direction) + (),
            tuple(self.up) + (),
            tuple(self.right) + (),
            self.view_plane_distance,
            self.ratio,
            self.shift_multiplier,
            self.width,
            self.height
        ], dtype=self.camera_dtype)

        return raw_camera[0]
