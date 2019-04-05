import math

import numpy as np
import pyopencl as cl
import pyopencl.cltypes
import pyopencl.tools
import os

from .quaternion import Quaternion


os.environ['SDL_VIDEO_CENTERED'] = '1'


class Camera:

    camera_dtype = np.dtype([
        ("pos", cl.cltypes.float3),
        ("dir", cl.cltypes.float3),
        ("up", cl.cltypes.float3),
        ("right", cl.cltypes.float3),
        ("zoom", cl.cltypes.float),
        ("shift_multiplier", cl.cltypes.float)
    ])

    def __init__(self,
                 device: cl.Device,
                 context: cl.Context,
                 queue: cl.CommandQueue,
                 position=np.array([-10, 0, 0], dtype=cl.cltypes.float),
                 direction=np.array([1, 0, 0], dtype=cl.cltypes.float),
                 up=np.array([0, 1, 0], dtype=cl.cltypes.float),
                 target=None,
                 zoom=1.0,
                 shift_multiplier=1.0,
                 mouse_speed=2.0):

        self.device = device
        self.context = context
        self.queue = queue

        self.world_up = up

        self.position = position
        self.direction = direction if target is None else (target - position)
        self.direction /= np.linalg.norm(self.direction)

        self.right = np.cross(self.direction, up)
        self.right /= np.linalg.norm(self.right)

        self.up = np.cross(self.right, self.direction)
        self.up /= np.linalg.norm(self.up)

        self.zoom = zoom
        self.shift_multiplier = shift_multiplier
        self.mouse_speed = mouse_speed

        self._camera_dtype, self._camera_decl = cl.tools.match_dtype_to_c_struct(
            self.device,
            "Camera",
            self.camera_dtype
        )

        self._camera_dtype = cl.tools.get_or_register_dtype(
            "Camera",
            self._camera_dtype
        )

        self._buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY,
            self._camera_dtype.itemsize
        )

        self.sync_with_device()

    def look_at(self, point):
        self.direction = point - self.position
        self.direction /= np.linalg.norm(self.direction)

        self.right = np.cross(self.direction, self.world_up)
        self.right /= np.linalg.norm(self.right)

        self.up = np.cross(self.right, self.direction)
        self.up /= np.linalg.norm(self.up)

    def _pitch(self, radians):
        q = Quaternion.from_axis_angle(self.right, radians / self.zoom)
        q *= Quaternion(*list(self.direction), 0.0)

        forward = np.array([q.x, q.y, q.z])

        self.look_at(self.position + self.direction + forward)

    def _yaw(self, radians):
        q = Quaternion.from_axis_angle(self.world_up, radians / self.zoom)
        q *= Quaternion(*list(self.direction), 0.0)

        forward = np.array([q.x, q.y, q.z])

        self.look_at(self.position + self.direction + forward)

    def sync_with_device(self):
        camera_instance = np.array([(
            tuple(self.position) + (0,),
            tuple(self.direction) + (0,),
            tuple(self.up) + (0,),
            tuple(self.right) + (0,),
            self.zoom,
            self.shift_multiplier
        )], dtype=self._camera_dtype)[0]

        cl.enqueue_copy(self.queue, self._buffer, camera_instance)

    @property
    def cl_type_declaration(self):
        return self._camera_decl

    @property
    def buffer(self):
        return self._buffer

    def rotate(self, dx=0.0, dy=0.0):
        deg_to_rad = math.pi / 180.0

        look_right_rads = self.mouse_speed * dx * deg_to_rad
        look_up_rads = self.mouse_speed * dy * deg_to_rad

        current_dec = math.acos(self.direction[1])
        requested_dec = current_dec - look_up_rads

        min_up_tilt_deg = 1
        zenith_min_dec = min_up_tilt_deg * deg_to_rad
        zenith_max_dec = (180.0 - min_up_tilt_deg) * deg_to_rad

        look_up_rads = (
            zenith_min_dec
            if requested_dec < zenith_min_dec
            else (
                zenith_max_dec
                if requested_dec > zenith_max_dec
                else look_up_rads
            )
        )

        self._pitch(-look_up_rads)
        self._yaw(-look_right_rads)
