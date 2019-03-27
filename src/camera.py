import numpy as np
import pyopencl as cl
import pyopencl.cltypes
import pyopencl.tools
import os


os.environ['SDL_VIDEO_CENTERED'] = '1'


class Camera:

    camera_dtype = np.dtype([
        ("pos", cl.cltypes.float3),
        ("dir", cl.cltypes.float3),
        ("up", cl.cltypes.float3),
        ("right", cl.cltypes.float3),
        ("view_plane_distance", cl.cltypes.float),
        ("ratio", cl.cltypes.float),
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
                 view_plane_distance=1.0,
                 ratio=1.0,
                 shift_multiplier=0.0001):

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

        self.view_plane_distance = view_plane_distance
        self.ratio = ratio
        self.shift_multiplier = shift_multiplier

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

    def zoom(self, value):
        self.view_plane_distance *= value

    def sync_with_device(self):
        camera_instance = np.array([(
            tuple(self.position) + (0,),
            tuple(self.direction) + (0,),
            tuple(self.up) + (0,),
            tuple(self.right) + (0,),
            self.view_plane_distance,
            self.ratio,
            self.shift_multiplier
        )], dtype=self._camera_dtype)[0]

        cl.enqueue_copy(self.queue, self._buffer, camera_instance)

    @property
    def cl_type_declaration(self):
        return self._camera_decl

    @property
    def buffer(self):
        return self._buffer

    def rotate(self, x=0.0, y=0.0):
        self.look_at(self.position + self.direction + self.right * x)

        up = np.cross(self.right, self.direction)

        self.look_at(self.position + self.direction + up * y)
