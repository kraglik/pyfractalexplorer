from abc import abstractmethod
import pyopencl as cl
import pyopencl.cltypes
import numpy as np

from .camera import Camera
from .fractals import Fractal


class Render:

    quality_props_dtype = np.dtype([
        ("iteration_limit", cl.cltypes.int),
        ("ray_steps_limit", cl.cltypes.int),
        ("epsilon", cl.cltypes.float),
        ("ray_shift_multiplier", cl.cltypes.float)
    ])

    def __init__(self,
                 camera: Camera,
                 fractal: Fractal,
                 context: cl.Context,
                 width: int = 500, height: int = 500):

        self.width = width
        self.height = height

        self.camera = camera
        self.context = context
        self.fractal = fractal

        self.buffer = cl.Image(
            context,
            cl.ImageFormat(cl.channel_order.RGB)
        )

    def resize(self, width, height):
        self.width = width
        self.height = height

        self.buffer = cl.Image

    def render(self):
        raise NotImplementedError

    def save(self, path):
        pass
