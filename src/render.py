from abc import abstractmethod
import pyopencl as cl
import pyopencl.cltypes
import pyopencl.tools
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from typing import Optional, Dict, Tuple

from .camera import Camera
from .fractals import Fractal


class Render:

    quality_props = np.dtype([
        ("iteration_limit", cl.cltypes.int),
        ("ray_steps_limit", cl.cltypes.int),
        ("epsilon", cl.cltypes.float),
        ("ray_shift_multiplier", cl.cltypes.float)
    ])

    def __init__(self,
                 device: cl.Device,
                 context: cl.Context,
                 queue: cl.CommandQueue,
                 camera: Camera,
                 fractal_class: type,
                 fractal_parameters: Optional[Dict],
                 fractal_color: Optional[Tuple[float, float, float]],
                 width: int = 500, height: int = 500,
                 iteration_limit=16,
                 ray_steps_limit=128,
                 epsilon=0.001,
                 ray_shift_multiplier=1.0):

        self.device = device
        self.context = context
        self.queue = queue

        self.iteration_limit = iteration_limit
        self.ray_steps_limit = ray_steps_limit
        self.epsilon = epsilon
        self.ray_shift_multiplier = ray_shift_multiplier

        self.width = max(1, width)
        self.height = max(1, height)

        self.camera = camera

        self._quality_props_dtype, self._quality_props_decl = cl.tools.match_dtype_to_c_struct(
            self.device,
            "QualityProps",
            self.quality_props
        )

        self._quality_props_dtype = cl.tools.get_or_register_dtype(
            "QualityProps",
            self._quality_props_dtype
        )

        self.fractal = fractal_class(
            device, context, queue,
            [camera.cl_type_declaration, self._quality_props_decl],
            fractal_parameters, fractal_color
        )

        self._host_image_buffer = np.zeros((self.width, self.height, 4), dtype=np.uint8)
        self._image_buffer = cl.Image(
            self.context,
            cl.mem_flags.READ_WRITE,
            cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
            shape=(self.width, self.height)
        )

        self._quality_props_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY,
            self._quality_props_dtype.itemsize
        )

        self._fractal_color_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY,
            cl.cltypes.uchar3.itemsize
        )

        self.sync_with_device()

    def resize(self, width, height):
        self.width = max(1, width)
        self.height = max(1, height)

        self._host_image_buffer = np.zeros((self.width, self.height, 4), dtype=np.uint8)
        self._image_buffer = cl.Image(
            self.context,
            cl.mem_flags.READ_WRITE,
            cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
            shape=(self.width, self.height)
        )

    def sync_with_device(self):
        quality_props_instance = np.array([(
            self.iteration_limit,
            self.ray_steps_limit,
            self.epsilon,
            self.ray_shift_multiplier
        )], dtype=self._quality_props_dtype)[0]

        cl.enqueue_copy(self.queue, self._quality_props_buffer, quality_props_instance)

        color = np.array([self.fractal.get_color() + (0,)], dtype=cl.cltypes.uchar3)
        cl.enqueue_copy(self.queue, self._fractal_color_buffer, color)

    def render(self):
        self.sync_with_device()
        self.camera.sync_with_device()

        self.fractal.render_function(
            self.queue,
            self._image_buffer.shape,
            None,
            self.camera.buffer,
            self._quality_props_buffer,
            self.fractal.get_parameters_buffer(),
            self._fractal_color_buffer,
            self._image_buffer
        )

        cl.enqueue_copy(
            self.queue,
            self._host_image_buffer,
            self._image_buffer,
            origin=(0, 0), region=(self.width, self.height)
        )

    def save(self, path):
        from PIL import Image

        image_array = np.zeros((self.width, self.height, 4), dtype=np.uint8)

        cl.enqueue_copy(self.queue, image_array, self._image_buffer, origin=(0, 0), region=(self.width, self.height))

        image = Image.fromarray(image_array)
        image.save(path)

    @property
    def cl_type_declaration(self):
        return self._quality_props_decl

    @property
    def host_buffer(self):
        return self._host_image_buffer
