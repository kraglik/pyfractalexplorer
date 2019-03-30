import pyopencl as cl
import pyopencl.cltypes
import pyopencl.tools
import numpy as np
from typing import Optional, Dict, Tuple

from .camera import Camera
from .fractals import fractals


class Render:

    quality_props = np.dtype([
        ("iteration_limit", cl.cltypes.int),
        ("ray_steps_limit", cl.cltypes.int),
        ("epsilon", cl.cltypes.float),
        ("ray_shift_multiplier", cl.cltypes.float),
        ("render_simple", cl.cltypes.int),
        ("sun_direction", cl.cltypes.float3),
        ("reflection_depth", cl.cltypes.int),
        ("use_orbit_trap", cl.cltypes.int)
    ])

    def __init__(self,
                 device: cl.Device,
                 context: cl.Context,
                 queue: cl.CommandQueue,
                 camera: Camera,
                 width: int = 500, height: int = 500,
                 iteration_limit=None,
                 ray_steps_limit=128,
                 epsilon=0.01,
                 render_simple=True,
                 sun_direction=(-1, 1, -1),
                 reflection_depth=1,
                 use_orbit_trap=True,
                 ray_shift_multiplier=1.0):

        self.device = device
        self.context = context
        self.queue = queue
        self.iteration_limit = iteration_limit
        self.ray_steps_limit = ray_steps_limit
        self.epsilon = epsilon
        self.ray_shift_multiplier = ray_shift_multiplier
        self.render_simple = render_simple
        self.sun_direction = sun_direction
        self.reflection_depth = reflection_depth
        self.use_orbit_trap = use_orbit_trap

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

        self.fractals = [
            fractal_class(
                device, context, queue,
                [camera.cl_type_declaration, self._quality_props_decl]
            )
            for fractal_class in fractals
        ]

        self.fractal_by_name = {fractal.get_name(): fractal for fractal in self.fractals}

        self.fractal = self.fractals[0]

        self._host_image_buffer = np.zeros(self.width * self.height * 4, dtype=np.uint8)
        self._image_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY,
            self._host_image_buffer.nbytes
        )

        self._quality_props_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY,
            self._quality_props_dtype.itemsize
        )

        self.sync_with_device()

    def resize(self, width, height):
        self.width = max(1, width)
        self.height = max(1, height)

        self._host_image_buffer = np.zeros(self.width * self.height * 4, dtype=np.uint8)
        self._image_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY,
            self._host_image_buffer.nbytes
        )

    def sync_with_device(self):
        quality_props_instance = np.array([(
            self.iteration_limit if self.iteration_limit is not None else self.fractal.get_default_iterations(),
            self.ray_steps_limit,
            self.epsilon,
            self.ray_shift_multiplier,
            self.render_simple,
            self.sun_direction + (0, ),
            self.reflection_depth,
            self.use_orbit_trap
        )], dtype=self._quality_props_dtype)[0]

        cl.enqueue_copy(self.queue, self._quality_props_buffer, quality_props_instance)

        self.camera.sync_with_device()
        self.fractal.sync_with_device()

    def render(self):
        self.sync_with_device()

        render_event = self.fractal.render_function(
            self.queue,
            (self.width, self.height),
            None,
            self.camera.buffer,
            self._quality_props_buffer,
            self.fractal.get_parameters_buffer(),
            self.fractal.get_material_buffer(),
            self._image_buffer
        )

        render_event.wait()

        cl.enqueue_copy(
            self.queue,
            self._host_image_buffer,
            self._image_buffer
        )

    def save(self, path):
        from PIL import Image

        image = Image\
            .fromarray(self._host_image_buffer.reshape((self.width, self.height, 4))[:, :, :3])\
            .transpose(Image.ROTATE_90)\
            .transpose(Image.FLIP_TOP_BOTTOM)
        image.save(path)

    @property
    def cl_type_declaration(self):
        return self._quality_props_decl

    @property
    def host_buffer(self):
        return self._host_image_buffer
