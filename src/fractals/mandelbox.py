from copy import copy
from typing import Optional, List, Tuple, Dict

import pyopencl as cl
import pyopencl.cltypes
import pyopencl.tools
import numpy as np

from .fractal import Fractal


class Mandelbox(Fractal):

    _default_parameters = {
        "r_min": 0.5,
        "escape_time": 100.0,
        "scale": 2.39128
    }

    mandelbox_parameters = np.dtype([
        ("r_min", cl.cltypes.float),
        ("escape_time", cl.cltypes.float),
        ("scale", cl.cltypes.float)
    ])

    # ---------------------------------------------------------------------------------------------------------------- #

    def _build_kernel(self, core_types_decl: List[str]):
        mandelbox_parameters_dtype, mandelbox_parameters_decl = cl.tools.match_dtype_to_c_struct(
            self.device,
            "MandelboxParameters",
            self.mandelbox_parameters
        )

        self._mandelbox_parameters_dtype = cl.tools.get_or_register_dtype(
            "MandelboxParameters",
            mandelbox_parameters_dtype
        )

        self._kernel = None
        self._kernel = mandelbox_parameters_decl + '\n'.join(core_types_decl) + self.get_kernel_code()

        self._program = cl.Program(self.context, self._kernel).build()

    def _set_parameters(self, parameters, color):
        if parameters is not None:
            assert set(parameters.keys()) == set(self._default_parameters.keys())
            self._parameters = parameters

        else:
            self._parameters = self._default_parameters

        self.set_color(color if color is not None else self.get_default_color())

    def _create_parameters_buffer(self):
        mandelbox_parameters_instance = np.array([(
            self._parameters["r_min"],
            self._parameters["escape_time"],
            self._parameters["scale"]
        )], dtype=self._mandelbox_parameters_dtype)[0]

        self._mandelbox_parameters_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY,
            self._mandelbox_parameters_dtype.itemsize
        )

        cl.enqueue_copy(self.queue, self._mandelbox_parameters_buffer, mandelbox_parameters_instance)

    # ---------------------------------------------------------------------------------------------------------------- #

    def get_kernel_code(self):
        if self._kernel is None:
            with open("src/fractals/kernels/mandelbox.cl", 'r') as f:
                self._kernel = f.read()

        return self._kernel

    def get_name(self):
        return "Mandelbox"

    def get_description(self):
        return "A MandelBox fractal"

    def get_default_parameters(self):
        return copy(self._default_parameters)

    def get_color(self):
        return float(self._color["x"]), float(self._color["y"]), float(self._color["z"])

    def get_color_cl(self):
        return self._color

    def set_color(self, color):
        self._color = np.zeros(1, dtype=cl.cltypes.float3)[0]
        self._color["x"], self._color["y"], self._color["z"] = color

    def get_default_color(self):
        return 0.8980392156862745, 0.8235294117647058, 0.7058823529411765

    def render(self, quality_props_buffer: cl.Buffer, image_buffer: cl.Image, camera_buffer: cl.Buffer):
        self._program.render(
            self.queue,
            image_buffer.shape,
            None,
            camera_buffer,
            quality_props_buffer,
            self._mandelbox_parameters_buffer,
            image_buffer
        )

