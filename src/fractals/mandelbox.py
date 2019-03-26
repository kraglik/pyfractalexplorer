from copy import copy
from typing import Optional, List, Tuple, Dict

import pyopencl as cl
import pyopencl.cltypes
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

    def __init__(self,
                 parameters: Optional[Dict] = None,
                 color: Optional[Tuple[float, float, float]] = None):

        self._kernel = None

        if parameters is not None:
            assert set(parameters.keys()) == set(self._default_parameters.keys())
            self._parameters = parameters

        else:
            self._parameters = self._default_parameters

        self.set_color(color if color is not None else self.get_default_color())

    def get_kernel(self):
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
        return float(self.color["x"]), float(self.color["y"]), float(self.color["z"])

    def get_color_cl(self):
        return self._color

    def set_color(self, color):
        self._color = np.zeros(1, dtype=cl.cltypes.float3)[0]
        self._color["x"], self._color["y"], self._color["z"] = color

    def get_default_color(self):
        return 0.8980392156862745, 0.8235294117647058, 0.7058823529411765

    def get_config_structures(self):
        return [
            ("MandelboxParameters", self.mandelbox_parameters)
        ]
