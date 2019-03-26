from copy import copy

from .fractal import Fractal


class Mandelbox(Fractal):

    _default_parameters = {
        "r_min": 0.5,
        "escape_time": 100.0
    }

    def __init__(self):
        self._kernel = None

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
