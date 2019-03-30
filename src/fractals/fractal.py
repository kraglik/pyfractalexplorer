from abc import abstractmethod
from typing import Optional, List, Dict, Tuple, Union
from string import Template

import numpy as np
import pyopencl as cl
import pyopencl.tools
import pyopencl.cltypes


class Fractal:

    _material_dtype = np.dtype([
        ("color_diffusive", cl.cltypes.uchar3),
        ("color_specular", cl.cltypes.uchar3),
        ("diffusion", cl.cltypes.float),
        ("specular", cl.cltypes.float),
        ("reflected", cl.cltypes.float)
    ])

    def __init__(self,
                 device: cl.Device,
                 context: cl.Context,
                 queue: cl.CommandQueue,
                 core_types_decl: List[str],
                 parameters: Optional[Dict] = None,
                 color: Optional[Tuple[float, float, float]] = None):

        self.device = device
        self.context = context
        self.queue = queue

        self._time = 0.0
        self._amplitude = 0.0

        self._material_buffer = None
        self._parameters_buffer = None

        self._material = self.get_default_material()

        self._build_kernel(core_types_decl)
        self.set_parameters(parameters, color)

    # PRIVATE METHODS

    def _build_kernel(self, core_types_decl: List[str]):
        parameters_dtype, parameters_decl = cl.tools.match_dtype_to_c_struct(
            self.device,
            self.get_parameters_typename(),
            self.get_numpy_dtype_parameters()
        )

        self._parameters_dtype = cl.tools.get_or_register_dtype(
            self.get_parameters_typename(),
            parameters_dtype
        )

        self._parameters_declaration = parameters_decl

        self._kernel = None
        self._kernel = self.get_kernel_code()

        template = Template(self._kernel)

        self._kernel = template.safe_substitute(
            dict(
                type_declarations=''.join(core_types_decl + [parameters_decl]),
                distance_function_declaration=self.get_distance_function_code(),
                outside_of_circumscribed_figure_declaration=self.get_check_circumscribed_figure_code(),
                fractal_parameters_typename=self.get_parameters_typename(),
                orbit_trap_declaration=self.get_orbit_trap_code()
            )
        )

        self._program = cl.Program(self.context, self._kernel).build()

    # PUBLIC METHODS

    def get_diffusive_color(self):
        return self._material["color_diffusive"]

    def get_specular_color(self):
        return self._material["color_specular"]

    @abstractmethod
    def get_default_material(self):
        raise NotImplementedError

    def set_material(self, material):
        self._material = material

    def get_material(self):
        return self._material

    def set_parameters(self, parameters, color):
        if parameters is not None:
            assert set(parameters.keys()) == set(self.get_default_parameters().keys())
            self._parameters = parameters

        else:
            self._parameters = self.get_default_parameters()

        self.set_color(color if color is not None else self.get_default_color())

        self.sync_with_device()

    def sync_with_device(self):
        parameters_instance = np.array([self.get_parameters_values()], dtype=self._parameters_dtype)[0]

        self._parameters_buffer = self._parameters_buffer or cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY,
            self._parameters_dtype.itemsize
        )

        cl.enqueue_copy(self.queue, self._parameters_buffer, parameters_instance)

        material_instance = np.array([(
            self.get_diffusive_color() + (0, ),
            self.get_specular_color() + (0, ),
            self._material["diffusive"],
            self._material["specular"],
            self._material["reflected"]
        )], dtype=self._material_dtype)[0]

        self._material_buffer = self._material_buffer or cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY,
            self._material_dtype.itemsize
        )

        cl.enqueue_copy(self.queue, self._material_buffer, material_instance)

    @abstractmethod
    def get_default_color(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    @abstractmethod
    def get_color_cl(self) -> np.void:
        raise NotImplementedError

    @abstractmethod
    def set_color(self, color: Tuple[int, int, int]):
        raise NotImplementedError

    def get_kernel_code(self):
        if self._kernel is None:
            with open("src/fractals/kernels/raymarch.cl", 'r') as f:
                self._kernel = f.read()

        return self._kernel

    @abstractmethod
    def get_distance_function_code(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_orbit_trap_code(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_check_circumscribed_figure_code(self) -> str:
        raise NotImplementedError

    def get_parameters_declaration(self) -> str:
        return self._parameters_declaration

    @abstractmethod
    def get_parameters_typename(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Union[int, Union[float, int]]]:
        raise NotImplementedError

    def get_parameters_buffer(self):
        return self._parameters_buffer

    def get_material_buffer(self):
        return self._material_buffer

    @abstractmethod
    def get_parameters_values(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def get_numpy_dtype_parameters(self) -> np.dtype:
        raise NotImplementedError

    @property
    def render_function(self):
        return self._program.render

    @abstractmethod
    def get_initial_camera_position(self):
        raise NotImplementedError

    @abstractmethod
    def get_initial_camera_target(self):
        raise NotImplementedError

    @abstractmethod
    def get_default_iterations(self):
        raise NotImplementedError

    def set_time(self, time: float):
        self._time = time

    def set_amplitude(self, amplitude: float):
        self._amplitude = amplitude
