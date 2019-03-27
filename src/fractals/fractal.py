from abc import abstractmethod
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pyopencl as cl


class Fractal:

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

        self._build_kernel(core_types_decl)
        self._set_parameters(parameters, color)
        self._create_parameters_buffer()

    # PRIVATE METHODS

    @abstractmethod
    def _build_kernel(self, core_types_decl: List[str]):
        raise NotImplementedError

    @abstractmethod
    def _set_parameters(self, parameters: Optional[Dict], color: Optional[Tuple[float, float, float]]):
        raise NotImplementedError

    @abstractmethod
    def _create_parameters_buffer(self):
        raise NotImplementedError

    # PUBLIC METHODS

    @abstractmethod
    def get_default_color(self) -> Tuple[float, float, float]:
        raise NotImplementedError

    @abstractmethod
    def get_color(self) -> Tuple[float, float, float]:
        raise NotImplementedError

    @abstractmethod
    def get_color_cl(self) -> np.void:
        raise NotImplementedError

    @abstractmethod
    def set_color(self, color: Tuple[float, float, float]):
        raise NotImplementedError

    @abstractmethod
    def get_kernel_code(self) -> str:
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

    @abstractmethod
    def render(self, quality_props_buffer: cl.Buffer, image_buffer: cl.Image, camera_buffer: cl.Buffer):
        raise NotImplementedError
