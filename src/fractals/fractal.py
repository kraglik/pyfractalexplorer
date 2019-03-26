from abc import abstractmethod
from typing import List, Dict, Tuple, Union
import numpy as np


class Fractal:

    @abstractmethod
    def get_default_color(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    @abstractmethod
    def set_color(self):
        raise NotImplementedError

    @abstractmethod
    def get_kernel(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Union[int, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_config_structures(self) -> Tuple[str, np.dtype]:
        raise NotImplementedError

    def __setattr__(self, instance, value):
        self._parameters[instance] = value

    def __getattr__(self, item):
        return self._parameters[item]
