import pyopencl as cl
from .fractals import Fractal


class App:
    def __init__(self,
                 platform_id=0,
                 device_id=0,
                 device_type=cl.device_type.ALL,
                 additional_fractals=[]
                 ):
        self.platform_id = platform_id or 0
        self.device_id = device_id or 0
        self.device_type = device_type or cl.device_type.ALL

        self.platform = cl.get_platforms()[self.platform_id]
        self.device = self.platform.get_devices()[self.device_id]

        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        self.structures = {}


