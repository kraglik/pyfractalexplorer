import math
from copy import copy

import numpy as np
import pyopencl as cl
import pyopencl.cltypes

from .fractal import Fractal


class SierpinskiTriangle(Fractal):
    _default_parameters = {
        "scale": 2.0,
        "offset": 1.5
    }

    _default_material = {
        "color_diffusive": (229, 210, 180),
        "color_specular": (255, 255, 255),
        "diffusive": 0.75,
        "specular": 0.25,
        "reflected": 0.25
    }

    sierpinski_parameters = np.dtype([
        ("scale", cl.cltypes.float),
        ("offset", cl.cltypes.float)
    ])

    # ---------------------------------------------------------------------------------------------------------------- #
    def get_default_material(self):
        return self._default_material

    def get_parameters_values(self):
        return (
            self._parameters["scale"] + math.cos(self._time) * self._amplitude,
            self._parameters["offset"]
        )

    def get_check_circumscribed_figure_code(self):
        return """
        inline bool outside_of_circumscribed_figure(float3 pos) {
            return pos.x < -5.2f || pos.x > 5.2f ||
                   pos.y < -5.2f || pos.y > 5.2f ||
                   pos.z < -5.2f || pos.z > 5.2f;  
        }
        """

    def get_distance_function_code(self):
        return """
        inline float fractal_distance(float3 z,
                              __global QualityProps * quality_props,
                              __global SierpinskiTriangleParameters * parameters) {
            
            float scale = parameters->scale;
            float offset = parameters->offset;
            
            float temp_x, temp_y, temp_z;
            
            for (int n = 0; n < quality_props->iteration_limit; n++) {
            
                if ((z.x + + z.y) < 0.0) {
                    temp_x = -z.y;
                    temp_y = -z.x;
                    z.x = temp_x;
                    z.y = temp_y;
                }
                
                if ((z.x + z.z) < 0.0) {
                    temp_x = -z.z;
                    temp_z = -z.x;
                    z.x = temp_x;
                    z.z = temp_z;
                }
                
                if ((z.z + z.y) < 0.0) {
                    temp_z = -z.y;
                    temp_y = -z.z;
                    z.z = temp_z;
                    z.y = temp_y;
                }
               
                z = scale * z - offset * (scale - 1.0f);
            }
         
            return fast_length(z) * pow(scale, (float)(-quality_props->iteration_limit));
        }
        """

    def get_name(self):
        return "Sierpinski Triangle"

    def get_description(self):
        return "A Sierpinski Triangle fractal"

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
        return 229, 210, 180

    def get_parameters_typename(self):
        return "SierpinskiTriangleParameters"

    def get_numpy_dtype_parameters(self):
        return self.sierpinski_parameters

    def get_initial_camera_position(self):
        return np.array([2, 0, 2], dtype=np.float32)

    def get_initial_camera_target(self):
        return np.array([0, 0.5, 0], dtype=np.float32)

    def get_default_iterations(self):
        return 16
