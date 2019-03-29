import math
from copy import copy

import numpy as np
import pyopencl as cl
import pyopencl.cltypes

from .fractal import Fractal


class MengerSponge(Fractal):
    _default_parameters = {
        "scale": 3.0,
    }

    _default_material = {
        "color_diffusive": (229, 210, 180),
        "color_specular": (255, 255, 255),
        "diffusive": 0.75,
        "specular": 0.25,
        "reflected": 0.25
    }

    mandelbox_parameters = np.dtype([
        ("scale", cl.cltypes.float),
    ])

    # ---------------------------------------------------------------------------------------------------------------- #
    def get_default_material(self):
        return self._default_material

    def get_parameters_values(self):
        return (
            self._parameters["scale"] + math.cos(self._time) * self._amplitude,
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
        inline float DEBox(float3 pos, float hlen) {
            return max(fabs(pos.x), max(fabs(pos.y), fabs(pos.z))) - hlen;
        }
        
        inline float fractal_distance(float3 pos,
                              __global QualityProps * quality_props,
                              __global MengerSpongeParameters * parameters) {
                              
            const float scale = parameters->scale;
            const float scaleM = 3.0f - 1.0f;
            const float3 offset = (float3)(1.0f, 1.0f, 1.0f);
            const int iters = quality_props->iteration_limit;
            const float psni = pow(scale, -(float)iters);
            
            for (int n = 0; n < iters; n++) {
                pos = fabs(pos);
                if (pos.x < pos.y)
                    pos.xy = pos.yx;
                if (pos.x < pos.z)
                    pos.xz = pos.zx;
                if (pos.y < pos.z)
                    pos.yz = pos.zy;
            
                pos = pos * scale - offset * (scaleM);
                if (pos.z < -0.5f * offset.z * (scaleM))
                    pos.z += offset.z * (scaleM);
            }
            
            return DEBox(pos, scale * 0.3333334f) * psni;
        }
        """

    def get_name(self):
        return "Menger Sponge"

    def get_description(self):
        return "A Menger Sponge fractal"

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
        return "MengerSpongeParameters"

    def get_numpy_dtype_parameters(self):
        return self.mandelbox_parameters

    def get_initial_camera_position(self):
        return np.array([2, 0, 2], dtype=np.float32)

    def get_initial_camera_target(self):
        return np.array([0, 0.5, 0], dtype=np.float32)

    def get_default_iterations(self):
        return 10
