from copy import copy

import numpy as np
import pyopencl as cl
import pyopencl.cltypes

from .fractal import Fractal


class KaleidoFractal(Fractal):
    _default_parameters = {
        "time": 0.0,
    }

    mandelbox_parameters = np.dtype([
        ("time", cl.cltypes.float),
    ])

    # ---------------------------------------------------------------------------------------------------------------- #
    def get_parameters_values(self):
        return (
            self._parameters["time"],
        )

    def get_check_circumscribed_figure_code(self):
        return """
        inline bool outside_of_circumscribed_figure(float3 pos) {
            return pos.x < -10.2f || pos.x > 10.2f ||
                   pos.y < -10.2f || pos.y > 10.2f ||
                   pos.z < -10.2f || pos.z > 10.2f;  
        }
        """

    def get_distance_function_code(self):
        return """
        float2 fold(float2 p, float ang){
            float2 n = {cos(-ang), sin(-ang)};
            p -= 2.0f * min(0.0f, dot(p, n)) * n;
            
            return p;
        }
        
        float3 tri_fold(float3 pt, float time) {
            float2 temp;
            
            temp.x = pt.x; temp.y = pt.y;
            temp = fold(temp, M_PI / 3.0f - cos(time)/10.0f);
            pt.x = temp.x; pt.y = temp.y;
            
            temp.x = pt.x; temp.y = pt.y;
            temp = fold(temp, M_PI / 3.0f);
            pt.x = temp.x; pt.y = temp.y;
        
            temp.x = pt.y; temp.y = pt.z;
            temp = fold(temp, -M_PI / 6.0f + sin(time) / 2.0f);
            pt.y = temp.x; pt.z = temp.y;
            
            temp.x = pt.y; temp.y = pt.z;
            temp = fold(temp, M_PI / 6.0f);
            pt.y = temp.x; pt.z = temp.y;
            
            return pt;
        }
        
        float3 tri_curve(float3 pt, float time, int iterations) {
            for(int i=0; i < iterations; i++){
                pt *= 2.0f;
                pt.x -= 2.6f;
                pt = tri_fold(pt, time);
            }
            return pt;
        }
        
        inline float fractal_distance(float3 p,
                              __global QualityProps * quality_props,
                              __global KaleidoParameters * parameters) {
            p *= 0.75f;
            p.x += 1.5f;
            p = tri_curve(p, parameters->time, quality_props->iteration_limit);
            
            return (fast_length( p * 0.004f ) - 0.01f);
        }
        """

    def get_name(self):
        return "Kaleido Fractal"

    def get_description(self):
        return "A Kaleido Fractal"

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
        return "KaleidoParameters"

    def get_numpy_dtype_parameters(self):
        return self.mandelbox_parameters

    def get_initial_camera_position(self):
        return np.array([2, 0, 2], dtype=np.float32)

    def get_initial_camera_target(self):
        return np.array([0, 0.5, 0], dtype=np.float32)

    def get_default_iterations(self):
        return 9
