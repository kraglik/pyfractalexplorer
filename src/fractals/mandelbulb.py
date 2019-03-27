from copy import copy

import numpy as np
import pyopencl as cl
import pyopencl.cltypes

from .fractal import Fractal


class Mandelbulb(Fractal):
    _default_parameters = {
        "power": 8.0,
    }

    mandelbox_parameters = np.dtype([
        ("power", cl.cltypes.float),
    ])

    # ---------------------------------------------------------------------------------------------------------------- #
    def get_parameters_values(self):
        return (
            self._parameters["power"],
        )

    def get_check_circumscribed_figure_code(self):
        return """
        inline bool outside_of_circumscribed_figure(float3 pos) {
            return pos.x < -5.1f || pos.x > 5.1f ||
                   pos.y < -5.1f || pos.y > 5.1f ||
                   pos.z < -5.1f || pos.z > 5.1f;  
        }
        """

    def get_distance_function_code(self):
        return """
        
        inline void pow_vec(float3 *v, float3 *result, float power) {
            float ph = atan(v->y / v->x);
            float th = acos(v->z / len(v));
        
            result->x = native_sin(power * th) * native_cos(power * ph);
            result->y = native_sin(power * th) * native_sin(power * ph);
            result->z = native_cos(power * th);
            
            *result *= pow(len(v), power);
        }
        
        inline float2 iterate_z(float dr, float3 z, float3 *c, float power, int limit) {
            float2 pair = { 0.0f, dr };
        
            for (int i = 0;; i++) {
        
                float r = len(&z);
                float3 zn;
                pow_vec(&z, &zn, power);
                
                zn += *c;
        
                if (i > limit || r > 2.0f) {
        
                    pair.x = r;
                    pair.y = dr;
        
                    break;
        
                } else {
                
                    dr = pow(r, power - 1.0f) * power * dr + 1.0f;
                    z = zn;
        
                }
            }
        
            return pair;
        }
        
        float distance(float3 *p0,
                       __global QualityProps * quality_props,
                       __global MandelbulbParameters * parameters) {
                              
            float2 p = iterate_z(1.0f, *p0, p0, parameters->power, quality_props->iteration_limit);
        
            return (0.5f * log(p.x) * p.x) / p.y;
        }
        """

    def get_name(self):
        return "Mandelbulb"

    def get_description(self):
        return "A Mandelbulb fractal"

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
        return "MandelbulbParameters"

    def get_numpy_dtype_parameters(self):
        return self.mandelbox_parameters
