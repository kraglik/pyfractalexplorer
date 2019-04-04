import math
from copy import copy

import numpy as np
import pyopencl as cl
import pyopencl.cltypes

from .fractal import Fractal


class Mandelbox(Fractal):

    _default_parameters = {
        "r_min": 0.5,
        "escape_time": 100.0,
        "scale": 2.39128  # -1.73
    }

    _default_material = {
        "color_diffusive": (229, 210, 180),
        "color_specular": (255, 255, 255),
        "diffusive": 0.75,
        "specular": 0.25,
        "reflected": 0.25
    }

    mandelbox_parameters = np.dtype([
        ("r_min", cl.cltypes.float),
        ("escape_time", cl.cltypes.float),
        ("scale", cl.cltypes.float)
    ])

    # ---------------------------------------------------------------------------------------------------------------- #
    def get_default_material(self):
        return self._default_material

    def get_parameters_values(self):
        return (
            self._parameters["r_min"],
            self._parameters["escape_time"],
            self._parameters["scale"] + math.cos(self._time) * self._amplitude
        )

    def get_check_circumscribed_figure_code(self):
        return """
        inline bool outside_of_circumscribed_figure(float3 pos) {
            return pos.x < -6.1f || pos.x > 6.1f ||
                   pos.y < -6.1f || pos.y > 6.1f ||
                   pos.z < -6.1f || pos.z > 6.1f;  
        }
        """

    def get_distance_function_code(self):
        return """
        #define COMPONENT_FOLD(x) ( (x>1) ? (2-x) : ((x<-1) ?(-2-x):x))
        
        inline float square(float x) { return x*x; }

        inline void fold_box(float3 *v) {
        
            v->x = COMPONENT_FOLD(v->x);
            v->y = COMPONENT_FOLD(v->y);
            v->z = COMPONENT_FOLD(v->z);
        
        }
        
        inline void fold_sphere(float3 *v, float r2, float r_min_2, float r_fixed_2)
        {
            if (r2 < r_min_2)
                *v *= r_fixed_2 / r_min_2;
            else
            if (r2 < r_fixed_2)
                *v *= r_fixed_2 / r2;
        }
        
        inline float fractal_distance(float3 point,
                              __global QualityProps * quality_props,
                              __global MandelboxParameters * parameters) {
            float3 p = point;
        
            float r_min_2 = square(parameters->r_min);
            float r_fixed_2 = 1.0f;
            float escape = square(parameters->escape_time);
            float d_factor = 1;
            float r2 = -1;
            float scale = parameters->scale;
        
            float c1 = fabs(scale - 1.0f);
            float c2 = pow(fabs(scale), 1 - quality_props->iteration_limit);
        
            for (int i = 0; i < quality_props->iteration_limit; i++) {
                fold_box(&p);
                r2 = dot(p, p);
        
                fold_sphere(&p, r2, r_min_2, r_fixed_2);
        
                p *= scale;
                p += point;
        
                if (r2 < r_min_2)
                    d_factor *= (r_fixed_2 / r_min_2);
                else if (r2<r_fixed_2)
                    d_factor *= (r_fixed_2 / r2);
        
                d_factor = d_factor * fabs(scale) + 1.0;
        
                if ( r2 > escape )
                    break;
            }
        
            r2 = sqrt(dot(p, p));
        
            return (r2 - c1) / d_factor - c2;
        }
        """

    def get_orbit_trap_code(self) -> str:
        return """
        float3 orbit_trap(float3 point, 
                          __global QualityProps * quality_props,
                          __global MandelboxParameters * parameters) {
        
            float3 color = {1e20f, 1e20f, 1e20f};
            float3 new_color;
            float3 orbit = {0, 0, 0};
            float3 m = {1.0f, 1.0f, 1.0f}; // {0.42f, 0.38f, 0.19f};
            
            float3 p = point;
        
            float r_min_2 = square(parameters->r_min);
            float r_fixed_2 = 1.0f;
            float escape = square(parameters->escape_time);
            float d_factor = 1;
            float r2 = -1;
            float scale = parameters->scale;
        
            float c1 = fabs(scale - 1.0f);
            float c2 = pow(fabs(scale), 1 - 10);
        
            for (int i = 0; i < quality_props->iteration_limit; i++) {
                fold_box(&p);
                r2 = dot(p, p);
        
                fold_sphere(&p, r2, r_min_2, r_fixed_2);
        
                p *= scale;
                p += point;
        
                if (r2 < r_min_2)
                    d_factor *= (r_fixed_2 / r_min_2);
                else if (r2<r_fixed_2)
                    d_factor *= (r_fixed_2 / r2);
        
                d_factor = d_factor * fabs(scale) + 1.0;
                
                orbit = max(orbit, p * m);
        
                if ( r2 > escape )
                    break;
            }
        
            r2 = sqrt(dot(p, p));
        
            return orbit;
        
        }
        """

    def get_name(self):
        return "Mandelbox"

    def get_description(self):
        return "A MandelBox fractal"

    def get_default_parameters(self):
        return copy(self._default_parameters)

    def get_color_cl(self):
        return self._color

    def set_color(self, color):
        self._color = np.zeros(1, dtype=cl.cltypes.float3)[0]
        self._color["x"], self._color["y"], self._color["z"] = color

    def get_default_color(self):
        return 229, 210, 180

    def get_parameters_typename(self):
        return "MandelboxParameters"

    def get_numpy_dtype_parameters(self):
        return self.mandelbox_parameters

    def get_initial_camera_position(self):
        return np.array([-10, 0, 0], dtype=np.float32)

    def get_initial_camera_target(self):
        return np.array([0, 0, 0], dtype=np.float32)

    def get_default_iterations(self):
        return 16

    def get_glow_color(self):
        return -80, 200, 255

    def get_glow_sharpness(self):
        return 18.0
