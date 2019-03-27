typedef struct Ray {
    float3 pos, dir;
} Ray;

typedef struct Hit {
    float distance;
    float3 position;
    int depth;
} Hit;

$type_declarations

/**********************************************************************************************************************/

$distance_function_declaration

$outside_of_circumscribed_figure_declaration

Hit march_ray(Ray *ray,
              __global QualityProps * quality_props,
              __global $fractal_parameters_typename * parameters,
               float path_len) {

    Hit hit = { .distance = 1e20f, .depth = quality_props->ray_steps_limit };
    float3 temp;

    float epsilon = quality_props->epsilon;

    for (int i = 0; i < quality_props->ray_steps_limit; i++) {
        float d = fractal_distance(ray->pos, quality_props, parameters);

        hit.position = ray->pos + d * ray->dir;

        if (d < epsilon && !(isinf(d) || isnan(d))) {
            hit.distance = path_len;
            hit.depth = i;

            break;

        } else {

            ray->pos += ray->dir * d * quality_props->ray_shift_multiplier;
            path_len += d * quality_props->ray_shift_multiplier;

        }
    }

    return hit;
}


Hit trace_ray(__global Camera * camera,
              __global QualityProps * quality_props,
              __global $fractal_parameters_typename * parameters,
              float x,
              float y) {

    float m = camera->shift_multiplier;

    Ray ray = {.pos = camera->pos };

    float3 initial_pos = camera->pos + camera->right * x + camera->up * y;

    ray.dir = normalize(camera->dir * camera->zoom + initial_pos - camera->pos);

    return march_ray(&ray, quality_props, parameters, 0.0f);
}

__kernel void render(__global Camera * camera,
                     __global QualityProps * quality_props,
                     __global $fractal_parameters_typename * parameters,
                     __global uchar3 * color,
                     __global uchar * output) {

    int idX = get_global_id(0);
    int idY = get_global_id(1);

    int width = get_global_size(0);
    int height = get_global_size(1);

    float ratio = (float) width / (float) height;

    int pixel_id = idY * width + idX;

    float hx = (float)width / 2.0f;
    float hy = (float)height / 2.0f;

    float x = ((float)idX - hx) / hx * ratio;
    float y = -((float)idY - hy) / hy;

    Hit hit = trace_ray(camera, quality_props, parameters, x, y);

    float color_strength = 1.0f - (float)hit.depth / (float)quality_props->ray_steps_limit;

    bool outside = outside_of_circumscribed_figure(hit.position);

    __global uchar * pixel = & output[idX * height * 4 + idY * 4];

    int2 pixel_pos = {idY, idX};

    if (hit.distance != 1e20f || !outside) {

        pixel[0] = (unsigned char)clamp((color_strength * (float)color->x), 0.0f, (float)color->x);
        pixel[1] = (unsigned char)clamp((color_strength * (float)color->y), 0.0f, (float)color->y);
        pixel[2] = (unsigned char)clamp((color_strength * (float)color->z), 0.0f, (float)color->z);
        pixel[3] = 255;

    } else {

        pixel[0] = 55;
        pixel[1] = 55;
        pixel[2] = 55;
        pixel[3] = 0;

    }


}
