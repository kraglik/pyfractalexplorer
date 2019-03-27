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

inline float len(float3 * vector) {
    return sqrt(vector->x * vector->x + vector->y * vector->y + vector->z * vector->z);
}

/**********************************************************************************************************************/

$distance_function_declaration

$outside_of_circumscribed_figure_declaration

Hit march_ray(Ray *ray,
              __global QualityProps * quality_props,
              __global $fractal_parameters_typename * parameters,
               float path_len) {

    Hit hit = { .distance = INFINITY, .depth = quality_props->ray_steps_limit };
    float3 temp;

    float epsilon = quality_props->epsilon;

    for (int i = 0; i < quality_props->ray_steps_limit; i++) {
        float d = distance(&ray->pos, quality_props, parameters);

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
                     __write_only image2d_t output) {

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

    uint4 pixel = {hit.depth, hit.depth, hit.depth, 0};
    int2 pixel_pos = {idY, idX};

    if (hit.distance != INFINITY || !outside) {

        pixel.x = (unsigned int)clamp((color_strength * (float)color->x), 0.0f, (float)color->x);
        pixel.y = (unsigned int)clamp((color_strength * (float)color->y), 0.0f, (float)color->y);
        pixel.z = (unsigned int)clamp((color_strength * (float)color->z), 0.0f, (float)color->z);
        pixel.w = 255;

    } else {

        pixel.x = 55;
        pixel.y = 55;
        pixel.z = 55;
        pixel.w = 0;

    }

    write_imageui(output, pixel_pos, pixel);
}
