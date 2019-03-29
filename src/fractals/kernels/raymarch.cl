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

Hit march_ray(Ray ray,
              __global QualityProps * quality_props,
              __global $fractal_parameters_typename * parameters) {

    float path_len = 0.0f;

    Hit hit = { .distance = 0.0f, .depth = quality_props->ray_steps_limit };

    float epsilon = quality_props->epsilon;

    for (int i = 0; i < quality_props->ray_steps_limit; i++) {
        float d = fractal_distance(ray.pos, quality_props, parameters);
        hit.distance += d * quality_props->ray_shift_multiplier;

        if (d > 100.0f) {
            hit.depth = quality_props->ray_steps_limit;
            hit.distance = 1e20f;

            break;
        }

        hit.position = ray.pos + d * ray.dir;

        if (d < epsilon && !isnan(d) && d <= 100.0f) {
            hit.distance = path_len;
            hit.depth = i;

            break;

        } else {
            ray.pos += ray.dir * d * quality_props->ray_shift_multiplier;
        }
    }

    return hit;
}


float3 normal_to_fractal(float3 point,
                         __global QualityProps * quality_props,
                         __global $fractal_parameters_typename * parameters) {

    float3 a = {quality_props->epsilon, 0.0f, 0.0f};
    float3 b = {0.0f, quality_props->epsilon, 0.0f};
    float3 c = {0.0f, 0.0f, quality_props->epsilon};

    float3 result = {
        fractal_distance(point + a, quality_props, parameters) -
            fractal_distance(point - a, quality_props, parameters),
        fractal_distance(point + b, quality_props, parameters) -
            fractal_distance(point - b, quality_props, parameters),
        fractal_distance(point + c, quality_props, parameters) -
            fractal_distance(point - c, quality_props, parameters)
    };

    return normalize(result);
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

    Ray ray = {.pos = camera->pos };
    ray.dir = camera->pos + camera->right * x + camera->up * y + camera->dir * camera->zoom;
    ray.dir = normalize(ray.dir - camera->pos);

    Hit hit = march_ray(ray, quality_props, parameters);

    bool outside = outside_of_circumscribed_figure(hit.position) ||
                   fast_length(hit.position) > 10.0f;

    __global uchar * pixel = & output[idX * height * 4 + idY * 4];

    int2 pixel_pos = {idY, idX};

    if (!outside) {
        if (quality_props->render_simple) {

            float color_strength = 1.0f - (float)hit.depth / (float)quality_props->ray_steps_limit;

            pixel[0] = (unsigned char)clamp((color_strength * (float)color->x), 0.0f, (float)color->x);
            pixel[1] = (unsigned char)clamp((color_strength * (float)color->y), 0.0f, (float)color->y);
            pixel[2] = (unsigned char)clamp((color_strength * (float)color->z), 0.0f, (float)color->z);
            pixel[3] = 255;

        } else {

            float3 normal = normal_to_fractal(hit.position, quality_props, parameters);

            float projection_length = dot(normal, quality_props->sun_direction);

            float diffusive = 0.0f;
            float specular = 0.0f;

            if (projection_length > 0.0f) {
                diffusive = max(0.0f, projection_length);
                specular = max(0.0f, pow(projection_length, 3.0f));

                Ray shadow_ray = {
                    .pos = hit.position + normal * (quality_props->epsilon * 2.0f),
                    .dir = quality_props->sun_direction
                };
                Hit shadow_hit = march_ray(shadow_ray, quality_props, parameters);

                if (shadow_hit.distance < 100.0f) {
                    diffusive *= 0.2f;
                    specular *= 0.2f;
                }
            }

            pixel[0] = (uchar) clamp(
                clamp(diffusive * (float)color->x, 0.0f, (float)color->x) * 0.7f +
                    clamp(specular * color->x, 0.0f, 255.0f) * 0.3f,
                0.0f,
                255.0f
            );
            pixel[1] = (uchar) clamp(
                clamp(diffusive * (float)color->y, 0.0f, (float)color->y) * 0.7f +
                    clamp(specular * color->y, 0.0f, 255.0f) * 0.3f,
                0.0f,
                255.0f
            );
            pixel[2] = (uchar) clamp(
                clamp(diffusive * (float)color->z, 0.0f, (float)color->z) * 0.7f +
                    clamp(specular * color->z, 0.0f, 255.0f) * 0.3f,
                0.0f,
                255.0f
            );
            pixel[3] = 255;
        }

    } else {

        pixel[0] = 135;
        pixel[1] = 206;
        pixel[2] = 235;
        pixel[3] = 0;

    }
}
