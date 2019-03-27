#define COMPONENT_FOLD(x) ( (x>1) ? (2-x) : ((x<-1) ?(-2-x):x))

typedef struct Ray {
    float3 pos, dir;
} Ray;

typedef struct Hit {
    float distance;
    float3 position;
    int depth;
} Hit;

/**********************************************************************************************************************/

float len(float3 * vector) {
    return sqrt(vector->x * vector->x + vector->y * vector->y + vector->z * vector->z);
}

/**********************************************************************************************************************/

float square(float x) { return x*x; }

void fold_box(float3 *v) {

    v->x = COMPONENT_FOLD(v->x);
    v->y = COMPONENT_FOLD(v->y);
    v->z = COMPONENT_FOLD(v->z);

}

void fold_sphere(float3 *v, float r2, float r_min_2, float r_fixed_2)
{
    if (r2 < r_min_2)
        *v *= r_fixed_2 / r_min_2;
    else
    if (r2 < r_fixed_2)
        *v *= r_fixed_2 / r2;
}

float mandelbox_distance(float3 *p0,
                         __global QualityProps * quality_props,
                         __global MandelboxParameters * mandelbox_parameters) {
    float3 p = *p0;

    float r_min_2 = square(mandelbox_parameters->r_min);
    float r_fixed_2 = 1.0f;
    float escape = square(mandelbox_parameters->escape_time);
    float d_factor = 1;
    float r2 = -1;
    float scale = mandelbox_parameters->scale;

    float c1 = fabs(scale - 1.0f);
    float c2 = pow(fabs(scale), 1 - quality_props->iteration_limit);

    for (int i = 0; i < quality_props->iteration_limit; i++) {
        fold_box(&p);
        r2 = dot(p, p);

        fold_sphere(&p, r2, r_min_2, r_fixed_2);

        p *= scale;
        p += *p0;

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

Hit march_ray(Ray *ray,
              __global QualityProps * quality_props,
              __global MandelboxParameters * mandelbox_parameters,
               float path_len) {

    Hit hit = { .distance = INFINITY, .depth = quality_props->ray_steps_limit };
    float3 temp;

    float epsilon = quality_props->epsilon;

    for (int i = 0; i < quality_props->ray_steps_limit; i++) {
        float d = mandelbox_distance(&ray->pos, quality_props, mandelbox_parameters);

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
              __global MandelboxParameters * mandelbox_parameters,
              float x,
              float y) {

    float m = camera->shift_multiplier;

    Ray ray = {.pos = camera->pos + camera->right * x * m + camera->up * y * m};

    float3 initial_pos = camera->pos + camera->right * x + camera->up * y;

    ray.dir = normalize(camera->dir * camera->view_plane_distance + initial_pos - camera->pos);

    return march_ray(&ray, quality_props, mandelbox_parameters, 0.0f);
}

__kernel void render(__global Camera * camera,
                     __global QualityProps * quality_props,
                     __global MandelboxParameters * mandelbox_parameters,
                     __read_write image2d_t output) {

    int idX = get_global_id(0);
    int idY = get_global_id(1);

    int width = get_global_size(0);
    int height = get_global_size(1);

    int pixel_id = idY * width + idX;

    float hx = (float)width / 2.0f;
    float hy = (float)height / 2.0f;

    float x = ((float)idX - hx) / hx * camera->ratio;
    float y = ((float)idY - hy) / hy;

    Hit hit = trace_ray(camera, quality_props, mandelbox_parameters, x, y);

    float color_strength = 1.0f - (float)hit.depth / (float)quality_props->ray_steps_limit;

    bool in_box = hit.position.x >= -5.1f && hit.position.x <= 5.1f &&
                  hit.position.y >= -5.1f && hit.position.y <= 5.1f &&
                  hit.position.z >= -5.1f && hit.position.z <= 5.1f;

    uint4 pixel = {hit.depth, hit.depth, hit.depth, 0};
    int2 pixel_pos = {idY, idX};

    if (hit.distance != INFINITY || in_box) {

        pixel.x = (unsigned int)clamp((color_strength * 229.0f), 0.0f, 229.0f);
        pixel.y = (unsigned int)clamp((color_strength * 210.0f), 0.0f, 210.0f);
        pixel.z = (unsigned int)clamp((color_strength * 180.0f), 0.0f, 180.0f);
        pixel.w = 255;

    } else {

        pixel.x = 55;
        pixel.y = 55;
        pixel.z = 55;
        pixel.w = 0;

    }

    write_imageui(output, pixel_pos, pixel);
}
