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

float mandelbox_distance(float3 *p0, __global WorldProps * props) {
    float3 p = *p0;

    float r_min_2 = square(props->r_min);
    float r_fixed_2 = 1.0f;
    float escape = square(props->escape_time);
    float d_factor = 1;
    float r2 = -1;
    float scale = props->scale;

    float c1 = fabs(scale - 1.0f);
    float c2 = pow(fabs(scale), 1 - props->it_limit);

    for (int i = 0; i < props->it_limit; i++) {
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

Hit march_ray(Ray *ray, __global WorldProps * props, float pathLen) {
    Hit hit = { .distance = INFINITY, .depth = props->it_limit };
    float3 temp;

    float epsilon = props->epsilon;

    for (int i = 0; i < props->it_limit; i++) {
        float d = mandelbox_distance(&ray->pos, props);

        hit.position = ray->pos + d * ray->dir;

        if (d < epsilon && !(isinf(d) || isnan(d))) {
            hit.distance = pathLen;
            hit.depth = i;

            break;

        } else {

            ray->pos += ray->dir * d * props->shift_value;
            pathLen += d * props->shift_value;

        }
    }

    return hit;
}


Hit trace_ray(__global Camera *camera, __global WorldProps * props, float x, float y) {
    float m = camera->shift_multiplier;

    Ray ray = {.pos = camera->pos + camera->right * x * m + camera->up * y * m};

    float3 initial_pos = camera->pos + camera->right * x + camera->up * y;

    ray.dir = normalize(camera->dir * camera->view_plane_distance + initial_pos - camera->pos);

    return march_ray(&ray, props, 0.0f);
}

__kernel void render(__global Camera * camera, __global WorldProps * props, __global uchar3 * pixels) {
    int idX = get_global_id(0);
    int idY = get_global_id(1);

    int width = get_global_size(0);
    int height = get_global_size(1);

    int pixel_id = idY * width + idX;

    float hx = (float)width / 2.0f;
    float hy = (float)height / 2.0f;

    float x = ((float)idX - hx) / hx * camera->ratio;
    float y = ((float)idY - hy) / hy;

    Hit hit = trace_ray(camera, props, x, y);

    float color_strength = 1.0f - (float)hit.depth / (float)props->it_limit;

    bool in_box = hit.position.x >= -5.0f && hit.position.x <= 5.0f &&
                  hit.position.y >= -5.0f && hit.position.y <= 5.0f &&
                  hit.position.z >= -5.0f && hit.position.z <= 5.0f;

    if (hit.distance != INFINITY || in_box) {

        pixels[pixel_id].x = (unsigned char)clamp((color_strength * 229.0f), 0.0f, 229.0f);
        pixels[pixel_id].y = (unsigned char)clamp((color_strength * 210.0f), 0.0f, 210.0f);
        pixels[pixel_id].z = (unsigned char)clamp((color_strength * 180.0f), 0.0f, 180.0f);

    } else {

        pixels[pixel_id].x = 25;
        pixels[pixel_id].y = 25;
        pixels[pixel_id].z = 25;

    }
}
