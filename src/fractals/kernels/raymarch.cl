typedef struct Ray {
    float3 pos, dir;
} Ray;

typedef struct Material {
    uchar3 color_diffusive;
    uchar3 color_specular;
    float diffusive;
    float specular;
    float reflected;
} Material;

typedef struct Hit {
    float distance;
    float3 position;
    int depth;
    uchar4 color;
} Hit;

$type_declarations

/**********************************************************************************************************************/

$distance_function_declaration

$outside_of_circumscribed_figure_declaration


float3 normal_to_fractal(float3 point,
                         __global QualityProps * quality_props,
                         __global $fractal_parameters_typename * parameters) {

    float3 a = {quality_props->epsilon * 0.05f, 0.0f, 0.0f};
    float3 b = {0.0f, quality_props->epsilon * 0.05f, 0.0f};
    float3 c = {0.0f, 0.0f, quality_props->epsilon * 0.05f};

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


float3 reflect(float3 direction, float3 normal) {
    return direction - 2 * dot(direction, normal) * normal;
}


bool in_shadow(float3 point,
               __global QualityProps * quality_props,
               __global $fractal_parameters_typename * parameters) {

    float epsilon = quality_props->epsilon;

    Hit first_hit;
    Ray ray = { .pos = point, .dir = quality_props->sun_direction };
    Hit hit = { .distance = epsilon * 2, .depth = quality_props->ray_steps_limit };

    for (int i = 0; i < quality_props->ray_steps_limit; i++) {
        float d = fractal_distance(ray.pos, quality_props, parameters);
        bool is_going_away = hit.distance < epsilon * 2;

        hit.distance += d * quality_props->ray_shift_multiplier;
        hit.position = ray.pos + d * ray.dir * quality_props->ray_shift_multiplier;
        ray.pos = hit.position;

        hit.depth = i;

        if (d > 100.0f) {
            hit.depth = quality_props->ray_steps_limit;
            hit.distance = 1e20f;

            break;
        }

        if (d < epsilon && !isnan(d))
            if (!is_going_away)
                break;
    }

    bool outside = outside_of_circumscribed_figure(hit.position) ||
                   fast_length(hit.position) > 15.0f ||
                   hit.distance > 100.0f;

    return !outside;
}


uchar4 blinn_phong(float3 position,
                  float3 normal,
                  float3 direction,
                  float shadow_coefficient,
                  __global QualityProps * quality_props,
                  __global $fractal_parameters_typename * parameters,
                  __global Material * material) {

    uchar3 color_diffusive = material->color_diffusive;
    uchar3 color_specular = material->color_specular;

    uchar4 color;

    float diffusive = 0.0f;
    float specular = 0.0f;

    float projection_length = dot(normal, quality_props->sun_direction);

    if (projection_length > 0.0f) {
        diffusive = max(0.0f, projection_length);
        specular = max(0.0f, pow(dot(reflect(-quality_props->sun_direction, normal), direction), 2.0f));

        Ray shadow_ray = {
            .pos = position,
            .dir = quality_props->sun_direction
        };

        if (in_shadow(position, quality_props, parameters)) {
            diffusive *= shadow_coefficient;
            specular *= shadow_coefficient;
        }
    }

    color.x = (unsigned char) clamp(
        clamp(diffusive * (float)color_diffusive.x, 0.0f, (float)color_diffusive.x) * material->diffusive +
            clamp(specular * (float)color_specular.x, 0.0f, (float)color_specular.x) * material->specular,
        0.0f,
        255.0f
    );
    color.y = (unsigned char) clamp(
        clamp(diffusive * (float)color_diffusive.y, 0.0f, (float)color_diffusive.y) * material->diffusive +
            clamp((float)specular * color_specular.y, 0.0f, (float)color_specular.y) * material->specular,
        0.0f,
        255.0f
    );
    color.z = (unsigned char) clamp(
        clamp(diffusive * (float)color_diffusive.z, 0.0f, (float)color_diffusive.z) * material->diffusive +
            clamp(specular * (float)color_specular.z, 0.0f, (float)color_specular.z) * material->specular,
        0.0f,
        255.0f
    );
    color.w = 255;

    return color;
}


Hit march_ray(Ray ray,
              __global QualityProps * quality_props,
              __global $fractal_parameters_typename * parameters,
              __global Material * material,
              int reflection_depth) {

    uchar3 color_diffusive = material->color_diffusive;
    uchar3 color_specular = material->color_specular;

    float epsilon = quality_props->epsilon;

    bool reflected = false, camera_in_shadow = in_shadow(ray.pos, quality_props, parameters);

    Hit first_hit;

    for (int it = 0; it < reflection_depth; it++) {
        Hit hit = { .distance = reflected ? epsilon * 2 : 0.0f, .depth = quality_props->ray_steps_limit };

        for (int i = 0; i < quality_props->ray_steps_limit; i++) {

            float d = fractal_distance(ray.pos, quality_props, parameters);
            bool is_going_away = (reflected && hit.distance < epsilon * 2);
            hit.distance += d * quality_props->ray_shift_multiplier;
            hit.position = ray.pos + d * ray.dir * quality_props->ray_shift_multiplier;
            ray.pos = hit.position;

            hit.depth = i;

            if (d > 100.0f) {
                hit.depth = quality_props->ray_steps_limit;
                hit.distance = 1e20f;

                break;
            }

            if (d < epsilon && !is_going_away)
                break;
        }

        bool outside = outside_of_circumscribed_figure(hit.position) ||
                       fast_length(hit.position) > 15.0f ||
                       hit.distance > 100.0f;

        if (!outside) {
            if (quality_props->render_simple) {

                float color_strength = 1.0f - (float)hit.depth / (float)quality_props->ray_steps_limit;

                hit.color.x = (unsigned char)clamp(
                    (color_strength * (float)color_diffusive.x),
                    0.0f,
                    (float)color_diffusive.x);
                hit.color.y = (unsigned char)clamp(
                    (color_strength * (float)color_diffusive.y),
                    0.0f,
                    (float)color_diffusive.y);
                hit.color.z = (unsigned char)clamp(
                    (color_strength * (float)color_diffusive.z),
                    0.0f,
                    (float)color_diffusive.z);
                hit.color.w = 255;

            } else {

                float3 normal = normal_to_fractal(hit.position, quality_props, parameters);

                hit.color = blinn_phong(
                    hit.position + normal * epsilon * 2,
                    normal,
                    ray.dir,
                    camera_in_shadow ? 0.6f : 0.2f,
                    quality_props,
                    parameters,
                    material
                );

                if (reflection_depth > 1) {

                    hit.color.x = (unsigned char)((float)hit.color.x * (1.0f - material->reflected));
                    hit.color.y = (unsigned char)((float)hit.color.y * (1.0f - material->reflected));
                    hit.color.z = (unsigned char)((float)hit.color.z * (1.0f - material->reflected));

                    ray.dir = reflect(ray.dir, normal);
                    ray.pos = hit.position + ray.dir * epsilon * 2;

                    reflected = true;
                }
            }

        } else {

            hit.color.x = 135;
            hit.color.y = 206;
            hit.color.z = 235;
            hit.color.w = 0;

        }

        float reflected_power = pow(material->reflected, it);

        if (it > 0) {
            first_hit.color.x += (unsigned char)(reflected_power * (float)hit.color.x);
            first_hit.color.y += (unsigned char)(reflected_power * (float)hit.color.y);
            first_hit.color.z += (unsigned char)(reflected_power * (float)hit.color.z);
        }

        if (it == 0)
            first_hit = hit;

        if (reflected_power * 255.0f < 1.0f || outside || quality_props->render_simple)
            break;

        if (it == 0)
            first_hit = hit;
    }

    return first_hit;
}


__kernel void render(__global Camera * camera,
                     __global QualityProps * quality_props,
                     __global $fractal_parameters_typename * parameters,
                     __global Material * material,
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

    Hit hit = march_ray(ray, quality_props, parameters, material, quality_props->reflection_depth + 1);

    __global uchar * pixel = & output[idX * height * 4 + idY * 4];

    pixel[0] = hit.color.x;
    pixel[1] = hit.color.y;
    pixel[2] = hit.color.z;
    pixel[3] = hit.color.w;
}
