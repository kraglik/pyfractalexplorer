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
    float3 normal;
    int depth;
    bool outside;
    float min_distance_to_fractal;
    float3 position_of_min_distance;
} Hit;

$type_declarations

/**********************************************************************************************************************/

$distance_function_declaration

$outside_of_circumscribed_figure_declaration

$orbit_trap_declaration


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
    return normalize(direction - 2 * dot(direction, normal) * normal);
}


uchar4 amplify_color(uchar4 color, float mul) {
    uchar4 new_color;

    new_color.x = (unsigned char)(mul * (float)color.x);
    new_color.y = (unsigned char)(mul * (float)color.y);
    new_color.z = (unsigned char)(mul * (float)color.z);
    new_color.w = color.w;

    return new_color;
}


Hit march_ray(float3 position,
               float3 direction,
               __global QualityProps * quality_props,
               __global $fractal_parameters_typename * parameters) {

    float epsilon = quality_props->epsilon;
    float glow_sharpness = quality_props->glow_sharpness;

    Hit hit = {
        .distance = 0.0f,
        .depth = quality_props->ray_steps_limit,
        .min_distance_to_fractal = 1.0f,
        .position_of_min_distance = position
    };

    for (int i = 0; i < quality_props->ray_steps_limit; i++) {
        float d = fractal_distance(position, quality_props, parameters);

        hit.distance += d * quality_props->ray_shift_multiplier;

        if (hit.min_distance_to_fractal > d)
            hit.position_of_min_distance = hit.position;

        hit.min_distance_to_fractal = min(hit.min_distance_to_fractal, glow_sharpness * d / hit.distance);
        hit.position = position + d * direction * quality_props->ray_shift_multiplier;
        position = hit.position;

        hit.depth = i;

        if (d > 100.0f) {
            hit.depth = quality_props->ray_steps_limit;
            hit.distance = 1e20f;

            break;
        }

        if (d < epsilon && !isnan(d)) {
            hit.normal = normal_to_fractal(hit.position, quality_props, parameters);
            hit.position = position + (epsilon - d) * hit.normal;
            position = hit.position;
            break;
        }
    }

    hit.outside = outside_of_circumscribed_figure(hit.position) ||
                  fast_length(hit.position) > 15.0f ||
                  hit.distance > 100.0f;

    return hit;
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

    if (quality_props->use_orbit_trap) {
        float3 ot = normalize(orbit_trap(position, quality_props, parameters));
        color_diffusive.x = (unsigned char)(ot.x * 255.0f);
        color_diffusive.y = (unsigned char)(ot.y * 255.0f);
        color_diffusive.z = (unsigned char)(ot.z * 255.0f);

        color_specular = color_diffusive;
    }

    uchar4 color;

    float diffusive = 0.0f;
    float specular = 0.0f;

    position += normal * quality_props->epsilon * 2;

    float projection_length = dot(normal, quality_props->sun_direction);

    if (projection_length > 0.0f) {
        diffusive = max(0.0f, projection_length);
        specular = max(0.0f, pow(dot(reflect(-quality_props->sun_direction, normal), direction), 3.0f));

        if (!march_ray(position, quality_props->sun_direction, quality_props, parameters).outside) {
            diffusive *= shadow_coefficient;
            specular *= shadow_coefficient;
        }
    } else {
        diffusive = 0.2f;
        specular = 0.0f;
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


uchar4 render_pixel(Ray ray,
           __global QualityProps * quality_props,
           __global $fractal_parameters_typename * parameters,
           __global Material * material) {

    uchar4 color = {0, 0, 0, 0};

    float epsilon = quality_props->epsilon;

    bool reflected = false;
    bool camera_in_shadow = !march_ray(ray.pos, quality_props->sun_direction, quality_props, parameters).outside;

    uchar4 fog_color = {255, 255, 255, 255};

    for (int i = 0; i < quality_props->reflection_depth + 1; i++) {
        Hit hit = march_ray(ray.pos, ray.dir, quality_props, parameters);

        uchar4 current_color;

        if (hit.outside) {

            current_color.x = 135;
            current_color.y = 206;
            current_color.z = 235;
            current_color.w = 0;

            int3 glow_color = quality_props->glow_color;

            if ((glow_color.x + glow_color.y + glow_color.z) == 0) {
                float3 ot = normalize(orbit_trap(hit.position, quality_props, parameters));
                glow_color.x = (int) ot.x * 255;
                glow_color.y = (int) ot.y * 255;
                glow_color.z = (int) ot.z * 255;
            }

            float glow_mul = (1.0f - hit.min_distance_to_fractal) * (1.0f - hit.min_distance_to_fractal);

            current_color.x = (uchar) clamp((int) current_color.x + (int) (glow_color.x * glow_mul), 0, 255);
            current_color.y = (uchar) clamp((int) current_color.y + (int) (glow_color.y * glow_mul), 0, 255);
            current_color.z = (uchar) clamp((int) current_color.z + (int) (glow_color.z * glow_mul), 0, 255);

        } else {

            if (quality_props->render_simple) {

                uchar3 color_diffusive = material->color_diffusive;

                if (quality_props->use_orbit_trap) {
                    float3 ot = normalize(orbit_trap(hit.position, quality_props, parameters));
                    color_diffusive.x = (unsigned char)(ot.x * 255.0f);
                    color_diffusive.y = (unsigned char)(ot.y * 255.0f);
                    color_diffusive.z = (unsigned char)(ot.z * 255.0f);
                }

                float color_strength = 1.0f - (float)hit.depth / (float)quality_props->ray_steps_limit;

                current_color.x = (unsigned char)clamp(
                    (color_strength * (float)color_diffusive.x),
                    0.0f,
                    (float)color_diffusive.x);
                current_color.y = (unsigned char)clamp(
                    (color_strength * (float)color_diffusive.y),
                    0.0f,
                    (float)color_diffusive.y);
                current_color.z = (unsigned char)clamp(
                    (color_strength * (float)color_diffusive.z),
                    0.0f,
                    (float)color_diffusive.z);
                current_color.w = 255;

            } else {

                current_color = blinn_phong(
                    hit.position,
                    hit.normal,
                    ray.dir,
                    camera_in_shadow ? 0.5f : 0.4f,
                    quality_props,
                    parameters,
                    material
                );

                float fog_mul = 1.0f - (hit.distance / 40.0f);

                current_color = amplify_color(current_color, fog_mul) + amplify_color(fog_color, 1.0f - fog_mul);

                ray.dir = reflect(ray.dir, hit.normal);
                ray.pos = hit.position + hit.normal * epsilon * 2;
            }
        }

        float reflected_power = 1.0f;

        if (i == 0) {

            reflected_power = (quality_props->render_simple || hit.outside) ? 1.0f : 1.0f - material->reflected;

        } else {

            reflected_power = pow(material->reflected, i);

        }

        color.x += (unsigned char)(reflected_power * (float)current_color.x);
        color.y += (unsigned char)(reflected_power * (float)current_color.y);
        color.z += (unsigned char)(reflected_power * (float)current_color.z);

        if (i == 0) {
            color.w = current_color.w;
        }

        if (hit.outside || quality_props->render_simple || (reflected_power * 255.0f) < 1.0f)
            break;
    }

    return color;
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

    uchar4 color = render_pixel(ray, quality_props, parameters, material);

    __global uchar * pixel = & output[idX * height * 4 + idY * 4];

    pixel[0] = color.x;
    pixel[1] = color.y;
    pixel[2] = color.z;
    pixel[3] = color.w;
}
