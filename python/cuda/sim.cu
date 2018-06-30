#include "cuda/helper_math.h"
#include "cuda/cudaNoise/cudaNoise.cuh"

__device__ const float timestep = %(sim_timestep)s;

__device__ const float SIM_HEIGHT_ABOVE = 0.2f;
__device__ const float SIM_HEIGHT_BELOW = 0.1f;

/*
 * 3D -> 1D index in contiguous array
 */
__device__ uint make_idx(uint i, uint j, uint k)
{
    return i * %(sim_h)s * %(sim_d)s + j * %(sim_d)s + k;
}

/*
 * 2D -> 1D index in contiguous array
 */
__device__ uint make_flat_idx(uint i, uint j)
{
    return i * %(sim_h)s + j;
}

// Simulation space spans heights from -0.1 below the lowest point to +0.2 above the highest one
__device__ float make_height(uint k)
{
    return float(k) / %(sim_d)s * (1.0f + SIM_HEIGHT_ABOVE + SIM_HEIGHT_BELOW) - SIM_HEIGHT_BELOW;
}

/*
 * Returns surface (ground of water) height at the location
 */
__device__ float make_surf_height(uint i, uint j, float *surface_height)
{
    return max(surface_height[make_flat_idx(i, j)], %(sea_level)s);
}

/*
 * Returns ground height at the location
 */
__device__ float make_ground_height(uint i, uint j, float *surface_height)
{
    return surface_height[make_flat_idx(i, j)];
}

/*
 * Calculates real-scale distance between simulation cells
 */
__device__ float make_cell_dist(int si, int sj, int sk,
                                int ti, int tj, int tk)
{
    return sqrtf(powf((ti-si) / %(sim_d_scale)s * %(cell_scale)s, 2) + powf((tj-sj) / %(sim_d_scale)s * %(cell_scale)s, 2) + powf((tk-sk) * %(cell_scale)s, 2));
}

/*
 * Tells whethes the cell is right at the surface (ground or water)
 */
__device__ bool is_surface(uint i, uint j, uint k, float *surface_height) {
    if(make_height(k) < make_surf_height(i, j, surface_height)) {
        if(
              (i > 0 && make_height(k+1) >= make_surf_height(i-1, j, surface_height))
           || (i < %(sim_w)s-1 && make_height(k+1) >= make_surf_height(i+1, j, surface_height))
           || (j > 0 && make_height(k+1) >= make_surf_height(i, j-1, surface_height))
           || (j < %(sim_h)s-1 && make_height(k+1) >= make_surf_height(i, j+1, surface_height))
        ) {
          return true;
        }
    }

    return false;
}

/*
 * Tells whether the cell if at the surface or above
 */
__device__ bool is_surface_or_above(uint i, uint j, uint k, float *surface_height) {
    return is_surface(i, j, k, surface_height) || make_height(k) >= make_surf_height(i, j, surface_height);
}

/*
 * Tells whether cells is not on the boundary (and >= surface)
 */
__device__ bool is_boundary(uint i, uint j, uint k, float *surface_height) {
    if(min(i %% (%(sim_w)s-1), 1) + min(j %% (%(sim_h)s-1), 1) + min(k %% (%(sim_d)s-1), 1) < 3
        || (make_height(k) < make_surf_height(i, j, surface_height) && !is_surface(i, j, k, surface_height))) {
        return true;
    }

    return false;
}

/*
 * Returns value interpolated and real-valued simulation coordinates
 */
template<typename T>
__device__ T interp3(float i, float j, float k, T *f) {
    float fraci = i - floorf(i);
    float fracj = j - floorf(j);
    float frack = k - floorf(k);
    T f000 = f[make_idx( (int)floorf(i), (int)floorf(j), (int)floorf(k) )];
    T f001 = f[make_idx( (int)floorf(i), (int)floorf(j), (int)ceilf(k) )];
    T f010 = f[make_idx( (int)floorf(i), (int)ceilf(j), (int)floorf(k) )];
    T f011 = f[make_idx( (int)floorf(i), (int)ceilf(j), (int)ceilf(k) )];
    T f100 = f[make_idx( (int)ceilf(i), (int)floorf(j), (int)floorf(k) )];
    T f101 = f[make_idx( (int)ceilf(i), (int)floorf(j), (int)ceilf(k) )];
    T f110 = f[make_idx( (int)ceilf(i), (int)ceilf(j), (int)floorf(k) )];
    T f111 = f[make_idx( (int)ceilf(i), (int)ceilf(j), (int)ceilf(k) )];

    return lerp(
                lerp(
                    lerp(f000, f001, frack),
                    lerp(f010, f011, frack),
                    fracj
                ),
                lerp(
                    lerp(f100, f101, frack),
                    lerp(f110, f111, frack),
                    fracj
                ),
                fraci
    );
}

/*
 * Jacobi iteration method for solving Poisson equations
 */
template<typename T>
__device__ void jacobi(float alpha, float rBeta, T *x, T *next_x, T *b, float *surface_height) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( is_boundary(i, j, k, surface_height) )
        return;

    T x001 = x[make_idx(i, j, k-1)];
    T x002 = x[make_idx(i, j, k+1)];
    T x010 = x[make_idx(i, j-1, k)];
    T x020 = x[make_idx(i, j+1, k)];
    T x100 = x[make_idx(i-1, j, k)];
    T x200 = x[make_idx(i+1, j, k)];

    T v = b[make_idx(i, j, k)];

    T r = (x001 + x002 + x010 + x020 + x100 + x200 + alpha * v) * rBeta;

    next_x[make_idx(i, j, k)] = r;
}

/*
 * Jacobi iteration method for solving Poisson equations, when solving for air movement field itself
 */
template<typename T>
__device__ void jacobi_auto(float alpha, float rBeta, T *x, T *next_x, float *surface_height) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( is_boundary(i, j, k, surface_height) )
        return;

    T x001 = x[make_idx(i, j, k-1)];
    T x002 = x[make_idx(i, j, k+1)];
    T x010 = x[make_idx(i, j-1, k)];
    T x020 = x[make_idx(i, j+1, k)];
    T x100 = x[make_idx(i-1, j, k)];
    T x200 = x[make_idx(i+1, j, k)];

    T v = x[make_idx(i, j, k)];

    T r = (x001 + x002 + x010 + x020 + x100 + x200 + alpha * v) * rBeta;

    next_x[make_idx(i, j, k)] = r;
}

extern "C" {

/*
 * Zeroes out the tensor
 */
__global__ void zero3d(float *x) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    x[make_idx(i, j, k)] = 0.0f;
}

__global__ void copy3d(float *from, float *to) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    to[make_idx(i, j, k)] = from[make_idx(i, j, k)];
}

/*
 * Transport matter attrobuted usign the velocity field
 */
__global__ void advect(float rdx, float3 *velocity, float *m, float *next_m, float *surface_height) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( is_boundary(i, j, k, surface_height) )
        return;

    float3 v = velocity[make_idx(i, j, k)];
    float3 p = make_float3(float(i), float(j), float(k));
    p -= v * rdx * timestep;

    // Clamp coords
    p.x = fminf(fmaxf(p.x, 0.0f), %(sim_w)s-1.0f);
    p.y = fminf(fmaxf(p.y, 0.0f), %(sim_h)s-1.0f);
    p.z = fminf(fmaxf(p.z, 0.0f), %(sim_d)s-1.0f);

    next_m[make_idx(i, j, k)] = interp3(p.x, p.y, p.z, m);
}

/*
 * Transport matter velocity field itself
 */
__global__ void advect_self(float rdx, float3 *velocity, float3 *next_m, float *surface_height) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( is_boundary(i, j, k, surface_height) )
        return;

    float3 v = velocity[make_idx(i, j, k)];
    float3 p = make_float3(float(i), float(j), float(k));
    p -= v * rdx * timestep;

    // Clamp coords
    p.x = fminf(fmaxf(p.x, 0.0f), %(sim_w)s-1.0f);
    p.y = fminf(fmaxf(p.y, 0.0f), %(sim_h)s-1.0f);
    p.z = fminf(fmaxf(p.z, 0.0f), %(sim_d)s-1.0f);

    next_m[make_idx(i, j, k)] = interp3(p.x, p.y, p.z, velocity);
}

__global__ void jacobi_f(float alpha, float rBeta, float *x, float *next_x, float *b, float *surface_height) {
    jacobi(alpha, rBeta, x, next_x, b, surface_height);
}

__global__ void jacobi_f3(float alpha, float rBeta, float3 *x, float3 *next_x, float *surface_height) {
    jacobi_auto(alpha, rBeta, x, next_x, surface_height);
}

/*
 * Sun heating the ground, air cooling with ascent, heat outflux into the outer scape
 */
__global__ void temperature_flux(float *temperature, float *next_temperature, float3 *velocity,
                                  float *surface_height, float sim_time, float *humidity,
                                  float *precipitation, float *last_precipitation) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( is_boundary(i, j, k, surface_height) )
        return;

    float sheight = make_height(k);

    float flux = 0.0f;

    // Different kinds of temperature dynamics
    //float time_flux = (sin(sim_time * 0.05f + float(i) * 0.01f) + 1.0f) / 2.0f;
    float time_flux = 1.0f; //max(sin(sim_time * 0.03f + float(i) * 0.02f), 0.0f); //  + float(j) * 0.008f
    //time_flux = (sin(sim_time * 0.1f) + 1.0f) / 2.0f;
    time_flux = (0.1f + 0.9f * (float(j) / %(sim_h)s)) // Sphericity - poles are colder
                * (sin(sim_time * 0.1f + float(i) * 0.02f) + 1.0f) / 2.0f; // Day-night cycles
    if(make_ground_height(i, j, surface_height) >= %(sea_level)s) {
        // How much rain in this column?
        float col_precipitation = 0.0f;
        for(int l = 0; l < %(sim_d)s; l++) {
            col_precipitation += precipitation[make_idx(i, j, l)] - last_precipitation[make_idx(i, j, l)];
        }
        // Sun heating the ground
        const float SUN_HEAT = 0.5f;
        flux += SUN_HEAT * (1.0f / pow(max((sheight - make_surf_height(i, j, surface_height)) * %(sim_h)s, 1.0f), 2.0f)) * timestep * time_flux;
        flux -= 0.1f * col_precipitation * timestep * time_flux;
    } else if(sheight < make_surf_height(i, j, surface_height) && sheight <= %(sea_level)s) {
        // How much rain in this column?
        float col_precipitation = 0.0f;
        for(int l = 0; l < %(sim_d)s; l++) {
            col_precipitation += precipitation[make_idx(i, j, l)] - last_precipitation[make_idx(i, j, l)];
        }
        // Sun heating the water
        const float SUN_WATER_HEAT = 0.05f;
        flux += SUN_WATER_HEAT * (1.0f / pow(max((sheight - make_surf_height(i, j, surface_height)) * %(sim_h)s, 1.0f), 2.0f)) * timestep * time_flux;
        //flux -= 0.1f * col_precipitation * timestep * time_flux;
    }

    // Cooling with ascent / heating with descent
    if(velocity[make_idx(i, j, k)].z >= 0) {
        const float ASCENT_COOL_RATE = 0.02f;
        flux += temperature[make_idx(i, j, k)] * -velocity[make_idx(i, j, k)].z * ASCENT_COOL_RATE * timestep;
    } else {
        const float DESCENT_HEAT_RATE = 0.015f;
        flux += temperature[make_idx(i, j, k)] * velocity[make_idx(i, j, k)].z * DESCENT_HEAT_RATE * timestep;
    }
    if( k == %(sim_d)s-2 ) {
        // Cool air in the top cells
        const float OUTFLUX_RATE = 0.005f;
        flux -= temperature[make_idx(i, j, k)] * OUTFLUX_RATE * timestep;
    }

    next_temperature[make_idx(i, j, k)] = temperature[make_idx(i, j, k)] + flux;
}

/*
 * Hot air floats, cold air falls
 */
__global__ void convection(float3 *velocity, float3 *next_velocity,
                            float *surface_height,
                            float *temperature, float ambient_temperature,
                            float *humidity, float ambient_humidity, float sim_time) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( is_boundary(i, j, k, surface_height) )
        return;

    float stemperature = temperature[make_idx(i, j, k)];
    float shumidity = humidity[make_idx(i, j, k)];
    float nval = 1e-4;
    float3 temperature_gradient = make_float3(0.0f, 0.0f, 0.0f);

    if(!is_boundary(i+1, j, k, surface_height)) {
        temperature_gradient += (stemperature - temperature[make_idx(i+1, j, k)]) * make_float3(1.0f, 0.0f, 0.0f);
        nval += 1;
    }
    if(!is_boundary(i-1, j, k, surface_height)) {
        temperature_gradient += (stemperature - temperature[make_idx(i-1, j, k)]) * make_float3(-1.0f, 0.0f, 0.0f);
        nval += 1;
    }
    if(!is_boundary(i, j+1, k, surface_height)) {
        temperature_gradient += (stemperature - temperature[make_idx(i, j+1, k)]) * make_float3(0.0f, 1.0f, 0.0f);
        nval += 1;
    }
    if(!is_boundary(i, j-1, k, surface_height)) {
        temperature_gradient += (stemperature - temperature[make_idx(i, j-1, k)]) * make_float3(0.0f, -1.0f, 0.0f);
        nval += 1;
    }
    temperature_gradient /= nval;
    temperature_gradient *= 1.0f; // Scale the horizontal gradient

    // Hot air floats, cold air falls
    const float BUOYANCY_RATE = 0.2f;
    //const uint seed = 1;
    uint seed = (uint)( sim_time * 1000 / 127 );

    // TODO: More humid air should be heavier
    float convection_speed = BUOYANCY_RATE * (
          stemperature - ambient_temperature
        - (shumidity - ambient_humidity) * 0.2f
    );

    uint cell_hash = cudaNoise::calcPerm12(seed + i + cudaNoise::calcPerm(seed + j + cudaNoise::calcPerm(seed + k)));
    next_velocity[make_idx(i, j, k)] = velocity[make_idx(i, j, k)]
        + make_float3(
              abs(convection_speed) * temperature_gradient.x,
              abs(convection_speed) * temperature_gradient.y,
              convection_speed * 1.0f
          );
}

/*
 * This is exactly how clouds form and make rain. Water from rivers, lakes, streams, or oceans evaporates into the air when it is heated up by the sun. As the water vapor rises up in the air, it condenses, or starts to cool down and turns back into a liquid. Then, droplets of water start to stick together as clouds. When enough droplets stick together in the clouds, they become large and heavy and are pulled down towards the earth by the force of gravity. When water drops fall from clouds, it is called rain. Sometimes the droplets freeze before they get to the ground and become hail, sleet, or snow!
 * Ref: https://learning-center.homesciencetools.com/article/clouds-and-rain/
 */
__global__ void water_cycle(float *humidity,
                            float *precipitation,
                            float *surface_height, float *temperature) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;
    uint idx = make_idx(i, j, k);

    float sheight = make_height(k);
    float stemperature = temperature[idx];

    if( is_boundary(i, j, k, surface_height) )
        return;

    if(make_ground_height(i, j, surface_height) < %(sea_level)s) { //  && sheight <= %(sea_level)s
        // Absorb surface water
        const float WATER_ABSORB_RATE = 1.0f;
        humidity[idx] += WATER_ABSORB_RATE * (1.0f / pow(max((sheight - make_surf_height(i, j, surface_height)) * %(sim_h)s, 1.0f), 2.0f)) * timestep;
    }
    if(make_ground_height(i, j, surface_height) >= %(sea_level)s && is_surface(i, j, k, surface_height)) {
        // Fresh water evaporation & evapotranspiration
        const float WATER_ABSORB_RATE = 0.1f;
        humidity[idx] += WATER_ABSORB_RATE * (1.0f / pow(max((sheight - make_surf_height(i, j, surface_height)) * %(sim_h)s, 1.0f), 2.0f)) * max(1.0f - sheight, 0.0f) * timestep;
    }

    float shumidity = humidity[idx];

    // Condensate excess vapor
    float dew_point = stemperature / sheight * 0.5f;
    if(shumidity > dew_point) {
        humidity[idx] += (dew_point - shumidity) * timestep;
        if(make_ground_height(i, j, surface_height) > %(sea_level)s)
            precipitation[idx] += fabsf(dew_point - shumidity) * timestep;
    }
}

/*
 * Calculate divergence. `halfrdx` is 0.5 / gridscale
 */
__global__ void divergence(float halfrdx, float3 *w, float *d, float *surface_height) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( is_boundary(i, j, k, surface_height) )
        return;

    float3 w100 = w[make_idx(i-1, j, k)];
    float3 w200 = w[make_idx(i+1, j, k)];
    float3 w010 = w[make_idx(i, j-1, k)];
    float3 w020 = w[make_idx(i, j+1, k)];
    float3 w001 = w[make_idx(i, j, k-1)];
    float3 w002 = w[make_idx(i, j, k+1)];

    d[make_idx(i, j, k)] = halfrdx * ((w200.x - w100.x) + (w020.y - w010.y) + (w002.z - w001.z));
}

/*
 * Deduce the pressure gradient from air velocity to preserve consistency
 */
__global__ void gradient(float halfrdx, float *p, float3 *w, float3 *u, float *surface_height) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( is_boundary(i, j, k, surface_height) )
        return;

    float p100 = p[make_idx(i-1, j, k)];
    float p200 = p[make_idx(i+1, j, k)];
    float p010 = p[make_idx(i, j-1, k)];
    float p020 = p[make_idx(i, j+1, k)];
    float p001 = p[make_idx(i, j, k-1)];
    float p002 = p[make_idx(i, j, k+1)];

    u[make_idx(i, j, k)] = w[make_idx(i, j, k)] - halfrdx * make_float3(p200-p100, p020-p010, p002-p001);
}

/*
 * Sets valid boundary values
 */
__global__ void boundary_f(float scale, float *x, float *surface_height) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( !is_boundary(i, j, k, surface_height) )
        return;

    float val = 0.0f * x[make_idx(i, j, k)];
    float nval = 1e-4;

    if( i > 0 && !is_boundary(i-1, j, k, surface_height) ) {
      val += x[make_idx(i-1, j, k)];
      nval += 1;
    }
    if( i < %(sim_w)s-1 && !is_boundary(i+1, j, k, surface_height) ) {
      val += x[make_idx(i+1, j, k)];
      nval += 1;
    }
    if( j > 0 && !is_boundary(i, j-1, k, surface_height) ) {
      val += x[make_idx(i, j-1, k)];
      nval += 1;
    }
    if( j < %(sim_h)s-1 && !is_boundary(i, j+1, k, surface_height) ) {
      val += x[make_idx(i, j+1, k)];
      nval += 1;
    }
    if( k > 0 && !is_boundary(i, j, k-1, surface_height) ) {
      val += x[make_idx(i, j, k-1)];
      nval += 1;
    }
    if( k < %(sim_d)s-1 && !is_boundary(i, j, k+1, surface_height) ) {
      val += x[make_idx(i, j, k+1)];
      nval += 1;
    }

    x[make_idx(i, j, k)] = scale * val / nval;
}

/*
 * Sets valid boundary values
 */
__global__ void boundary_f3(float scale, float3 *x, float *surface_height) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if( !is_boundary(i, j, k, surface_height) )
        return;

    float3 val = 0.0f * x[make_idx(i, j, k)];
    float nval = 1e-4;

    if( i > 0 && !is_boundary(i-1, j, k, surface_height) ) {
      val += make_float3(scale * x[make_idx(i-1, j, k)].x, x[make_idx(i-1, j, k)].y, x[make_idx(i-1, j, k)].z); // x[make_idx(i-1, j, k)];
      nval += 1;
    }
    if( i < %(sim_w)s-1 && !is_boundary(i+1, j, k, surface_height) ) {
      val += make_float3(scale * x[make_idx(i+1, j, k)].x, x[make_idx(i+1, j, k)].y, x[make_idx(i+1, j, k)].z); // x[make_idx(i+1, j, k)];
      nval += 1;
    }
    if( j > 0 && !is_boundary(i, j-1, k, surface_height) ) {
      val += make_float3(x[make_idx(i, j-1, k)].x, scale * x[make_idx(i, j-1, k)].y, x[make_idx(i, j-1, k)].z); // x[make_idx(i, j-1, k)];
      nval += 1;
    }
    if( j < %(sim_h)s-1 && !is_boundary(i, j+1, k, surface_height) ) {
      val += make_float3(x[make_idx(i, j+1, k)].x, scale * x[make_idx(i, j+1, k)].y, x[make_idx(i, j+1, k)].z); // x[make_idx(i, j+1, k)];
      nval += 1;
    }
    if( k > 0 && !is_boundary(i, j, k-1, surface_height) ) {
      val += make_float3(x[make_idx(i, j, k-1)].x, x[make_idx(i, j, k-1)].y, scale * x[make_idx(i, j, k-1)].z); // x[make_idx(i, j, k-1)];
      nval += 1;
    }
    if( k < %(sim_d)s-1 && !is_boundary(i, j, k+1, surface_height) ) {
      val += make_float3(x[make_idx(i, j, k+1)].x, x[make_idx(i, j, k+1)].y, scale * x[make_idx(i, j, k+1)].z); // x[make_idx(i, j, k+1)];
      nval += 1;
    }

    x[make_idx(i, j, k)] = val / nval;
}
}
