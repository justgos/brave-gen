#include "cuda/helper_math.h"
#include "cuda/cudaNoise/cudaNoise.cuh"

const int BIOME_WATER_DEEP = 0;
const int BIOME_WATER_SHALLOW = 1;
const int BIOME_GRASS = 2;
const int BIOME_ROCKS = 3;
const int BIOME_SNOW = 4;
const int BIOME_DIRT = 5;

extern "C" {

  /*
   * 2D -> 1D index in contiguous array
   */
__device__ uint make_flat_idx(uint i, uint j)
{
    return i * %(map_h)s + j;
}

/*
 * 3D -> 1D index in contiguous array
 */
__device__ uint make_biome_idx(uint i, uint j, int biome)
{
    return make_flat_idx(i, j) * %(nbiomes)s + biome;
}

/*
 * Deforms a geosphere to make a stone-like object
 */
__global__ void make_stone(float3 *verts, float2 *uvs, float3 *orig_verts, uint numv, float3 scale, int seed) {
  uint i = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(i >= numv)
    return;

  float3 v = orig_verts[i];

  // float3 uvw2 = make_float3(
  //                 cudaNoise::repeaterSimplex(v + make_float3(0.0f, 0.0f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
  //                 cudaNoise::repeaterSimplex(v + make_float3(0.2f, 0.7f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
  //                 0.0f
  // );
  // float3 uvw3 = make_float3(
  //                 cudaNoise::repeaterSimplex(v + 0.5f * uvw2 + make_float3(1.4f, 2.1f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
  //                 cudaNoise::repeaterSimplex(v + 0.5f * uvw2 + make_float3(2.7f, 1.8f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
  //                 0.0f
  // );

  v.x *= scale.x;
  v.y *= scale.y;
  v.z *= scale.z;

  // float3 pos, float scale, int seed, float size, int minNum, int maxNum, float jitter
  verts[i] = v + 0.15f * cudaNoise::worleyNoise(v, 1.0f, seed, 1.0f, 1, 100, 0.0f) * v;
  verts[i] += 0.2f * cudaNoise::repeaterPerlin(verts[i] + make_float3(10.0f, 10.0f, 10.0f), 1.0f, seed, 15, 2.0f, 0.5f) * verts[i];

  uvs[i] = make_float2( acos(verts[i].y / norm3df(verts[i].x, verts[i].y, verts[i].z)), atan2(verts[i].z, verts[i].x) );
}

/*
 * Generate the base terrain mesh
 */
__global__ void make_noise(float *res)
{
  uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
  float3 uvw = make_float3(float(i), float(j), 0.0f) / %(map_scale)s;

  // float3 pos, float scale, int seed, int n, float lacunarity, float decay
  float3 uvw2 = make_float3(
                  cudaNoise::repeaterSimplex(uvw + make_float3(0.0f, 0.0f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                  cudaNoise::repeaterSimplex(uvw + make_float3(0.2f, 0.7f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                  0.0f
  );
  float3 uvw3 = make_float3(
                  cudaNoise::repeaterSimplex(uvw + 0.5f * uvw2 + make_float3(1.4f, 2.1f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                  cudaNoise::repeaterSimplex(uvw + 0.5f * uvw2 + make_float3(2.7f, 1.8f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                  0.0f
  );
  res[make_flat_idx(i, j)] = cudaNoise::repeaterPerlin(uvw + 0.3f * uvw3, 1.0f, 1, 15, 2.0f, 0.5f);
  //res[make_flat_idx(i, j)] *= 1.0f + cudaNoise::repeaterPerlin_d(uvw, 1.0f, 1, 15, 2.0f, 0.5f) * 0.2f;
  res[make_flat_idx(i, j)] *= 0.6f + pow(fabsf(cudaNoise::repeaterHybrid(uvw + 0.3f * uvw3, 1.0f, 1, 15, 2.0f, 0.9f, 0.7f)), 2.0f);

  // TODO: derivative noise, http://www.iquilezles.org/www/articles/morenoise/morenoise.htm

  // TODO: custom amplitudes for different octaves

  // TODO: Billow noise: https://www.google.com.ua/search?q=billow+noise&oq=billow+noise
  // other noise types:
  // https://thebookofshaders.com/11/
  // http://www.iquilezles.org/www/articles/voronoise/voronoise.htm
  // http://www.upvector.com/?section=Tutorials&subsection=Intro to Procedural Textures

  //res[make_flat_idx(i, j)] += -cudaNoise::repeaterPerlinAbs(uvw + 0.3f * uvw2, 1.0f, 1, 15, 2.0f, 0.5f) * 0.5;
  //res[make_flat_idx(i, j)] = cudaNoise::repeaterSimplexExp(uvw, 1.0f, 1, 15, 2.0f, 0.5f);
}

/*
 * Generate some dirt biomes
 */
__device__ float get_dirt_density(uint i, uint j, float3 uvw, float *h, float sea_level, float *biomes)
{
  float land_h = max(h[make_flat_idx(i, j)] - sea_level, 0.0f) / (1.0f - sea_level);

  float3 uvw2 = make_float3(
                  cudaNoise::repeaterSimplex(uvw + make_float3(2.0f, 7.0f, 5.0f), 0.3f, 1, 10, 2.0f, 0.5f),
                  cudaNoise::repeaterSimplex(uvw + make_float3(6.2f, 2.7f, 0.0f), 0.3f, 1, 10, 2.0f, 0.5f),
                  0.0f
  );
  float3 uvw3 = make_float3(
                  cudaNoise::repeaterSimplex(uvw + 0.5f * uvw2 + make_float3(1.4f, 2.1f, 0.0f), 0.3f, 1, 10, 2.0f, 0.5f),
                  cudaNoise::repeaterSimplex(uvw + 0.5f * uvw2 + make_float3(2.7f, 1.8f, 0.0f), 0.3f, 1, 10, 2.0f, 0.5f),
                  0.0f
  );

  float noise = fabsf(cudaNoise::perlinNoise(uvw + make_float3(21.2f, 3.1f, 5.1f) + 0.2f * uvw3, 12.0f, 1));
  if(noise < 0.55f)
    noise = 0.0f;
  return (1.0f - land_h) * pow(noise, 4.0f) * 4.0f;
}

__device__ float get_grass_density(uint i, uint j, float3 uvw, float *h, float sea_level, float *biomes)
{
  float land_h = max(h[make_flat_idx(i, j)] - sea_level, 0.0f) / (1.0f - sea_level);
  return (1.0f - land_h) * max(1.0f - biomes[make_biome_idx(i, j, BIOME_DIRT)], 0.0f);
}

/*
 * Generate some tree biomes
 */
__global__ void get_tree_density(float *h, float sea_level, float *biomes, float *res)
{
  uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
  float3 uvw = make_float3(float(i), float(j), 0.0f) / %(map_scale)s;

  float sheight = h[make_flat_idx(i, j)];

  if(sheight > sea_level) {
    float3 uvw2 = make_float3(
                    cudaNoise::repeaterSimplex(uvw + make_float3(0.0f, 0.0f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                    cudaNoise::repeaterSimplex(uvw + make_float3(0.2f, 0.7f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                    0.0f
    );
    float3 uvw3 = make_float3(
                    cudaNoise::repeaterSimplex(uvw + 0.5f * uvw2 + make_float3(1.4f, 2.1f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                    cudaNoise::repeaterSimplex(uvw + 0.5f * uvw2 + make_float3(2.7f, 1.8f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                    0.0f
    );

    /*return pow(1.0 - sheight, 2.0)
                              * pow((cudaNoise::perlinNoise(uvw + make_float3(23.2f, 35.1f, 5.1f) + 0.5f * uvw3, 10.0f, 1) + 1.0f) / 2.0f, 1.0f) * 2.0f;*/

    float noise = fabsf(cudaNoise::perlinNoise(uvw + make_float3(23.2f, 35.1f, 5.1f) + 0.5f * uvw3, 7.0f, 1));
    // Cut off the noise and use another layer for density
    // idea from `Continuous World Generation in No Man s Sky`, https://www.youtube.com/watch?v=sCRzxEEcO2Y
    if(noise < 0.15f)
      noise = 0.0f;
    noise *= fabsf(cudaNoise::perlinNoise(uvw + make_float3(63.2f, 25.1f, 55.1f) + 0.4f * uvw3, 7.0f, 1));

    res[make_flat_idx(i, j)] = pow(1.0 - sheight, 2.0) * noise * 8.0f * biomes[make_biome_idx(i, j, BIOME_GRASS)]
                                * 0.0015;
  } else {
    res[make_flat_idx(i, j)] = 0.0f;
  }
}

/*
 * Generate some grass
 */
__global__ void get_vegetation_density(float *h, float sea_level, float *biomes, float *res)
{
  uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
  float3 uvw = make_float3(float(i), float(j), 0.0f) / %(map_scale)s;

  float sheight = h[make_flat_idx(i, j)];

  if(sheight > sea_level) {
    float3 uvw2 = make_float3(
                    cudaNoise::repeaterSimplex(uvw + make_float3(0.0f, 0.0f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                    cudaNoise::repeaterSimplex(uvw + make_float3(0.2f, 0.7f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                    0.0f
    );
    float3 uvw3 = make_float3(
                    cudaNoise::repeaterSimplex(uvw + 0.5f * uvw2 + make_float3(1.4f, 2.1f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                    cudaNoise::repeaterSimplex(uvw + 0.5f * uvw2 + make_float3(2.7f, 1.8f, 0.0f), 0.2f, 1, 10, 2.0f, 0.5f),
                    0.0f
    );

    /*return pow(1.0 - sheight, 2.0)
                              * pow((cudaNoise::perlinNoise(uvw + make_float3(23.2f, 35.1f, 5.1f) + 0.5f * uvw3, 10.0f, 1) + 1.0f) / 2.0f, 1.0f) * 2.0f;*/

    float noise = fabsf(cudaNoise::perlinNoise(uvw + make_float3(23.2f, 35.1f, 5.1f) + 0.5f * uvw3, 7.0f, 1));
    // Cut off the noise and use another layer for density
    // idea from `Continuous World Generation in No Man s Sky`, https://www.youtube.com/watch?v=sCRzxEEcO2Y
    if(noise < 0.45f)
      noise = 0.0f;
    noise *= fabsf(cudaNoise::perlinNoise(uvw + make_float3(63.2f, 25.1f, 55.1f) + 0.4f * uvw3, 7.0f, 1));

    res[make_flat_idx(i, j)] = pow(1.0 - sheight, 2.0) * noise * 8.0f * biomes[make_biome_idx(i, j, BIOME_GRASS)]
                                * 0.15;
  } else {
    res[make_flat_idx(i, j)] = 0.0f;
  }
}

/*
 * Generate stone map
 */
__global__ void get_stone_density(float *h, float sea_level, float *biomes, float *res)
{
  uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
  float3 uvw = make_float3(float(i), float(j), 0.0f) / %(map_scale)s;

  float sheight = h[make_flat_idx(i, j)];

  if(sheight > sea_level) {
    float noise = fabsf(cudaNoise::perlinNoise(uvw + make_float3(73.2f, 32.1f, 35.1f), 5.0f, 1));
    // Cut off the noise and use another layer for density
    // idea from `Continuous World Generation in No Man s Sky`, https://www.youtube.com/watch?v=sCRzxEEcO2Y
    if(noise < 0.15f)
      noise = 0.0f;
    noise *= fabsf(cudaNoise::perlinNoise(uvw + make_float3(23.2f, 15.1f, 57.1f), 7.0f, 1));

    res[make_flat_idx(i, j)] = pow(sheight, 2.0f) * noise * 8.0f * biomes[make_biome_idx(i, j, BIOME_DIRT)]
                                * 0.008;
  } else {
    res[make_flat_idx(i, j)] = 0.0f;
  }
}

/*
 * Generate biomes over the base terrain
 */
__global__ void assign_biome(float *biomes, float *h, float sea_level)
{
  uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
  float3 uvw = make_float3(float(i), float(j), 0.0f) / %(map_scale)s;

  float slope = 0.0f;
  // If can get a gradient - compute slope
  // if(i > 0 && j > 0 && i < %(map_w)s-1 && j < %(map_h)s-1) {
  //     slope = (abs(h[make_flat_idx(i, j+1)] - h[make_flat_idx(i, j)]) \
  //                  + abs(h[make_flat_idx(i+1, j)] - h[make_flat_idx(i, j)]) \
  //                  + abs(h[make_flat_idx(i, j-1)] - h[make_flat_idx(i, j)]) \
  //                  + abs(h[make_flat_idx(i-1, j)] - h[make_flat_idx(i, j)])) \
  //                  / 4.0f * %(slope_scale)s;
  // }
  if(i < %(map_w)s-1 && j < %(map_h)s-1) {
      slope = max(
                  abs(h[make_flat_idx(i, j+1)] - h[make_flat_idx(i, j)]),
                  abs(h[make_flat_idx(i+1, j)] - h[make_flat_idx(i, j)])
                ) * %(slope_scale)s;
  }

  if(h[make_flat_idx(i, j)] < sea_level + 1e-4f) {
      float water_h = max(-(h[make_flat_idx(i, j)] - (sea_level - 1e-4f)), 0.0f) / sea_level;

      // Deep Water
      biomes[make_biome_idx(i, j, BIOME_WATER_DEEP)] = water_h;
      // Water
      biomes[make_biome_idx(i, j, BIOME_WATER_SHALLOW)] = 1.0f - water_h;
  }
  if(h[make_flat_idx(i, j)] > sea_level) {
      float land_h = max(h[make_flat_idx(i, j)] - sea_level, 0.0f) / (1.0f - sea_level);

      float cliff_slope = 1.0f;
      // if(slope >= cliff_slope)
      //     biomes[make_biome_idx(i, j, BIOME_ROCKS)] = (slope-cliff_slope) * 4.0f * pow(land_h, 8.0f) * 10.0f;
      if(slope >= cliff_slope)
        biomes[make_biome_idx(i, j, BIOME_ROCKS)] = (slope-cliff_slope) * pow(land_h, 1.0f) * 2.0f;

      // Dirt
      biomes[make_biome_idx(i, j, BIOME_DIRT)] = get_dirt_density(i, j, uvw, h, sea_level, biomes);
      // Grass
      biomes[make_biome_idx(i, j, BIOME_GRASS)] = get_grass_density(i, j, uvw, h, sea_level, biomes);
      // Mountains - snow
      biomes[make_biome_idx(i, j, BIOME_SNOW)] = pow(land_h, 12.0f) * 10.0f;

      if(slope < cliff_slope + 1e-4) {
          //
      }
  }
}

}
