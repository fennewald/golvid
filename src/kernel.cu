#include "src/kernel.cuh"

#include <cstdint>

#include <curand_kernel.h>
#include <fmt/format.h>

#include "src/pixel.hh"

static constexpr long k_block_width = 4;
static constexpr long k_block_height = 4;

static constexpr pixel_t k_on_pixel = k_white;
static constexpr pixel_t k_off_pixel = k_black;

inline __device__ unsigned int index(int x, int y, res_t res, int pitch) {
	x %= res.width;
	y %= res.height;
	return x + (y * pitch);
}

inline __device__ uint8_t rule(bool alive, uint8_t neighbors) {
	switch (neighbors) {
	case 0: [[fallthrough]];
	case 1: return 0;
	case 2: return alive ? 1 : 0;
	case 3: return 1;
	case 4: [[fallthrough]];
	case 5: return 0;
	case 6: [[fallthrough]];
	case 7: return 1;
	case 8: return 0;
	default: __builtin_unreachable();
	}
}

__device__ uint8_t
get_cell(const uint8_t * prev, int x, int y, res_t res, int pitch) {
	int  idx = index(x, y, res, pitch);
	bool alive = prev[idx];

	// clang-format off
	uint8_t neighbors = prev[index(x - 1, y - 1, res, pitch)] +
	                    prev[index(  x  , y - 1, res, pitch)] +
	                    prev[index(x + 1, y - 1, res, pitch)] +
	                    prev[index(x - 1,   y  , res, pitch)] +
	                    prev[index(x + 1,   y  , res, pitch)] +
	                    prev[index(x - 1, y + 1, res, pitch)] +
	                    prev[index(  x  , y + 1, res, pitch)] +
	                    prev[index(x + 1, y + 1, res, pitch)];
	// clang-format on

	return rule(alive, neighbors);
}

inline __device__ uint8_t scale_channel(uint8_t raw) {
	/*
	double  factor = 0.9;
	uint8_t a = 0xff - raw;
	uint8_t b = a * 0.9;
	return 0xff - b;
	*/
	return (uint8_t)((double)raw * 0.9);
}

inline __device__ void update_pixel(pixel_t * pixel, uint8_t val) {
	if (val) {
		*pixel = k_on_pixel;
	} else {
		pixel->r = scale_channel(pixel->r);
		pixel->g = scale_channel(pixel->g);
		pixel->b = scale_channel(pixel->b);
	}
}

inline __device__ int job_x(void) {
	return (blockIdx.x * blockDim.x) + threadIdx.x;
}

inline __device__ int job_y(void) {
	return (blockIdx.y * blockDim.y) + threadIdx.y;
}

__global__ void step_gol(
    const uint8_t * prev,
    uint8_t *       next,
    res_t           res,
    int             pitch,
    pixel_t *       pixels,
    int             pix_pitch) {
	int x = job_x();
	int y = job_y();
	if (x >= (int)res.width) return;
	if (y >= (int)res.height) return;

	uint8_t val = get_cell(prev, x, y, res, pitch);
	update_pixel(&pixels[index(x, y, res, pix_pitch)], val);
	next[index(x, y, res, pitch)] = val;
}

__global__ void kernel_initalize(
    uint8_t * w0, uint8_t * w1, res_t res, int pitch, pixel_t * pixels, int pix_pitch) {
	int x = job_x();
	int y = job_y();
	if (x >= (int)res.width) return;
	if (y >= (int)res.height) return;

	int w_idx = index(x, y, res, pitch);
	int p_idx = index(x, y, res, pix_pitch);

	curandState rng;
	curand_init(10881, w_idx, 0, &rng);
	int  r = curand(&rng);
	bool val = r & 1;

	w0[w_idx] = val ? 1 : 0;
	w1[w_idx] = val ? 1 : 0;
	pixels[p_idx] = val ? k_on_pixel : k_off_pixel;
}

void initalize(
    uint8_t * w0, uint8_t * w1, res_t res, int pitch, pixel_t * pixels, int pix_pitch) {
	dim3 block_dim = dim3(k_block_width, k_block_height);
	dim3 grid_dim(
	    (res.width + k_block_width - 1) / k_block_width,
	    (res.height + k_block_height - 1) / k_block_height);

	fmt::println("block_dim: {}x{}", block_dim.x, block_dim.y);
	fmt::println("grid_dim: {}x{}", grid_dim.x, grid_dim.y);
	kernel_initalize<<<grid_dim, block_dim>>>(
	    w0, w1, res, pitch, pixels, pix_pitch / sizeof(pixel_t));
}

void step(
    const uint8_t * prev,
    uint8_t *       next,
    res_t           res,
    int             pitch,
    pixel_t *       pixels,
    int             pix_pitch) {
	dim3 block_dim(k_block_width, k_block_height);
	dim3 grid_dim(
	    (res.width + k_block_width - 1) / k_block_width,
	    (res.height + k_block_height - 1) / k_block_height);
	step_gol<<<grid_dim, block_dim>>>(
	    prev, next, res, pitch, pixels, pix_pitch / sizeof(pixel_t));
}
