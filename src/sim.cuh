#pragma once

// #include "src/pixel.cuh"
// #include "src/cell.cuh"

// #include <cuda_runtime.h>

/*
namespace details {

constexpr size_t ceil_div(size_t num, size_t denom) {
	return (num + denom - 1) / denom;
}

constexpr float deg_to_rad(float deg) {
	return deg * 3.14159265358979323846 / 180;
}

}  // namespace details

namespace params {
static constexpr int   deposit = 0xff;
static constexpr float sensor_angle_deg = 45;
static constexpr float sensor_angle_rad = details::deg_to_rad(sensor_angle_deg);
static constexpr float agent_turn_deg = 45;
static constexpr float agent_turn_rad = details::deg_to_rad(agent_turn_deg);
static constexpr float sensor_distance = 2;
static constexpr float agent_move_distance = 2;

static constexpr size_t n_agents = 2;
static constexpr size_t width = 200;
static constexpr size_t height = 200;

static constexpr size_t agent_block_dim = 128;
static constexpr size_t agent_grid_dim =
    details::ceil_div(n_agents, agent_block_dim);

static constexpr size_t cells_block_width = 32;
static constexpr size_t cells_block_height = 32;
static constexpr size_t cells_grid_width =
    details::ceil_div(width, cells_block_width);
static constexpr size_t cells_grid_height =
    details::ceil_div(height, cells_block_height);
static constexpr dim3 cells_block_dim =
    dim3{cells_block_width, cells_block_height};
static constexpr dim3 cells_grid_dim = dim3{cells_grid_width, cells_grid_height};

}  // namespace params

struct Cell {
	int x = 0;

	inline __device__ void operator+=(const Cell rhs) { x += rhs.x; }

	__device__ Cell operator/(int n) const { return Cell{x / n}; }
};
*/

// namespace initialize {

// __host__ void cells(Cell * cells, int pitch);

// __host__ void agents(float * x, float * y, float * dir);

// }  // namespace initialize

// __host__ void step(
//     Cell ** prev,
//     Cell ** next,
//     int     cell_pitch,
//     float * x,
//     float * y,
//     float * dir,
//     Pixel * pixels,
//     int     pixel_pitch);
