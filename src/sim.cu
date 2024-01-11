#include "src/sim.cuh"

#include "src/vec.cuh"

#include <cstdint>

inline __device__ uint32_t idx_1(void) {
	return (blockIdx.x * blockDim.x) + threadIdx.x;
}

inline __device__ IVec2 idx_2(void) {
	return IVec2{
	    static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x),
	    static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y),
	};
}

template<typename T>
inline __device__ T * pitch_ptr(T * base, IVec2 idx, int pitch) {
	uint32_t x = idx.x % params::width;
	uint32_t y = idx.y % params::height;
	return ((T *)((char *)base + (y * pitch))) + x;
}

namespace initialize {

__global__ void k_cells(Cell * cells, int pitch) {
	auto coords = idx_2();
	if (coords.x >= static_cast<int>(params::width)) return;
	if (coords.y >= static_cast<int>(params::height)) return;

	Cell * output = pitch_ptr(cells, coords, pitch);
	output->x = 0;
}


__global__ void k_agents(float * x, float * y, float * dir) {
	auto idx = idx_1();
	if (idx > params::n_agents) return;

	x[idx] = static_cast<float>(params::width) / 2;
	y[idx] = static_cast<float>(params::height) / 2;
	dir[idx] = 0;
}

__host__ void cells(Cell * cells, int pitch) {
	k_cells<<<params::cells_grid_dim, params::cells_block_dim>>>(cells, pitch);
}

__host__ void agents(float * x, float * y, float * dir) {
	k_agents<<<params::agent_grid_dim, params::agent_block_dim>>>(x, y, dir);
}

}  // namespace initialize


inline __device__ void render_cell(const Cell * cell, Pixel * pixel) {
	pixel->r = cell->x & 0xff;
	pixel->g = cell->x & 0xff;
	pixel->b = cell->x & 0xff;
	pixel->a = 0xff;
}

__global__ void
k_render(const Cell * cells, int cell_pitch, Pixel * pixels, int pix_pitch) {
	auto coords = idx_2();
	if (coords.x >= static_cast<int>(params::width)) return;
	if (coords.y >= static_cast<int>(params::height)) return;

	render_cell(
	    pitch_ptr(cells, coords, cell_pitch),
	    pitch_ptr(pixels, coords, pix_pitch));
}

__host__ void
render(const Cell * cells, int cell_pitch, Pixel * pixels, int pix_pitch) {
	k_render<<<params::cells_grid_dim, params::cells_block_dim>>>(
	    cells, cell_pitch, pixels, pix_pitch);
}


__device__ Cell get_avg_cell(const Cell * cells, IVec2 coords, int pitch) {
	Cell res;

#pragma unroll
	for (int dy = -1; dy < 2; ++dy) {
		int y = coords.y + dy;
#pragma unroll
		for (int dx = -1; dx < 2; ++dx) {
			int x = coords.x + dx;
			res += *pitch_ptr(cells, {x, y}, pitch);
		}
	}

	return res / 9;
}

inline __device__ FVec2 carts(float theta) {
	FVec2 res;
	sincosf(theta, &res.y, &res.x);
	return res;
}

inline __device__ FVec2 moved(FVec2 src, float dir) {
	return src + (carts(dir) * params::sensor_distance);
}

inline __device__ Cell sample(Cell * cells, FVec2 coords, int pitch) {
	IVec2 i_coords = {static_cast<int>(coords.x), static_cast<int>(coords.y)};
	return *pitch_ptr(cells, i_coords, pitch);
}

__global__ void
agent_step(float * xs, float * ys, float * dirs, Cell * cells, int pitch) {
	auto idx = idx_1();
	if (idx >= params::n_agents) return;

	auto coords = FVec2{xs[idx], ys[idx]};
	auto d = dirs[idx];

	// sense
	auto fl = sample(cells, moved(coords, d - params::sensor_angle_rad), pitch);
	auto c = sample(cells, moved(coords, d), pitch);
	auto fr = sample(cells, moved(coords, d + params::sensor_angle_rad), pitch);

	if (fl.x > c.x && fl.x > fr.x) {
		d += params::agent_turn_rad;
	} else if (fr.x > fl.x && fr.x > c.x) {
		d -= params::agent_turn_rad;
	}

	// move
	coords += carts(d) * params::agent_move_distance;

	// Deposit
	IVec2 i_coords = {static_cast<int>(coords.x), static_cast<int>(coords.y)};
	atomicAdd(&(pitch_ptr(cells, i_coords, pitch)->x), params::deposit);

	// Write back coordinates
	xs[idx] = coords.x;
	ys[idx] = coords.y;
	dirs[idx] = d;
}

inline __device__ void decay(Cell * cell) { cell->x *= 0.8; }

__global__ void media_step(const Cell * prev, Cell * next, int pitch) {
	auto coords = idx_2();
	if (coords.x >= static_cast<int>(params::width)) return;
	if (coords.y >= static_cast<int>(params::height)) return;

	Cell * output = pitch_ptr(next, coords, pitch);
	Cell   c = get_avg_cell(prev, coords, pitch);

	// decay(&c);
	*output = c;
}

__host__ void step(
    Cell ** prev,
    Cell ** next,
    int     cell_pitch,
    float * x,
    float * y,
    float * dir,
    Pixel * pixels,
    int     pix_pitch) {
	agent_step<<<params::agent_grid_dim, params::agent_block_dim>>>(
	    x, y, dir, *prev, cell_pitch);

	media_step<<<params::cells_grid_dim, params::cells_block_dim>>>(
	    *prev, *next, cell_pitch);

	render(*next, cell_pitch, pixels, pix_pitch);

	Cell * tmp = *prev;
	*prev = *next;
	*next = tmp;
}
