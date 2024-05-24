#include "src/agent_vec.cuh"

#include "src/cuda_util.cuh"
#include "src/medium.cuh"
#include "src/params.hh"
#include "src/pixel.cuh"
#include "src/util.hh"

#include <curand_kernel.h>

__global__ void k_init(AgentVec agents) {
	auto idx = IDX_1;
	if (idx >= params::n_agents) return;

	curandState * rng = &agents.rng()[idx];
	curand_init(idx, 0, 0, rng);
	agents.xs()[idx] = (float)(curand(rng) % params::width);
	agents.ys()[idx] = (float)(curand(rng) % params::height);

	auto d = cu_util::unit_vec((float)(curand(rng) % 360) * M_PI / 180.0);
	agents.dxs()[idx] = d.x;
	agents.dys()[idx] = d.y;
}

__host__ void AgentVec::init() {
	k_init<<<params::agent_grid_dim, params::agent_block_dim>>>(*this);
	cudaError_t res = cudaGetLastError();
	if (res != cudaSuccess)
		throw Exception::format("Failed to initalize AgentVec, {}", res);
}


__global__ void k_step(AgentVec agents, Medium medium) {
	auto idx = IDX_1;
	if (idx >= params::n_agents) return;

	auto agent = agents.at(idx);

	auto l = agent.sense_l(medium);
	auto c = agent.sense_c(medium);
	auto r = agent.sense_r(medium);

	auto sl = l - params::hard_turn;
	auto sc = c - params::hard_turn;
	auto sr = r - params::hard_turn;

	// if ((agent.rand() & 0b1) == 0) {
	// 	agent.turn_rand();
	// } else
	if (sl > c && sl > r) {
		agent.turn_left();
	} else if (sr > c && sr > l) {
		agent.turn_right();
	} else if (sr > c || sl > c) {
		agent.turn_rand();
	} else {
		// Don't turn
	}

	agent.move();

	agents.store(agent, idx);
}

__host__ void AgentVec::step(Medium medium) {
	k_step<<<params::agent_grid_dim, params::agent_block_dim>>>(*this, medium);
	cudaError_t res = cudaGetLastError();
	if (res != cudaSuccess) throw Exception::format("Failed to step, {}", res);
}


__global__ void k_deposit(AgentVec agents, Medium medium) {
	auto idx = IDX_1;
	if (idx >= params::n_agents) return;

	auto c = int2{
	    static_cast<int>(agents.xs()[idx]), static_cast<int>(agents.ys()[idx])};
	medium.get(c)->deposit();
}

__host__ void AgentVec::deposit(Medium medium) {
	k_deposit<<<params::agent_grid_dim, params::agent_block_dim>>>(*this, medium);
	cudaError_t res = cudaGetLastError();
	if (res != cudaSuccess)
		throw Exception::format("Failed to deposit, {}", res);
}

inline __device__ float2 fadd(float2 a, float2 b) {
	return float2{a.x + b.x, a.y + b.y};
}

inline __device__ float2 fmul(float2 vec, float p) {
	return float2{vec.x * p, vec.y * p};
}

inline __device__ bool ieq(int2 lhs, int2 rhs) {
	return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}

// Plot a single point on the line, with brightness b
inline __device__ void plot_line_point(int2 tgt, Pixel * pixels, int pitch) {
	Pixel * out = cu_util::pitch_ptr(pixels, tgt, pitch);
	*out = color::blue;
}

// Plot a single point on the line, with brightness b
inline __device__ void plot_line_point(int x, int y, Pixel * pixels, int pitch) {
	plot_line_point(int2{x, y}, pixels, pitch);
}

template<typename T>
inline __device__ void swap(T & lhs, T & rhs) {
	T tmp = rhs;
	rhs = lhs;
	lhs = tmp;
}

// Bresenham's line algorithim
inline __device__ void line(int2 p0, int2 p1, Pixel * pixels, int pitch) {
	if (p0.x > p1.x) swap(p0, p1);
	int dx, dy, Po;
	int k = 0;
	dx = (p1.x - p0.x);
	dy = (p1.y - p0.y);
	if (dy <= dx && dy > 0) {
		dx = abs(dx);
		dy = abs(dy);
		Po = (2 * dy) - dx;
		plot_line_point(p0.x, p0.y, pixels, pitch);
		int xk = p0.x;
		int yk = p0.y;
		for (k = p0.x; k < p1.x; k++) {
			if (Po < 0) {
				plot_line_point(++xk, yk, pixels, pitch);
				Po = Po + (2 * dy);
			} else {
				plot_line_point(++xk, ++yk, pixels, pitch);
				Po = Po + (2 * dy) - (2 * dx);
			}
		}
	} else if (dy > dx && dy > 0) {
		dx = abs(dx);
		dy = abs(dy);
		Po = (2 * dx) - dy;
		plot_line_point(p0.x, p0.y, pixels, pitch);
		int xk = p0.x;
		int yk = p0.y;
		for (k = p0.y; k < p1.y; k++) {
			if (Po < 0) {
				plot_line_point(xk, ++yk, pixels, pitch);
				Po = Po + (2 * dx);
			} else {
				plot_line_point(++xk, ++yk, pixels, pitch);
				Po = Po + (2 * dx) - (2 * dy);
			}
		}
	} else if (dy >= -dx) {
		dx = abs(dx);
		dy = abs(dy);
		Po = (2 * dy) - dx;
		plot_line_point(p0.x, p0.y, pixels, pitch);
		int xk = p0.x;
		int yk = p0.y;
		for (k = p0.x; k < p1.x; k++) {
			if (Po < 0) {
				plot_line_point(++xk, yk, pixels, pitch);
				Po = Po + (2 * dy);
			} else {
				plot_line_point(++xk, --yk, pixels, pitch);
				Po = Po + (2 * dy) - (2 * dx);
			}
		}
	} else if (dy < -dx) {
		dx = abs(dx);
		dy = abs(dy);
		Po = (2 * dy) - dx;
		plot_line_point(p0.x, p0.y, pixels, pitch);
		int xk = p0.x;
		int yk = p0.y;
		for (k = p0.y; k > p1.y; k--) {
			if (Po < 0) {
				plot_line_point(xk, --yk, pixels, pitch);
				Po = Po + (2 * dx);
			} else {
				plot_line_point(++xk, --yk, pixels, pitch);
				Po = Po + (2 * dx) - (2 * dy);
			}
		}
	}
}

__global__ void k_render_dirs(AgentVec agents, Pixel * pixels, int pitch) {
	using cu_util::round_f2;

	auto idx = IDX_1;
	if (idx >= params::n_agents) return;

	auto agent = agents.at(idx);

	auto c = agent.coords();
	auto p0 = round_f2(c);
	auto p1 = round_f2(fadd(c, fmul(agent.dir(), params::agent_dir_hint_len)));
	line(p0, p1, pixels, pitch);
}

__host__ void AgentVec::render_dirs(Pixel * pixels, int pitch) {
	k_render_dirs<<<params::agent_grid_dim, params::agent_block_dim>>>(
	    *this, pixels, pitch);
	cudaError_t res = cudaGetLastError();
	if (res != cudaSuccess)
		throw Exception::format("Failed to render Agent vectors, {}", res);
}
