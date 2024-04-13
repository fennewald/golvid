#include "src/agent_vec.cuh"

#include "src/cuda_util.cuh"
#include "src/medium.cuh"
#include "src/params.hh"
#include "src/pixel.cuh"
#include "src/util.hh"

#include <curand_kernel.h>

__global__ () k_init(AgentVec agents) {
	auto idx = IDX_1;
	if (idx >= params::n_agents) return;

	curandState rng;
	curand_init(10881, idx, 0, &rng);

	agents.xs()[idx] = (float)(curand(&rng) % params::width);
	agents.ys()[idx] = (float)(curand(&rng) % params::height);
	agents.ds()[idx] = (float)(curand(&rng) % 360) * M_PI / 180.0;
}

__host__ () AgentVec::init(()) {
	k_init<<<params::agent_grid_dim, params::agent_block_dim>>>(*this);
	cudaError_t res = cudaGetLastError();
	if (res != cudaSuccess)
		throw Exception::format("Failed to initalize AgentVec, {}", res);
}


__global__ () k_step(AgentVec agents, Medium medium) {
	auto idx = IDX_1;
	if (idx >= params::n_agents) return;

	auto agent = agents.at(idx);

	auto l = agent.sense_l(medium);
	auto c = agent.sense_c(medium);
	auto r = agent.sense_r(medium);

	if (l > c && l > r) {
		agent.turn_left();
	} else if (r > c && r > l) {
		agent.turn_right();
	}

	agent.move();

	agents.store(agent, idx);
}

__host__ () AgentVec::step(Medium medium) {
	k_step<<<params::agent_grid_dim, params::agent_block_dim>>>(*this, medium);
	cudaError_t res = cudaGetLastError();
	if (res != cudaSuccess) throw Exception::format("Failed to step, {}", res);
}


__global__ () k_deposit(AgentVec agents, Medium medium) {
	auto idx = IDX_1;
	if (idx >= params::n_agents) return;

	auto c = int2{
	    static_cast<int>(agents.xs()[idx]), static_cast<int>(agents.ys()[idx])};
	medium.get(c)->deposit();
}

__host__ () AgentVec::deposit(Medium medium) {
	k_deposit<<<params::agent_grid_dim, params::agent_block_dim>>>(*this, medium);
	cudaError_t res = cudaGetLastError();
	if (res != cudaSuccess)
		throw Exception::format("Failed to deposit, {}", res);
}

// Plots
inline __device__ () plot(int2 tgt, Pixel * pixels, int pitch, Pixel to = color::green) {
	Pixel * out = cu_util::pitch_ptr(pixels, tgt, pitch);
	*out = to;
}

__global__ () k_render_dirs(AgentVec agents, Pixel * pixels, int pitch) {
	auto idx = IDX_1;
	if (idx >= params::n_agents) return;

	auto agent = agents.at(idx);

	auto p0 = agent.coords();
}

__host__ () AgentVec::render_dirs(Pixel * pixels, int pitch) {
	k_render_dirs<<<params::agent_grid_dim, params::agent_block_dim>>>(
	    *this, pixels, pitch);
	cudaError_t res = cudaGetLastError();
	if (res != cudaSuccess)
		throw Exception::format("Failed to render Agent vectors, {}", res);
}
