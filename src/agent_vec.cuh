#pragma once

#include "src/agent.cuh"
#include "src/cuda_util.cuh"
#include "src/exception.hh"
#include "src/medium.cuh"
#include "src/params.hh"
#include "src/pixel.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstddef>

class AgentVec {
public:
	static __host__ AgentVec alloc() {
		float *xs = nullptr, *ys = nullptr, *dxs = nullptr, *dys = nullptr;
		curandState * states;

		cudaError_t res;
		res = cudaMalloc(&xs, params::n_agents * sizeof(float));
		if (res != cudaSuccess)
			throw Exception::format("failed to allocated xs, {}", res);
		res = cudaMalloc(&ys, params::n_agents * sizeof(float));
		if (res != cudaSuccess)
			throw Exception::format("failed to allocated ys, {}", res);
		res = cudaMalloc(&dxs, params::n_agents * sizeof(float));
		if (res != cudaSuccess)
			throw Exception::format("failed to allocated dxs, {}", res);
		res = cudaMalloc(&dys, params::n_agents * sizeof(float));
		if (res != cudaSuccess)
			throw Exception::format("failed to allocated dys, {}", res);
		res = cudaMalloc(&states, params::n_agents * sizeof(curandState));
		if (res != cudaSuccess)
			throw Exception::format("failed to allocated ds, {}", res);

		llog::trace(
		    "allocated agents: {}, {}, {}, {}",
		    fmt::ptr(xs),
		    fmt::ptr(ys),
		    fmt::ptr(dxs),
		    fmt::ptr(dys),
		    fmt::ptr(states));

		return AgentVec{xs, ys, dxs, dys, states};
	}

	__device__ __host__ constexpr AgentVec(
	    float * xs, float * ys, float * dxs, float * dys, curandState * states) :
	    m_xs{xs}, m_ys{ys}, m_dxs{dxs}, m_dys{dys}, m_states{states} {}

	__device__ __host__ inline constexpr float * xs() { return m_xs; }
	__device__ __host__ inline constexpr float * ys() { return m_ys; }
	__device__ __host__ inline constexpr float * dxs() { return m_dxs; }
	__device__ __host__ inline constexpr float * dys() { return m_dys; }

	__device__ __host__ inline constexpr curandState * rng() {
		return m_states;
	}

	__device__ inline Agent at(size_t idx) {
		return Agent{m_xs[idx], m_ys[idx], m_dxs[idx], m_dys[idx]};
	}

	__device__ inline void store(Agent a, size_t idx) {
		m_xs[idx] = a.x();
		m_ys[idx] = a.y();
		m_dxs[idx] = a.dx();
		m_dys[idx] = a.dy();
	}

	__host__ void init();
	__host__ void step(Medium medium);
	__host__ void deposit(Medium medium);
	// Render directions of all agents
	__host__ void render_dirs(Pixel * pixels, int pitch);

private:
	float *       m_xs;
	float *       m_ys;
	float *       m_dxs;
	float *       m_dys;
	curandState * m_states;
};
