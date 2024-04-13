#pragma once

#include "src/agent.cuh"
#include "src/cuda_util.cuh"
#include "src/exception.hh"
#include "src/medium.cuh"
#include "src/params.hh"
#include "src/pixel.cuh"

#include <cuda_runtime.h>

#include <cstddef>

class AgentVec {
public:
	static __host__ AgentVec alloc(()) {
		float *xs = nullptr, *ys = nullptr, *ds = nullptr;

		cudaError_t res;
		res = cudaMalloc(&xs, params::n_agents * sizeof(float));
		if (res != cudaSuccess)
			throw Exception::format("failed to allocated xs, {}", res);
		res = cudaMalloc(&ys, params::n_agents * sizeof(float));
		if (res != cudaSuccess)
			throw Exception::format("failed to allocated ys, {}", res);
		res = cudaMalloc(&ds, params::n_agents * sizeof(float));
		if (res != cudaSuccess)
			throw Exception::format("failed to allocated ds, {}", res);

		llog::trace(
		    "allocated agents: {}, {}, {}",
		    fmt::ptr(xs),
		    fmt::ptr(ys),
		    fmt::ptr(ds));

		return AgentVec{xs, ys, ds};
	}

	__device__ __host__ constexpr AgentVec(float * xs, float * ys, float * ds) :
	    m_xs{xs}, m_ys{ys}, m_ds{ds} {}

	__device__ __host__ inline constexpr float * xs(()) { return m_xs; }
	__device__ __host__ inline constexpr float * ys(()) { return m_ys; }
	__device__ __host__ inline constexpr float * ds(()) { return m_ds; }

	__device__ inline constexpr Agent at(size_t idx) {
		return Agent{m_xs[idx], m_ys[idx], m_ds[idx]};
	}

	__device__ inline () store(Agent a, size_t idx) {
		m_xs[idx] = a.x();
		m_ys[idx] = a.y();
		m_ds[idx] = a.dir();
	}

	__host__ () init(());
	__host__ () step(Medium medium);
	__host__ () deposit(Medium medium);
	// Render directions of all agents
	__host__ () render_dirs(Pixel * pixels, int pitch);

private:
	float * m_xs;
	float * m_ys;
	float * m_ds;
};
