#pragma once

#include "src/cell.cuh"
#include "src/cuda_util.cuh"
#include "src/exception.hh"
#include "src/params.hh"

#include <cuda_runtime.h>

class Medium {
public:
	static __host__ Medium alloc(void) {
		size_t pitch;
		Cell * cells;

		cudaError_t res = cudaMallocPitch(
		    &cells, &pitch, params::width * sizeof(Cell), params::height);
		if (res != cudaSuccess)
			throw Exception::format(
			    "Failed to allocate medium, {}: {}",
			    cudaGetErrorName(res),
			    cudaGetErrorString(res));

		return Medium{cells, pitch};
	}

	static __device__ __host__ constexpr bool contains(int2 c) {
		return (c.x >= 0) && (c.y >= 0) && (c.x < params::width) &&
		       (c.y < params::height);
	}

	__device__ __host__ constexpr Medium(Cell * cells, size_t pitch) :
	    m_cells{cells}, m_pitch{pitch} {}

	__device__ __host__ inline constexpr Cell * cells(void) { return m_cells; }
	__device__ __host__ inline constexpr size_t pitch(void) { return m_pitch; }

	__device__ inline Cell * get_raw(int2 idx) {
		return cu_util::pitch_ptr_raw(cells(), idx, pitch());
	}

	__device__ inline Cell * get(int2 idx) {
		return cu_util::pitch_ptr(cells(), idx, pitch());
	}

	__device__ inline Cell at(int2 idx) {
		if (!contains(idx)) return Cell::empty();
		return *cu_util::pitch_ptr_raw(cells(), idx, pitch());
	}

	// Returns the average of the neighborhood centered at int2
	__device__ Cell avg_at(int2 idx);

	__host__ void init(void);

private:
	Cell * m_cells;
	size_t m_pitch;
};
