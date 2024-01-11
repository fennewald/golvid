#pragma once

#include "src/params.hh"

#include <cuda_runtime.h>

#include <cstdint>

namespace cu_util {

#define IDX_1 ((blockIdx.x * blockDim.x) + threadIdx.x)

#define IDX_2                                                         \
	int2 {                                                            \
		static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x),    \
		    static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y) \
	}

template<typename T>
__device__ inline T * pitch_ptr(T * base, int2 idx, int pitch) {
	uint32_t x = idx.x % params::width;
	uint32_t y = idx.y % params::height;
	return ((T *)((char *)base + (y * pitch))) + x;
}

template<typename T>
__device__ inline T * pitch_ptr_raw(T * base, int2 idx, int pitch) {
	return ((T *)((char *)base + (idx.y * pitch))) + idx.x;
}

}  // namespace cu_util
