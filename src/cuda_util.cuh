#pragma once

#include "src/params.hh"

#include <cuda_runtime.h>
#include <fmt/format.h>

namespace cu_util {

#define IDX_1 ((blockIdx.x * blockDim.x) + threadIdx.x)

#define IDX_2                                                         \
	int2 {                                                            \
		static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x),    \
		    static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y) \
	}

template<typename T>
__device__ inline T * pitch_ptr_raw(T * base, int2 idx, int pitch) {
	return base + (idx.y * pitch / sizeof(T)) + idx.x;
}

template<typename T>
__device__ inline T * pitch_ptr(T * base, int2 idx, int pitch) {
	idx.x %= params::width;
	idx.y %= params::height;
	while (idx.x < 0) idx.x += params::width;
	while (idx.y < 0) idx.y += params::height;
	return pitch_ptr_raw(base, idx, pitch);
}

__device__ int2 round_f2(float2 ns);

}  // namespace cu_util

template<>
struct fmt::formatter<cudaError_t> {
	constexpr auto parse(format_parse_context & ctx)
	    -> format_parse_context::iterator {
		auto it = ctx.begin();
		if (it != ctx.end() && *it != '}') throw_format_error("Invalid format");
		return it;
	}

	auto format(cudaError_t err, format_context & ctx) const
	    -> format_context::iterator {
		return fmt::format_to(
		    ctx.out(), "{}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
	}
};
