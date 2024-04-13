#include "src/cuda_util.cuh"

namespace cu_util {

__device__ int2 round_f2(float2 ns) {
	return int2{__float2int_rd(ns.x), __float2int_rd(ns.y)};
}

}  // namespace cu_util
