#include "src/medium.cuh"

#include "src/cuda_util.cuh"


__global__ void k_init(Medium m) {
	auto idx = IDX_2;
	if (m.contains(idx)) m.get_raw(idx)->clear();
}

__device__ Cell Medium::avg_at(int2 idx) {
	auto res = Cell{};

	static constexpr int kernel_dim = 3;
	static constexpr int n_cells = kernel_dim * kernel_dim;
	static constexpr int max_val = (kernel_dim - 1) / 2;
	static constexpr int min_val = max_val * -1;
	static_assert(kernel_dim % 2 == 1, "kernel dim must be odd");

#pragma unroll
	for (int dy = min_val; dy <= max_val; ++dy) {
		int y = idx.y + dy;
#pragma unroll
		for (int dx = min_val; dx <= max_val; ++dx) {
			int x = idx.x + dx;
			res += at({x, y});
		}
	}
	return res / n_cells;
}

__host__ void Medium::init(void) {
	k_init<<<params::cells_grid_dim, params::cells_block_dim>>>(*this);
}
